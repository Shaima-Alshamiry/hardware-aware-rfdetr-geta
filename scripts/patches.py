"""Shared monkey-patches required for GETA-based training and ONNX export."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def force_static(s: Any) -> Any:
    """Convert a potentially dynamic tensor dimension to a static Python int."""
    if hasattr(s, 'item'):
        return int(s.item())
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def _coerce_shape(shape: tuple) -> list:
    flat = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return [force_static(s) for s in flat]


def apply_interpolate_patch() -> None:
    """Replace bicubic/antialias interpolation with bilinear for ONNX compatibility."""
    _orig = F.interpolate

    def _patched(
        x: torch.Tensor,
        size=None,
        scale_factor=None,
        mode: str = 'nearest',
        align_corners=None,
        recompute_scale_factor=None,
        antialias: bool = False,
    ) -> torch.Tensor:
        if mode == 'bicubic' or antialias:
            mode = 'bilinear'
            antialias = False
            if align_corners is None:
                align_corners = False
        return _orig(x, size, scale_factor, mode, align_corners,
                     recompute_scale_factor, antialias)

    F.interpolate = _patched


def apply_shape_patches() -> None:
    """Force dynamic tensor dimensions to static ints for ONNX tracing."""
    _orig_view = torch.Tensor.view
    _orig_reshape_method = torch.Tensor.reshape
    _orig_reshape_fn = torch.reshape

    def _view(self: torch.Tensor, *shape: Any) -> torch.Tensor:
        return _orig_view(self, *_coerce_shape(shape))

    def _reshape_method(self: torch.Tensor, *shape: Any) -> torch.Tensor:
        return _orig_reshape_method(self, *_coerce_shape(shape))

    def _reshape_fn(tensor: torch.Tensor, shape: Any) -> torch.Tensor:
        return _orig_reshape_fn(tensor, [force_static(s) for s in shape])

    torch.Tensor.view = _view
    torch.Tensor.reshape = _reshape_method
    torch.reshape = _reshape_fn


def apply_oto_patch() -> None:
    """Guard OTO against misaligned group sizes and stale node IDs in graph post-processing."""
    import only_train_once.transform.tensor_transform as tensor_transform
    _orig = tensor_transform.basic_transformation

    def _patched(tensor, num_groups):
        if tensor.numel() % num_groups != 0:
            return torch.ones((num_groups, 1), device=tensor.device)
        return _orig(tensor, num_groups)

    tensor_transform.basic_transformation = _patched

    import only_train_once.graph.graph as graph_module

    def _post_process_for_quantize_linear(self):
        # Replicated from OTO source with one added guard (marked # GUARD) to skip
        # QuantizeLinear nodes whose downstream gemm node was removed from self.nodes
        # during ONNX graph optimisation. This happens with RF-DETR's attention/matmul
        # subgraph, where _find_closest_node_outgoing traverses into topology that
        # OTO's parser leaves out of self.nodes (NodePattern matmul/transpose None).
        class QuantizeLinear:
            pass

        quantize_linear_to_linear = dict()
        linear_to_quantize_linear = dict()
        from only_train_once.graph.utils import (
            _find_closest_node_outgoing,
            _find_nodes_between_start_end_nodes,
        )

        for node in self.nodes.values():
            if node.op_name == "lessorequal" and type(node.op.module).__name__ in [
                "QuantizeLinear",
                "BertAttention",
            ]:
                if self.incoming(node)[0].op.module is None:
                    abs_node = self.incoming(node)[0]
                    abs_node.op.module = QuantizeLinear()
                    common_node = self.incoming(abs_node)[0]
                    for child_node in self.outgoing(common_node):
                        if child_node.op_name == "sign":
                            child_node.op.module = QuantizeLinear()

        for node in self.nodes.values():
            if node.op is None:
                continue
            if type(node.op.module).__name__ in [
                "QuantizeLinear",
                "BertAttention",
                "LlamaAttention",
                "SimpleViTAttention",
                "ViTAttention",
                "PhiMHA",
            ] and (
                len(node.param_names) == 0 or "LayerNorm" not in node.param_names[0]
            ):
                node_linear = _find_closest_node_outgoing(
                    self, node, "gemm", quantize_linear_to_linear
                )
                if node_linear is None or node_linear.id not in self.nodes:  # GUARD
                    continue
                if node_linear.id not in linear_to_quantize_linear:
                    linear_to_quantize_linear[node_linear.id] = list()
                linear_to_quantize_linear[node_linear.id].append(node)

        for linear_node_id in linear_to_quantize_linear:
            if linear_node_id not in self.nodes:  # GUARD: removed as connected node in a prior iteration
                continue
            linear_node = self.nodes[linear_node_id]
            quant_linear_nodes = linear_to_quantize_linear[linear_node_id]
            connected_nodes = _find_nodes_between_start_end_nodes(
                self, quant_linear_nodes, linear_node
            )
            for node in quant_linear_nodes:
                if len(node.param_names) == 0:
                    start_node = self.incoming(node)[0]
                    self.add_edge(start_node, linear_node)
                    break
            all_param_names = set(
                sum([node.param_names for node in quant_linear_nodes], [])
            )
            temp_cfg_params = linear_node.op.cfg_params
            linear_node.op_name = "QuantizeLinear"
            linear_node.op = quant_linear_nodes[0].op
            linear_node.op.cfg_params = temp_cfg_params
            linear_node.param_names = list(all_param_names)
            for node_to_remove in connected_nodes:
                if node_to_remove.id == linear_node.id:
                    continue
                self.remove(node_to_remove)
                for node_group in self.op_name_to_node_group_comp_op.values():
                    if node_group.contain_node(node_to_remove):
                        node_group.remove_node(node_to_remove)

    graph_module.Graph._post_process_for_quantize_linear = _post_process_for_quantize_linear

    def _post_process_for_quantize_conv2d(self):
        # Replicated from OTO source with the same two guards as the linear version.
        class QuantizeConv2d:
            pass

        quantize_conv2d_to_conv2d = dict()
        conv2d_to_quantize_conv2d = dict()
        from only_train_once.graph.utils import (
            _find_closest_node_outgoing,
            _find_nodes_between_start_end_nodes,
        )

        for node in self.nodes.values():
            if (
                node.op_name == "lessorequal"
                and type(node.op.module).__name__ == "QuantizeConv2d"
            ):
                if self.incoming(node)[0].op.module is None:
                    abs_node = self.incoming(node)[0]
                    abs_node.op.module = QuantizeConv2d()
                    common_node = self.incoming(abs_node)[0]
                    for child_node in self.outgoing(common_node):
                        if child_node.op_name == "sign":
                            child_node.op.module = QuantizeConv2d()

        for node in self.nodes.values():
            if node.op is None:
                continue
            if type(node.op.module).__name__ == "QuantizeConv2d":
                node_conv2d = _find_closest_node_outgoing(
                    self, node, "conv", quantize_conv2d_to_conv2d
                )
                quantize_conv2d_to_conv2d[node.id] = node_conv2d
                if node_conv2d is None or node_conv2d.id not in self.nodes:  # GUARD
                    continue
                if node_conv2d.id not in conv2d_to_quantize_conv2d:
                    conv2d_to_quantize_conv2d[node_conv2d.id] = list()
                conv2d_to_quantize_conv2d[node_conv2d.id].append(node)

        for conv2d_node_id in conv2d_to_quantize_conv2d:
            if conv2d_node_id not in self.nodes:  # GUARD
                continue
            conv2d_node = self.nodes[conv2d_node_id]
            quant_conv2d_nodes = conv2d_to_quantize_conv2d[conv2d_node_id]
            connected_nodes = _find_nodes_between_start_end_nodes(
                self, quant_conv2d_nodes, conv2d_node
            )
            for node in quant_conv2d_nodes:
                if len(node.param_names) == 0:
                    start_node = self.incoming(node)[0]
                    self.add_edge(start_node, conv2d_node)
                    break
            all_param_names = set(
                sum([node.param_names for node in quant_conv2d_nodes], [])
            )
            temp_cfg_params = conv2d_node.op.cfg_params
            conv2d_node.op_name = "quantizeconv2d"
            conv2d_node.op = quant_conv2d_nodes[0].op
            conv2d_node.op.cfg_params = temp_cfg_params
            conv2d_node.param_names = list(all_param_names)
            for node_to_remove in connected_nodes:
                if node_to_remove.id == conv2d_node.id:
                    continue
                self.remove(node_to_remove)
                for node_group in self.op_name_to_node_group_comp_op.values():
                    if node_group.contain_node(node_to_remove):
                        node_group.remove_node(node_to_remove)

    graph_module.Graph._post_process_for_quantize_conv2d = _post_process_for_quantize_conv2d


def apply_layer_norm_patch() -> None:
    """Force F.layer_norm normalized_shape to static ints for ONNX tracing."""
    _orig = F.layer_norm

    def _patched(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        return _orig(input, [force_static(s) for s in normalized_shape], weight, bias, eps)

    F.layer_norm = _patched


def apply_optimizer_patch() -> None:
    """Skip duplicate parameters when adding a param group to an optimizer.

    Filters both params AND p_names together so their lists stay in sync.
    Filtering only params (without matching p_names) would cause zip() to pair
    the wrong name with the wrong tensor, leading to shape mismatches in
    GETA's gradient_descent_step.
    """
    _orig = torch.optim.Optimizer.add_param_group

    def _robust(self, param_group):
        existing_ids = {id(p) for group in self.param_groups for p in group['params']}
        if 'p_names' in param_group:
            keep = [(n, p) for n, p in zip(param_group['p_names'], param_group['params'])
                    if id(p) not in existing_ids]
            if not keep:
                return
            param_group['p_names'] = [n for n, _ in keep]
            param_group['params'] = [p for _, p in keep]
        else:
            unique = [p for p in param_group['params'] if id(p) not in existing_ids]
            if not unique:
                return
            param_group['params'] = unique
        return _orig(self, param_group)

    torch.optim.Optimizer.add_param_group = _robust
