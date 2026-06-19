import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def quantize_weights_dq_only(input_path: str, output_path: str) -> None:
    """
    Replaces QuantizeLinear(weight) + DequantizeLinear pairs with DQ-only nodes
    backed by pre-quantized INT8 weight initializers.

    torch.onnx.export produces float32_weight → Q → DQ → Conv, which TensorRT's
    compiler backend fails to fuse. TensorRT's canonical INT8 weight format is
    int8_weight → DQ → Conv (DQ-only). This script performs that conversion using
    the per-channel scales already embedded in the ONNX graph.
    """
    model = onnx.load(input_path)
    graph = model.graph

    initializer_map = {init.name: init for init in graph.initializer}

    # Count how many nodes reference each initializer so we don't remove shared ones
    usage_count: dict[str, int] = {}
    for node in graph.node:
        for inp in node.input:
            usage_count[inp] = usage_count.get(inp, 0) + 1

    nodes_to_remove = []
    dq_input_remap: dict[str, str] = {}
    new_initializers = []
    fp32_inits_to_remove: set[str] = set()
    q_tensors_to_remove: set[str] = set()
    replaced = 0

    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if not node.input or node.input[0] not in initializer_map:
            continue  # Activation quantization — leave untouched

        fp32_name = node.input[0]
        scale_name = node.input[1]
        zp_name = node.input[2] if len(node.input) > 2 else ""

        if scale_name not in initializer_map:
            continue

        w_fp32 = numpy_helper.to_array(initializer_map[fp32_name]).astype(np.float32)
        scale = numpy_helper.to_array(initializer_map[scale_name]).astype(np.float32)
        zp = np.zeros(scale.shape, dtype=np.int8)
        if zp_name and zp_name in initializer_map:
            zp = numpy_helper.to_array(initializer_map[zp_name]).astype(np.int8)

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)

        # Per-channel quantization along the specified axis
        broadcast_shape = [1] * w_fp32.ndim
        broadcast_shape[axis] = -1
        scale_bc = scale.reshape(broadcast_shape) if scale.size > 1 else scale
        zp_bc = zp.reshape(broadcast_shape) if zp.size > 1 else zp
        w_int8 = np.clip(np.round(w_fp32 / scale_bc) + zp_bc, -128, 127).astype(np.int8)

        int8_name = fp32_name + "_int8"
        new_initializers.append(numpy_helper.from_array(w_int8, name=int8_name))

        # Remap downstream DQ node to use the INT8 initializer directly
        dq_input_remap[node.output[0]] = int8_name
        q_tensors_to_remove.add(node.output[0])
        nodes_to_remove.append(node)

        if usage_count.get(fp32_name, 0) == 1:
            fp32_inits_to_remove.add(fp32_name)

        replaced += 1

    # Patch DequantizeLinear inputs
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in dq_input_remap:
            node.input[0] = dq_input_remap[node.input[0]]

    # Remove QuantizeLinear nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Clean up stale value_info entries for the removed Q output tensors
    for vi in list(graph.value_info):
        if vi.name in q_tensors_to_remove:
            graph.value_info.remove(vi)

    # Remove superseded FP32 weight initializers
    for init in list(graph.initializer):
        if init.name in fp32_inits_to_remove:
            graph.initializer.remove(init)

    graph.initializer.extend(new_initializers)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Converted {replaced} weight Q+DQ pairs to DQ-only with INT8 initializers")
    print(f"Saved to {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Q+DQ weight pairs to DQ-only INT8 initializers for TensorRT."
    )
    p.add_argument("--input", default="rfdetr_production.onnx")
    p.add_argument("--output", default="rfdetr_production_int8w.onnx")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    quantize_weights_dq_only(args.input, args.output)
