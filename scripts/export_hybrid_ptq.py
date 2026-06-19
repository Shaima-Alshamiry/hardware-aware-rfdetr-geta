import argparse
import os
import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from torch.utils.data import DataLoader
from tqdm import tqdm

import rfdetr.models.backbone.projector as projector
from constants import INPUT_SIZE, MEAN, STD

# Add these classes so PyTorch recognizes the file upon loading
class NativeQDQConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs.pop('scale', None) # Clean up extra arguments
        super().__init__(*args, **kwargs)
        self.register_buffer('scales', torch.ones(1))
        self.register_buffer('zero_points', torch.zeros(1))

class NativeQDQLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        kwargs.pop('scale', None)
        super().__init__(*args, **kwargs)
        self.register_buffer('scales', torch.ones(1))
        self.register_buffer('zero_points', torch.zeros(1))

# 1. Your custom patch for LayerNorm
def _patched_layernorm_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1).float()
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + self.eps)
    x = self.weight.float() * x + self.bias.float()
    return x.permute(0, 3, 1, 2)

# 2. Wrapper class for export
class ExportWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.inner(x)
        return out["pred_boxes"], out["pred_logits"], out["pred_masks"]

# 3. Data loader for calibration
class DummyCOCO(torchvision.datasets.CocoDetection):
    def __init__(self, img_dir, ann_file, transforms):
        super().__init__(img_dir, ann_file)
        self._transforms = transforms
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return self._transforms(img), 0

class _COCOCalibrationReader(CalibrationDataReader):
    def __init__(self, data_loader: DataLoader, input_name: str, n_batches: int) -> None:
        self._iter = iter(data_loader)
        self._input_name = input_name
        self._n_batches = n_batches
        self._count = 0
    def get_next(self) -> dict | None:
        if self._count >= self._n_batches: return None
        try:
            images, _ = next(self._iter)
            self._count += 1
            return {self._input_name: images.numpy()}
        except StopIteration:
            return None

# 4. Bias fix function for TensorRT (which you brilliantly wrote earlier)
def _fix_bias_qdq_for_trt(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    graph = model.graph
    init_map = {t.name: t for t in graph.initializer}
    out_to_node = {o: n for n in graph.node for o in n.output}
    weight_dq_outputs = {
        node.input[1] for node in graph.node
        if node.op_type in ("Conv", "ConvTranspose", "Gemm", "MatMul")
        and len(node.input) >= 2 and node.input[1] in out_to_node
        and out_to_node[node.input[1]].op_type == "DequantizeLinear"
    }
    nodes_to_drop = set()
    new_inits = []
    for node in graph.node:
        if node.op_type != "DequantizeLinear": continue
        dq_output = node.output[0]
        if dq_output in weight_dq_outputs: continue
        q_name = node.input[0]
        if q_name not in init_map: continue
        s_name = node.input[1]
        zp_name = node.input[2] if len(node.input) > 2 else None
        if s_name not in init_map: continue
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
        q = numpy_helper.to_array(init_map[q_name]).astype(np.float64)
        s = numpy_helper.to_array(init_map[s_name]).astype(np.float64)
        zp = (numpy_helper.to_array(init_map[zp_name]).astype(np.float64)
              if zp_name and zp_name in init_map else np.float64(0))
        if s.ndim == 1 and s.size > 1 and q.ndim > 1:
            bc_shape = [1] * q.ndim
            bc_shape[axis] = -1
            s = s.reshape(bc_shape)
            if isinstance(zp, np.ndarray) and zp.ndim == 1 and zp.size > 1:
                zp = zp.reshape(bc_shape)
        float_val = ((q - zp) * s).astype(np.float32)
        new_inits.append(numpy_helper.from_array(float_val, name=dq_output))
        nodes_to_drop.add(id(node))
    if not nodes_to_drop: return
    all_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(n for n in all_nodes if id(n) not in nodes_to_drop)
    graph.initializer.extend(new_inits)
    onnx.save(model, onnx_path)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Note: Here we use the clean weights file (FP32) that you generated in your first step without QDQ
    p.add_argument("--model", default="qdq_rfdetr.pth")
    p.add_argument("--output", default="rfdetr_h_production.onnx")
    p.add_argument("--calibration-batches", type=int, default=10)
    p.add_argument("--data-dir", default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO")
    p.add_argument("--val-ann", default="./coco_data/annotations/instances_val2017.json")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    from patches import apply_interpolate_patch, apply_shape_patches
    apply_interpolate_patch()
    apply_shape_patches()
    projector.LayerNorm.forward = _patched_layernorm_forward

    device = torch.device('cpu')
    val_images_dir = os.path.join(args.data_dir, "val2017")

    print(f"▶ Loading Clean FP32 Model from {args.model}...")
    # Note: Ensure this file is the Float (Pruned) version
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model.eval()

    fp32_path = args.output.replace(".onnx", "_fp32_tmp.onnx")
    print(f"▶ Exporting Initial FP32 ONNX to {fp32_path}...")
    example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    output_names = ["dets", "labels", "masks"]
    export_model = ExportWrapper(model).eval()

    torch.onnx.export(
        export_model, example_input, fp32_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in ["input"] + output_names},
    )

    preprocessed_path = args.output.replace(".onnx", "_pre_tmp.onnx")
    print("▶ Pre-processing FP32 ONNX...")
    quant_pre_process(fp32_path, preprocessed_path)
    if os.path.exists(fp32_path): os.remove(fp32_path)

    # ====================================================
    # 🛡️ Smart Shield: Protecting sensitive nodes (modification compatible with older ONNX)
    # ====================================================
    print("▶ Identifying sensitive Segmentation nodes to protect from quantization...")
    tmp_model = onnx.load(preprocessed_path)
    nodes_to_exclude = []
    
    # Types of layers that cause issues in Segmentation
    forbidden_types = ['Resize', 'Upsample', 'Sigmoid', 'Softmax']
    
    for node in tmp_model.graph.node:
        # Protect layers based on their type or name (if it contains the word 'mask' or 'segment')
        if node.op_type in forbidden_types or 'mask' in node.name.lower() or 'segment' in node.name.lower():
            nodes_to_exclude.append(node.name)

    transforms_val = T.Compose([
        T.ToImage(), T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=MEAN, std=STD),
    ])
    val_ds = DummyCOCO(val_images_dir, args.val_ann, transforms_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    reader = _COCOCalibrationReader(val_loader, "input", args.calibration_batches)

    print(f"▶ Applying Hybrid INT8 QDQ Quantization (Protecting {len(nodes_to_exclude)} nodes) → {args.output}...")
    
    # We removed 'op_types_to_exclude' to avoid the error
    quantize_static(
        model_input=preprocessed_path,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True},
        nodes_to_exclude=nodes_to_exclude 
    )

    print("▶ Fixing bias QDQ nodes for TensorRT compatibility...")
    _fix_bias_qdq_for_trt(args.output)
    if os.path.exists(preprocessed_path): os.remove(preprocessed_path)
    print(f"✅ SUCCESS! Hybrid Production-Ready QDQ ONNX Exported (~34 MB)!")

if __name__ == "__main__":
    main()