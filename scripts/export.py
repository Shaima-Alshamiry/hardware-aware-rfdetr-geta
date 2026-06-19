import argparse
import torch
import torch.nn as nn
import rfdetr.models.backbone.projector as projector
from constants import INPUT_SIZE
from qdq_layers import NativeQDQConv2d, NativeQDQLinear  # noqa: F401 — needed for torch.load deserialization

# 1. Patch for LayerNorm to fix ONNX export compatibility issues.
# This manually calculates normalization to avoid unsupported ONNX ops for specific runtimes.
def _patched_layernorm_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1).float()
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + self.eps)
    x = self.weight.float() * x + self.bias.float()
    return x.permute(0, 3, 1, 2)

class ExportWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.inner(x)
        
        # Enforcing output formats (Boxes: [1, 100, 4], Logits: [1, 100, CLASSES], Masks: [1, 100, H, W])
        boxes = out["pred_boxes"].float()
        logits = out["pred_logits"].float()
        masks = out["pred_masks"].float()
        
        return boxes, logits, masks
        
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Native QDQ pruned RF-DETR to ONNX.")
    p.add_argument("--model", default="finetuned_rfdetr_vsu.pth",
                   help="Path to the QDQ pruned model saved by the previous step.")
    p.add_argument("--output", default="rfdetr_production_fvsu.onnx",
                   help="Path for the exported ONNX file.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    # Apply external patches for dynamic shapes and interpolation
    from patches import apply_interpolate_patch, apply_shape_patches
    apply_interpolate_patch()
    apply_shape_patches()

    # Activate the custom LayerNorm patch
    projector.LayerNorm.forward = _patched_layernorm_forward

    device = torch.device('cpu')

    print(f"▶ Loading QDQ Pruned Model from {args.model}...")
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    
    # Force the model to FP32 to ensure it matches the example_input dtype
    model = model.float()
    
    model.eval()

    print(f"▶ Exporting Native QDQ ONNX to {args.output}...")
    example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    export_model = ExportWrapper(model).eval()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Hides the harmless PyTorch Tracer warnings
        torch.onnx.export(
            export_model,
            example_input,
            args.output,
            export_params=True,
            opset_version=17, # Opset 17 is generally the most stable for modern QDQ nodes
            do_constant_folding=True,
            input_names=["input"],
            output_names=["dets", "labels", "masks"],
            # Disabling dynamic axes forces Static Shapes, which TensorRT prefers
            dynamic_axes=None 
        )
        
    print("✅ Model Exported with Static Shapes!")
    print("✅ SUCCESS! Production-Ready QDQ ONNX Exported!")

if __name__ == "__main__":
    main()