import argparse
import torch
import torch.nn as nn
import rfdetr.models.backbone.projector as projector
from constants import INPUT_SIZE
from only_train_once import OTO
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model
from rfdetr import RFDETRSegNano

# 1. Patch LayerNorm to ensure clean ONNX tracing without unsupported operations
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
        boxes = out["pred_boxes"].float()
        logits = out["pred_logits"].float()
        masks = out["pred_masks"].float()
        return boxes, logits, masks

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-side End-to-End Pruning & Clean ONNX Export.")
    parser.add_argument("--checkpoint", default="clean_pruned_rfdetr_lv.pth")
    parser.add_argument("--output", default="rfdetr_production_lv.onnx")
    args = parser.parse_args()

    from patches import apply_interpolate_patch, apply_shape_patches
    apply_interpolate_patch()
    apply_shape_patches()

    projector.LayerNorm.forward = _patched_layernorm_forward
    device = torch.device('cpu')

    print("▶ Step 1: Initializing OTO Architecture...")
    model = model_to_quantize_model(
        RFDETRSegNano().model.model, quant_mode=QuantizationMode.WEIGHT_ONLY
    ).to(device)

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)

    print(f"▶ Step 2: Loading Checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)

    print("▶ Step 3: Committing Structural Pruning Subnet...")
    oto.construct_subnet()
    model.eval()
    model = model.float()

    print(f"▶ Step 4: Exporting Clean Production-Ready ONNX to {args.output}...")
    export_model = ExportWrapper(model).eval()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            export_model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["dets", "labels", "masks"],
            dynamic_axes=None
        )

    print(f"✅ SUCCESS: Complete cluster-side processing finished. Exported file: {args.output}")

if __name__ == "__main__":
    main()
