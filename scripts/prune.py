import argparse
import torch
import torch.nn as nn
from constants import INPUT_SIZE
from only_train_once import OTO
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model
from rfdetr import RFDETRSegNano

def strip_geta_wrappers(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if child.__class__.__name__ in ("QuantizeLinear", "QuantizeConv2d"):
            
            with torch.no_grad():
                if hasattr(child, 'quantize_weight'):
                    # 🔴 Modification here: passing child.weight inside the function 🔴
                    fused_weight = child.quantize_weight(child.weight).detach()
                else:
                    fused_weight = child.weight.detach()

            bias = child.bias.detach() if child.bias is not None else None
            
            if "Linear" in child.__class__.__name__:
                new_layer = nn.Linear(child.in_features, child.out_features, bias=(bias is not None))
            else:
                new_layer = nn.Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    groups=child.groups, dilation=child.dilation,
                    bias=(bias is not None)
                )
            
            new_layer = new_layer.to(fused_weight.dtype)
            new_layer.weight.data.copy_(fused_weight)
            if bias is not None:
                new_layer.bias.data.copy_(bias)
            
            setattr(module, name, new_layer)
        else:
            strip_geta_wrappers(child)
    return module
    
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Commit structural pruning and strip GETA wrappers.")
    p.add_argument("--checkpoint", default="./checkpoints_fp16/geta_best.pth")
    p.add_argument("--checkpoint-dir", default="./checkpoints_fp16")
    p.add_argument("--output", default="clean_pruned_rfdetr_vsu.pth")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    from patches import apply_interpolate_patch, apply_layer_norm_patch, apply_oto_patch, apply_shape_patches
    apply_interpolate_patch()
    apply_layer_norm_patch()
    apply_shape_patches()
    apply_oto_patch()

    device = torch.device('cpu')

    print("▶ Phase 1: Rebuilding OTO Graph...")
    model = model_to_quantize_model(
        RFDETRSegNano().model.model, quant_mode=QuantizationMode.WEIGHT_ONLY
    ).to(device)

    # Disable Quantization for sensitive parts
    for name, module in model.named_modules():
        if any(k in name for k in ['class_embed', 'bbox_embed', 'query_embed', 'attention']):
            if hasattr(module, 'set_quant_state'):
                module.set_quant_state(False, False)

    oto = OTO(model=model, dummy_input=torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))

    print(f"▶ Phase 2: Loading Checkpoint: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)

    print("▶ Phase 3: Committing Structural Pruning...")
    oto.construct_subnet(out_dir=args.checkpoint_dir)

    print("▶ Phase 4: Stripping GETA Wrappers using Quantized Weights...")
    strip_geta_wrappers(model)

    torch.save(model, args.output)
    print(f"✅ SUCCESS: Clean slimmed model saved as '{args.output}'")

if __name__ == "__main__":
    main()