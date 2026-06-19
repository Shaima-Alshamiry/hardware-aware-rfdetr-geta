import torch
import torch.nn as nn
import torch.nn.functional as F  
from rfdetr import RFDETRNano
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

DEVICE = torch.device('cpu')

# ==========================================
# 1. PATCHES (To bypass ONNX errors)
# ==========================================
_orig_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' or antialias is True:
        mode = 'bilinear'; antialias = False
        if align_corners is None: align_corners = False
    return _orig_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

def force_static(s): 
    if hasattr(s, 'item'): return int(s.item())
    try: return int(s)
    except: return s

_original_view = torch.Tensor.view
torch.Tensor.view = lambda self, *shape: _original_view(self, *[force_static(s) for s in (shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)])

_original_reshape = torch.Tensor.reshape
torch.Tensor.reshape = lambda self, *shape: _original_reshape(self, *[force_static(s) for s in (shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)])

_orig_torch_reshape = torch.reshape
torch.reshape = lambda input, shape: _orig_torch_reshape(input, [force_static(s) for s in shape])

import only_train_once.transform.tensor_transform as tensor_transform
_orig_basic = tensor_transform.basic_transformation
def patched_basic_transformation(tensor, num_groups):
    if tensor.numel() % num_groups != 0: return torch.ones((num_groups, 1), device=tensor.device)
    return _orig_basic(tensor, num_groups)
tensor_transform.basic_transformation = patched_basic_transformation

# ==========================================
# 2. LOAD, COMMIT & CLEAN
# ==========================================
print("▶ Phase 1: Rebuilding OTO Graph on CPU...")
model = model_to_quantize_model(RFDETRNano().model.model, quant_mode=QuantizationMode.WEIGHT_ONLY).to(DEVICE)

# Defining OTO here is sufficient to apply the GETA structure
oto = OTO(model=model, dummy_input=torch.randn(1, 3, 384, 384))

print("▶ Phase 2: Loading Trained Checkpoint...")
checkpoint = torch.load('./checkpoints/geta_epoch_15_int8.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# construct_subnet identifies zero-norm groups from the trained weights and physically
# slices them out (prune_out_dim + prune_in_dim), producing truly smaller tensors.
# Without this step the pruned channels are zero but layer dimensions are unchanged,
# so the exported ONNX runs at full model size with no throughput benefit.
print("▶ Phase 3: Committing Structural Pruning (construct_subnet)...")
oto.construct_subnet(out_dir='./checkpoints/')

print("▶ Phase 4: Stripping GETA Wrappers...")
def strip_geta_wrappers(module):
    for name, child in module.named_children():
        if child.__class__.__name__ in ["QuantizeLinear", "QuantizeConv2d"]:
            if "Linear" in child.__class__.__name__:
                new = nn.Linear(child.in_features, child.out_features, bias=(child.bias is not None))
            else:
                new = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size, 
                                stride=child.stride, padding=child.padding, groups=child.groups, 
                                dilation=child.dilation, bias=(child.bias is not None))
            
            # Transfer weights to clean layers
            new.weight.data = child.weight.data.clone()
            if child.bias is not None: new.bias.data = child.bias.data.clone()
            setattr(module, name, new)
        else: strip_geta_wrappers(child)

strip_geta_wrappers(model)

# Save full model object (not just state dict) so the slimmed architecture is preserved.
# Script 3 loads this with torch.load() and must not create a fresh RFDETRNano skeleton.
torch.save(model, 'clean_pruned50_rfdetr.pth')
print("✅ SUCCESS: Clean slimmed model saved as 'clean_pruned_rfdetr.pth'")