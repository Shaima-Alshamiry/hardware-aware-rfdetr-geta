import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant  
import torchvision.transforms.v2 as T
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from rfdetr import RFDETRNano
import rfdetr.models.backbone.projector as projector

# ==========================================
# 1. LOCAL EXPORT PATCHES (To silence ONNX errors permanently)
# ==========================================
# 1.1 Bypass Bicubic error
_orig_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' or antialias is True:
        mode = 'bilinear'; antialias = False
        if align_corners is None: align_corners = False
    return _orig_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

# 1.2 Bypass LayerNorm error encountered in ONNX
def patched_layernorm_forward(self, x):
    x = x.permute(0, 2, 3, 1).float()
    mean = x.mean(dim=-1, keepdim=True); var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + self.eps); x = self.weight.float() * x + self.bias.float()
    return x.permute(0, 3, 1, 2)
projector.LayerNorm.forward = patched_layernorm_forward

# 1.3 Bypass dimension errors (ListConstruct)
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

# ==========================================
# 2. LOAD CLEAN MODEL
# ==========================================
print("▶ Loading Clean Pruned Model...")
DEVICE = torch.device('cpu')
# Load the full model object saved by script 2. The slimmed layer dimensions differ from
# a fresh RFDETRNano skeleton, so we cannot use load_state_dict here.
model = torch.load('clean_pruned10_rfdetr.pth', map_location=DEVICE, weights_only=False).to(DEVICE)

model.train() # Training mode to plant nodes

# ==========================================
# 3. EXPLICIT QDQ INJECTION (Partial Quantization)
# ==========================================
print("▶ Inserting ONNX-Compatible Explicit QDQ Nodes...")

# 1. إعادة مقاييس البيانات العابرة لكي لا ينهار سكريبت Jetson الخاص بكم
act_fq = quant.FakeQuantize.with_args(
    observer=quant.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
    quant_min=-128, quant_max=127, dtype=torch.qint8, 
    qscheme=torch.per_tensor_symmetric, reduce_range=False
)

weight_fq = quant.FakeQuantize.with_args(
    observer=quant.PerChannelMinMaxObserver,
    quant_min=-128, quant_max=127, dtype=torch.qint8, 
    qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0
)

model.qconfig = quant.QConfig(activation=act_fq, weight=weight_fq)

# 2. 🚨 الدرع السحري: حماية طبقات الماسك وكل طبقات LayerNorm من التكميم
for name, module in model.named_modules():
    # حماية الكلمات المفتاحية
    if any(k in name for k in ["out_proj", "input_proj", "query_embed", "class_embed", "bbox_embed", "mask", "seg", "decoder", "norm"]):
        module.qconfig = None
    # حماية صريحة لأي طبقة LayerNorm في الموديل بأكمله (لتجنب التقطيع البطئ)
    if isinstance(module, nn.LayerNorm):
        module.qconfig = None

quant.prepare_qat(model, inplace=True)
model.eval()

# ==========================================
# 4. CALIBRATION 
# ==========================================
transforms_val = T.Compose([T.ToImage(), T.Resize((384, 384)), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=[0.485, 0.455, 0.406], std=[0.229, 0.224, 0.225])])
class DummyCOCO(torchvision.datasets.CocoDetection):
    def __getitem__(self, idx): return transforms_val(super().__getitem__(idx)[0]), 0
val_ds = DummyCOCO("/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO/val2017", "./coco_data/annotations/instances_val2017.json")
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

CALIBRATION_BATCHES = 200
print(f"▶ Calibrating QDQ scales over {CALIBRATION_BATCHES} images...")
with torch.no_grad():
    for i, (images, _) in enumerate(tqdm(val_loader, total=CALIBRATION_BATCHES)):
        if i >= CALIBRATION_BATCHES: break
        model(images)

# ==========================================
# 5. FREEZE SCALES 
# ==========================================
print("▶ Freezing Scales as requested by developer...")
model.apply(quant.disable_observer)
model.apply(quant.enable_fake_quant)

# ==========================================
# 6. ONNX QDQ EXPORT
# ==========================================
ONNX_PATH = "rfdetr_f.onnx"
print(f"▶ Exporting Explicit QDQ ONNX to {ONNX_PATH}...")
example_input = torch.randn(1, 3, 384, 384, device=DEVICE)

torch.onnx.export(
    model, example_input, ONNX_PATH,
    export_params=True, 
    opset_version=17, 
    do_constant_folding=True,
    input_names=["input"], 
    output_names=["pred_logits", "pred_boxes"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "pred_logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    }
)
print("✅ SUCCESS! Production-Ready QDQ ONNX Exported Perfectly!")