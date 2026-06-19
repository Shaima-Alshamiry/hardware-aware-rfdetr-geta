import os
import torch
import csv
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from rfdetr import RFDETRNano
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

# ==========================================
# 0. CUDNN STABILITY PATCH
# ==========================================
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda')
CPU = torch.device('cpu')

# ==========================================
# 1. OPTIMIZER & GRAPH PATCHES
# ==========================================
_orig_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' or antialias is True:
        mode = 'bilinear'; antialias = False
        if align_corners is None: align_corners = False
    return _orig_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

import torch.optim.optimizer
original_add_param_group = torch.optim.Optimizer.add_param_group
def robust_add_param_group(self, param_group):
    existing_ids = {id(p) for group in self.param_groups for p in group['params']}
    unique_params = [p for p in param_group['params'] if id(p) not in existing_ids]
    if not unique_params: return 
    param_group['params'] = unique_params
    return original_add_param_group(self, param_group)
torch.optim.Optimizer.add_param_group = robust_add_param_group 

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
# 2. DATASETS & CONFIG
# ==========================================
BATCH_SIZE = 4  
GRAD_ACCUM_STEPS = 4
START_EPOCH = 7      # 🌟 We start after the 8th epoch which completed
TARGET_EPOCHS = 50   # 🌟 The new target is 50 epochs
os.makedirs('./checkpoints', exist_ok=True)

IMAGES_DIR = "/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO"
TRAIN_ANN_FILE = "./coco_data/annotations/instances_train2017.json"

transforms_train = T.Compose([
    T.ToImage(), T.Resize((384, 384)), T.ToDtype(torch.float32, scale=True), 
    T.Normalize(mean=[0.485, 0.455, 0.406], std=[0.229, 0.224, 0.225])
])

class RFDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file); self._transforms = transforms
    def __getitem__(self, idx):
        img, target = super().__getitem__(int(idx))
        orig_w, orig_h = img.size 
        boxes, labels = [], []
        for obj in target:
            x_min, y_min, w, h = obj['bbox']
            boxes.append([max(0., min((x_min + w/2)/orig_w, 1.)), max(0., min((y_min + h/2)/orig_h, 1.)), max(1e-4, min(w/orig_w, 1.)), max(1e-4, min(h/orig_h, 1.))])
            labels.append(obj['category_id'])
        return self._transforms(img) if self._transforms else img, {"boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)), "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)}

def collate_fn(batch): return tuple(zip(*batch))
full_train_ds = RFDetrDataset(os.path.join(IMAGES_DIR, 'train2017'), TRAIN_ANN_FILE, transforms_train)
torch.manual_seed(42)

train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

# ==========================================
# 3. INITIALIZE MODEL & RESUME OTO
# ==========================================
print("▶ Preparing Model for Resuming Training...")

base_model = RFDETRNano().model.model

# First convert the model to the same structure (INT8) to ensure compatibility of the saved weights
quantized_model = model_to_quantize_model(
    base_model, 
    quant_mode=QuantizationMode.WEIGHT_ONLY,
    num_bits=8
).to(CPU).eval()

oto = OTO(model=quantized_model, dummy_input=torch.rand(1, 3, 384, 384, device=CPU))

# Setup the Optimizer with the same parameters
optimizer = oto.geta(
    variant="adamw", 
    lr=1e-5, 
    weight_decay=1e-4, 
    target_group_sparsity=0.50,
    group_divisible=16,            
    bit_reduction=8,
    start_projection_step=2 * len(train_loader),  
    projection_periods=3, 
    projection_steps=1 * len(train_loader), 
    start_pruning_step=5 * len(train_loader),     
    pruning_periods=3, 
    pruning_steps=1 * len(train_loader)
)

# 🌟 The atomic solution: load the weights and reset the Optimizer to ensure it runs
checkpoint_path = f'./checkpoints/geta_epoch_{START_EPOCH}.pth'
if os.path.exists(checkpoint_path):
    print(f"▶ Loading checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=CPU)
    
    # Load the model weights (this will work because the weights are compatible)
    quantized_model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded.")

    # Skip loading the Optimizer state because it causes conflicts
    print("⚠ Optimizer state skipped to avoid shape mismatch. Starting fresh optimizer momentum.")
    
    print(f"🚀 Resuming from Epoch {START_EPOCH + 1}...")
else:
    raise FileNotFoundError(f"❌ Checkpoint {checkpoint_path} not found.")

print("▶ Moving Model to GPU...")
quantized_model = quantized_model.to(DEVICE)

class RobustArgs:
    def __getattr__(self, name): return False
    def __init__(self): 
        self.num_classes = 91; self.dec_layers = 2; self.hidden_dim = 256
        self.num_queries = 300; self.num_select = 300; self.group_detr = 13
        self.eval_max_dets = 300; self.two_stage = True; self.aux_loss = True
        self.ia_bce_loss = True; self.device = "cuda"; self.cls_loss_coef = 1.0
        self.bbox_loss_coef = 5.0; self.giou_loss_coef = 2.0; self.focal_alpha = 0.25
        self.masks = False; self.set_cost_class = 1.0; self.set_cost_bbox = 5.0; self.set_cost_giou = 2.0
        
criterion = build_criterion_and_postprocessors(RobustArgs())[0].to(DEVICE)

# ==========================================
# 4. TRAINING LOOP (RESUME MODE)
# ==========================================
print(f"\n🚀 RESUMING: EPOCH {START_EPOCH+1} -> {TARGET_EPOCHS}")

for epoch in range(START_EPOCH, TARGET_EPOCHS):
    quantized_model.train(); criterion.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TARGET_EPOCHS}")
    epoch_loss = 0.0
    optimizer.zero_grad() 
    
    for step, (images, targets) in enumerate(pbar):
        valid = [i for i, t in enumerate(targets) if t['boxes'].numel() > 0]
        if not valid: continue
        imgs = torch.stack([images[i].to(DEVICE) for i in valid])
        tgts = [{k: v.to(DEVICE) for k, v in targets[i].items()} for i in valid]
        
        outputs = quantized_model(imgs)
        loss_dict = criterion(outputs, tgts)
        loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict if k in criterion.weight_dict) / GRAD_ACCUM_STEPS
            
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0: 
            torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), 0.1)
            optimizer.step(); optimizer.zero_grad() 
        epoch_loss += loss.item() * GRAD_ACCUM_STEPS

    print(f"✅ Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
    # Save the new checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': quantized_model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
    }, f'./checkpoints/geta_epoch_{epoch+1}.pth')

print("✅ Training Complete.")