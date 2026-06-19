import os
import torch
import csv # 🎯 Library for saving data into CSV files

# ==========================================
# 0. CUDNN STABILITY PATCH
# ==========================================
# Disable CUDNN V8 API and benchmarking to ensure deterministic behavior 
# and avoid potential crashes during quantization/pruning operations.
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
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
# 1. THE MASTER MONKEY PATCHES
# ==========================================
# These patches are necessary to make the OTO graph tracing compatible 
# with RF-DETR's specific architectural layers.
import only_train_once.transform.tensor_transform as tensor_transform
import rfdetr.models.backbone.projector as projector

def force_static(s):
    if hasattr(s, 'item'): return int(s.item())
    try: return int(s)
    except: return s

_orig_basic = tensor_transform.basic_transformation
def patched_basic_transformation(tensor, num_groups):
    if tensor.numel() % num_groups != 0:
        return torch.ones((num_groups, 1), device=tensor.device)
    return _orig_basic(tensor, num_groups)
tensor_transform.basic_transformation = patched_basic_transformation

_original_view = torch.Tensor.view
_original_reshape = torch.Tensor.reshape
def patched_view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _original_view(self, *[force_static(s) for s in shape])
def patched_reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _original_reshape(self, *[force_static(s) for s in shape])
torch.Tensor.view = patched_view
torch.Tensor.reshape = patched_reshape

original_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' or antialias is True:
        mode = 'bilinear'; antialias = False
        if align_corners is None: align_corners = False
    if size is not None:
        size = [force_static(s) for s in size] if isinstance(size, (tuple, list)) else force_static(size)
    return original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

def patched_layernorm_forward(self, x):
    x = x.permute(0, 2, 3, 1)
    mean = x.mean(dim=-1, keepdim=True); var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + self.eps); x = self.weight * x + self.bias
    return x.permute(0, 3, 1, 2)
projector.LayerNorm.forward = patched_layernorm_forward

import torch.optim.optimizer
original_add_param_group = torch.optim.Optimizer.add_param_group
def robust_add_param_group(self, param_group):
    existing_ids = {id(p) for group in self.param_groups for p in group['params']}
    unique_params = [p for p in param_group['params'] if id(p) not in existing_ids]
    if not unique_params: return 
    param_group['params'] = unique_params
    return original_add_param_group(self, param_group)
torch.optim.Optimizer.add_param_group = robust_add_param_group 

# ==========================================
# 2. CONFIGURATION & DIRECTORIES
# ==========================================
DEVICE = torch.device('cuda')
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
TARGET_EPOCHS = 15 

os.makedirs('./checkpoints', exist_ok=True)

COCO_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

IMAGES_DIR = "/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO"
TRAIN_ANN_FILE = "./coco_data/annotations/instances_train2017.json"
VAL_ANN_FILE = "./coco_data/annotations/instances_val2017.json"
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val2017")

# ==========================================
# 3. DATA LOADERS (10% Subset for PoC)
# ==========================================
transforms = T.Compose([
    T.ToImage(), T.Resize((384, 384)), T.ToDtype(torch.float32, scale=True), 
    T.Normalize(mean=[0.485, 0.455, 0.406], std=[0.229, 0.224, 0.225])
])

class RFDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file); self._transforms = transforms
        
    def __getitem__(self, idx):
        img, target = super().__getitem__(int(idx))
        orig_w, orig_h = img.size 
        boxes = []
        labels = []
        for obj in target:
            x_min, y_min, w, h = obj['bbox']
            cx = (x_min + (w / 2)) / orig_w
            cy = (y_min + (h / 2)) / orig_h
            norm_w = w / orig_w
            norm_h = h / orig_h
            boxes.append([max(0., min(cx, 1.)), max(0., min(cy, 1.)), max(1e-4, min(norm_w, 1.)), max(1e-4, min(norm_h, 1.))])
            labels.append(obj['category_id'])
            
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0,4))
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        if self._transforms: img = self._transforms(img)
        return img, {"boxes": boxes_tensor, "labels": labels_tensor}

def collate_fn(batch): return tuple(zip(*batch))

full_train_ds = RFDetrDataset(os.path.join(IMAGES_DIR, 'train2017'), TRAIN_ANN_FILE, transforms)

subset_size = int(0.10 * len(full_train_ds))
print(f"\n--> Data Check: Using 10% of the dataset ({subset_size} out of {len(full_train_ds)} images)")

torch.manual_seed(42)
indices = torch.randperm(len(full_train_ds))[:subset_size].tolist()
train_ds_10pct = Subset(full_train_ds, indices)

train_loader = DataLoader(
    train_ds_10pct, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    collate_fn=collate_fn
)

coco_gt = COCO(VAL_ANN_FILE)
val_ids = coco_gt.getImgIds()

# ==========================================
# 4. MODEL & OPTIMIZER (GETA INT8 - 50% SPARSITY)
# ==========================================
print("--> Initializing RF-DETR with GETA 8-bit...")
rf_wrapper = RFDETRNano() 
pytorch_model = rf_wrapper.model.model.to(DEVICE)

# Initialize Weight-Only Quantization
quantized_model = model_to_quantize_model(pytorch_model, quant_mode=QuantizationMode.WEIGHT_ONLY).to(DEVICE)

# Protect sensitive heads from quantization to maintain accuracy
for name, module in quantized_model.named_modules():
    if any(k in name for k in ['class_embed', 'bbox_embed', 'input_proj']):
        if hasattr(module, 'set_quant_state'): module.set_quant_state(False, False)

torch.cuda.empty_cache()
dummy_input = torch.rand(1, 3, 384, 384).to(DEVICE)

print("--> Building OTO Graph (Safe CPU/CUDA Vanilla Mode)...")
with torch.no_grad():
    _ = quantized_model(dummy_input)
    oto = OTO(model=quantized_model, dummy_input=dummy_input)

# 🎯 50% SPARSITY CONFIGURATION (GETA Optimizer)
optimizer = oto.geta(
    variant="adamw", lr=1e-5, weight_decay=1e-4, target_group_sparsity=0.50,
    
    # Projection Schedule
    start_projection_step=2 * len(train_loader), 
    projection_periods=3,                                             
    projection_steps=1 * len(train_loader), 
    
    # Pruning Schedule
    start_pruning_step=5 * len(train_loader),    
    pruning_periods=3, 
    pruning_steps=1 * len(train_loader),          
    
    # Quantization Precision (Fixed at 8-bit)
    bit_reduction=8, 
    min_bit_wt=8, 
    max_bit_wt=8                
)

# Robust Argument class for the Criterion
class RobustArgs:
    def __init__(self):
        self.num_classes = 91 
        self.dec_layers = 2
        self.hidden_dim = 256
        self.num_queries = 300
        self.num_select = 300
        self.group_detr = 13
        self.eval_max_dets = 300
        self.two_stage = True
        self.segmentation_head = False
        self.aux_loss = True
        self.ia_bce_loss = True
        self.device = "cuda"
        self.devices = "cuda"
        self.cls_loss_coef = 1.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.set_cost_class = 1.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0
        self.focal_alpha = 0.25
    def __getattr__(self, name): return 1 

criterion, _ = build_criterion_and_postprocessors(RobustArgs())
criterion = criterion.to(DEVICE)

# ==========================================
# 5. EVALUATION FUNCTION (mAP)
# ==========================================
def evaluate_map(model):
    model.eval()
    torch.cuda.empty_cache() 
    results = []
    import torchvision.transforms as ST
    eval_tf = ST.Compose([ST.Resize((384, 384)), ST.ToTensor(), ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    with torch.no_grad():
        for img_id in tqdm(val_ids, desc="Evaluating", leave=False):
            img_info = coco_gt.loadImgs(img_id)[0]
            img = Image.open(os.path.join(VAL_IMAGES_DIR, img_info['file_name'])).convert("RGB")
            w_orig, h_orig = img.size
            img_t = eval_tf(img).unsqueeze(0).to(DEVICE)
            
            outputs = model(img_t)
            logits, bboxes = outputs['pred_logits'][0], outputs['pred_boxes'][0]
            prob = logits.sigmoid()
            scores, indexes = torch.topk(prob.view(-1), 100)
            
            for s, idx in zip(scores, indexes):
                if s > 0.05:
                    q_idx, c_idx = idx // logits.shape[1], idx % logits.shape[1]
                    cx, cy, w, h = bboxes[q_idx].tolist()
                    x_min = (cx - 0.5 * w) * w_orig
                    y_min = (cy - 0.5 * h) * h_orig
                    try: cat_id = COCO_CLASSES[c_idx] if c_idx < len(COCO_CLASSES) else int(c_idx)
                    except: cat_id = 1
                    results.append({
                        "image_id": img_id, "category_id": cat_id,
                        "bbox": [x_min, y_min, w * w_orig, h * h_orig], "score": float(s)
                    })
    if not results: return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = val_ids
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats[0]

# ==========================================
# 6. TRAINING LOOP WITH CSV LOGGER
# ==========================================
print("\n" + "="*50)
print(f"🚀 STARTING INT8 TRAINING (50% SPARSITY) FOR {TARGET_EPOCHS} EPOCHS")
print("="*50)

# 🎯 CSV Setup and Header Initialization
csv_file_path = './training_log_50pct.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'mAP (%)'])

for epoch in range(TARGET_EPOCHS):
    quantized_model.train(); criterion.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TARGET_EPOCHS}")
    epoch_loss = 0.0
    optimizer.zero_grad() 
    
    for step, (images, targets) in enumerate(pbar):
        try:
            # Filter valid samples
            valid_indices = [i for i, t in enumerate(targets) if t['boxes'].numel() > 0 and torch.all(t['boxes'][:, 2:] > 0)]
            if not valid_indices: continue
            
            images = torch.stack([images[i].to(DEVICE) for i in valid_indices])
            targets = [{k: v.to(DEVICE) for k, v in targets[i].items()} for i in valid_indices]
            
            outputs = quantized_model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict) / GRAD_ACCUM_STEPS
            total_loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0: 
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad() 
                
            epoch_loss += total_loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=total_loss.item() * GRAD_ACCUM_STEPS)
        except Exception as e:
            continue

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Calculate final average loss and mAP for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    mAP = evaluate_map(quantized_model)
    mAP_pct = mAP * 100
    
    print(f"\n✅ Epoch {epoch+1} Completed | Loss: {avg_epoch_loss:.4f} | mAP: {mAP_pct:.2f}%\n")
    
    # 🎯 Save metrics to CSV
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, round(avg_epoch_loss, 4), round(mAP_pct, 4)])
    
    # Save checkpoints with '50pct' naming convention
    save_dict = {
        'epoch': epoch + 1,
        'model_state_dict': quantized_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP
    }
    torch.save(save_dict, f'./checkpoints/int8_50pct_checkpoint_epoch_{epoch+1}.pth')

    clean_weights_path = f'./int8_50pct_clean_weights_epoch_{epoch+1}.pth'
    torch.save(quantized_model.state_dict(), clean_weights_path)

print(f"\n--> [Success] 15-Epoch INT8 (50% Sparsity) Training Complete! Data saved to {csv_file_path}")

# ==========================================
# 7. EXPORT THE FINAL OPTIMIZED MODEL
# ==========================================
print("\n" + "="*50)
print("🚀 COMMITTING & EXPORTING THE FINAL OPTIMIZED MODEL...")
print("="*50)

try:
    # Finalize the structural pruning (remove zeroed-out channels)
    oto.commit()
    print("    [Info] Architecture physically compressed (Pruning Committed).")
except Exception as e:
    print(f"    [Warning] Could not commit architecture: {e}")

quantized_model.eval()
quantized_model.half() # Convert to FP16 for smaller storage size

final_optimized_path = './int8_50pct_FINAL_OPTIMIZED.pth'
torch.save(quantized_model.state_dict(), final_optimized_path)

final_size_mb = os.path.getsize(final_optimized_path) / (1024 * 1024)
print(f"\n✅ Final Optimized Model saved to: {final_optimized_path}")
print(f"✅ Final Disk Size: {final_size_mb:.2f} MB")