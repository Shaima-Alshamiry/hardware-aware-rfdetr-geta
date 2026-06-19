import os
import torch
import csv
import random
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F # 🎯 Confirmed definition is present here
from torch.cuda.amp import autocast, GradScaler 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from only_train_once import OTO
from rfdetr import RFDETRNano
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

# ==========================================
# 0. SYSTEM & STABILITY SETTINGS
# ==========================================
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
TARGET_EPOCHS = 15 

# ==========================================
# 1. THE MASTER MONKEY PATCHES
# ==========================================
import only_train_once.transform.tensor_transform as tensor_transform
import rfdetr.models.backbone.projector as projector
import rfdetr.models.matcher as matcher
import rfdetr.models.criterion as criterion_module
from scipy.optimize import linear_sum_assignment

def force_static(s):
    if hasattr(s, 'item'): return int(s.item())
    try: return int(s)
    except: return s

# Patch 1: OTO Transformation Fix
_orig_basic = tensor_transform.basic_transformation
def patched_basic_transformation(tensor, num_groups):
    if tensor.numel() % num_groups != 0:
        return torch.ones((num_groups, 1), device=tensor.device)
    return _orig_basic(tensor, num_groups)
tensor_transform.basic_transformation = patched_basic_transformation

# Patch 2: Static View/Reshape
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

# Patch 3: Interpolation Fix
original_interpolate = F.interpolate 
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' or antialias is True:
        mode = 'bilinear'; antialias = False
        if align_corners is None: align_corners = False
    if size is not None:
        size = [force_static(s) for s in size] if isinstance(size, (tuple, list)) else force_static(size)
    return original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

# Patch 4: LayerNorm Fix
def patched_layernorm_forward(self, x):
    x = x.permute(0, 2, 3, 1)
    mean = x.mean(dim=-1, keepdim=True); var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + self.eps); x = self.weight * x + self.bias
    return x.permute(0, 3, 1, 2)
projector.LayerNorm.forward = patched_layernorm_forward

# Patch 5: Matcher NaN Fix
def patched_matcher_forward(self, outputs, targets, group_detr=1):
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid().float()
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).float()
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -matcher.generalized_box_iou(matcher.box_cxcywh_to_xyxy(out_bbox), matcher.box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = torch.nan_to_num(C.view(bs, num_queries, -1), nan=1e5).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
matcher.HungarianMatcher.forward = patched_matcher_forward

# Patch 6: Criterion Fix
original_criterion_init = criterion_module.SetCriterion.__init__
def patched_criterion_init(self, *args, **kwargs):
    if 'losses' in kwargs: kwargs['losses'] = [l for l in kwargs['losses'] if l != 'masks']
    args_list = list(args)
    if len(args_list) > 3 and isinstance(args_list[3], (list, tuple)):
        args_list[3] = [l for l in args_list[3] if l != 'masks']
    return original_criterion_init(self, *tuple(args_list), **kwargs)
criterion_module.SetCriterion.__init__ = patched_criterion_init

# ==========================================
# 2. DATA LOADERS
# ==========================================
IMAGES_DIR = "/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO"
TRAIN_ANN_FILE = "./coco_data/annotations/instances_train2017.json"
transforms = T.Compose([
    T.ToImage(), T.Resize((384, 384)), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RFDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file); self._transforms = transforms
    def __getitem__(self, idx):
        img, target = super().__getitem__(int(idx))
        orig_w, orig_h = img.size 
        boxes = []; labels = []
        for obj in target:
            x_min, y_min, w, h = obj['bbox']
            boxes.append([(x_min + w/2)/orig_w, (y_min + h/2)/orig_h, w/orig_w, h/orig_h])
            labels.append(obj['category_id'])
        if self._transforms: img = self._transforms(img)
        return img, {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

def collate_fn(batch): return tuple(zip(*batch))
full_ds = RFDetrDataset(os.path.join(IMAGES_DIR, 'train2017'), TRAIN_ANN_FILE, transforms)
subset_idx = torch.randperm(len(full_ds))[:int(0.10 * len(full_ds))].tolist()
train_loader = DataLoader(Subset(full_ds, subset_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 3. MODEL & OPTIMIZER (Stable Manual Setup)
# ==========================================
print("--> Initializing Stable RF-DETR Pruning Pipeline...")
rf_wrapper = RFDETRNano() 
model = rf_wrapper.model.model.to(DEVICE)
dummy_input = torch.rand(1, 3, 384, 384).to(DEVICE)
oto = OTO(model=model, dummy_input=dummy_input)

from only_train_once.optimizer import GETA
oto_params = oto.get_optimizer_params(lr=1e-5, weight_decay=1e-4)

# Filter out duplicate parameters
seen_params = set()
clean_param_groups = []
for group in oto_params:
    unique_in_group = []
    for p in group['params']:
        if p not in seen_params:
            unique_in_group.append(p)
            seen_params.add(p)
    if unique_in_group:
        group['params'] = unique_in_group
        clean_param_groups.append(group)

optimizer = GETA(
    params=clean_param_groups, lr=1e-5, weight_decay=1e-4, 
    target_group_sparsity=0.10, 
    start_pruning_step=2 * len(train_loader),
    pruning_periods=3, pruning_steps=1 * len(train_loader)
)

scaler = GradScaler()
class RobustArgs:
    def __init__(self):
        self.num_classes = 91; self.dec_layers = 2; self.hidden_dim = 256; self.num_queries = 300
        self.num_select = 300; self.group_detr = 13; self.eval_max_dets = 300; self.two_stage = True
        self.cls_loss_coef = 1.0; self.bbox_loss_coef = 5.0; self.giou_loss_coef = 2.0; self.focal_alpha = 0.25
        self.set_cost_class = 1.0; self.set_cost_bbox = 5.0; self.set_cost_giou = 2.0; self.device = "cuda"
        self.segmentation_head = False 
    def __getattr__(self, name): return 1 

criterion, _ = build_criterion_and_postprocessors(RobustArgs())
criterion = criterion.to(DEVICE)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
csv_path = './training_log_stable_final.csv'
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f); writer.writerow(['Epoch', 'Loss'])

print(f"🚀 STARTING SECURE 10% PRUNING TRAINING")
for epoch in range(TARGET_EPOCHS):
    model.train(); epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for step, (images, targets) in enumerate(pbar):
        images = torch.stack([img.to(DEVICE) for img in images])
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict) / GRAD_ACCUM_STEPS
        
        if not torch.isnan(total_loss):
            scaler.scale(total_loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            epoch_loss += total_loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=total_loss.item() * GRAD_ACCUM_STEPS)

    with open(csv_path, mode='a', newline='') as f:
        csv.writer(f).writerow([epoch+1, round(epoch_loss/len(train_loader), 6)])

print("🚀 SAVING...")
model.half()
torch.save(model.state_dict(), './fp16_10pct_final_stable.pth')