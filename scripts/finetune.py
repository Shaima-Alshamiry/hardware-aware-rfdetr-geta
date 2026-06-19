import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from constants import INPUT_SIZE, MEAN, STD

# --- Configuration ---
GRAD_ACCUM_STEPS = 4
EPOCHS = 5

class RFDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(int(idx))
        orig_w, orig_h = img.size
        boxes, labels, masks = [], [], []
        for obj in target:
            if obj.get('iscrowd', 0):
                continue
            x_min, y_min, w, h = obj['bbox']
            cx = max(0., min((x_min + w / 2) / orig_w, 1.))
            cy = max(0., min((y_min + h / 2) / orig_h, 1.))
            bw = max(1e-4, min(w / orig_w, 1.))
            bh = max(1e-4, min(h / orig_h, 1.))
            boxes.append([cx, cy, bw, bh])
            labels.append(obj['category_id'])

            seg = obj.get('segmentation', [])
            if seg:
                rles = coco_mask.frPyObjects(seg, orig_h, orig_w)
                m = coco_mask.decode(coco_mask.merge(rles))
            else:
                m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            masks.append(m)

        img_t = self._transforms(img) if self._transforms else img
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_t = (torch.tensor(labels, dtype=torch.int64)
                    if labels else torch.zeros((0,), dtype=torch.int64))
        if masks:
            raw = torch.from_numpy(np.stack(masks)).float().unsqueeze(0)
            masks_t = torch.nn.functional.interpolate(
                raw, size=(INPUT_SIZE, INPUT_SIZE), mode='nearest'
            ).squeeze(0).bool()
        else:
            masks_t = torch.zeros((0, INPUT_SIZE, INPUT_SIZE), dtype=torch.bool)
        return img_t, {"boxes": boxes_t, "labels": labels_t, "masks": masks_t}

@dataclass
class CriterionConfig:
    num_classes: int = 91
    dec_layers: int = 4
    hidden_dim: int = 256
    num_queries: int = 100
    num_select: int = 100
    group_detr: int = 13
    eval_max_dets: int = 300
    two_stage: bool = True
    aux_loss: bool = True
    ia_bce_loss: bool = True
    device: str = "cuda"
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25
    masks: bool = True
    set_cost_class: float = 1.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    segmentation_head: bool = True
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    mask_point_sample_ratio: int = 16
    use_varifocal_loss: bool = False
    use_position_supervised_loss: bool = False

def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch))

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-End Healing for Pruned RF-DETR.")
    p.add_argument("--model", default="clean_pruned_rfdetr_vsu.pth",
                   help="Path to the clean pruned model saved by prune.py.")
    p.add_argument("--output", default="finetuned_rfdetr_vsu.pth",
                   help="Path to save the healed, fine-tuned model.")
    p.add_argument("--data-dir", default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO")
    p.add_argument("--train-ann", default="./coco_data/annotations/instances_train2017.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=90)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fix CuDNN issue to avoid sudden memory errors
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from patches import apply_interpolate_patch, apply_shape_patches
    apply_interpolate_patch()
    apply_shape_patches()

    print(f"▶ 1. Loading Clean Pruned Model from {args.model}...")
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model = model.float()

    # 🔴 Completely unfreeze to allow the model to compensate for what it lost during pruning (End-to-End) 🔴
    for param in model.parameters():
        param.requires_grad = True

    model.train()

    print("▶ 2. Preparing Dataset and Dataloader...")
    transforms = T.Compose([
        T.ToImage(), T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=MEAN, std=STD),
    ])

    train_images_dir = os.path.join(args.data_dir, "train2017")
    
    full_train_ds = RFDetrDataset(train_images_dir, args.train_ann, transforms)
    subset_size = int(0.20 * len(full_train_ds))
    indices = torch.randperm(len(full_train_ds))[:subset_size].tolist()
    train_ds = torch.utils.data.Subset(full_train_ds, indices)

    print(f"⚡ TIME SAVER MODE: Using {subset_size} images (20% of dataset).")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=1, collate_fn=collate_fn
    )

    print("▶ 3. Initializing Optimizer and Criterion...")
    
    # 🔴 Setting up Differential LRs (Differential Learning Rates) 🔴
    # The Backbone learns very slowly to improve box accuracy without destroying it.
    # The rest of the model (including masks) learns 10 times faster to patch the holes.
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": 5e-5,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 5e-6, 
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

    criterion, _ = build_criterion_and_postprocessors(
        CriterionConfig(num_classes=args.num_classes + 1)
    )
    criterion = criterion.to(device)
    criterion.train()

    # 🔴 Tripling the mask penalty to force the model to fill holes and smooth edges 🔴
    if 'loss_mask' in criterion.weight_dict:
        criterion.weight_dict['loss_mask'] *= 3.0
    if 'loss_dice' in criterion.weight_dict:
        criterion.weight_dict['loss_dice'] *= 3.0

    print(f"\n🚀 STARTING FINE-TUNING FOR {EPOCHS} EPOCHS TO HEAL MASKS & BOXES...")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(pbar):
            valid = [i for i, t in enumerate(targets) if t['boxes'].numel() > 0]
            if not valid:
                continue
                
            imgs = torch.stack([images[i].to(device) for i in valid])
            tgts = [{k: v.to(device) for k, v in targets[i].items()} for i in valid]

            outputs = model(imgs)
            loss_dict = criterion(outputs, tgts)
            
            weighted = [
                loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                for k in loss_dict if k in criterion.weight_dict
            ]
            loss = torch.stack(weighted).sum() / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({'Loss': f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")

        torch.save(model, args.output)
        
    print(f"\n🎉 SUCCESS! Fully healed model saved as '{args.output}'.")

if __name__ == "__main__":
    main()