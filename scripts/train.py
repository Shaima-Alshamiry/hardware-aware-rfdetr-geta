import argparse
import contextlib
import io
import os
import csv
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from only_train_once import OTO
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model
from rfdetr import RFDETRSegNano
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

from constants import INPUT_SIZE, MEAN, STD

GRAD_ACCUM_STEPS = 4
FP32_MODULES = ("segmentation_head", "class_embed", "bbox_embed")


def _to_plain_layer(layer: nn.Module) -> nn.Module | None:
    cn = layer.__class__.__name__
    if cn == "QuantizeLinear":
        new: nn.Module = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
    elif cn == "QuantizeConv2d":
        new = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups, bias=layer.bias is not None)
    else: return None
    new.weight.data.copy_(layer.weight.data)
    if layer.bias is not None: new.bias.data.copy_(layer.bias.data)
    return new

def revert_quant_layers(module: nn.Module) -> None:
    for name, child in module.named_children():
        plain = _to_plain_layer(child)
        if plain is not None: setattr(module, name, plain)
        else: revert_quant_layers(child)

def keep_fp32(model: nn.Module) -> None:
    for name in FP32_MODULES:
        sub = getattr(model, name, None)
        if sub is None: continue
        plain = _to_plain_layer(sub)
        if plain is not None: setattr(model, name, plain)
        else: revert_quant_layers(sub)


class RFDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file); self._transforms = transforms
    def __getitem__(self, idx):
        img, target = super().__getitem__(int(idx))
        orig_w, orig_h = img.size; boxes, labels, masks = [], [], []
        for obj in target:
            if obj.get('iscrowd', 0): continue
            x_min, y_min, w, h = obj['bbox']
            cx = max(0., min((x_min + w / 2) / orig_w, 1.))
            cy = max(0., min((y_min + h / 2) / orig_h, 1.))
            bw = max(1e-4, min(w / orig_w, 1.))
            bh = max(1e-4, min(h / orig_h, 1.))
            boxes.append([cx, cy, bw, bh]); labels.append(obj['category_id'])
            seg = obj.get('segmentation', [])
            if seg:
                rles = coco_mask.frPyObjects(seg, orig_h, orig_w)
                m = coco_mask.decode(coco_mask.merge(rles))
            else: m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            masks.append(m)
        img_t = self._transforms(img) if self._transforms else img
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        if masks:
            raw = torch.from_numpy(np.stack(masks)).float().unsqueeze(0)
            masks_t = torch.nn.functional.interpolate(raw, size=(INPUT_SIZE, INPUT_SIZE), mode='nearest').squeeze(0).bool()
        else: masks_t = torch.zeros((0, INPUT_SIZE, INPUT_SIZE), dtype=torch.bool)
        return img_t, {"boxes": boxes_t, "labels": labels_t, "masks": masks_t}


class ValDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file); self._transforms = transforms
    def __getitem__(self, idx):
        img, _ = super().__getitem__(int(idx)); orig_w, orig_h = img.size
        return self._transforms(img), self.ids[idx], torch.tensor([orig_h, orig_w])


@dataclass
class CriterionConfig:
    num_classes: int = 91; dec_layers: int = 4; hidden_dim: int = 256; num_queries: int = 100; num_select: int = 100
    group_detr: int = 13; eval_max_dets: int = 300; two_stage: bool = True; aux_loss: bool = True; ia_bce_loss: bool = True
    device: str = "cuda"; cls_loss_coef: float = 1.0; bbox_loss_coef: float = 5.0; giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25; masks: bool = True; set_cost_class: float = 1.0; set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0; segmentation_head: bool = True; mask_ce_loss_coef: float = 5.0; mask_dice_loss_coef: float = 5.0; mask_point_sample_ratio: int = 16; use_varifocal_loss: bool = False; use_position_supervised_loss: bool = False


def collate_fn(batch: list) -> tuple: return tuple(zip(*batch))
def val_collate_fn(batch: list) -> tuple:
    images, image_ids, orig_sizes = zip(*batch)
    return list(images), list(image_ids), torch.stack(list(orig_sizes))


def _coco_ap(results: list, coco_gt: COCO, iou_type: str) -> float:
    if not results: return 0.0
    coco_dt = coco_gt.loadRes(cast(Any, results))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate(); ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()): ev.summarize()
    return float(ev.stats[0])


def evaluate(model, postprocess, val_loader, coco_gt: COCO, device) -> tuple[float, float]:
    model.eval()
    bbox_results, segm_results = [], []
    with torch.no_grad():
        for images, image_ids, orig_sizes in tqdm(val_loader, desc="  Val", leave=False):
            imgs = torch.stack(images).to(device); outputs = model(imgs)
            detections = postprocess(outputs, orig_sizes.to(device))
            for res, img_id in zip(detections, image_ids):
                scores, labels, boxes = res['scores'].cpu(), res['labels'].cpu(), res['boxes'].cpu()
                masks = res['masks'].cpu() if 'masks' in res else None
                for j in range(scores.numel()):
                    if scores[j] < 0.05: continue
                    x1, y1, x2, y2 = boxes[j].tolist()
                    cat_id = int(labels[j])
                    bbox_results.append({'image_id': int(img_id), 'category_id': cat_id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(scores[j])})
                    if masks is not None:
                        m = np.asfortranarray(masks[j, 0].numpy().astype(np.uint8))
                        rle = coco_mask.encode(m); rle['counts'] = rle['counts'].decode('ascii')
                        segm_results.append({'image_id': int(img_id), 'category_id': cat_id, 'segmentation': rle, 'score': float(scores[j])})
    return _coco_ap(bbox_results, coco_gt, 'bbox'), _coco_ap(segm_results, coco_gt, 'segm')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GETA stable transformer execution loop.")
    p.add_argument("--data-dir", default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO")
    p.add_argument("--train-ann", default="./coco_data/annotations/instances_train2017.json")
    p.add_argument("--val-ann", default="./coco_data/annotations/instances_val2017.json")
    p.add_argument("--is-nuimages", dest="is_nuimages", action="store_true")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-subset", type=float, default=0.10)
    p.add_argument("--val-subset", type=int, default=500)
    p.add_argument("--sparsity", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-classes", type=int, default=90)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from patches import apply_interpolate_patch, apply_layer_norm_patch, apply_optimizer_patch, apply_oto_patch, apply_shape_patches
    apply_interpolate_patch(); apply_layer_norm_patch(); apply_shape_patches(); apply_oto_patch(); apply_optimizer_patch()

    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
    torch.backends.cudnn.enabled = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    device = torch.device('cuda'); cpu = torch.device('cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_subdir, val_subdir = ("train", "valid") if args.is_nuimages else ("train2017", "val2017")
    transforms = T.Compose([T.ToImage(), T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=MEAN, std=STD)])

    full_train_ds = RFDetrDataset(os.path.join(args.data_dir, train_subdir), args.train_ann, transforms)
    subset_size = int(args.train_subset * len(full_train_ds))
    torch.manual_seed(42)
    train_loader = DataLoader(Subset(full_train_ds, torch.randperm(len(full_train_ds))[:subset_size].tolist()), batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(ValDataset(os.path.join(args.data_dir, val_subdir), args.val_ann, transforms), torch.randperm(500)[:args.val_subset].tolist()), batch_size=4, shuffle=False, num_workers=4, collate_fn=val_collate_fn)
    coco_gt = COCO(args.val_ann)

    print("▶ Preparing OTO Graph on CPU...")
    base_model = RFDETRSegNano(num_classes=args.num_classes).model.model
    quantized_model = model_to_quantize_model(base_model, quant_mode=QuantizationMode.WEIGHT_ONLY).to(cpu).eval()
    keep_fp32(quantized_model)

    oto = OTO(model=quantized_model, dummy_input=torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=cpu))

    unprunable_keywords = ['segmentation_head', 'class_embed', 'bbox_embed', 'proj', 'attn', 'transformer']
    for group_id, node_group in list(oto._graph.node_groups.items()):
        is_protected = any(any(kw in param_name for kw in unprunable_keywords) for param_name in node_group.param_names)
        if is_protected:
            node_group.is_prunable = False
            if hasattr(node_group, 'auxiliary_parameters'): node_group.auxiliary_parameters = []

    print("▶ Moving Model to GPU...")
    quantized_model = quantized_model.to(device)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    
    optimizer = oto.geta(
        variant="adamw", lr=args.lr, weight_decay=1e-4, target_group_sparsity=args.sparsity, group_divisible=16,                
        bit_reduction=0, min_bit_wt=8, max_bit_wt=8, # Fix the quantization scale/range to protect Attention features
        start_projection_step=1 * steps_per_epoch, projection_periods=1, projection_steps=steps_per_epoch,
        start_pruning_step=5 * steps_per_epoch, pruning_periods=4, pruning_steps=1 * steps_per_epoch,          
    )

    criterion, postprocessors = build_criterion_and_postprocessors(CriterionConfig(num_classes=args.num_classes + 1))
    criterion = criterion.to(device); postprocess = postprocessors['bbox'] if isinstance(postprocessors, dict) else postprocessors

    print(f"\n🚀 STARTING HIGH-RECOVERY OPTIMIZATION LOOP")
    best_segm_map = 0.0; is_committed = False

    for epoch in range(args.epochs):
        # --- Critical engineering magic here: pruning ends completely at epoch 11 ---
        # We physically prune zero-channels from memory to leave the final epochs for pure recovery
        if epoch >= 10 and not is_committed:
            print("\n✂ [GETA Action] Pruning finished. Committing compressed architecture to flash variables...")
            try:
                oto.commit()
                is_committed = True
                print("✅ Architecture physically optimized and compressed successfully.")
            except Exception as commit_err:
                print(f"⚠ Commit warning: {commit_err}")
                
        quantized_model.train(); criterion.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0; optimizer.zero_grad()

        for step, (images, targets) in enumerate(pbar):
            valid = [i for i, t in enumerate(targets) if t['boxes'].numel() > 0]
            if not valid: continue
            imgs = torch.stack([images[i].to(device) for i in valid])
            tgts = [{k: v.to(device) for k, v in targets[i].items()} for i in valid]

            outputs = quantized_model(imgs); loss_dict = criterion(outputs, tgts)
            weighted = [loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict if k in criterion.weight_dict]
            loss = torch.stack(weighted).sum() / GRAD_ACCUM_STEPS; loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), 1.0)
                # Only update parameters if the architecture hasn't been physically locked yet
                if not is_committed:
                    optimizer.step()
                else:
                    # In pure recovery mode, we use standard AdamW on the surviving weights
                    torch.optim.AdamW(quantized_model.parameters(), lr=args.lr).step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = epoch_loss / len(train_loader)
        bbox_map, segm_map = evaluate(quantized_model, postprocess, val_loader, coco_gt, device)
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | bbox mAP: {bbox_map:.4f} | segm mAP: {segm_map:.4f}")

        try:
            torch.save({'model_state_dict': quantized_model.state_dict()}, os.path.join(args.checkpoint_dir, f"geta_epoch_{epoch+1}.pth"))
            if segm_map > best_segm_map:
                best_segm_map = segm_map
                torch.save({'model_state_dict': quantized_model.state_dict()}, os.path.join(args.checkpoint_dir, "geta_best.pth"))
        except RuntimeError as io_err: pass

    print(f"✅ Training Complete. Best segm mAP: {best_segm_map:.4f}")

if __name__ == "__main__": main()
