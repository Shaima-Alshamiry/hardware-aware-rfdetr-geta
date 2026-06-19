import argparse
import contextlib
import io
import os
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


# Submodules kept in FP32: the mask head (per-pixel masks are quantization-
# sensitive) plus the class/bbox prediction heads (logit/coordinate outputs are
# sensitive and tiny, so quantizing them risks accuracy for ~no speedup).
# Attention is deliberately NOT here — it is the compute INT8 is meant to speed up.
FP32_MODULES = ("segmentation_head", "class_embed", "bbox_embed")


def _to_plain_layer(layer: nn.Module) -> nn.Module | None:
    """Return a plain FP32 nn layer copying a Quantize* wrapper's weights/bias.

    Returns None if `layer` is not a Quantize* wrapper (i.e. a container to recurse
    into). The GETA Quantize* wrappers always fake-quantize in forward() and expose
    no off-switch, so the only way to keep something FP32 is to swap the wrapper out.
    """
    cn = layer.__class__.__name__
    if cn == "QuantizeLinear":
        new: nn.Module = nn.Linear(
            layer.in_features, layer.out_features, bias=layer.bias is not None
        )
    elif cn == "QuantizeConv2d":
        new = nn.Conv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups,
            bias=layer.bias is not None,
        )
    else:
        return None
    new.weight.data.copy_(layer.weight.data)
    if layer.bias is not None:
        new.bias.data.copy_(layer.bias.data)
    return new


def revert_quant_layers(module: nn.Module) -> None:
    """Replace every GETA Quantize* wrapper under `module` with a plain FP32 layer."""
    for name, child in module.named_children():
        plain = _to_plain_layer(child)
        if plain is not None:
            setattr(module, name, plain)
        else:
            revert_quant_layers(child)


def keep_fp32(model: nn.Module) -> None:
    """Unwrap the FP32_MODULES so they train/run in FP32, in place.

    Handles both leaf wrappers (e.g. class_embed is a bare QuantizeLinear) and
    containers (segmentation_head, bbox_embed), keeping train.py and prune.py
    architecturally identical so checkpoints load.
    """
    for name in FP32_MODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        plain = _to_plain_layer(sub)
        if plain is not None:
            setattr(model, name, plain)
        else:
            revert_quant_layers(sub)


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


class ValDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, _ = super().__getitem__(int(idx))
        orig_w, orig_h = img.size
        return self._transforms(img), self.ids[idx], torch.tensor([orig_h, orig_w])


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


def val_collate_fn(batch: list) -> tuple:
    images, image_ids, orig_sizes = zip(*batch)
    return list(images), list(image_ids), torch.stack(list(orig_sizes))


def _coco_ap(results: list, coco_gt: COCO, iou_type: str) -> float:
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(cast(Any, results))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return float(ev.stats[0])


def evaluate(model, postprocess, val_loader, coco_gt: COCO, device) -> tuple[float, float]:
    """Return (bbox mAP@[.5:.95], segm mAP@[.5:.95]).

    The mask head is the capacity being protected, so segmentation must be
    measured directly — selecting checkpoints on bbox AP alone is blind to it.
    PostProcess already emits per-detection masks (resized to the original
    image size) when the model outputs ``pred_masks``; we RLE-encode them for
    a 'segm' COCOeval pass.
    """
    model.eval()
    bbox_results, segm_results = [], []
    with torch.no_grad():
        for images, image_ids, orig_sizes in tqdm(val_loader, desc="  Val", leave=False):
            imgs = torch.stack(images).to(device)
            outputs = model(imgs)
            detections = postprocess(outputs, orig_sizes.to(device))
            for res, img_id in zip(detections, image_ids):
                scores = res['scores'].cpu()
                labels = res['labels'].cpu()
                boxes = res['boxes'].cpu()
                masks = res['masks'].cpu() if 'masks' in res else None
                for j in range(scores.numel()):
                    score = float(scores[j])
                    if score < 0.001:
                        continue
                    label = int(labels[j])
                    x1, y1, x2, y2 = boxes[j].tolist()
                    bbox_results.append({
                        'image_id': int(img_id),
                        'category_id': label,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': score,
                    })
                    if masks is not None:
                        m = np.asfortranarray(masks[j, 0].numpy().astype(np.uint8))
                        rle = coco_mask.encode(m)
                        rle['counts'] = rle['counts'].decode('ascii')
                        segm_results.append({
                            'image_id': int(img_id),
                            'category_id': label,
                            'segmentation': rle,
                            'score': score,
                        })
    return (_coco_ap(bbox_results, coco_gt, 'bbox'),
            _coco_ap(segm_results, coco_gt, 'segm'))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GETA-based pruning + quantization training.")
    p.add_argument("--data-dir", default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO",
                   help="Root directory of the MS-COCO dataset.")
    p.add_argument("--train-ann", default="./coco_data/annotations/instances_train2017.json")
    p.add_argument("--val-ann", default="./coco_data/annotations/instances_val2017.json")
    p.add_argument("--is-nuimages", dest="is_nuimages", action="store_true",
                   help="Use the nuImages (RF-DETR/Roboflow) image-folder names "
                        "(train/valid). Default is the raw MS-COCO layout (train2017/val2017).")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-subset", type=float, default=0.10,
                   help="Fraction of training data to use (0, 1].")
    p.add_argument("--val-subset", type=int, default=500,
                   help="Number of validation images.")
    p.add_argument("--sparsity", type=float, default=0.50,
                   help="Target group sparsity for GETA.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-classes", type=int, default=90,
                   help="RF-DETR num_classes for the detection head. 90 for MS-COCO; "
                        "for nuImages (single 'person' category, id 1) use 1. The COCO "
                        "backbone/transformer are still loaded pretrained — only the head "
                        "is reinitialised to width num_classes+1 to match the dataset.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from patches import (
        apply_interpolate_patch,
        apply_layer_norm_patch,
        apply_optimizer_patch,
        apply_oto_patch,
        apply_shape_patches,
    )
    apply_interpolate_patch()
    apply_layer_norm_patch()
    apply_shape_patches()
    apply_oto_patch()
    apply_optimizer_patch()

    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')
    cpu = torch.device('cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_subdir, val_subdir = ("train", "valid") if args.is_nuimages else ("train2017", "val2017")
    train_images_dir = os.path.join(args.data_dir, train_subdir)
    val_images_dir = os.path.join(args.data_dir, val_subdir)

    transforms = T.Compose([
        T.ToImage(), T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=MEAN, std=STD),
    ])

    full_train_ds = RFDetrDataset(train_images_dir, args.train_ann, transforms)
    subset_size = int(args.train_subset * len(full_train_ds))
    torch.manual_seed(42)
    train_loader = DataLoader(
        Subset(full_train_ds, torch.randperm(len(full_train_ds))[:subset_size].tolist()),
        batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn,
    )

    val_ds = ValDataset(val_images_dir, args.val_ann, transforms)
    torch.manual_seed(0)
    val_loader = DataLoader(
        Subset(val_ds, torch.randperm(len(val_ds))[:args.val_subset].tolist()),
        batch_size=4, shuffle=False, num_workers=4, collate_fn=val_collate_fn,
    )
    coco_gt = COCO(args.val_ann)

    print("▶ Preparing OTO Graph on CPU...")
    # RFDETRSegNano re-heads to width num_classes+1 when num_classes differs from the
    # COCO default (90), loading the backbone/transformer pretrained but reinitialising
    # class_embed/enc_out_class_embed. For nuImages (--num-classes 1) this yields the
    # 2-wide head the dataset needs; the old hard-coded default built a 91-wide head and
    # trained it on single-class data, which is why mAP collapsed.
    base_model = RFDETRSegNano(num_classes=args.num_classes).model.model
    quantized_model = model_to_quantize_model(
        base_model, quant_mode=QuantizationMode.WEIGHT_ONLY
    ).to(cpu).eval()

    # Keep the mask head and prediction heads (FP32_MODULES) in FP32. The previous
    # set_quant_state() call was a silent no-op (the method does not exist on the
    # Quantize* wrappers, so the hasattr guard always failed) and the wrappers have
    # no way to disable quant in forward() — the only fix is to unwrap them.
    keep_fp32(quantized_model)

    oto = OTO(model=quantized_model, dummy_input=torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=cpu))

    # Exclude the segmentation head from GETA's structured-pruning search space.
    # set_quant_state only ever concerned quantization; without this the head and
    # its share of the hidden dim were pruned at the target sparsity, collapsing
    # the per-pixel mask dot-product even when its weights stayed FP32.
    seg_param_names = [n for n, _ in quantized_model.named_parameters()
                       if 'segmentation_head' in n]
    oto.mark_unprunable_by_param_names(seg_param_names)
    still_prunable = [ng.id for ng in oto._graph.node_groups.values()
                      if ng.is_prunable
                      and any('segmentation_head' in pn for pn in ng.param_names)]
    if still_prunable:
        print(f"  ⚠ segmentation_head groups still prunable: {still_prunable}")

    print("▶ Moving Model to GPU...")
    quantized_model = quantized_model.to(device)

    # GETA's clock (num_steps) ticks once per optimizer.step(), and with grad
    # accumulation that fires only every GRAD_ACCUM_STEPS batches. The schedule
    # must therefore be expressed in optimizer steps, not batches, or pruning
    # lands GRAD_ACCUM_STEPS-times too late (and never triggers within the epoch
    # budget).
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    optimizer = oto.geta(
        variant="adamw", lr=args.lr, weight_decay=1e-4,
        target_group_sparsity=args.sparsity, group_divisible=16, bit_reduction=8,
        start_projection_step=2 * steps_per_epoch,
        projection_periods=3,
        projection_steps=steps_per_epoch,
        start_pruning_step=5 * steps_per_epoch,
        pruning_periods=3,
        pruning_steps=1 * steps_per_epoch,
    )

    # Criterion/PostProcess head width must equal the model's (num_classes+1), or the
    # PostProcess label decode (topk_index % num_classes) misaligns with the GT
    # category ids and mAP reads ~0.
    criterion, postprocessors = build_criterion_and_postprocessors(
        CriterionConfig(num_classes=args.num_classes + 1))
    criterion = criterion.to(device)
    # build_criterion_and_postprocessors returns a single PostProcess; tolerate
    # older dict-style returns too.
    postprocess = postprocessors['bbox'] if isinstance(postprocessors, dict) else postprocessors

    print(f"\n🚀 STARTING TRAINING ({args.sparsity:.0%} SPARSITY, group_divisible=16)")
    best_segm_map = 0.0
    for epoch in range(args.epochs):
        quantized_model.train()
        criterion.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(pbar):
            valid = [i for i, t in enumerate(targets) if t['boxes'].numel() > 0]
            if not valid:
                continue
            imgs = torch.stack([images[i].to(device) for i in valid])
            tgts = [{k: v.to(device) for k, v in targets[i].items()} for i in valid]

            outputs = quantized_model(imgs)
            loss_dict = criterion(outputs, tgts)
            weighted = [
                loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                for k in loss_dict if k in criterion.weight_dict
            ]
            loss = torch.stack(weighted).sum() / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = epoch_loss / len(train_loader)
        bbox_map, segm_map = evaluate(quantized_model, postprocess, val_loader, coco_gt, device)
        quantized_model.train()
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
              f"bbox mAP: {bbox_map:.4f} | segm mAP: {segm_map:.4f}")

        ckpt = {'model_state_dict': quantized_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"geta_epoch_{epoch+1}.pth"))
        # Select on segmentation mAP — that is the capacity being protected.
        if segm_map > best_segm_map:
            best_segm_map = segm_map
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "geta_best.pth"))
            print(f"  ⭐ New best segm mAP: {best_segm_map:.4f} — saved to geta_best.pth")

    print(f"✅ Training Complete. Best segm mAP@[.5:.95]: {best_segm_map:.4f}. "
          f"Use geta_best.pth for Step 2.")


if __name__ == "__main__":
    main()
