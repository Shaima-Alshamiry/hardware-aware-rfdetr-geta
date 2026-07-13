import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

# Use RFDETRSegNano (which includes the segmentation head)
from rfdetr import RFDETRSegNano
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model

# Disable cuDNN to avoid server-side library version errors
torch.backends.cudnn.enabled = False

def convert_to_coco_format(outputs, orig_sizes, image_ids):
    results = []
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    masks = outputs.get("pred_masks", None) 

    for i in range(len(image_ids)):
        img_id = image_ids[i].item()
        orig_w, orig_h = orig_sizes[i].tolist()
        
        # SLICE the batch first to avoid indexing errors
        img_logits = logits[i]
        img_boxes = boxes[i]
        
        scores, labels = img_logits.max(-1)
        keep = scores > 0.05
        
        cur_scores = scores[keep]
        cur_labels = labels[keep]
        cur_boxes = img_boxes[keep]

        if masks is not None:
            # Interpolate and threshold segmentation masks
            cur_masks = masks[i][keep]
            cur_masks = torch.nn.functional.interpolate(
                cur_masks.unsqueeze(1), size=(int(orig_h), int(orig_w)), 
                mode="bilinear", align_corners=False
            ).squeeze(1).gt(0.5)

        for j in range(len(cur_scores)):
            s = cur_scores[j].item()
            l = cur_labels[j].item()
            b = cur_boxes[j].tolist()
            
            # Coordinate conversion (CXCYWH to XYWH)
            cx, cy, w, h = b
            abs_w = w * orig_w
            abs_h = h * orig_h
            x_min = (cx - w / 2) * orig_w
            y_min = (cy - h / 2) * orig_h
            
            res_item = {
                "image_id": int(img_id),
                "category_id": int(l), 
                "bbox": [float(x_min), float(y_min), float(abs_w), float(abs_h)],
                "score": float(s)
            }

            if masks is not None:
                # Encode mask to RLE for COCO compatibility
                m = cur_masks[j].cpu().numpy().astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(m))
                rle['counts'] = rle['counts'].decode('ascii')
                res_item["segmentation"] = rle

            results.append(res_item)
    return results

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}")

    # 1. INITIALIZE MODEL (Always SegNano to ensure mask head exists)
    model_wrapper = RFDETRSegNano()
    model = model_wrapper.model.model

    if args.mode == "baseline":
        print(f"▶ Loading Baseline Model weights: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        
        # Extract state_dict from dictionary wrapper
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        # Use strict=False to ignore missing OTO/Quantization keys in baseline weights
        model.load_state_dict(state_dict, strict=False)
        
    elif args.mode == "geta":
        print(f"▶ Loading GETA Optimized Model weights: {args.weights}")
        
        # Convert architecture to quantized structure BEFORE loading weights
        model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_ONLY)
        
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        # Load weights with strict=False to handle potential key mismatches during research
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    if args.fp16:
        print("▶ Running in FP16 Mode.")
        model = model.half()
    model.eval()

    # 2. PREPARE DATASET
    print("▶ Preparing COCO Dataset...")
    transforms = T.Compose([
        T.ToImage(),
        T.Resize((384, 384)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.455, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class CustomCocoDetect(CocoDetection):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            img_id = self.ids[index]
            orig_size = torch.tensor([img.width, img.height])
            return transforms(img), orig_size, img_id

    dataset = CustomCocoDetect(root=args.img_dir, annFile=args.ann_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 3. INFERENCE
    print(f"▶ Starting Inference on {len(dataset)} images...")
    all_results = []
    
    with torch.inference_mode():
        for images, orig_sizes, image_ids in tqdm(dataloader):
            images = images.to(device)
            if args.fp16:
                images = images.half()
                
            outputs = model(images)
            batch_results = convert_to_coco_format(outputs, orig_sizes, image_ids)
            all_results.extend(batch_results)

    # 4. EVALUATION
    print("\n▶ Calculating mAP...")
    res_name = f"{args.mode}_fp16" if args.fp16 else f"{args.mode}_fp32"
    res_file = f"results_{res_name}.json"
    
    with open(res_file, "w") as f:
        json.dump(all_results, f)

    coco_gt = COCO(args.ann_file)
    coco_dt = coco_gt.loadRes(res_file)
    
    # Run standard Box evaluation
    print("\n--- Bounding Box Evaluation ---")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Optional: Run Segmentation evaluation if masks are present
    if "segmentation" in all_results[0]:
        print("\n--- Segmentation Mask Evaluation ---")
        coco_eval_seg = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_seg.evaluate()
        coco_eval_seg.accumulate()
        coco_eval_seg.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "geta"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--fp16", action="store_true", help="Run in FP16 mode")
    parser.add_argument("--img_dir", type=str, default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO/val2017")
    parser.add_argument("--ann_file", type=str, default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO/annotations/instances_val2017.json")
    parser.add_argument("--batch_size", type=int, default=32)
    
    main(parser.parse_args())