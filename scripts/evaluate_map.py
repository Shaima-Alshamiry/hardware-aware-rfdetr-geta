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

from rfdetr import RFDETRSegNano
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model

torch.backends.cudnn.enabled = False

def convert_to_coco_format(outputs, orig_sizes, image_ids):
    results = []
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    masks = outputs.get("pred_masks", None) 

    for i in range(len(image_ids)):
        img_id = image_ids[i].item()
        orig_w, orig_h = orig_sizes[i].tolist()
        
        img_logits = logits[i]
        img_boxes = boxes[i]
        
        scores, labels = img_logits.max(-1)
        keep = scores > 0.05
        
        cur_scores = scores[keep]
        cur_labels = labels[keep]
        cur_boxes = img_boxes[keep]

        if masks is not None:
            cur_masks = masks[i][keep]
            cur_masks = torch.nn.functional.interpolate(
                cur_masks.unsqueeze(1), size=(int(orig_h), int(orig_w)), 
                mode="bilinear", align_corners=False
            ).squeeze(1).gt(0.5)

        # The loop is now correctly defined to ensure l, s, and b remain within range
        for j in range(len(cur_scores)):
            s = cur_scores[j].item()
            l = cur_labels[j].item()
            b = cur_boxes[j].tolist()
            
            # 1. Clamping to prevent out-of-bounds values [0, 1]
            cx = max(0.0, min(b[0], 1.0))
            cy = max(0.0, min(b[1], 1.0))
            w = max(0.0, min(b[2], 1.0))
            h = max(0.0, min(b[3], 1.0))
            
            # 2. Convert coordinates to pixels
            x_min = max(0.0, (cx - w / 2) * orig_w)
            y_min = max(0.0, (cy - h / 2) * orig_h)
            abs_w = w * orig_w
            abs_h = h * orig_h
            
            # 3. Adjust bbox format (if the problem persists, use the second line)
            # Format [x_min, y_min, w, h] (COCO default format)
            res_item = {
                "image_id": int(img_id),
                "category_id": int(l), 
                "bbox": [float(x_min), float(y_min), float(abs_w), float(abs_h)],
                "score": float(s)
            }

            if masks is not None:
                m = cur_masks[j].cpu().numpy().astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(m))
                rle['counts'] = rle['counts'].decode('ascii')
                res_item["segmentation"] = rle

            results.append(res_item)
    return results

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}")

    model_wrapper = RFDETRSegNano()
    model = model_wrapper.model.model

    if args.mode == "baseline":
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
    elif args.mode == "geta":
        print(f"▶ Loading GETA Optimized Model weights: {args.weights}")
        model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_ONLY)
        
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        
        # Modification here: Check if the checkpoint is a dictionary or a model
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
            model.load_state_dict(state_dict, strict=False)
        else:
            # If it's the model itself (via direct export), load it directly
            model = checkpoint 
            print("▶ Direct model loaded from checkpoint.")
    
    model = model.to(device)
    
    model.eval()

    transforms = T.Compose([
        T.ToImage(),
        T.Resize((312, 312)), # 312 confirmed
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class CustomCocoDetect(CocoDetection):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            img_id = self.ids[index]
            orig_size = torch.tensor([img.width, img.height])
            return transforms(img), orig_size, img_id

    dataset = CustomCocoDetect(root=args.img_dir, annFile=args.ann_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    all_results = []
    with torch.inference_mode():
        for images, orig_sizes, image_ids in tqdm(dataloader):
            images = images.to(device)
            if args.fp16:
                images = images.half()
            outputs = model(images)
            batch_results = convert_to_coco_format(outputs, orig_sizes, image_ids)
            all_results.extend(batch_results)

    res_name = f"{args.mode}_fp16" if args.fp16 else f"{args.mode}_fp32"
    res_file = f"results_{res_name}.json"
    with open(res_file, "w") as f:
        json.dump(all_results, f)

    coco_gt = COCO(args.ann_file)
    coco_dt = coco_gt.loadRes(res_file)
    
    print("\n--- Bounding Box Evaluation ---")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "geta"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--img_dir", type=str, default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO/val2017")
    parser.add_argument("--ann_file", type=str, default="/gpfs/VICOMTECH/Databases/GeneralDatabases/MS-COCO/annotations/instances_val2017.json")
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())