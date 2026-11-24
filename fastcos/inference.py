# infer_demo.py
import json, os
from pathlib import Path
import torch, cv2
import numpy as np

# --- import bits from your training file (copy these small helpers if needed) ---
from auto_train_nuscenes import (FusionFCOS, CocoFusionDataset, collate_fn,
                                 generate_points)
def nms_torch(boxes, scores, iou_thr=0.5):
    # boxes: [N,4] (x1,y1,x2,y2), scores: [N]
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        rest = idxs[1:]
        x1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_r = (boxes[rest, 2]-boxes[rest, 0]) * (boxes[rest, 3]-boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thr]
    return torch.tensor(keep, device=boxes.device)

def fcos_decode(outputs, strides, score_thr=0.65, iou_thr=0.5, topk=100):
    levels = ["P3","P4","P5"]
    B = next(iter(outputs.values()))[0].shape[0]
    results = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for lv in levels:
            logits, ltrb, ctn = outputs[lv]  # [B,C,H,W], [B,4,H,W], [B,1,H,W]
            C, H, W = logits.shape[1], logits.shape[2], logits.shape[3]
            pts = generate_points(H, W, strides[lv], device=logits.device)  # (N,2)
            cls = torch.sigmoid(logits[b].permute(1,2,0).reshape(-1, C))    # (N,C)
            reg = ltrb[b].permute(1,2,0).reshape(-1, 4)                      # (N,4)
            cen = torch.sigmoid(ctn[b].permute(1,2,0).reshape(-1))           # (N,)

            # per-class scores
            scores = cls * cen.unsqueeze(1)                                  # (N,C)
            keep = scores.max(dim=1).values > score_thr
            if keep.any():
                pts_k = pts[keep]
                reg_k = reg[keep]
                sc_k  = scores[keep]                                         # (M,C)

                x1 = pts_k[:,0] - reg_k[:,0]; y1 = pts_k[:,1] - reg_k[:,1]
                x2 = pts_k[:,0] + reg_k[:,2]; y2 = pts_k[:,1] + reg_k[:,3]
                boxes = torch.stack([x1,y1,x2,y2], dim=1)                    # (M,4)

                # class-wise NMS
                for c in range(C):
                    s = sc_k[:, c]
                    k = s > score_thr
                    if not k.any(): 
                        continue
                    b_c = boxes[k]
                    s_c = s[k]
                    keep_idx = nms_torch(b_c, s_c, iou_thr)
                    all_boxes.append(b_c[keep_idx])
                    all_scores.append(s_c[keep_idx])
                    all_labels.append(torch.full((keep_idx.numel(),), c, device=boxes.device, dtype=torch.long))

        if len(all_boxes):
            boxes = torch.cat(all_boxes, 0)
            scores = torch.cat(all_scores, 0)
            labels = torch.cat(all_labels, 0)

            # keep top-k
            if boxes.size(0) > topk:
                top = scores.topk(topk).indices
                boxes, scores, labels = boxes[top], scores[top], labels[top]

            det = torch.cat([boxes, labels.float().unsqueeze(1), scores.unsqueeze(1)], 1).cpu().numpy()
            results.append(det)
        else:
            results.append(np.zeros((0,6), dtype=np.float32))
    return results

def draw_and_save(img, dets, out_path, class_names):
    im = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    for x1,y1,x2,y2,cls,score in dets:
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(im,f"{class_names[int(cls)]}:{score:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imwrite(str(out_path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

def main():
    data_root = r"C:\Users\BTC\Desktop\nus2d_mini"  
    weights = Path(data_root)/"checkpoints"/"fusion_ep369.pth"  
    out_dir = Path(data_root)/"viz_latest"; out_dir.mkdir(parents=True, exist_ok=True)

    # class names (order used in training)
    names = ["car","truck","bus","trailer","construction_vehicle","pedestrian","motorcycle","bicycle","traffic_cone","barrier"]

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionFCOS(num_classes=len(names)).to(device)
    ckpt = torch.load(str(weights), map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # dataset (we’ll just take a few samples)
    ds = CocoFusionDataset(data_root, img_size=(640,960), num_classes=len(names))
    idxs = list(range(0, min(8, len(ds))))   # 8 examples
    for i in idxs:
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device)   
        lid = sample["lidar"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs, _ = model(img, lid)
        dets = fcos_decode(outputs, model.strides, score_thr=0.65)[0]
        draw_and_save(sample["image"], dets, out_dir/f"viz_{i}.png", names)

    print(f"Saved visualizations to: {out_dir}")

if __name__ == "__main__":
    main()
