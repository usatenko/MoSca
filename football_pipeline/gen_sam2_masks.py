"""YOLO11 (frame-0 detect) + SAM2 video propagation -> MoSca vos_deva instance masks.
Usage: python gen_sam2_masks.py <ws_dir>   (expects <ws_dir>/images, writes <ws_dir>/vos_deva/Annotations)
Needs: ultralytics, sam2 (facebookresearch/sam2) + checkpoints/sam2.1_hiera_small.pt
"""
import sys, os, glob, numpy as np, torch, cv2, imageio.v2 as iio
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

ws = sys.argv[1]
frames_dir = os.path.join(ws, "images")
out_dir = os.path.join(ws, "vos_deva", "Annotations"); os.makedirs(out_dir, exist_ok=True)
SAM2_CKPT = os.environ.get("SAM2_CKPT", "/workspace/sam2/checkpoints/sam2.1_hiera_small.pt")
SAM2_CFG  = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")

imgs = sorted(glob.glob(f"{frames_dir}/*.jpg"))
det = YOLO("yolo11n.pt")
boxes = det(imgs[0], classes=[0], conf=0.30, verbose=False)[0].boxes.xyxy.cpu().numpy()
print("frame0 players:", len(boxes))

predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
seg = {}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=frames_dir)
    for i, b in enumerate(boxes):
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=i+1, box=b)
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        seg[fidx] = (obj_ids, (mask_logits > 0).cpu().numpy())

def id2color(i):
    i = int(i)+1
    return (i % 250 + 1, (i*37) % 250 + 1, (i*97) % 250 + 1)

H, W = iio.imread(imgs[0]).shape[:2]
for fidx, fn in enumerate(imgs):
    canvas = np.zeros((H, W, 3), np.uint8)
    if fidx in seg:
        obj_ids, masks = seg[fidx]
        for k, oid in enumerate(obj_ids):
            m = masks[k];  m = m[0] if m.ndim == 3 else m
            canvas[m] = id2color(oid)
    iio.imwrite(os.path.join(out_dir, os.path.basename(fn).replace(".jpg", ".png")), canvas)
print("wrote masks to", out_dir)
