"""Probe: which output label does a fine-tuned expert use for 'car', and how good is it?"""
import json, os, sys, collections
import torch
from PIL import Image
from torchvision.ops import box_iou
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

ROOT = r"D:\tool_build_phase\ai_router"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "expert_1"
split, n = (sys.argv[2], int(sys.argv[3])) if len(sys.argv) > 3 else ("val", 150)
names_file = sys.argv[4] if len(sys.argv) > 4 else None

dev = torch.device("cuda")
from h1common_model import load_expert, load_processor
proc = load_processor()
model = load_expert(ckpt, dev).half()

jdir = os.path.join(ROOT, "data", "100k", split)
if names_file:
    names = [l.strip() for l in open(names_file)][:n]
else:
    names = sorted(f[:-5] for f in os.listdir(jdir) if f.endswith(".json"))[:n]
stats = collections.Counter()
for name in names:
    d = json.load(open(os.path.join(jdir, name + ".json")))
    gt = [[o["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for o in d["frames"][0]["objects"]
          if o["category"] == "car" and "box2d" in o]
    gt = torch.tensor(gt).reshape(-1, 4)
    im = Image.open(os.path.join(jdir, name + ".jpg")).convert("RGB")
    inp = proc(images=im, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(pixel_values=inp["pixel_values"].half())
    out.logits = out.logits.float(); out.pred_boxes = out.pred_boxes.float()
    r = proc.post_process_object_detection(out, target_sizes=torch.tensor([im.size[::-1]]).to(dev), threshold=0.5)[0]
    stats["gt"] += len(gt)
    for lab in r["labels"].unique().tolist():
        m = r["labels"] == lab
        b = r["boxes"][m].cpu()
        stats[("n", lab)] += len(b)
        if len(gt) and len(b):
            iou = box_iou(b, gt)
            stats[("tp50", lab)] += int((iou.max(1).values >= 0.5).sum())
            stats[("tp75", lab)] += int((iou.max(1).values >= 0.75).sum())
print(ckpt, split, len(names), "GT cars", stats["gt"])
for k, v in sorted(stats.items(), key=lambda kv: str(kv[0])):
    if isinstance(k, tuple) and k[0] == "n" and v >= 5:
        lab = k[1]
        print(f"label {lab:2d} {model.config.id2label[lab]:>12s}: n={v:5d}  prec@.5={stats[('tp50',lab)]/v:.3f}  prec@.75={stats[('tp75',lab)]/v:.3f}")
