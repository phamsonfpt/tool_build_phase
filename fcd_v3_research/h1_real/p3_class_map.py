"""P3.0 — class-index mapping and memorisation check, using Meta-Train (100K test rides) and 100K train only.

For every expert and every output index l, queries with sigmoid score >= 0.5 at l are matched to car GT
(IoU >= 0.5) on 300 Meta-Train images; the car index is the one with the most true positives.
Memorisation check: precision of that index on 150 random 100K-train images (the experts' training split)
versus 150 Meta-Train images. Writes cache/class_map.json. No Val image or label is touched.
"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from build_features import iou_mat
from h1common import CACHE, DATA, EXPERTS, write_json
from h1common_model import load_expert, load_processor


def car_gt(split, name):
    d = json.load(open(os.path.join(DATA, "100k", split, name + ".json")))
    b = [[o["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for o in d["frames"][0].get("objects", [])
         if o["category"] == "car" and "box2d" in o]
    return np.array(b, np.float32).reshape(-1, 4)


def run(model, proc, split, names, dev):
    """Per output index: (#queries with score >= 0.5, #those matching a car GT at IoU 0.5 / 0.75)."""
    n = np.zeros(model.config.num_labels); tp50 = np.zeros_like(n); tp75 = np.zeros_like(n)
    with torch.no_grad():
        for name in names:
            im = Image.open(os.path.join(DATA, "100k", split, name + ".jpg")).convert("RGB")
            out = model(pixel_values=proc(images=im, return_tensors="pt")["pixel_values"].to(dev).half())
            s = out.logits.float().sigmoid()[0].cpu().numpy()                       # 300 x labels
            c = out.pred_boxes.float()[0].cpu().numpy()
            W, H = im.size
            xyxy = np.stack([(c[:, 0] - c[:, 2] / 2) * W, (c[:, 1] - c[:, 3] / 2) * H,
                             (c[:, 0] + c[:, 2] / 2) * W, (c[:, 1] + c[:, 3] / 2) * H], 1)
            g = car_gt(split, name)
            for lab in np.nonzero((s >= 0.5).any(0))[0]:
                q = s[:, lab] >= 0.5
                n[lab] += q.sum()
                if len(g):
                    best = iou_mat(xyxy[q], g).max(1)
                    tp50[lab] += (best >= 0.5).sum(); tp75[lab] += (best >= 0.75).sum()
    return n, tp50, tp75


def main():
    units = json.load(open(os.path.join(CACHE, "p1_units.json")))
    rng = random.Random(30)
    meta = rng.sample(sorted(n for ns in units["meta"].values() for n in ns), 300)
    train_names = sorted(f[:-5] for f in os.listdir(os.path.join(DATA, "100k", "train")) if f.endswith(".json"))
    train = rng.sample(train_names, 150)
    dev = torch.device("cuda")
    proc = load_processor()
    out = {}
    for e in EXPERTS:
        model = load_expert(e, dev).half()
        n, t50, t75 = run(model, proc, "test", meta, dev)
        lab = int(np.argmax(t50))
        nt, tt50, tt75 = run(model, proc, "train", train, dev)
        out[e] = {"car_label": lab, "id2label_name": model.config.id2label[lab],
                  "meta": {"n": int(n[lab]), "prec50": t50[lab] / max(n[lab], 1), "prec75": t75[lab] / max(n[lab], 1)},
                  "meta_150": None,
                  "train_150": {"n": int(nt[lab]), "prec50": tt50[lab] / max(nt[lab], 1), "prec75": tt75[lab] / max(nt[lab], 1)},
                  "other_labels_with_tp": {int(l): int(t50[l]) for l in np.nonzero(t50)[0] if l != lab}}
        n2, a50, a75 = run(model, proc, "test", meta[:150], dev)
        out[e]["meta_150"] = {"n": int(n2[lab]), "prec50": a50[lab] / max(n2[lab], 1), "prec75": a75[lab] / max(n2[lab], 1)}
        print(e, json.dumps(out[e], default=float))
        del model; torch.cuda.empty_cache()
    write_json(os.path.join(CACHE, "class_map.json"), out)


if __name__ == "__main__":
    main()
