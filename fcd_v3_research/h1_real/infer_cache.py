"""Run each expert once on the 100K keyframes of Meta-Train and Val; cache car-score, box and decoder feature
of every query with car score >= SCORE_FLOOR. Labels are NOT read here.

usage: python infer_cache.py <meta|val> <expert_k>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from h1common import CACHE, CAR_LABEL, DATA, SCORE_FLOOR
from h1common_model import load_expert, load_processor

SRC = {"meta": "test", "val": "val"}


class Keyframes(Dataset):
    def __init__(self, names, split):
        self.names, self.split = names, split
        self.proc = load_processor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(os.path.join(DATA, "100k", self.split, self.names[i] + ".jpg")).convert("RGB")
        pv = self.proc(images=im, return_tensors="pt")["pixel_values"][0]
        return pv, torch.tensor(im.size, dtype=torch.float32)  # (W, H)


def main(which, expert):
    units = json.load(open(os.path.join(CACHE, "p1_units.json")))[which]
    names = [n for r in sorted(units) for n in units[r]]
    dev = torch.device("cuda")
    model = load_expert(expert, dev).half()
    dl = DataLoader(Keyframes(names, SRC[which]), batch_size=8, num_workers=6, pin_memory=True)
    scores, boxes, feats, offsets = [], [], [], [0]
    t0 = time.time()
    with torch.no_grad():
        for b, (pv, wh) in enumerate(dl):
            out = model(pixel_values=pv.to(dev).half())
            s = out.logits.float().sigmoid()[..., CAR_LABEL]           # (B, 300)
            cxcywh = out.pred_boxes.float()
            xyxy = torch.cat([cxcywh[..., :2] - cxcywh[..., 2:] / 2, cxcywh[..., :2] + cxcywh[..., 2:] / 2], -1)
            xyxy = xyxy * torch.cat([wh, wh], -1).to(dev)[:, None, :]
            h = out.last_hidden_state                                  # (B, 300, 256) decoder output
            for j in range(s.shape[0]):
                keep = s[j] >= SCORE_FLOOR
                order = torch.argsort(s[j][keep], descending=True)
                scores.append(s[j][keep][order].cpu().numpy().astype(np.float32))
                boxes.append(xyxy[j][keep][order].cpu().numpy().astype(np.float32))
                feats.append(h[j][keep][order].cpu().numpy().astype(np.float16))
                offsets.append(offsets[-1] + int(keep.sum()))
            if b % 100 == 0:
                print(f"{which} {expert}: {(b + 1) * 8}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(CACHE, f"pred_{which}_{expert}.npz"), names=np.array(names),
             offsets=np.array(offsets, dtype=np.int64), scores=np.concatenate(scores),
             boxes=np.concatenate(boxes), feats=np.concatenate(feats))
    print(f"done {which} {expert} in {time.time() - t0:.0f}s, {offsets[-1]} queries", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
