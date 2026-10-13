"""Per-image counts on the whole grid G = TAU_LOC x TAU_FCD x TAU_CLS, for theta_IoU in {0.75, 0.5}.

Signals of an anchor box b (anchor expert = the detector M under test, the others form the ensemble):
  cls  = anchor car score
  loc  = mean over other experts of max IoU(b, boxes of that expert with score >= MATCH_FLOOR)   (0 if none)
  fcd  = mean over other experts of 1 - cos(h_b, h_match) on decoder output features            (1 if none)
Acceptance at tau = (t_cls, t_loc, t_fcd):  cls >= t_cls  and  loc >= t_loc  and  fcd <= t_fcd.

Greedy matching (Section 2.1) runs once per (loc, fcd) filter in score order; because a box's match only
depends on higher-scored accepted boxes, the counts at every t_cls are prefix sums of that single pass.

usage: python build_features.py <meta|val> <anchor_expert>
"""
import json
import os
import sys

import numba
import numpy as np

from h1common import CACHE, DATA, EXPERTS, FCD_QUANTILES, TAU_CLS, TAU_LOC

THETAS = (0.75, 0.5)
MATCH_FLOOR = 0.10
SRC = {"meta": "test", "val": "val"}


def load_pred(which, expert):
    z = np.load(os.path.join(CACHE, f"pred_{which}_{expert}.npz"))
    return {k: z[k] for k in z.files}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


@numba.njit(cache=True)
def greedy_tp(iou, accept, theta):
    """iou: (n_pred sorted by score desc, n_gt in file order). Returns TP flag per prediction."""
    n_p, n_g = iou.shape
    used = np.zeros(n_g, np.bool_)
    tp = np.zeros(n_p, np.bool_)
    for i in range(n_p):
        if not accept[i]:
            continue
        best, bj = -1.0, -1
        for j in range(n_g):          # strict '>' keeps the first GT in file order on ties
            if not used[j] and iou[i, j] >= theta and iou[i, j] > best:
                best, bj = iou[i, j], j
        if bj >= 0:
            used[bj] = True
            tp[i] = True
    return tp


def gt_boxes(split, name):
    d = json.load(open(os.path.join(DATA, "100k", split, name + ".json")))
    b = [[o["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for o in d["frames"][0].get("objects", [])
         if o["category"] == "car" and "box2d" in o]
    return np.array(b, np.float32).reshape(-1, 4)


def main(which, anchor):
    others = [e for e in EXPERTS if e != anchor]
    A = load_pred(which, anchor)
    O = [load_pred(which, e) for e in others]
    names = A["names"]
    for o in O:
        assert (o["names"] == names).all()
    n_img = len(names)
    sig = []                                           # per-image (cls, loc, fcd, boxes)
    for i in range(n_img):
        a0, a1 = A["offsets"][i], A["offsets"][i + 1]
        keep = A["scores"][a0:a1] >= TAU_CLS[0]
        s, bx = A["scores"][a0:a1][keep], A["boxes"][a0:a1][keep]
        f = A["feats"][a0:a1][keep].astype(np.float32)
        loc = np.zeros(len(s), np.float32); fcd = np.zeros(len(s), np.float32)
        for o in O:
            o0, o1 = o["offsets"][i], o["offsets"][i + 1]
            ok = o["scores"][o0:o1] >= MATCH_FLOOR
            ob, of = o["boxes"][o0:o1][ok], o["feats"][o0:o1][ok].astype(np.float32)
            if len(s) == 0:
                continue
            if len(ob) == 0:
                fcd += 1.0
                continue
            m = iou_mat(bx, ob)
            j = m.argmax(1); best = m.max(1)
            cos = (f * of[j]).sum(1) / np.maximum(np.linalg.norm(f, axis=1) * np.linalg.norm(of[j], axis=1), 1e-9)
            loc += best
            fcd += np.where(best > 0, 1 - cos, 1.0)
        sig.append((s, loc / len(O), fcd / len(O), bx))

    thr_path = os.path.join(CACHE, f"fcd_thresholds_{anchor}.json")
    if which == "meta":   # tau_fcd grid = quantiles of fcd among Meta-Train boxes with score >= 0.3
        allf = np.concatenate([g[2][g[0] >= 0.3] for g in sig])
        tau_fcd = [float("inf") if q == 1.0 else float(np.quantile(allf, q)) for q in FCD_QUANTILES]
        json.dump({"quantiles": FCD_QUANTILES, "tau_fcd": tau_fcd}, open(thr_path, "w"))
    else:                 # Val always reuses the Meta-Train thresholds
        tau_fcd = json.load(open(thr_path))["tau_fcd"]
    tau_fcd = np.array(tau_fcd)

    shape = (n_img, len(THETAS), len(TAU_LOC), len(tau_fcd), len(TAU_CLS))
    N = np.zeros(shape, np.int16); TP = np.zeros(shape, np.int16); VH = np.zeros(shape, np.int16)
    M = np.zeros(n_img, np.int16)
    for i in range(n_img):
        s, loc, fcd, bx = sig[i]
        g = gt_boxes(SRC[which], names[i])
        M[i] = len(g)
        if len(s) == 0:
            continue
        iou = iou_mat(bx, g)
        maxiou = iou.max(1) if len(g) else np.zeros(len(s), np.float32)
        cls_bin = np.searchsorted(TAU_CLS, s, side="right")    # box counts for every t_cls <= its score
        for li, tl in enumerate(TAU_LOC):
            for fi, tf in enumerate(tau_fcd):
                acc = (loc >= tl) & (fcd <= tf)
                for ti, th in enumerate(THETAS):
                    tp = greedy_tp(iou, acc, th)
                    hal = acc & ~tp & (maxiou < 0.5)
                    # count per bin then reverse-cumsum: N(t_k) = #accepted boxes with score >= t_k
                    for arr, flag in ((N, acc), (TP, tp), (VH, hal)):
                        c = np.bincount(cls_bin[flag], minlength=len(TAU_CLS) + 1)[1:]
                        arr[i, ti, li, fi] = np.cumsum(c[::-1])[::-1]
    np.savez_compressed(os.path.join(CACHE, f"counts_{which}_{anchor}.npz"), names=names, N=N, TP=TP, VH=VH, M=M,
                        tau_fcd=tau_fcd, thetas=np.array(THETAS))
    print(which, anchor, "images", n_img, "boxes>=0.05", sum(len(g[0]) for g in sig), "GT", int(M.sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
