"""Ground-truth statistics of the BDD100K train split (the data the experts were trained on).

    python train_stats.py            # about 1-2 min for 70,000 label files

Writes
  data/train_stats.json   per class: share of boxes, mean / variance of box width, height, area, aspect
                          ratio and centre; COCO size buckets; objects per image; scene attributes; the
                          FCD latent space (centroids + real train vectors projected to 3-D)
  data/train_index.npz    every train image with its attributes and GT boxes (for the filtered gallery)
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from settings import AI_ROUTER, SPLITS, STORE, WORKTREE

sys.path.insert(0, str(AI_ROUTER))
from src.config import load_config  # noqa: E402

OUT = STORE / "train_stats.json"
INDEX = STORE / "train_index.npz"
GEOM = ("w", "h", "area", "aspect", "cx", "cy")
ATTRS = ("weather", "scene", "timeofday")
SMALL, LARGE = 32 ** 2, 96 ** 2                    # COCO size buckets (px²)
TRAIN_VECTORS = WORKTREE / "threshold_sweep" / "component" / "train_anchor_vectors.pkl"
POINTS_PER_CLASS = 400


def read(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    objs = d["frames"][0].get("objects", []) if d.get("frames") else (d.get("labels") or [])
    return path.stem, d.get("attributes", {}), objs


def latent_space(cfg, names):
    """FCD space of ai_router: 256-D anchor vector -> PCA 64-D (fitted on train) -> per-class mean and
    covariance. Here the 64-D space is reduced once more to 3-D (PCA on the class centroids' space) only
    to draw it; distances are always computed in 64-D."""
    import torch
    cdir = AI_ROUTER / cfg["paths"]["calibrated_params"]
    p = torch.load(cdir / "centroids.pt", map_location="cpu", weights_only=True)
    meta = json.loads((cdir / "centroids_meta.json").read_text(encoding="utf-8"))
    pmean, comps = p["pca_mean"].numpy().astype(np.float64), p["pca_components"].numpy().astype(np.float64)
    mu, inv = p["mu"].numpy().astype(np.float64), p["inv_cov"].numpy().astype(np.float64)
    ids = p["class_ids"].numpy().tolist()
    slot = {c: i for i, c in enumerate(ids)}

    pts = None
    if TRAIN_VECTORS.is_file():                     # the 23,023 train vectors setup_centroids.py fitted on
        # our own cache, written earlier by threshold_sweep/component (not downloaded data)
        with open(TRAIN_VECTORS, "rb") as fh:
            emb, lab = pickle.load(fh)
        z = (emb.astype(np.float64) - pmean) @ comps.T
        center = z.mean(0)
        _, sv, vt = np.linalg.svd(z - center, full_matrices=False)
        var = sv ** 2
        P = vt[:3]
        rng = np.random.default_rng(0)
        keep = np.concatenate([rng.permutation(np.flatnonzero(lab == c))[:POINTS_PER_CLASS] for c in np.unique(lab)])
        keep.sort()
        d = np.full(len(keep), np.nan)
        for j, i in enumerate(keep):
            s = slot.get(int(lab[i]))
            if s is not None:
                dz = z[i] - mu[s]
                d[j] = np.sqrt(max(dz @ inv[s] @ dz, 0))
        xyz = (z[keep] - center) @ P.T
        pts = {"xyz": np.round(xyz, 3).tolist(), "cls": lab[keep].astype(int).tolist(), "d": np.round(d, 3).tolist(),
               "n_total": int(len(z)), "per_class_cap": POINTS_PER_CLASS,
               "explained": np.round(var[:3] / var.sum(), 4).tolist()}
    else:                                           # fall back to the centroids only
        center = mu.mean(0)
        _, _, vt = np.linalg.svd(mu - center, full_matrices=False)
        P = vt[:3]
    classes = []
    for i, cid in enumerate(ids):
        cov = np.linalg.inv(inv[i])
        nm = names[cid]
        ev = np.linalg.eigvalsh(cov)
        classes.append({"name": nm, "n_vectors": meta["per_class"].get(nm, {}).get("n"),
                        "pooled_cov": meta["per_class"].get(nm, {}).get("pooled_cov"),
                        "mu_norm": float(np.linalg.norm(mu[i])), "trace_cov": float(np.trace(cov)),
                        "max_eig": float(ev.max()), "min_eig": float(ev.min()),
                        "xyz": np.round((mu[i] - center) @ P.T, 3).tolist()})
    dist = [[float(np.linalg.norm(mu[a] - mu[b])) for b in range(len(ids))] for a in range(len(ids))]
    return {"source": "ai_router calibrated_params/centroids.pt", "images": meta["images"],
            "pca_dim": int(mu.shape[1]), "emb_dim": int(comps.shape[1]),
            "pca_explained_variance": meta["pca_explained_variance"], "classes": classes,
            "centroid_dist": np.round(dist, 3).tolist(), "points": pts}


def main():
    cfg = load_config()
    names = cfg["classes"]["names"]
    gt_map = cfg["classes"]["gt_map"]
    files = sorted(SPLITS["train"].glob("*.json"))
    print(len(files), "label files", flush=True)
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(read, files, chunksize=256))

    geom = defaultdict(lambda: defaultdict(list))
    per_image, attrs, by_class_images = [], defaultdict(Counter), defaultdict(int)
    attr_class = {k: defaultdict(Counter) for k in ATTRS}          # attribute value -> class -> boxes
    idx_names, idx_attr, idx_off, idx_boxes, idx_cls = [], {k: [] for k in ATTRS}, [0], [], []
    for stem, at, objs in rows:
        kept = []
        for o in objs:
            b, c = o.get("box2d"), gt_map.get(o.get("category"))
            if b is None or c is None:
                continue
            x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
            w, h = x2 - x1, y2 - y1
            if w <= 0.5 or h <= 0.5:
                continue
            g = geom[c]
            g["w"].append(w); g["h"].append(h); g["area"].append(w * h); g["aspect"].append(w / h)
            g["cx"].append((x1 + x2) / 2); g["cy"].append((y1 + y2) / 2)
            kept.append((x1, y1, x2, y2, names.index(c)))
        per_image.append(len(kept))
        for k in ATTRS:
            v = at.get(k, "undefined")
            attrs[k][v] += 1
            idx_attr[k].append(v)
            for b in kept:
                attr_class[k][v][names[b[4]]] += 1
        for c in {b[4] for b in kept}:
            by_class_images[names[c]] += 1
        idx_names.append(stem + ".jpg")
        idx_boxes.extend(b[:4] for b in kept); idx_cls.extend(b[4] for b in kept)
        idx_off.append(len(idx_cls))

    total = sum(len(geom[c]["w"]) for c in names)
    classes = []
    for c in names:
        g = geom[c]
        n = len(g["w"])
        row = {"name": c, "boxes": n, "share": n / total if total else 0, "images": by_class_images[c]}
        for k in GEOM:
            v = np.asarray(g[k], dtype=np.float64)
            row[k] = {"mean": float(v.mean()) if n else None, "var": float(v.var()) if n else None,
                      "std": float(v.std()) if n else None, "p50": float(np.median(v)) if n else None}
        a = np.asarray(g["area"], dtype=np.float64)
        row["size"] = {"small": int((a < SMALL).sum()), "medium": int(((a >= SMALL) & (a < LARGE)).sum()), "large": int((a >= LARGE).sum())}
        la = np.log10(np.maximum(a, 1))
        row["log_area_hist"] = np.histogram(la, bins=24, range=(0, 6))[0].tolist() if n else [0] * 24
        classes.append(row)

    n_img = np.asarray(per_image)
    out = {"split": str(SPLITS["train"]), "images": len(rows), "boxes": total, "class_names": names,
           "images_with_boxes": int((n_img > 0).sum()), "classes": classes,
           "size_buckets": {"small_lt": SMALL, "large_ge": LARGE},
           "objects_per_image": {"mean": float(n_img.mean()), "var": float(n_img.var()), "median": float(np.median(n_img)),
                                 "max": int(n_img.max()), "zero": int((n_img == 0).sum()),
                                 "hist": np.bincount(np.minimum(n_img, 80)).tolist()},
           "attributes": {k: dict(v.most_common()) for k, v in attrs.items()},
           "attribute_classes": {k: {v: dict(c) for v, c in d.items()} for k, d in attr_class.items()},
           "latent": latent_space(cfg, names)}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    np.savez_compressed(INDEX, names=np.asarray(idx_names), offsets=np.asarray(idx_off, np.int64),
                        boxes=np.asarray(idx_boxes, np.float32).reshape(-1, 4), cls=np.asarray(idx_cls, np.int8),
                        n=n_img.astype(np.int16), **{k: np.asarray(v) for k, v in idx_attr.items()})
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB", INDEX, f"{INDEX.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
