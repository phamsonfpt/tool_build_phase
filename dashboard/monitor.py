"""Monitoring of one job: system (per-batch speed, GPU memory, time per step from the pipeline's own
tracer) and data (is the uploaded data like the train data and like the reference batch?)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import ks_2samp

from settings import JOBS, data_file

REFERENCE_JOB = "example-batch"


def _jsonl(p: Path):
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _arr(x):
    return np.asarray([v for v in x if v is not None], dtype=np.float64)


def _per_image(s: Dict, key: str) -> np.ndarray:
    """Mean of an object quantity per image (images without a finite value are skipped)."""
    o = s["objects"]
    if "img" not in o:                                  # summary written before the image index was stored
        return np.zeros(0)
    acc = defaultdict(list)
    for i, v in zip(o["img"], o[key]):
        if v is not None:
            acc[i].append(v)
    return np.asarray([np.mean(v) for v in acc.values()], dtype=np.float64)


def job_monitoring(job_dir: Path, s: Dict) -> Dict:
    tr = Path(job_dir) / "trace"
    spans = _jsonl(tr / "spans.jsonl")
    agg = defaultdict(lambda: {"n": 0, "ms": 0.0})
    for sp in spans:
        a = agg[sp["name"]]
        a["n"] += 1
        a["ms"] += sp["dur_ms"]
    system = {"metrics": _jsonl(tr / "metrics.jsonl"),
              "spans": [{"name": k, "n": v["n"], "total_ms": round(v["ms"], 1), "mean_ms": round(v["ms"] / v["n"], 2)}
                        for k, v in agg.items()],
              "timeline": [{k: sp.get(k) for k in ("name", "start_s", "dur_ms", "batch")} for sp in spans[:3000]],
              "run": json.loads((tr / "run.json").read_text(encoding="utf-8")) if (tr / "run.json").is_file() else None}

    alerts = []
    data: Dict = {}
    # ---- against train ground truth
    tp = data_file("train_stats.json")
    if tp.is_file():
        t = json.loads(tp.read_text(encoding="utf-8"))
        names = t["class_names"]
        cls = np.asarray(s["objects"]["cls"], int)
        det_share = np.bincount(cls, minlength=len(names)) / max(len(cls), 1)
        gt_share = np.array([c["share"] for c in t["classes"]])
        tvd = 0.5 * float(np.abs(det_share - gt_share).sum())
        data["class_share"] = [{"name": n, "detected": round(float(a), 4), "train_gt": round(float(b), 4)}
                               for n, a, b in zip(names, det_share, gt_share)]
        data["class_share_tvd"] = round(tvd, 4)
        if tvd > 0.15 and len(cls) >= 50:
            alerts.append({"level": "warn", "text": f"Tỉ lệ lớp được phát hiện lệch so với GT train: TVD = {tvd:.2f} (> 0.15)."})
        n_obj = np.array([im["n"] for im in s["images"]])
        data["objects_per_image"] = {"batch_mean": round(float(n_obj.mean()), 2) if len(n_obj) else None,
                                     "train_gt_mean": round(t["objects_per_image"]["mean"], 2),
                                     "hist": np.bincount(np.minimum(n_obj, 80)).tolist() if len(n_obj) else []}
    sizes = defaultdict(int)
    for im in s["images"]:
        sizes[f"{im['size'][0]}×{im['size'][1]}"] += 1
    data["image_sizes"] = dict(sizes)
    if any(k != "1280×720" for k in sizes):
        alerts.append({"level": "info", "text": "Có ảnh khác cỡ 1280×720 của BDD100K: ngưỡng được calibrate trên ảnh 1280×720."})
    empty = sum(im["n"] == 0 for im in s["images"])
    data["empty_images"] = empty
    if s["images"] and empty / len(s["images"]) > 0.10:
        alerts.append({"level": "warn", "text": f"{empty} ảnh không có object nào ≥ ngưỡng score ({100 * empty / len(s['images']):.0f} %)."})

    # ---- against the reference batch (1,000 BDD100K test images run through the same pipeline)
    ref_p = JOBS / REFERENCE_JOB / "summary.json"
    if ref_p.is_file() and Path(job_dir).name != REFERENCE_JOB:
        r = json.loads(ref_p.read_text(encoding="utf-8"))
        tests = []
        # one value per image: objects of the same image are not independent, so a KS test over
        # objects would report p-values far too small
        for key, label, a, b in [
            ("img_unc", "img_unc (max entropy ảnh)", _arr(im["img_unc"] for im in s["images"] if im["n"]), _arr(im["img_unc"] for im in r["images"] if im["n"])),
            ("img_conf", "img_conf (min confidence ảnh)", _arr(im["img_conf"] for im in s["images"] if im["n"]), _arr(im["img_conf"] for im in r["images"] if im["n"])),
            ("obj_conf", "mean_conf trung bình của ảnh", _per_image(s, "conf"), _per_image(r, "conf")),
            ("obj_fcd", "D_M trung bình của ảnh (FCD)", _per_image(s, "d"), _per_image(r, "d"))]:
            if len(a) >= 2 and len(b) >= 2:
                ks = ks_2samp(a, b)
                tests.append({"key": key, "label": label, "n": int(len(a)), "n_ref": int(len(b)),
                              "mean": round(float(a.mean()), 4), "mean_ref": round(float(b.mean()), 4),
                              "ks": round(float(ks.statistic), 4), "p": float(ks.pvalue)})
                if ks.pvalue < 0.01 and len(a) >= 30:
                    alerts.append({"level": "warn", "text": f"{label} lệch so với lô tham chiếu (KS = {ks.statistic:.2f}, p = {ks.pvalue:.1e})."})
        data["vs_reference"] = {"job": REFERENCE_JOB, "tests": tests,
                                "zones_ref": r["zones"], "red_share_ref": round(r["zones_b5"]["RED"] / r["n_images"], 4)}
    data["red_share"] = round(s["zones_b5"]["RED"] / max(s["n_images"], 1), 4)
    if s["zones"]["GREEN"] == 0:
        alerts.append({"level": "info", "text": f"Vùng Xanh rỗng: tau_conf = {s['thresholds']['tau_conf']} > 1 nên không ảnh nào đạt (LTT không chứng nhận được)."})
    return {"system": system, "data": data, "alerts": alerts}
