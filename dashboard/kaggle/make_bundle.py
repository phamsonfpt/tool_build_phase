"""Build the folder that becomes the private Kaggle dataset used by the GPU worker.

    python dashboard/kaggle/make_bundle.py --user <kaggle user> [--out D:/m29_kaggle_bundle]
    kaggle datasets create  -p <out> --dir-mode zip            # first time (private by default)
    kaggle datasets version -p <out> --dir-mode zip -m "..."    # after code or checkpoint changes

Contents: ai_router code + calibration (centroids.pt, thresholds.yaml), the 3 expert checkpoints, the
RT-DETR r50vd architecture config from the local Hugging Face cache (the engine loads it offline), and the
dashboard files the worker imports. No BDD100K image or label goes into the bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from settings import AI_ROUTER, DATA_ROOT  # noqa: E402

DASH_FILES = ("engine.py", "settings.py", "worker.py", "jobs.py", "monitor.py")
CALIB_FILES = ("centroids.pt", "centroids_meta.json", "thresholds.yaml", "ltt_trace.json")
HF_MODEL = "models--PekingU--rtdetr_r50vd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle user name (dataset owner)")
    ap.add_argument("--slug", default="m29-ai-router-bundle")
    ap.add_argument("--out", default=str(Path.home() / "m29_kaggle_bundle"))
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache" / "huggingface" / "hub"))
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(AI_ROUTER / "src", out / "ai_router" / "src", ignore=ign)
    shutil.copytree(AI_ROUTER / "configs", out / "ai_router" / "configs", ignore=ign)
    (out / "ai_router" / "calibrated_params").mkdir(parents=True)
    for f in CALIB_FILES:
        if (AI_ROUTER / "calibrated_params" / f).is_file():
            shutil.copy2(AI_ROUTER / "calibrated_params" / f, out / "ai_router" / "calibrated_params" / f)
    (out / "ai_router" / "experiments").mkdir(parents=True)
    for i in (1, 2, 3):
        shutil.copy2(DATA_ROOT / "ai_router" / "experiments" / f"expert_{i}.pth", out / "ai_router" / "experiments" / f"expert_{i}.pth")
    src = Path(a.hf_cache) / HF_MODEL
    ref = (src / "refs" / "main").read_text().strip()
    dst = out / "hf_cache" / HF_MODEL
    (dst / "refs").mkdir(parents=True)
    (dst / "refs" / "main").write_text(ref)
    (dst / "snapshots" / ref).mkdir(parents=True)
    for f in ("config.json", "preprocessor_config.json"):          # architecture only, no pretrained weights
        if (src / "snapshots" / ref / f).exists():
            shutil.copyfile(src / "snapshots" / ref / f, dst / "snapshots" / ref / f)
    (out / "dashboard").mkdir()
    for f in DASH_FILES:
        shutil.copy2(HERE / f, out / "dashboard" / f)
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": a.slug, "id": f"{a.user}/{a.slug}", "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}  {size / 2**20:.0f} MB")
    print(f"kaggle datasets create -p \"{out}\" --dir-mode zip")


if __name__ == "__main__":
    main()
