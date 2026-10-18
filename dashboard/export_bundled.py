"""Write dashboard/bundled/: the derived data a GPU-less web server needs, safe for a public repository.

    python dashboard/export_bundled.py          # needs the local GPU setup (loads the 3 experts)

- train_stats.json      aggregate statistics only (no per-image labels; train_index.npz stays out)
- system.json           detector / threshold summary shown on the Test page
- centroids.json        FCD class centroids and inverse covariances (step B5)
- examples/<job>/       results of the two examples: job.json, summary.json, monitoring.json, trace/,
                        explain/*.json.gz (the single example + the first 30 of the review queue). Image
                        paths and ground truth are removed: BDD100K images and
                        labels stay out of the public repository (license: non-profit use with its copyright notice). The pages get the images from a local worker.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

from settings import BUNDLED, JOBS, STORE

EXAMPLES = ("example-single", "example-batch")
BATCH_EXPLAINS = 30


def scrub(x):
    """Local absolute paths (D:/..., C:\\...) -> file name only; the bundle goes into a public repository."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str) and len(x) > 2 and x[1] == ":" and x[2] in "/\\":
        return x.replace("\\", "/").rsplit("/", 1)[-1]
    return x


def scrub_tree(root: Path):
    for p in root.rglob("*"):
        if p.suffix == ".json":
            p.write_text(json.dumps(scrub(json.loads(p.read_text(encoding="utf-8"))), ensure_ascii=False), encoding="utf-8")
        elif p.suffix == ".jsonl":
            lines = [json.dumps(scrub(json.loads(l)), ensure_ascii=False) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_gt(d: dict) -> dict:
    d.pop("gt", None)
    for o in d.get("objects", []):
        o.pop("tp", None)
    return d


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from engine import Engine
    from jobs import batch_context
    eng = Engine()
    BUNDLED.mkdir(exist_ok=True)
    shutil.copy2(STORE / "train_stats.json", BUNDLED / "train_stats.json")
    sysinfo = eng.system()
    sysinfo["ai_router"] = None
    (BUNDLED / "system.json").write_text(json.dumps(sysinfo), encoding="utf-8")
    cents = {c: eng.centroid(c) for c in eng.names}
    (BUNDLED / "centroids.json").write_text(json.dumps({k: v for k, v in cents.items() if not v.get("missing")}), encoding="utf-8")

    for jid in EXAMPLES:
        src, dst = JOBS / jid, BUNDLED / "examples" / jid
        if not (src / "summary.json").is_file():
            sys.exit(f"{jid} chưa chạy xong: mở server ở chế độ local một lần")
        shutil.rmtree(dst, ignore_errors=True)
        (dst / "trace").mkdir(parents=True)
        job = json.loads((src / "job.json").read_text(encoding="utf-8"))
        paths = job["images"]
        job["images"] = {n: "" for n in paths}
        job.pop("traceback", None)
        (dst / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        for f in ("summary.json", "monitoring.json"):
            shutil.copy2(src / f, dst / f)
        for f in ("spans.jsonl", "metrics.jsonl", "final_zones.jsonl", "run.json"):
            if (src / "trace" / f).is_file():
                shutil.copy2(src / "trace" / f, dst / "trace" / f)
        (dst / "explain").mkdir()
        # every explain is ~90 KB compressed: ship the single example and the head of the review queue;
        # any other image is explained on demand by a worker that has the original
        red = sorted((im for im in summary["images"] if im["zone"] == "RED"),
                     key=lambda im: (im["priority"], -(im["img_unc"] or 0)))
        keep = list(paths) if len(paths) == 1 else [im["name"] for im in red[:BATCH_EXPLAINS]]
        for i, name in enumerate(keep):
            cache = src / "explain" / f"{name}.json"
            if cache.is_file():
                d = json.loads(cache.read_text(encoding="utf-8"))
            else:
                d = eng.explain(Path(paths[name]), batch_context(summary, name))
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(json.dumps(d), encoding="utf-8")
            (dst / "explain" / f"{name}.json.gz").write_bytes(gzip.compress(json.dumps(strip_gt(d)).encode(), 9))
            if (i + 1) % 100 == 0:
                print(jid, i + 1, "/", len(paths), flush=True)
    scrub_tree(BUNDLED)
    size = sum(p.stat().st_size for p in BUNDLED.rglob("*") if p.is_file())
    print(f"{BUNDLED}  {size / 2**20:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
