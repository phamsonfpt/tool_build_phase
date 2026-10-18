"""Smoke test: GPU engine on the example image and a 60-image batch (needs CUDA + checkpoints)."""
import json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EXAMPLE_SINGLE, EXAMPLE_BATCH_DIR, STORE
from engine import Engine

t = time.time(); e = Engine(); print("load", round(time.time() - t, 1), "s", e.device)
d = e.explain(EXAMPLE_SINGLE)
bad = [c for c in d["checks"] if not c["ok"]]
print("explain:", len(d["objects"]), "objects, zone", d["B5"]["zone"], d["B5"]["reasons"], "checks", len(d["checks"]), "failed", bad)
print("json kb", len(json.dumps(d)) // 1024, "gt", d.get("gt", {}).get("tp"), d.get("gt", {}).get("fp"), d.get("gt", {}).get("fn"))
jd = STORE / "smoke_job"; shutil.rmtree(jd, ignore_errors=True); jd.mkdir()
paths = sorted(EXAMPLE_BATCH_DIR.glob("*.jpg"))[:60]
t = time.time(); s = e.run_batch(paths, jd, lambda a, b: None); print("batch", round(time.time() - t, 1), "s")
print("zones", s["zones"], "b5", s["zones_b5"], "budget", s["budget"], "kmeans ran", s["kmeans"]["ran"], "replicates", s["kmeans"].get("replicates_pipeline"))
print("latent pts", len(s["latent"]["x"]), "explained", s["latent"]["explained"], "summary kb", (jd / "summary.json").stat().st_size // 1024)
from monitor import job_monitoring
m = job_monitoring(jd, s); print("alerts", m["alerts"]); print("spans", [x["name"] for x in m["system"]["spans"]])
