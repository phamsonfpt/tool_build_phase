"""Web mode without a GPU: optional password, worker token, claim -> result -> done, explain request round
trip, BDD seed refused on a public server, upload limits, path traversal. Needs only the repository: the
results come from bundled/, the image is generated (no BDD100K image is involved).

    python dashboard/tests/test_web_mode.py
"""
import base64
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC_JOB = HERE / "bundled" / "examples" / "example-single"
store = Path(tempfile.mkdtemp(prefix="m29web_"))
os.environ.update(M29_MODE="web", M29_DASH_STORE=str(store), WORKER_TOKEN="tok", DASH_PASSWORD="pw")
sys.path.insert(0, str(HERE))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from jobs import JobManager  # noqa: E402

server.jm = JobManager(local_worker=False)
server.ensure_examples()
c = TestClient(server.app)
USER = {"Authorization": "Basic " + base64.b64encode(b"nhom:pw").decode()}
W = {"X-Worker-Token": "tok", "X-Worker-Name": "pc"}
fails = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def tar(files):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return b.getvalue()


check("pages need the team password", c.get("/").status_code == 401 and c.get("/api/jobs").status_code == 401)
check("wrong password refused", c.get("/api/jobs", headers={"Authorization": "Basic " + base64.b64encode(b"x:no").decode()}).status_code == 401)
check("health check is open", c.get("/healthz").json()["mode"] == "web")
check("worker API needs the token", c.post("/api/worker/claim", json={"name": "pc"}).status_code == 401)
check("bundled examples are loaded", {"example-single", "example-batch"} <= set(server.jm.jobs))
check("no worker yet", c.get("/api/system", headers=USER).json()["engine_loaded"] is False)
check("nothing to do -> 204", c.post("/api/worker/claim", json={"name": "pc"}, headers=W).status_code == 204)

from PIL import Image  # noqa: E402
_b = io.BytesIO()
Image.new("RGB", (1280, 720), (90, 110, 130)).save(_b, "JPEG")
img = _b.getvalue()
r = c.post("/api/jobs", files=[("files", ("my photo.jpg", img, "image/jpeg"))], data={"title": "t"}, headers=USER)
jid = r.json()["id"]
task = c.post("/api/worker/claim", json={"name": "pc"}, headers=W).json()
check("worker gets the job", task["type"] == "run" and task["job"] == jid and task["images"] == ["my_photo.jpg"])
check("job is running on pc", c.get(f"/api/jobs/{jid}", headers=USER).json()["worker"] == "pc")
check("worker downloads the original", c.get(f"/api/worker/jobs/{jid}/image/my_photo.jpg", headers=W).content == img)
c.post(f"/api/worker/jobs/{jid}/progress", json={"name": "pc", "done": 1, "total": 1, "phase": "B0→B7"}, headers=W)
check("progress recorded", c.get(f"/api/jobs/{jid}", headers=USER).json()["progress"]["phase"] == "B0→B7")

summary = json.loads((SRC_JOB / "summary.json").read_text(encoding="utf-8"))
summary["images"][0]["name"] = "my_photo.jpg"
evil = tar({"../../escape.txt": b"x", "summary.json": json.dumps(summary).encode(),
            "trace/spans.jsonl": (SRC_JOB / "trace" / "spans.jsonl").read_bytes(), "secret/other.txt": b"y"})
r = c.post(f"/api/worker/jobs/{jid}/result", content=evil, headers={**W, "X-Worker-Meta": json.dumps({"pipeline_s": 1.5})})
check("result accepted", r.status_code == 200)
jd = c.get(f"/api/jobs/{jid}", headers=USER).json()
check("job done with monitoring", jd["status"] == "done" and jd["monitoring_ready"])
check("archive cannot write outside the job", not (store / "escape.txt").exists() and not (store / "jobs" / "escape.txt").exists()
      and not (store / "jobs" / jid / "secret").exists())
check("gpu stage carries the worker timing", any(s["name"] == "gpu_worker" and s.get("pipeline_s") == 1.5 for s in jd["stages"]))

r = c.get(f"/api/jobs/{jid}/explain/my_photo.jpg", headers=USER)
check("missing explain -> 202 pending", r.status_code == 202 and r.json()["pending"])
task = c.post("/api/worker/claim", json={"name": "pc"}, headers=W).json()
check("worker gets the explain request first", task["type"] == "explain" and task["image"] == "my_photo.jpg")
ex = json.dumps({"image": "my_photo.jpg", "objects": []}).encode()
c.post(f"/api/worker/jobs/{jid}/explain/my_photo.jpg", content=ex, headers=W)
check("explain served after upload", c.get(f"/api/jobs/{jid}/explain/my_photo.jpg", headers=USER).json()["image"] == "my_photo.jpg")
other = next(n for n in json.loads((HERE / "bundled" / "examples" / "example-batch" / "job.json").read_text(encoding="utf-8"))["images"]
             if not (HERE / "bundled" / "examples" / "example-batch" / "explain" / f"{n}.json.gz").is_file())
check("example image without bundled explain -> 202", c.get(f"/api/jobs/example-batch/explain/{other}", headers=USER).status_code == 202)
check("kaggle (no originals) is not given an example image", c.post("/api/worker/claim", json={"name": "kaggle"}, headers=W).status_code == 204)
c.post("/api/worker/hello", json={"name": "pc", "has_originals": True}, headers=W)
task = c.post("/api/worker/claim", json={"name": "pc"}, headers=W).json()
check("a worker with originals gets it", task["type"] == "explain" and task["image"] == other)
c.post(f"/api/worker/jobs/example-batch/explain-fail/{other}", json={"name": "pc", "error": "không có ảnh gốc"}, headers=W)
r = c.get(f"/api/jobs/example-batch/explain/{other}", headers=USER)
check("worker failure is shown once", r.status_code == 409 and "ảnh gốc" in r.json()["detail"])
check("failure is remembered: no new GPU work", c.get(f"/api/jobs/example-batch/explain/{other}", headers=USER).status_code == 409
      and not server.jm.explain_q)
key = ("example-batch", other)
server.jm.explain_errors[key] = (server.jm.explain_errors[key][0], 0)       # pretend 10 minutes passed
check("after the wait it retries", c.get(f"/api/jobs/example-batch/explain/{other}", headers=USER).status_code == 202)
check("example explains alone do not start Kaggle", server.jm.pending() and not server.jm.pending(gpu_can_do=True))
import jobs as jobs_mod
jobs_mod.MAX_EXPLAIN_QUEUE = len(server.jm.explain_q)
names = list(json.loads((HERE / "bundled" / "examples" / "example-batch" / "job.json").read_text(encoding="utf-8"))["images"])
more = next(n for n in names if (("example-batch", n) not in server.jm.explain_q) and not (HERE / "bundled" / "examples" / "example-batch" / "explain" / f"{n}.json.gz").is_file())
check("on-demand explain queue is capped", c.get(f"/api/jobs/example-batch/explain/{more}", headers=USER).status_code == 429)
(server.jm.jobs[jid].dir / "explain" / "my_photo.jpg.json").unlink()
check("example requests cannot block uploaded images", c.get(f"/api/jobs/{jid}/explain/my_photo.jpg", headers=USER).status_code == 202)
for e in server.jm.explain_q.values():
    e["t"] = 0                                                          # nobody picked them up for 10 minutes
check("stale requests expire", not server.jm.pending() and not server.jm.explain_q)
jobs_mod.MAX_EXPLAIN_QUEUE = 20
(store / "jobs" / "leak.json").write_text('{"secret": 1}', encoding="utf-8")
check("explain name cannot leave the job folder", all(
    c.get(f"/api/jobs/{jid}/explain/{n}", headers=USER).status_code == 404
    for n in ("..%5C..%5Cleak", "..%5Cleak", "%2E%2E%5C%2E%2E%5Cleak", "not-in-job.jpg")))
check("bundled example explain is served (gzip)", c.get("/api/jobs/example-single/explain/cd92dc38-3ee11e29.jpg", headers=USER).status_code == 200)
check("bundled explain has no ground truth", "gt" not in c.get("/api/jobs/example-single/explain/cd92dc38-3ee11e29.jpg", headers=USER).json())

check("example image missing before seed", c.get("/api/jobs/example-single/image/cd92dc38-3ee11e29.jpg", headers=USER).status_code == 404)
check("public server refuses BDD seed", c.post("/api/worker/seed", content=tar({"examples/cd92dc38-3ee11e29.jpg": img}), headers=W).status_code == 403)
check("hello tells the worker seeding is off", c.post("/api/worker/hello", json={"name": "pc"}, headers=W).json()["seed"]["allowed"] is False)
check("gallery explains why there are no train images", "license" in c.get("/api/train/gallery", headers=USER).json()["detail"])
server.ALLOW_SEED = True                                   # private deployment
r = c.post("/api/worker/seed", content=tar({"examples/cd92dc38-3ee11e29.jpg": img, "../x.jpg": b"no", "train/a/b.jpg": b"no"}), headers=W)
check("seed stores only allowed files", r.json()["example_images"] == 1 and not (store / "x.jpg").exists())
check("example image served after seed", c.get("/api/jobs/example-single/image/cd92dc38-3ee11e29.jpg", headers=USER).status_code == 200)
check("gallery explains it is not seeded", c.get("/api/train/gallery", headers=USER).status_code == 503)

# public pages and upload limits
server.PASSWORD = ""
check("no password -> pages open to everyone", c.get("/api/jobs").status_code == 200 and c.get("/").status_code == 200)
check("system reports public + no BDD images", c.get("/api/system").json()["public"] is True)
server.MAX_JOB_IMAGES = 2
r = c.post("/api/jobs", files=[("files", (f"a{i}.jpg", img, "image/jpeg")) for i in range(3)])
check("too many images refused", r.status_code == 413 and "Tối đa 2" in r.json()["detail"])
server.MAX_JOB_IMAGES, server.MAX_QUEUED_JOBS = 300, 1
c.post("/api/jobs", files=[("files", ("q1.jpg", img, "image/jpeg"))])
r = c.post("/api/jobs", files=[("files", ("q2.jpg", img, "image/jpeg"))])
check("queue cap refuses a new job", r.status_code == 429)
server.MAX_IMAGE_MB, server.MAX_QUEUED_JOBS = 0.01, 50
check("oversized image refused", c.post("/api/jobs", files=[("files", ("big.jpg", img, "image/jpeg"))]).status_code == 413)

shutil.rmtree(store, ignore_errors=True)
print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
