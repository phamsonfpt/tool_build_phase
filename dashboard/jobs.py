"""Job queue. Each job keeps its own trace: our stages (upload, wait for GPU, GPU work, monitoring) in
job.json, plus the pipeline tracer files under trace/ (spans, metrics, final zones).

local mode: one in-process GPU worker thread runs the jobs (engine.py).
web mode:   remote workers (worker.py on Kaggle or a PC) claim jobs over HTTP and send the results back.
"""
from __future__ import annotations

import io
import json
import os
import queue
import tarfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from settings import JOBS

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORKER_TIMEOUT_S = 180          # a running job whose worker is silent this long goes back to the queue
MAX_EXPLAIN_QUEUE = int(os.environ.get("MAX_EXPLAIN_QUEUE", 20))   # on-demand explains waiting for a GPU (public server)
EXPLAIN_ERROR_TTL_S = 600       # an image a worker could not explain is not sent to a GPU again for this long
EXPLAIN_WAIT_TTL_S = 600        # a request nobody picked up in this long is dropped (the page asks again if still open)
RESULT_FILES = ("summary.json", "run.json")
RESULT_DIRS = ("trace", "queue", "explain")


class QueueFull(Exception):
    """Too many on-demand explains are already waiting for a GPU."""


def now() -> float:
    return round(time.time(), 3)


class Job:
    def __init__(self, job_id: str, d: Optional[Dict] = None):
        self.id = job_id
        self.dir = JOBS / job_id
        self.d = d or json.loads((self.dir / "job.json").read_text(encoding="utf-8"))
        self._lock = threading.Lock()

    def save(self):
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(self.d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / "job.json")

    def stage(self, name: str, **attrs):
        with self._lock:
            st = self.d["stages"]
            if st and st[-1].get("end") is None:
                st[-1]["end"] = now()
            st.append({"name": name, "start": now(), "end": None, **attrs})
            self.save()

    def close_stage(self, **attrs):
        with self._lock:
            if self.d["stages"] and self.d["stages"][-1].get("end") is None:
                self.d["stages"][-1]["end"] = now()
                self.d["stages"][-1].update(attrs)
            self.save()

    def update(self, **kw):
        with self._lock:
            self.d.update(kw)
            self.save()

    def image_path(self, name: str) -> Optional[Path]:
        p = self.d["images"].get(name)
        return Path(p) if p else None

    def batch_context(self, name: str) -> Optional[Dict]:
        sp = self.dir / "summary.json"
        return batch_context(json.loads(sp.read_text(encoding="utf-8")), name) if sp.is_file() else None


def batch_context(s: Dict, name: str) -> Optional[Dict]:
    """What K-Means did to this image inside its batch (shown at step B6)."""
    im = next((x for x in s["images"] if x["name"] == name), None)
    km = s["kmeans"]
    if im is None or s["n_images"] <= 1:
        return None
    return {"n_images": s["n_images"], "budget": km["budget"], "red_before": km["red_before"], "ran": km["ran"],
            "zone_b5": im["zone_b5"], "zone_final": im["zone"], "cluster": im.get("cluster"),
            "dist_to_centroid": im.get("dist_to_centroid"), "kept": im.get("kept_by_kmeans"),
            "cluster_size": km["sizes"][im["cluster"]] if im.get("cluster") is not None else None}


def write_monitoring(j: Job) -> Dict:
    from monitor import job_monitoring
    summary = json.loads((j.dir / "summary.json").read_text(encoding="utf-8"))
    (j.dir / "monitoring.json").write_text(json.dumps(job_monitoring(j.dir, summary)), encoding="utf-8")
    return summary


class JobManager:
    def __init__(self, engine_factory=None, local_worker: bool = True):
        self._engine_factory = engine_factory
        self.local_worker = local_worker
        self.engine = None
        self._engine_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self.q: "queue.Queue[str]" = queue.Queue()             # local mode only
        # web mode: (job_id, image) -> {"t": queued at, "original": the image file is on this server}.
        # Without the original (bundled examples) only a worker that has BDD100K locally can explain it.
        self.explain_q: Dict[tuple, Dict] = {}
        self.explain_errors: Dict[tuple, tuple] = {}             # web mode: (job, image) -> (reason, time)
        self.workers: Dict[str, Dict] = {}                       # web mode: name -> device, last_seen, job
        self.jobs: Dict[str, Job] = {}
        for p in sorted(JOBS.glob("*/job.json")):
            j = Job(p.parent.name)
            if j.d["status"] in ("queued", "running"):        # server restarted mid-job
                j.update(status="error", error="Server khởi động lại khi job đang chạy")
            self.jobs[j.id] = j
        if local_worker:
            threading.Thread(target=self._worker, daemon=True).start()
        else:
            threading.Thread(target=self._watchdog, daemon=True).start()

    # ------------------------------------------------------------------ shared
    def get_engine(self):
        with self._engine_lock:                       # startup preload and the worker may ask at once
            if self.engine is None:
                self.engine = self._engine_factory()
        return self.engine

    def create(self, kind: str, files: List[tuple], title: str = "", job_id: Optional[str] = None,
               link_paths: Optional[List[Path]] = None, enqueue: bool = True) -> Job:
        """files: [(filename, bytes)] uploaded; link_paths: images used in place (examples)."""
        job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        jdir = JOBS / job_id
        (jdir / "input").mkdir(parents=True, exist_ok=True)
        t0 = now()
        images = {}
        for name, data in files:
            stem, ext = Path(name).stem, Path(name).suffix.lower()
            if ext not in IMG_EXT:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:80] or "image"
            fn, i = f"{safe}{ext}", 1
            while fn in images:
                fn, i = f"{safe}_{i}{ext}", i + 1
            (jdir / "input" / fn).write_bytes(data)
            images[fn] = str(jdir / "input" / fn)
        for p in link_paths or []:
            images[p.name] = str(p)
        d = {"id": job_id, "kind": kind, "title": title, "created": t0, "status": "queued", "error": None,
             "n_images": len(images), "images": images, "progress": {"done": 0, "total": len(images)},
             "stages": [{"name": "upload", "start": t0, "end": now(), "files": len(images),
                         "mb": round(sum(Path(p).stat().st_size for p in images.values()) / 2**20, 2)},
                        {"name": "wait_gpu", "start": now(), "end": None}]}
        j = Job(job_id, d)
        j.save()
        self.jobs[job_id] = j
        if enqueue and self.local_worker:
            self.q.put(job_id)
        return j

    def explain(self, j: Job, name: str) -> Optional[Dict]:
        """The step-by-step record of one image. In web mode a missing one is queued for a worker -> None."""
        if name not in j.d["images"]:                    # names come from the URL: check before touching the disk
            raise KeyError(name)
        cache = j.dir / "explain" / f"{name}.json"
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        if not self.local_worker:
            with self._claim_lock:
                err = self.explain_errors.get((j.id, name))
                if err and now() - err[1] < EXPLAIN_ERROR_TTL_S:
                    raise RuntimeError(err[0])               # not sent to a GPU again until the reason expires
                self.explain_errors.pop((j.id, name), None)
                self._prune()
                if (j.id, name) not in self.explain_q:
                    p = j.image_path(name)
                    original = p is not None and p.is_file()
                    # two separate caps: example images nobody can explain must not block uploaded ones
                    same = sum(e["original"] == original for e in self.explain_q.values())
                    if same >= MAX_EXPLAIN_QUEUE:
                        raise QueueFull(f"Đang có {same} ảnh chờ máy GPU tính từng bước. Hãy thử lại sau ít phút.")
                    self.explain_q[(j.id, name)] = {"t": now(), "original": original}
            return None
        d = self.get_engine().explain(j.image_path(name), j.batch_context(name))
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(d), encoding="utf-8")
        return d

    # ------------------------------------------------------------------ local mode
    def _worker(self):
        while True:
            jid = self.q.get()
            j = self.jobs[jid]
            try:
                j.update(status="running")
                if self.engine is None:
                    j.stage("load_models")
                    self.get_engine()
                eng = self.engine
                names = list(j.d["images"])
                paths = [Path(j.d["images"][n]) for n in names]
                j.stage("pipeline", device=eng.device)

                def progress(done, total):
                    j.update(progress={"done": done, "total": total})

                summary = eng.run_batch(paths, j.dir, progress)
                j.close_stage(images_per_s=summary["images_per_s"])
                if j.d["kind"] == "single" or len(paths) == 1:
                    j.stage("explain")
                    self.explain(j, names[0])
                j.stage("monitoring")
                write_monitoring(j)
                j.close_stage()
                j.update(status="done", zones=summary["zones"], zones_b5=summary["zones_b5"], finished=now())
            except Exception as e:                     # noqa: BLE001 — shown on the job page
                j.close_stage(error=str(e))
                j.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])

    # ------------------------------------------------------------------ web mode: remote workers
    def seen(self, worker: str, **info):
        w = self.workers.setdefault(worker, {"name": worker, "first_seen": now()})
        w.update(info, last_seen=now())

    def alive_workers(self) -> List[Dict]:
        t = now()
        return [w for w in self.workers.values() if t - w["last_seen"] < 45]

    def _prune(self):
        t = now()
        for k in [k for k, e in self.explain_q.items() if t - e["t"] > EXPLAIN_WAIT_TTL_S]:
            del self.explain_q[k]

    def pending(self, gpu_can_do: bool = False) -> bool:
        """Work waiting for a GPU. gpu_can_do: only work a worker without BDD100K (Kaggle) can do, so that
        opening example images does not start Kaggle runs that cannot explain them (no original image)."""
        if any(j.d["status"] == "queued" for j in self.jobs.values()):
            return True
        with self._claim_lock:
            self._prune()
            return any(e["original"] or not gpu_can_do for e in self.explain_q.values())

    def claim(self, worker: str) -> Optional[Dict]:
        """Hand the next piece of work to a worker: explain requests first (one image, seconds), then jobs."""
        with self._claim_lock:
            self.seen(worker)
            self._prune()
            has_originals = self.workers[worker].get("has_originals", False)
            for key, e in list(self.explain_q.items()):
                if not e["original"] and not has_originals:
                    continue                                 # leave it for a worker that has BDD100K locally
                del self.explain_q[key]
                jid, name = key
                j = self.jobs.get(jid)
                if j is None or (j.dir / "explain" / f"{name}.json").is_file():
                    continue
                self.workers[worker]["job"] = f"{jid} (giải thích {name})"
                return {"type": "explain", "job": jid, "image": name, "ctx": j.batch_context(name)}
            queued = sorted((j for j in self.jobs.values() if j.d["status"] == "queued"), key=lambda j: j.d["created"])
            if not queued:
                self.workers[worker]["job"] = None
                return None
            j = queued[0]
            w = self.workers[worker]
            j.stage("gpu_worker", worker=worker, device=w.get("device"))
            j.update(status="running", worker=worker)
            w["job"] = j.id
            return {"type": "run", "job": j.id, "kind": j.d["kind"], "images": list(j.d["images"])}

    def progress(self, worker: str, jid: str, done: int, total: int, phase: str = ""):
        self.seen(worker)
        j = self.jobs.get(jid)
        if j is not None and j.d["status"] == "running" and j.d.get("worker") == worker:
            j.update(progress={"done": done, "total": total, "phase": phase})

    def finish(self, worker: str, jid: str, tar_bytes: bytes, meta: Dict):
        """Results of a job from a worker: a tar.gz of summary.json, run.json, trace/, queue/, explain/."""
        self.seen(worker, job=None)
        j = self.jobs[jid]
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            for m in tf.getmembers():
                p = Path(m.name)
                top = p.parts[0] if p.parts else ""
                if not m.isfile() or p.is_absolute() or ".." in p.parts or not (m.name in RESULT_FILES or top in RESULT_DIRS):
                    continue                                   # only the files we asked for, inside the job folder
                dst = j.dir / p
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
        j.close_stage(**{k: meta.get(k) for k in ("download_s", "pipeline_s", "explain_s", "n_explained", "images_per_s")})
        j.stage("monitoring")
        summary = write_monitoring(j)
        j.close_stage()
        j.update(status="done", zones=summary["zones"], zones_b5=summary["zones_b5"], finished=now())

    def fail(self, worker: str, jid: str, error: str, tb: str = ""):
        self.seen(worker, job=None)
        j = self.jobs.get(jid)
        if j is not None:
            j.close_stage(error=error)
            j.update(status="error", error=error, traceback=tb[-4000:])

    def store_explain(self, worker: str, jid: str, name: str, data: bytes):
        self.seen(worker, job=None)
        j = self.jobs[jid]
        if name not in j.d["images"]:
            raise KeyError(name)
        json.loads(data)                                    # must be valid JSON
        (j.dir / "explain").mkdir(exist_ok=True)
        (j.dir / "explain" / f"{name}.json").write_bytes(data)

    def explain_failed(self, worker: str, jid: str, name: str, error: str):
        self.seen(worker, job=None)
        self.explain_errors[(jid, name)] = (error, now())

    def _watchdog(self):
        """A Kaggle session can end mid-job: give its job back to the queue so the next worker takes it."""
        while True:
            time.sleep(15)
            t = now()
            for j in list(self.jobs.values()):
                if j.d["status"] != "running":
                    continue
                w = self.workers.get(j.d.get("worker") or "")
                if w is None or t - w["last_seen"] > WORKER_TIMEOUT_S:
                    j.close_stage(error="worker mất kết nối")
                    j.stage("wait_gpu", note="worker trước mất kết nối, chờ worker khác")
                    j.update(status="queued", worker=None, progress={"done": 0, "total": j.d["n_images"]})
