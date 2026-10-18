"""GPU worker for a dashboard running in web mode (Render). Same code on Kaggle and on a PC with a GPU.

    python worker.py --server https://<app>.onrender.com --token <WORKER_TOKEN> [--name may-son]

Loop: say hello (device, detector summary, FCD centroids) -> claim work -> for a job: download its images,
run the shipped ai_router pipeline (engine.py), explain the images the pages will open, send everything
back as one tar.gz; for an explain request: one image, one JSON. Ground truth is never sent: BDD100K labels
stay off the public server (license: non-profit use with its copyright notice). With --seed (private deployments with ALLOW_SEED=1 only) a PC that holds BDD100K
also uploads the example and train images the pages show; they never go into git.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import socket
import tarfile
import threading
import time
import traceback
from pathlib import Path

import requests

from jobs import batch_context
from settings import EXAMPLE_BATCH_DIR, EXAMPLE_SINGLE, SPLITS, STORE

EXPLAIN_ALL_UP_TO = 60        # batches this small get every image explained
EXPLAIN_MAX = 400             # larger batches: the review queue, the images K-Means dropped, a few grey ones
SEED_TRAIN_IMAGES = 1500
SEED_CHUNK_BYTES = 25 * 2**20


class Client:
    def __init__(self, server: str, token: str, name: str):
        self.base = server.rstrip("/")
        self.name = name
        self.s = requests.Session()
        self.s.headers.update({"X-Worker-Token": token, "X-Worker-Name": name})

    def post(self, path, **kw):
        r = self.s.post(self.base + path, timeout=kw.pop("timeout", 120), **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{path}: {r.status_code} {r.text[:300]}")
        return r

    def get(self, path, **kw):
        r = self.s.get(self.base + path, timeout=kw.pop("timeout", 120), **kw)
        r.raise_for_status()
        return r


class Progress:
    """Throttled progress reports; also the heartbeat that keeps the job assigned to this worker."""

    def __init__(self, c: Client, job: str, total: int):
        self.c, self.job, self.total = c, job, total
        self.done, self.phase, self.t = 0, "", 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()

    def __call__(self, done, total=None, phase=None):
        self.done, self.total = done, total or self.total
        self.phase = phase or self.phase
        if time.time() - self.t > 1.0:
            self.send()

    def send(self):
        self.t = time.time()
        try:
            self.c.post(f"/api/worker/jobs/{self.job}/progress", json={"name": self.c.name, "done": self.done,
                                                                         "total": self.total, "phase": self.phase}, timeout=20)
        except Exception as e:                     # noqa: BLE001 — a missed heartbeat is not fatal
            print("progress:", e, flush=True)

    def _beat(self):
        while not self._stop.wait(15):
            self.send()

    def stop(self):
        self._stop.set()


def hello(c: Client, eng, can_seed: bool) -> dict:
    cents = {}
    for cls in eng.names:
        cen = eng.centroid(cls)
        if not cen.get("missing"):
            cents[cls] = cen
    sysinfo = eng.system()
    sysinfo["ai_router"] = None                   # a local path means nothing on the server
    r = c.post("/api/worker/hello", json={"name": c.name, "device": eng.device, "system": sysinfo,
                                          "centroids": cents, "can_seed": can_seed,
                                          "has_originals": EXAMPLE_BATCH_DIR.is_dir()}, timeout=60)
    return r.json()


# ------------------------------------------------------------------ seeding (PC with BDD100K only)
def _jpeg(src: Path, width: int) -> bytes:
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return b.getvalue()


def seed(c: Client, status: dict):
    """Upload what the pages need and the server lacks: train label index, train thumbnails, example images."""
    items = []                                       # (archive name, bytes or path)
    idx = STORE / "train_index.npz"
    if not status.get("train_index") and idx.is_file():
        items.append(("train_index.npz", idx))
    if idx.is_file() and SPLITS["train"].is_dir():
        import numpy as np
        z = np.load(idx, allow_pickle=False)
        names, cls, off = z["names"], z["cls"], z["offsets"]
        img_of_box = np.repeat(np.arange(len(names)), np.diff(off))
        rng = random.Random(0)
        pick = set(rng.sample(range(len(names)), SEED_TRAIN_IMAGES - 300))
        for c_ in range(int(cls.max()) + 1):           # rare classes must be visible too
            imgs = sorted(set(img_of_box[cls == c_].tolist()))
            pick.update(rng.sample(imgs, min(30, len(imgs))))
        if status.get("train_images", 0) < len(pick):
            cache = STORE / "seed_cache" / "train"
            for i in sorted(pick):
                n = str(names[i])
                items.append((f"train/{n}", ("train", SPLITS["train"] / n, cache / n, 360)))
    examples = ([EXAMPLE_SINGLE] if EXAMPLE_SINGLE.is_file() else []) + (sorted(EXAMPLE_BATCH_DIR.glob("*.jpg")) if EXAMPLE_BATCH_DIR.is_dir() else [])
    if status.get("example_images", 0) < len(examples):
        cache = STORE / "seed_cache" / "examples"
        for p in examples:                             # the single example keeps full size for its step-by-step page
            items.append((f"examples/{p.name}", ("ex", p, cache / p.name, 1280 if p == EXAMPLE_SINGLE else 640)))
    if not items:
        return
    print(f"seed: {len(items)} file lên server", flush=True)
    buf, size, sent = io.BytesIO(), 0, 0
    tf = tarfile.open(fileobj=buf, mode="w:gz")

    def flush():
        nonlocal buf, size, tf, sent
        tf.close()
        c.post("/api/worker/seed", data=buf.getvalue(), timeout=600)
        sent += size
        buf, size = io.BytesIO(), 0
        tf = tarfile.open(fileobj=buf, mode="w:gz")

    for arc, src in items:
        if isinstance(src, Path):
            data = src.read_bytes()
        else:
            _, path, cached, width = src
            if not cached.is_file():
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(_jpeg(path, width))
            data = cached.read_bytes()
        ti = tarfile.TarInfo(arc)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
        size += len(data)
        if size > SEED_CHUNK_BYTES:
            flush()
    flush()
    print(f"seed xong: {sent / 2**20:.1f} MB", flush=True)


# ------------------------------------------------------------------ work
def without_gt(d: dict) -> dict:
    """An uploaded image that happens to be in BDD100K would carry its labels: drop them."""
    d.pop("gt", None)
    for o in d.get("objects", []):
        o.pop("tp", None)
    return d


def explain_names(summary: dict) -> list:
    ims = summary["images"]
    if len(ims) <= EXPLAIN_ALL_UP_TO:
        return [im["name"] for im in ims]
    red = [im["name"] for im in ims if im["zone"] == "RED"]
    dedup = [im["name"] for im in ims if im.get("dedup")]
    grey = sorted((im for im in ims if im["zone"] != "RED" and not im.get("dedup")), key=lambda im: -(im["img_unc"] or 0))
    return (red + dedup + [im["name"] for im in grey[:20]])[:EXPLAIN_MAX]


def run_job(c: Client, eng, task: dict):
    jid, names = task["job"], task["images"]
    jdir = STORE / "worker" / jid
    shutil.rmtree(jdir, ignore_errors=True)
    (jdir / "input").mkdir(parents=True)
    prog = Progress(c, jid, len(names))
    meta = {}
    try:
        t = time.time()
        paths = []
        for i, n in enumerate(names):
            p = jdir / "input" / n
            p.write_bytes(c.get(f"/api/worker/jobs/{jid}/image/{n}").content)
            paths.append(p)
            prog(i + 1, len(names), "tải ảnh về worker")
        meta["download_s"] = round(time.time() - t, 2)
        t = time.time()
        summary = eng.run_batch(paths, jdir, lambda d, n: prog(d, n, "B0→B7"))
        meta["pipeline_s"] = round(time.time() - t, 2)
        meta["images_per_s"] = summary["images_per_s"]
        t = time.time()
        todo = explain_names(summary)
        (jdir / "explain").mkdir(exist_ok=True)
        for i, n in enumerate(todo):
            d = without_gt(eng.explain(jdir / "input" / n, batch_context(summary, n)))
            (jdir / "explain" / f"{n}.json").write_text(json.dumps(d), encoding="utf-8")
            prog(i + 1, len(todo), "giải thích từng ảnh")
        meta["explain_s"], meta["n_explained"] = round(time.time() - t, 2), len(todo)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(jdir / "summary.json", "summary.json")
            for sub in ("trace", "queue", "explain"):
                for f in sorted((jdir / sub).glob("*")) if (jdir / sub).is_dir() else []:
                    if f.is_file() and f.name != "images.jsonl":     # per-image trace is not shown; keep the upload small
                        tf.add(f, f"{sub}/{f.name}")
        prog(len(todo), len(todo), "gửi kết quả")
        c.post(f"/api/worker/jobs/{jid}/result", data=buf.getvalue(), headers={"X-Worker-Meta": json.dumps(meta)}, timeout=600)
        print(f"job {jid}: {len(names)} ảnh, {meta}", flush=True)
    except Exception as e:                         # noqa: BLE001 — reported to the job page
        traceback.print_exc()
        c.post(f"/api/worker/jobs/{jid}/fail", json={"name": c.name, "error": f"{type(e).__name__}: {e}",
                                                      "traceback": traceback.format_exc()})
    finally:
        prog.stop()
        shutil.rmtree(jdir, ignore_errors=True)


def run_explain(c: Client, eng, task: dict):
    jid, n = task["job"], task["image"]
    tmp = STORE / "worker" / f"{jid}-explain"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        r = c.s.get(f"{c.base}/api/worker/jobs/{jid}/image/{n}", timeout=120)
        if r.status_code == 404:
            p = EXAMPLE_BATCH_DIR / n                  # a bundled example: only a PC with BDD100K has the original
            if not p.is_file():
                raise FileNotFoundError(f"Máy GPU '{c.name}' không có ảnh gốc {n}. Ảnh của lô ví dụ chỉ tính được "
                                        "bằng worker chạy ở máy có dữ liệu BDD100K.")
        else:
            r.raise_for_status()
            p = tmp / n
            p.write_bytes(r.content)
        d = without_gt(eng.explain(p, task.get("ctx")))
        c.post(f"/api/worker/jobs/{jid}/explain/{n}", data=json.dumps(d).encode(), timeout=120)
        print(f"explain {jid}/{n}", flush=True)
    except Exception as e:                         # noqa: BLE001 — the page shows the reason
        print(f"explain {jid}/{n} lỗi: {e}", flush=True)
        c.post(f"/api/worker/jobs/{jid}/explain-fail/{n}", json={"name": c.name, "error": str(e)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("M29_SERVER", ""))
    ap.add_argument("--token", default=os.environ.get("M29_WORKER_TOKEN", ""))
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--idle-exit", type=float, default=0, help="seconds without work before exiting (0 = never)")
    ap.add_argument("--max-hours", type=float, default=0, help="stop taking new work after this many hours (0 = never)")
    ap.add_argument("--seed", action="store_true", help="upload BDD100K images (only accepted by a server with ALLOW_SEED=1)")
    a = ap.parse_args()
    if not a.server or not a.token:
        ap.error("cần --server và --token (hoặc M29_SERVER, M29_WORKER_TOKEN)")
    c = Client(a.server, a.token, a.name)
    from engine import Engine
    t = time.time()
    eng = Engine()
    print(f"nạp 3 expert: {time.time() - t:.1f} s trên {eng.device}", flush=True)
    can_seed = a.seed and SPLITS["train"].is_dir()
    started, last_work, last_hello = time.time(), time.time(), 0.0
    while True:
        try:
            if time.time() - last_hello > 300:         # the web server forgets everything when it restarts
                st = hello(c, eng, can_seed)
                last_hello = time.time()
                if can_seed and st.get("seed", {}).get("allowed"):
                    seed(c, st["seed"])
                elif can_seed:
                    print("server không nhận ảnh BDD (bản công khai), bỏ qua --seed", flush=True)
            if a.max_hours and time.time() - started > a.max_hours * 3600:
                print("hết thời gian chạy, thoát", flush=True)
                return
            r = c.post("/api/worker/claim", json={"name": c.name}, timeout=60)
            if r.status_code == 204:
                if a.idle_exit and time.time() - last_work > a.idle_exit:
                    print("không có việc, thoát", flush=True)
                    return
                time.sleep(3)
                continue
            task = r.json()
            last_work = time.time()
            (run_job if task["type"] == "run" else run_explain)(c, eng, task)
            last_work = time.time()
        except requests.RequestException as e:          # server asleep or restarting: wait and retry
            print("server:", e, flush=True)
            last_hello = 0.0
            time.sleep(10)


if __name__ == "__main__":
    main()
