"""M29 dashboard: web pages + API.

    python server.py                      # local: this machine's GPU runs the jobs   http://127.0.0.1:8770
    M29_MODE=web python server.py         # production (Render): no GPU, jobs wait for remote workers (worker.py)

Production environment: DASH_PASSWORD (team password for the pages), WORKER_TOKEN (shared with the
workers), and for the Kaggle worker the variables listed in kaggle_launch.py.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import os
import re
import secrets
import tarfile
import threading
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from jobs import Job, JobManager, QueueFull, now
from settings import BUNDLED, EXAMPLE_BATCH_DIR, EXAMPLE_SINGLE, HERE, JOBS, MODE, SEED, SPLITS, STORE, THUMBS, data_file

NAME = re.compile(r"^[\w\-.]+$")
WEB = MODE == "web"
PASSWORD = os.environ.get("DASH_PASSWORD", "")          # empty = public pages
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
# BDD100K images and labels: redistribution is allowed only for non-profit use with the BDD100K copyright notice,
# so the public server stays on the safe side and refuses them unless ALLOW_SEED=1
ALLOW_SEED = os.environ.get("ALLOW_SEED", "") == "1"
# anyone can upload on a public server: keep one person from using up the GPU quota
MAX_JOB_IMAGES = int(os.environ.get("MAX_JOB_IMAGES", 300))
MAX_QUEUED_JOBS = int(os.environ.get("MAX_QUEUED_JOBS", 5))
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", 15))
app = FastAPI(title="M29 dashboard")
jm: JobManager = None      # set in main()
launcher = None            # KaggleLauncher in web mode


# ---------------------------------------------------------------------------- access
@app.middleware("http")
async def guard(request: Request, call_next):
    """Team password on every page and API except the worker API (which has its own token)."""
    path = request.url.path
    if PASSWORD and not path.startswith("/api/worker/") and path != "/healthz":
        ok = False
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:                      # noqa: BLE001 — malformed header = not logged in
                ok = False
        if not ok:
            return Response("Cần mật khẩu của nhóm", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="M29 dashboard", charset="UTF-8"'})
    resp = await call_next(request)
    if not path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"   # pages and scripts change often: always revalidate
    return resp


def worker_auth(x_worker_token: str = Header("")):
    if not WORKER_TOKEN or not secrets.compare_digest(x_worker_token, WORKER_TOKEN):
        raise HTTPException(401, "worker token sai")


@app.get("/healthz")
def healthz():
    return {"ok": True, "mode": MODE}


def _job(job_id: str):
    j = jm.jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"Không có job {job_id}")
    return j


def _thumb(src: Path, key: str, w: int) -> Response:
    w = max(64, min(int(w), 1280))
    out = THUMBS / f"{key}_{w}.jpg"
    if not out.is_file():
        from PIL import Image
        im = Image.open(src).convert("RGB")
        im.thumbnail((w, w * 2))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, "JPEG", quality=80)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


# ---------------------------------------------------------------------------- data train
@app.get("/api/train/stats")
def train_stats():
    p = data_file("train_stats.json")
    if not p.is_file():
        raise HTTPException(503, "Chưa có thống kê train: chạy python train_stats.py")
    return FileResponse(p, media_type="application/json")


_index = {}


def train_index():
    """Every train image with attributes and GT boxes. Local: computed by train_stats.py. Web: uploaded by a
    local worker (it holds BDD100K labels, so it is never committed to the public repository)."""
    p = SEED / "train_index.npz" if WEB else STORE / "train_index.npz"
    if not p.is_file():
        raise HTTPException(503, ("Ảnh train chưa có trên server: bật worker ở máy có dữ liệu BDD100K để nạp lên" if ALLOW_SEED else
                                  "Bản công khai không hiển thị ảnh train BDD100K (license chỉ cho phân phối vì mục đích phi lợi nhuận, kèm thông báo bản quyền; nhóm chờ ban tổ chức xác nhận). "
                                  "Các thống kê ở trên được tính trên đủ 70.000 ảnh.")
                            if WEB else "Chưa có chỉ mục ảnh train: chạy python train_stats.py")
    key = (str(p), p.stat().st_mtime)
    if _index.get("key") != key:
        z = np.load(p, allow_pickle=False)
        ix = {k: z[k] for k in z.files}
        ix["img_of_box"] = np.repeat(np.arange(len(ix["n"])), np.diff(ix["offsets"]))
        _index.update(key=key, i=ix)
    return _index["i"]


@app.get("/api/train/gallery")
def train_gallery(weather: str = "", scene: str = "", timeofday: str = "", cls: str = "",
                  min_obj: int = 0, sort: str = "random", offset: int = 0, limit: int = 24):
    """Every train image, filtered by attributes and by a class it contains; paged. On the web server only
    the images a worker uploaded can be shown; the count over all 70,000 is still reported."""
    ix = train_index()
    class_names = json.loads(data_file("train_stats.json").read_text(encoding="utf-8"))["class_names"]
    m = np.ones(len(ix["names"]), bool)
    for key, val in (("weather", weather), ("scene", scene), ("timeofday", timeofday)):
        if val:
            m &= ix[key] == val
    if min_obj:
        m &= ix["n"] >= min_obj
    if cls:
        if cls not in class_names:
            raise HTTPException(400, "lớp không hợp lệ")
        has = np.zeros(len(m), bool)
        has[np.unique(ix["img_of_box"][ix["cls"] == class_names.index(cls)])] = True
        m &= has
    total_all = int(m.sum())
    if WEB:
        avail = np.isin(ix["names"], [p.name for p in (SEED / "train").glob("*.jpg")])
        m &= avail
    sel = np.flatnonzero(m)
    if sort == "most":
        sel = sel[np.argsort(-ix["n"][sel], kind="stable")]
    elif sort == "fewest":
        sel = sel[np.argsort(ix["n"][sel], kind="stable")]
    elif sort != "name":
        sel = sel[np.random.default_rng(0).permutation(len(sel))]
    limit = max(1, min(limit, 96))
    page = []
    for i in sel[offset:offset + limit]:
        a, b = ix["offsets"][i], ix["offsets"][i + 1]
        boxes = np.round(ix["boxes"][a:b], 1).tolist()
        page.append({"name": str(ix["names"][i]), "boxes": [bx + [int(c)] for bx, c in zip(boxes, ix["cls"][a:b])],
                     **{k: str(ix[k][i]) for k in ("weather", "scene", "timeofday")}})
    return {"total": int(len(sel)), "total_all": total_all, "subset": WEB, "offset": offset, "items": page}


@app.get("/api/train/image/{name}")
def train_image(name: str, w: int = 0):
    if not NAME.match(name):
        raise HTTPException(400)
    if WEB:
        p = SEED / "train" / name
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
    p = SPLITS["train"] / name
    if not p.is_file():
        raise HTTPException(404)
    return _thumb(p, f"train/{p.stem}", w) if w else FileResponse(p, media_type="image/jpeg")


# ---------------------------------------------------------------------------- system
_system_cache = {}


@app.get("/api/system")
def system():
    out = {"mode": MODE, "public": not PASSWORD, "bdd_images": ALLOW_SEED or not WEB,
           "limits": {"images": MAX_JOB_IMAGES, "queued_jobs": MAX_QUEUED_JOBS, "image_mb": MAX_IMAGE_MB} if WEB else None, "queue": sum(j.d["status"] == "queued" for j in jm.jobs.values()) + len(jm.explain_q),
           "running": [j.id for j in jm.jobs.values() if j.d["status"] == "running"]}
    if not WEB:
        eng = jm.engine
        out.update(engine_loaded=eng is not None, engine_loading=eng is None and jm._engine_lock.locked())
        if eng is not None:
            if "s" not in _system_cache:
                _system_cache["s"] = eng.system()
            out.update(_system_cache["s"])
        return out
    alive = jm.alive_workers()
    out.update(engine_loaded=bool(alive), engine_loading=False,
               workers=[{k: w.get(k) for k in ("name", "device", "job", "last_seen")} for w in alive],
               kaggle=launcher.status() if launcher else None, now=now())
    p = data_file("system.json")
    if p.is_file():
        out.update(json.loads(p.read_text(encoding="utf-8")))
    if alive:
        out["device"] = alive[0].get("device")
    return out


@app.get("/api/centroid/{cls}")
def centroid(cls: str):
    if not WEB:
        return jm.get_engine().centroid(cls)
    p = data_file("centroids.json")
    if not p.is_file():
        raise HTTPException(503, "Chưa có centroid: cần một worker kết nối")
    c = json.loads(p.read_text(encoding="utf-8")).get(cls)
    if c is None:
        raise HTTPException(404)
    return c


@app.post("/api/kaggle/start")
def kaggle_start():
    if launcher is None:
        raise HTTPException(400, "Chỉ có ở chế độ web")
    try:
        launcher.launch()
    except Exception as e:                         # noqa: BLE001 — the message is shown on the page
        raise HTTPException(500, f"Không bật được Kaggle: {e}")
    return launcher.status()


# ---------------------------------------------------------------------------- jobs
def _public(j):
    d = {k: v for k, v in j.d.items() if k not in ("images", "traceback")}
    d["image_names"] = list(j.d["images"])[:5]
    return d


@app.get("/api/jobs")
def list_jobs():
    return sorted((_public(j) for j in jm.jobs.values()), key=lambda d: -d["created"])


@app.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(...), title: str = Form("")):
    if WEB and len(files) > MAX_JOB_IMAGES:
        raise HTTPException(413, f"Tối đa {MAX_JOB_IMAGES} ảnh mỗi lần gửi (bạn gửi {len(files)})")
    if WEB and sum(j.d["status"] in ("queued", "running") for j in jm.jobs.values()) >= MAX_QUEUED_JOBS:
        raise HTTPException(429, f"Đang có {MAX_QUEUED_JOBS} job chờ hoặc đang chạy. Hãy thử lại sau ít phút.")
    data = []
    for f in files:
        b = await f.read()
        if WEB and len(b) > MAX_IMAGE_MB * 2**20:
            raise HTTPException(413, f"Ảnh {f.filename} lớn hơn {MAX_IMAGE_MB:g} MB")
        data.append((f.filename or "image.jpg", b))
    if not data:
        raise HTTPException(400, "Không có file")
    j = jm.create("single" if len(data) == 1 else "batch", data, title)
    if j.d["n_images"] == 0:
        raise HTTPException(400, "Không có file ảnh hợp lệ (jpg, png, bmp, webp)")
    return _public(j)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = _job(job_id)
    d = _public(j)
    d["traceback"] = j.d.get("traceback")
    for key, fn in (("summary_ready", "summary.json"), ("monitoring_ready", "monitoring.json")):
        d[key] = (j.dir / fn).is_file()
    return d


@app.get("/api/jobs/{job_id}/summary")
def job_summary(job_id: str):
    p = _job(job_id).dir / "summary.json"
    if not p.is_file():
        raise HTTPException(409, "Job chưa xong")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/monitoring")
def job_monitoring(job_id: str):
    p = _job(job_id).dir / "monitoring.json"
    if not p.is_file():
        raise HTTPException(409, "Job chưa xong")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/image/{name}")
def job_image(job_id: str, name: str, w: int = 0):
    j = _job(job_id)
    if not NAME.match(name):
        raise HTTPException(400)
    p = j.image_path(name)
    if p is not None and p.is_file():
        return _thumb(p, f"{job_id}/{p.stem}", w) if w else FileResponse(p)
    seeded = SEED / "examples" / name                   # example images uploaded by a local worker (web mode)
    if seeded.is_file():
        return FileResponse(seeded, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
    raise HTTPException(404, "Ảnh không có trên server")


@app.get("/api/jobs/{job_id}/explain/{name}")
def job_explain(job_id: str, name: str):
    j = _job(job_id)
    if j.d["status"] != "done":
        raise HTTPException(409, "Job chưa xong")
    if not NAME.match(name) or name not in j.d["images"]:   # never build a path from an unknown name
        raise HTTPException(404)
    gz = j.dir / "explain" / f"{name}.json.gz"           # bundled examples are stored compressed
    if gz.is_file():
        return Response(gzip.decompress(gz.read_bytes()), media_type="application/json")
    try:
        d = jm.explain(j, name)
    except KeyError:
        raise HTTPException(404)
    except RuntimeError as e:                            # a worker reported it cannot explain this image
        raise HTTPException(409, str(e))
    except QueueFull as e:                               # public server: cap on-demand GPU work
        raise HTTPException(429, str(e))
    if d is None:
        return JSONResponse({"pending": True, "workers": len(jm.alive_workers()),
                             "kaggle": launcher.status()["enabled"] if launcher else False}, status_code=202)
    return JSONResponse(d)


@app.get("/api/examples")
def examples():
    return {k: _public(jm.jobs[k]) if k in jm.jobs else None for k in ("example-single", "example-batch")}


# ---------------------------------------------------------------------------- worker API (web mode)
@app.post("/api/worker/hello", dependencies=[Depends(worker_auth)])
async def worker_hello(request: Request):
    """A worker announces itself with its device, the detector/threshold summary and the FCD centroids."""
    d = await request.json()
    name = d.get("name") or "worker"
    jm.seen(name, device=d.get("device"), seeds=d.get("can_seed", False), has_originals=bool(d.get("has_originals")))
    if d.get("system"):
        (STORE / "system.json").write_text(json.dumps(d["system"]), encoding="utf-8")
    if d.get("centroids"):
        (STORE / "centroids.json").write_text(json.dumps(d["centroids"]), encoding="utf-8")
    return {"ok": True, "seed": _seed_status()}


@app.post("/api/worker/claim", dependencies=[Depends(worker_auth)])
async def worker_claim(request: Request):
    d = await request.json()
    task = jm.claim(d.get("name") or "worker")
    return task if task else Response(status_code=204)


@app.get("/api/worker/jobs/{job_id}/image/{name}", dependencies=[Depends(worker_auth)])
def worker_image(job_id: str, name: str):
    p = _job(job_id).image_path(name)
    if p is None or not p.is_file():
        raise HTTPException(404, "ảnh gốc không có trên server")
    return FileResponse(p)


@app.post("/api/worker/jobs/{job_id}/progress", dependencies=[Depends(worker_auth)])
async def worker_progress(job_id: str, request: Request):
    d = await request.json()
    jm.progress(d.get("name") or "worker", job_id, int(d.get("done", 0)), int(d.get("total", 0)), d.get("phase", ""))
    return {"ok": True}


@app.post("/api/worker/jobs/{job_id}/result", dependencies=[Depends(worker_auth)])
async def worker_result(job_id: str, request: Request, x_worker_name: str = Header("worker"), x_worker_meta: str = Header("{}")):
    _job(job_id)
    jm.finish(x_worker_name, job_id, await request.body(), json.loads(x_worker_meta or "{}"))
    return {"ok": True}


@app.post("/api/worker/jobs/{job_id}/fail", dependencies=[Depends(worker_auth)])
async def worker_fail(job_id: str, request: Request):
    d = await request.json()
    jm.fail(d.get("name") or "worker", job_id, d.get("error", "lỗi"), d.get("traceback", ""))
    return {"ok": True}


@app.post("/api/worker/jobs/{job_id}/explain/{name}", dependencies=[Depends(worker_auth)])
async def worker_explain(job_id: str, name: str, request: Request, x_worker_name: str = Header("worker")):
    _job(job_id)
    try:
        jm.store_explain(x_worker_name, job_id, name, await request.body())
    except KeyError:
        raise HTTPException(404)
    return {"ok": True}


@app.post("/api/worker/jobs/{job_id}/explain-fail/{name}", dependencies=[Depends(worker_auth)])
async def worker_explain_fail(job_id: str, name: str, request: Request):
    d = await request.json()
    jm.explain_failed(d.get("name") or "worker", job_id, name, d.get("error", "lỗi"))
    return {"ok": True}


def _seed_status():
    return {"allowed": ALLOW_SEED, "train_index": (SEED / "train_index.npz").is_file(),
            "train_images": sum(1 for _ in (SEED / "train").glob("*.jpg")) if (SEED / "train").is_dir() else 0,
            "example_images": sum(1 for _ in (SEED / "examples").glob("*.jpg")) if (SEED / "examples").is_dir() else 0}


@app.post("/api/worker/seed", dependencies=[Depends(worker_auth)])
async def worker_seed(request: Request):
    """BDD100K images and the train label index, uploaded by a local worker (kept only on this server's disk).
    Only for a private deployment (ALLOW_SEED=1): a public one must not serve them."""
    if not ALLOW_SEED:
        raise HTTPException(403, "Server công khai không nhận ảnh hay nhãn BDD100K (đặt ALLOW_SEED=1 cho bản riêng tư)")
    with tarfile.open(fileobj=io.BytesIO(await request.body()), mode="r:gz") as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            ok = m.isfile() and not p.is_absolute() and ".." not in p.parts and (
                m.name == "train_index.npz" or (len(p.parts) == 2 and p.parts[0] in ("train", "examples") and p.suffix == ".jpg"))
            if ok:
                dst = SEED / p
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
    return _seed_status()


# ---------------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index():
    """index.html with ?v=<mtime> on the script and stylesheet, so an edited file is never served from cache."""
    st = HERE / "static"
    html = (st / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "style.css"):
        html = html.replace(f'"{name}"', f'"{name}?v={int((st / name).stat().st_mtime)}"')
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


def ensure_examples():
    if WEB:
        # results computed once on a GPU and committed (bundled/examples); images come from a local worker
        import shutil
        for src in sorted((BUNDLED / "examples").glob("*/job.json")):
            dst = JOBS / src.parent.name
            if not (dst / "job.json").is_file():
                shutil.copytree(src.parent, dst, dirs_exist_ok=True)
                jm.jobs[dst.name] = Job(dst.name)
        return
    if "example-single" not in jm.jobs and EXAMPLE_SINGLE.is_file():
        jm.create("single", [], "Ví dụ: 1 ảnh test BDD100K", job_id="example-single", link_paths=[EXAMPLE_SINGLE])
    if "example-batch" not in jm.jobs and EXAMPLE_BATCH_DIR.is_dir():
        paths = sorted(EXAMPLE_BATCH_DIR.glob("*.jpg"))
        jm.create("batch", [], f"Ví dụ: batch {len(paths)} ảnh test BDD100K", job_id="example-batch", link_paths=paths)


def make_engine():
    from engine import Engine
    return Engine()


def main():
    global jm, launcher
    if WEB:
        if not WORKER_TOKEN:
            print("WARNING: WORKER_TOKEN chưa đặt, sẽ không worker nào nhận được job", flush=True)
        jm = JobManager(local_worker=False)
        ensure_examples()
        from kaggle_launch import KaggleLauncher
        launcher = KaggleLauncher(jm, WORKER_TOKEN)
    else:
        jm = JobManager(make_engine)
        threading.Thread(target=jm.get_engine, daemon=True).start()   # load the 3 experts now, not on the first job
        ensure_examples()
    host = os.environ.get("HOST", "0.0.0.0" if WEB else "127.0.0.1")
    uvicorn.run(app, host=host, port=int(os.environ.get("PORT", 8770)), log_level="warning")


if __name__ == "__main__":
    main()
