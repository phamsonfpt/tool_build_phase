"""Start the Kaggle GPU worker when work is waiting and no worker is connected (web mode).

A Kaggle run is started by pushing a new version of a private script kernel (Kaggle API). The kernel
mounts the private dataset that holds ai_router + the 3 checkpoints, runs dashboard/worker.py, claims
jobs from this server over HTTPS and exits after 10 idle minutes.

Environment (set on Render):
  KAGGLE_USERNAME                         Kaggle user name
  KAGGLE_API_TOKEN  or  KAGGLE_KEY        API token (Settings > API on kaggle.com)
  KAGGLE_KERNEL     (m29-gpu-worker)      kernel slug that gets created / re-run
  KAGGLE_DATASET    (m29-ai-router-bundle) private dataset made by kaggle/make_bundle.py
  KAGGLE_ACCELERATOR (optional)           e.g. NvidiaTeslaT4; empty = Kaggle's default GPU
  PUBLIC_URL / RENDER_EXTERNAL_URL        address the worker calls back
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path

from settings import HERE

COOLDOWN_S = 20 * 60          # a pushed run needs a few minutes to boot; do not push again meanwhile
RETRY_AFTER_ERROR_S = 5 * 60


class KaggleLauncher:
    def __init__(self, jm, worker_token: str):
        self.jm = jm
        self.token = worker_token
        self.user = os.environ.get("KAGGLE_USERNAME", "")
        self.kernel = os.environ.get("KAGGLE_KERNEL", "m29-gpu-worker")
        self.dataset = os.environ.get("KAGGLE_DATASET", "m29-ai-router-bundle")
        self.acc = os.environ.get("KAGGLE_ACCELERATOR") or None
        self.url = (os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
        has_key = bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"))
        self.enabled = bool(self.user and has_key and self.url and self.token)
        self.last_push = None
        self.last_error = None
        self.last_result = None
        self._lock = threading.Lock()
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def status(self):
        return {"enabled": self.enabled, "kernel": f"{self.user}/{self.kernel}" if self.user else None,
                "dataset": f"{self.user}/{self.dataset}" if self.user else None,
                "last_push": self.last_push, "last_error": self.last_error, "last_result": self.last_result,
                "missing": [k for k, ok in (("KAGGLE_USERNAME", self.user),
                                            ("KAGGLE_API_TOKEN/KAGGLE_KEY", os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")),
                                            ("PUBLIC_URL", self.url), ("WORKER_TOKEN", self.token)) if not ok]}

    def _loop(self):
        while True:
            time.sleep(20)
            try:
                self.maybe_launch()
            except Exception:                      # noqa: BLE001 — kept for the status line
                self.last_error = traceback.format_exc()[-1500:]

    def maybe_launch(self):
        if not self.jm.pending(gpu_can_do=True) or self.jm.alive_workers():
            return
        t = time.time()
        if self.last_push and t - self.last_push < (RETRY_AFTER_ERROR_S if self.last_error else COOLDOWN_S):
            return
        self.launch()

    def launch(self):
        with self._lock:
            if not self.enabled:
                raise RuntimeError("Kaggle chưa được cấu hình: thiếu " + ", ".join(self.status()["missing"]))
            self.last_push, self.last_error = time.time(), None
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                with tempfile.TemporaryDirectory() as d:
                    code = (HERE / "kaggle" / "kernel_run.py").read_text(encoding="utf-8")
                    code = code.replace("__SERVER_URL__", self.url).replace("__WORKER_TOKEN__", self.token)
                    (Path(d) / "run.py").write_text(code, encoding="utf-8")
                    meta = {"id": f"{self.user}/{self.kernel}", "title": self.kernel, "code_file": "run.py",
                            "language": "python", "kernel_type": "script", "is_private": True,
                            "enable_gpu": True, "enable_internet": True,
                            "dataset_sources": [f"{self.user}/{self.dataset}"], "competition_sources": [], "kernel_sources": []}
                    (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                    r = api.kernels_push(d, acc=self.acc)
                self.last_result = {k: getattr(r, k, None) for k in ("url", "version_number", "error")}
                if self.last_result.get("error"):
                    self.last_error = str(self.last_result["error"])
            except Exception as e:                 # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                raise
