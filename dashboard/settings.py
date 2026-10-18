"""Paths and mode shared by the dashboard. Override with environment variables when the layout differs.

M29_MODE=local  (default) the server owns the GPU and runs jobs itself.
M29_MODE=web    the server has no GPU (Render): jobs wait for remote workers (worker.py on Kaggle or a PC).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MODE = os.environ.get("M29_MODE", "local")
_SIBLING = WORKTREE / "ai_router"      # the repo's own ai_router, once its calibration has been generated
AI_ROUTER = Path(os.environ.get("M29_AI_ROUTER") or (
    _SIBLING if (_SIBLING / "calibrated_params" / "centroids.pt").is_file()
    else "D:/tool_build_phase/.claude/worktrees/m29-codebase-pipeline-design-d4c799/ai_router"))
DATA_ROOT = Path(os.environ.get("M29_DATA_ROOT", "D:/tool_build_phase"))
SPLITS = {s: DATA_ROOT / "ai_router" / "data" / "100k" / s for s in ("train", "val", "test")}
STORE = Path(os.environ.get("M29_DASH_STORE", HERE / "data"))          # our own outputs, never committed
JOBS = STORE / "jobs"
THUMBS = STORE / "thumbs"
SEED = STORE / "seed"            # BDD100K images a local worker uploads to a web server at run time (never in git)
BUNDLED = HERE / "bundled"       # derived data committed with the code: aggregate statistics, example results
EXAMPLE_SINGLE = WORKTREE / "one_image_walkthrough" / "input" / "cd92dc38-3ee11e29.jpg"
EXAMPLE_BATCH_DIR = WORKTREE / "flow_compare" / "batch_input"
for p in (STORE, JOBS, THUMBS, SEED):
    p.mkdir(parents=True, exist_ok=True)


def data_file(name: str) -> Path:
    """Our own computed file if present, else the copy shipped in bundled/."""
    p = STORE / name
    return p if p.is_file() else BUNDLED / name
