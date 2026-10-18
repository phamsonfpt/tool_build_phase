"""Kaggle GPU worker for the M29 dashboard. The Render server fills in the two values below and pushes this
file as a private Kaggle script kernel; never publish a copy that contains a real token."""
import glob
import os
import subprocess
import sys

SERVER = "__SERVER_URL__"
TOKEN = "__WORKER_TOKEN__"

# the private dataset made by kaggle/make_bundle.py: ai_router code + checkpoints + dashboard + HF config
found = glob.glob("/kaggle/input/**/dashboard/worker.py", recursive=True)
if not found:
    sys.exit("Không tìm thấy dataset m29-ai-router-bundle: hãy gắn dataset vào kernel")
root = found[0][: -len("/dashboard/worker.py")]
print("bundle:", root, flush=True)

# the experts' state dicts were converted for transformers 4.57 module names
try:
    import transformers
    have = transformers.__version__
except ImportError:
    have = None
if have != "4.57.3":
    print("transformers", have, "-> 4.57.3", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3"], check=False)

env = dict(os.environ, M29_MODE="worker", M29_AI_ROUTER=f"{root}/ai_router", M29_DATA_ROOT=root,
           M29_DASH_STORE="/kaggle/working/store", HF_HUB_CACHE=f"{root}/hf_cache", HF_HUB_OFFLINE="1")
subprocess.run([sys.executable, f"{root}/dashboard/worker.py", "--server", SERVER, "--token", TOKEN,
                "--name", "kaggle", "--idle-exit", "600", "--max-hours", "8.5"], env=env, check=True)
