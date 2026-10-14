"""Tạo 3 notebook huấn luyện (mỗi chuyên gia một notebook, EXPERT_ID điền sẵn) và 1 notebook kiểm tra ensemble.
Notebook chuyên gia 1 ghi đè src/Train_RTDETR_Ensemble.ipynb. Dùng: python build_nb.py <thư mục src>"""
import json, sys, os, ast
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells")
rd = lambda f: open(os.path.join(C, f), encoding="utf-8").read()
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
DATASET = "solesensei/solesensei_bdd100k"
USER = "YOUR_KAGGLE_USERNAME"

def intro(k):
    return f"""# 🚗 Build phase — Chuyên gia {k}/3: RT-DETR-R50vd trên 10 lớp BDD100K

## Thành viên nhóm: chỉ cần 3 thao tác
1. Mở notebook được giao → **Copy & Edit**.
2. Nhìn thanh bên phải: *Accelerator = GPU T4 x2*, *Internet = On*, và mục *Input* có dataset **BDD100K** (`solesensei_bdd100k`). Nếu thiếu, notebook sẽ dừng ngay trong phút đầu kèm lời nhắc cụ thể, không tốn quota.
3. **Save Version → Save** (để mặc định *Save & Run All*). Có thể đóng trình duyệt; khoảng 8.5 giờ sau, kết quả nằm ở tab *Output*.

Không sửa gì trong notebook. `EXPERT_ID = {k}` đã được điền sẵn.

## Notebook tự làm các bước sau
1. **Kiểm tra môi trường:** GPU, Internet, phiên bản `transformers`, và đủ dữ liệu.
2. **Chuẩn bị dữ liệu.** Giữ **mọi box của 10 lớp**:
   * **BDD train** (69.863 ảnh) → train: dùng toàn bộ.
   * **BDD val** (10.000 ảnh) → **chỉ để xem**: đường cong loss trong lúc train (trên 1.000 ảnh cố định) và báo cáo từng lớp sau khi train. Không chọn checkpoint, không chỉnh siêu tham số theo val, để val vẫn dùng được cho bài báo.
   * **BDD test** → không đọc.
3. **Đo tốc độ** trong 40 bước, rồi tự chọn số bước train cho vừa `TRAIN_HOURS` = 8 giờ (vừa giới hạn 12 giờ mỗi phiên và hạn mức GPU khoảng 30 giờ mỗi tuần của Kaggle).
4. **Train** với seed riêng của chuyên gia {k}. Có chốt chặn: tự lưu và dừng nếu phiên chạy tới 11 giờ.
5. **Kiểm tra** trên toàn bộ val: recall và precision theo từng lớp, recall theo thời tiết, thời điểm trong ngày và loại cảnh.

## Kết quả trong Output
`expert_{k}.pth`, `expert_{k}_final/` (định dạng HuggingFace), `expert_{k}_manifest.json` (seed, số epoch, sha256 trọng số), `expert_{k}_class_report.txt`, `expert_{k}_loss_curve.csv` (loss train và val theo bước), `data_stats.json`.
"""

def config(k):
    return f"""# ==============================================================================
# CẤU HÌNH (đã điền sẵn cho chuyên gia {k}; thường không cần sửa)
# ==============================================================================
EXPERT_ID = {k}       # seed: 1 -> 42, 2 -> 142, 3 -> 242
TRAIN_HOURS = 8.0   # quỹ thời gian cho riêng bước train
NUM_WORKERS = 2     # đặt 0 nếu dataloader bị treo
RESUME_FROM = ""    # để trống; chỉ dùng khi tiếp tục từ checkpoint-* của một phiên trước

import os, time
os.environ.update(EXPERT_ID=str(EXPERT_ID), NUM_WORKERS=str(NUM_WORKERS), RESUME_FROM=RESUME_FROM,
                  NB_START=str(time.time()))
os.chdir("/kaggle/working")
"""

env_check = """# ==============================================================================
# KIỂM TRA MÔI TRƯỜNG
# ==============================================================================
import subprocess, sys, urllib.request, torch
n_gpu = torch.cuda.device_count()
print(f"[*] PyTorch {torch.__version__} | số GPU: {n_gpu} | " + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)))
assert n_gpu >= 1, "Chưa bật GPU: Settings -> Accelerator -> GPU T4 x2."
if n_gpu < 2:
    print("[!] Chỉ có 1 GPU: vẫn chạy được (batch hiệu dụng giữ nguyên 16) nhưng số epoch đạt được sẽ ít hơn.")
try:
    urllib.request.urlopen("https://huggingface.co", timeout=15)
except Exception as e:
    raise RuntimeError("Không có Internet (cần để tải trọng số PekingU/rtdetr_r50vd): Settings -> Internet -> On") from e
import transformers
from packaging.version import Version
if Version(transformers.__version__) < Version("4.44"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.44", "accelerate"])
    print("[*] Đã nâng cấp transformers (các script chạy ở tiến trình riêng nên dùng ngay bản mới).")
else:
    print(f"[V] transformers {transformers.__version__}")
"""

bench = """# ==============================================================================
# ĐO TỐC ĐỘ (40 bước) VÀ CHỌN SỐ BƯỚC TRAIN CHO VỪA QUỸ THỜI GIAN
# ==============================================================================
import json, shutil
!BENCH=1 torchrun --nproc_per_node={n_gpu} train.py
b = json.load(open(f"/kaggle/working/bench_{EXPERT_ID}.json"))
shutil.rmtree(f"/kaggle/working/bench_tmp_{EXPERT_ID}", ignore_errors=True)
MAX_STEPS = int(TRAIN_HOURS * 3600 * b["steps_per_sec"] * 0.90)   # chừa 10% cho lưu checkpoint và đo loss val
epochs = MAX_STEPS * b["effective_batch"] / b["images_per_epoch"]
os.environ["MAX_STEPS"] = str(MAX_STEPS)
print(f"[*] {b['steps_per_sec'] * b['effective_batch']:.1f} ảnh/giây -> {MAX_STEPS} bước ≈ {epochs:.1f} epoch trong {TRAIN_HOURS} giờ")
if epochs < 3:
    print("[!] Dưới 3 epoch: mô hình sẽ chưa hội tụ tốt. Hãy kiểm tra đã chọn GPU T4 x2 chưa.")
"""

summary = """# ==============================================================================
# TÓM TẮT KẾT QUẢ
# ==============================================================================
import json, os
m = json.load(open(f"/kaggle/working/expert_{EXPERT_ID}_manifest.json"))
print(json.dumps({k: m[k] for k in ("expert_id", "seed", "steps_done", "epochs_done", "effective_batch",
                                    "train_filtered", "weights_sha256")}, indent=2))
for f in sorted(os.listdir("/kaggle/working")):
    p = os.path.join("/kaggle/working", f)
    print("    ->", f + ("/" if os.path.isdir(p) else f"  ({os.path.getsize(p) / 2**20:.1f} MB)"))
"""

verify_src = '''# ==============================================================================
# KIỂM TRA ENSEMBLE: 3 chuyên gia có thật sự khác nhau không, và đã học đủ 10 lớp chưa
# Add Input: Output của 3 notebook "bdd100k-rtdetr-expert-1/2/3". Chạy được trên CPU.
# ==============================================================================
import glob, json, os, itertools, torch
pths = sorted(glob.glob("/kaggle/input/**/expert_*.pth", recursive=True))
mans = sorted(glob.glob("/kaggle/input/**/expert_*_manifest.json", recursive=True))
reps = sorted(glob.glob("/kaggle/input/**/expert_*_class_report.txt", recursive=True))
print(f"[*] Tìm thấy {len(pths)} file trọng số, {len(mans)} manifest")
assert len(pths) >= 2, "Cần Output của ít nhất 2 chuyên gia trong Input."
ok = True
M = [json.load(open(p)) for p in mans]
for m in M:
    print(f"    chuyên gia {m['expert_id']}: seed={m['seed']} epoch={m['epochs_done']:.2f} sha={m['weights_sha256'][:12]}")
if len({m["seed"] for m in M}) < len(M) or len({m["weights_sha256"] for m in M}) < len(M):
    print("[X] Có chuyên gia trùng seed hoặc trùng trọng số!"); ok = False
sd = {p: torch.load(p, map_location="cpu", weights_only=True) for p in pths}
for a, b in itertools.combinations(pths, 2):
    for part, key in (("đầu phân loại", lambda k: "class_embed" in k or "enc_score_head" in k), ("backbone", lambda k: "backbone" in k)):
        ks = [k for k in sd[a] if key(k) and sd[a][k].is_floating_point()]
        va = torch.cat([sd[a][k].flatten().float() for k in ks]); vb = torch.cat([sd[b][k].flatten().float() for k in ks])
        d = ((va - vb).norm() / va.norm()).item()
        print(f"    {os.path.basename(a)} vs {os.path.basename(b)} | {part}: lệch tương đối {d:.4f}")
        if d < 1e-6:
            print("[X] Hai chuyên gia GIỐNG HỆT nhau!"); ok = False
for r in reps:
    print("\\n" + open(r, encoding="utf-8").read())
    if "CHƯA HỌC ĐƯỢC" in open(r, encoding="utf-8").read():
        ok = False
print("\\n[V] ENSEMBLE HỢP LỆ: các chuyên gia khác nhau và đã học đủ các lớp." if ok else "\\n[X] ENSEMBLE CHƯA ĐẠT, xem các dòng [X] / CHƯA HỌC ĐƯỢC ở trên.")
'''

def check_syntax(cells):
    for c in cells:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%%", "!"))))

def save(cells, path, gpu=True):
    check_syntax(cells)
    kaggle = {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isGpuEnabled": gpu, "isInternetEnabled": gpu,
              "language": "python", "sourceType": "notebook"}
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}, "kaggle": kaggle}, "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

out = sys.argv[1]  # thư mục src/ của dự án
for k in (1, 2, 3):
    cells = [md(intro(k)), code(config(k)), code(env_check),
             md("## Chuẩn bị dữ liệu"), code(rd("common.py")), code(rd("prepare.py")), code("!python prepare_data.py\n"),
             md("## Huấn luyện"), code(rd("train.py")), code(bench),
             code("# TRAIN (đọc MAX_STEPS từ cell trên)\n!torchrun --nproc_per_node={n_gpu} train.py\n"),
             md("## Kiểm tra từng lớp và edge case"), code(rd("check.py")), code("!python check_classes.py {EXPERT_ID}\n"),
             code(summary)]
    name = "Train_RTDETR_Ensemble.ipynb" if k == 1 else f"Train_RTDETR_Ensemble_Expert{k}.ipynb"
    save(cells, os.path.join(out, name))
vcells = [md("# Kiểm tra ensemble 3 chuyên gia\nAdd Input → *Notebook Output* của 3 notebook chuyên gia, rồi **Run All** (không cần GPU)."),
          code(verify_src)]
save(vcells, os.path.join(out, "Verify_Ensemble.ipynb"), gpu=False)
print("Đã ghi:", [f for f in sorted(os.listdir(out)) if f.endswith(".ipynb")])
