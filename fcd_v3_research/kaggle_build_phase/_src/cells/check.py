%%writefile check_classes.py
# ==============================================================================
# KIỂM TRA SAU KHI TRAIN trên build_check (nửa BDD test, không dùng khi train):
#  (1) recall/precision theo từng lớp -> mô hình đã học đủ 10 lớp chưa
#  (2) recall theo nhóm điều kiện (thời tiết, thời điểm, loại cảnh) -> edge case
#  (3) so sánh với các chuyên gia khác (nếu có trong Input) -> có phải bản sao không
# Kết quả ghi vào expert_K_class_report.txt.
# ==============================================================================
import os, sys, json, glob, itertools, collections
import numpy as np, torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from common import *

class Tee:  # in ra màn hình đồng thời ghi vào file báo cáo
    def __init__(self, path): self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()

EXPERT_ID = int(sys.argv[1]); N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
sys.stdout = Tee(os.path.join(WORK, f"expert_{EXPERT_ID}_class_report.txt"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
final_dir = os.path.join(WORK, f"expert_{EXPERT_ID}_final")
model = RTDetrForObjectDetection.from_pretrained(final_dir).to(dev).eval()
proc = RTDetrImageProcessor.from_pretrained(final_dir)
with open(COMPACT["build_check"]) as f:
    items = json.load(f)[:N_IMG]

def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)

gt = np.zeros(10, int); tp = np.zeros(10, int); npred = np.zeros(10, int); pred_tp = np.zeros(10, int)
strata = collections.defaultdict(lambda: [0, 0, 0])  # (thuộc tính, giá trị) -> [số ảnh, GT, TP]
for k in range(0, len(items), 16):
    batch = items[k:k + 16]
    ims = [Image.open(r["file"]).convert("RGB") for r in batch]
    inp = proc(images=ims, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        out = model(**inp)
    out["logits"] = out.logits.float(); out["pred_boxes"] = out.pred_boxes.float()
    res = proc.post_process_object_detection(out, threshold=0.3, target_sizes=[im.size[::-1] for im in ims])
    for r, p in zip(batch, res):
        g = np.array(r["boxes"], float).reshape(-1, 5)
        pb, pl = p["boxes"].cpu().numpy(), p["labels"].cpu().numpy()
        img_gt = img_tp = 0
        for c in range(10):
            gc, pc = g[g[:, 4] == c, :4], pb[pl == c]
            gt[c] += len(gc); npred[c] += len(pc); img_gt += len(gc)
            if len(gc) and len(pc):
                m = iou(gc, pc) >= 0.5
                t = int(m.any(1).sum()); tp[c] += t; img_tp += t; pred_tp[c] += int(m.any(0).sum())
        for a in ("weather", "timeofday", "scene"):
            s = strata[(a, r.get("attributes", {}).get(a, "?"))]
            s[0] += 1; s[1] += img_gt; s[2] += img_tp

print(f"[*] Chuyên gia {EXPERT_ID}: {len(items)} ảnh build_check (nửa BDD test, không dùng khi train), ngưỡng 0.3, IoU 0.5")
print(f"\n{'lớp':14s} {'GT':>7s} {'dự đoán':>8s} {'recall':>7s} {'precision':>9s}")
weak = []
for c, n in enumerate(BDD_CLASSES):
    rec = tp[c] / gt[c] if gt[c] else float("nan"); prec = pred_tp[c] / npred[c] if npred[c] else float("nan")
    note = "  (dưới 20 GT: không đủ tin cậy)" if gt[c] < 20 else ""
    print(f"{n:14s} {gt[c]:7d} {npred[c]:8d} {rec:7.3f} {prec:9.3f}{note}")
    if gt[c] >= 20 and rec < 0.05:
        weak.append(n)
print("\n[!] CHƯA HỌC ĐƯỢC:" if weak else "\n[V] Mọi lớp đủ dữ liệu kiểm tra đều đã được học.", weak or "")

print(f"\n{'nhóm điều kiện':32s} {'ảnh':>6s} {'GT':>7s} {'recall':>7s}")
for (a, v), (n_img, g_, t_) in sorted(strata.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
    note = "  (dưới 30 ảnh: không đủ tin cậy)" if n_img < 30 else ""
    print(f"{a + ' = ' + str(v):32s} {n_img:6d} {g_:7d} {t_ / g_ if g_ else float('nan'):7.3f}{note}")

paths = sorted(set(glob.glob(os.path.join(WORK, "expert_*.pth")) + glob.glob("/kaggle/input/**/expert_*.pth", recursive=True)))
heads = {}
for p in paths:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    heads[p] = torch.cat([sd[k].flatten().float() for k in sorted(sd) if "class_embed" in k or "enc_score_head" in k])
for a, b in itertools.combinations(heads, 2):
    d = (heads[a] - heads[b]).norm() / heads[a].norm()
    flag = "  <-- GIỐNG HỆT, không phải ensemble!" if d < 1e-6 else ""
    print(f"[*] Lệch tương đối đầu phân loại {os.path.basename(a)} vs {os.path.basename(b)}: {d:.4f}{flag}")
