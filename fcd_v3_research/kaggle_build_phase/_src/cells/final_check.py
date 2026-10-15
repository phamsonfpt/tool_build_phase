# ==============================================================================
# KIỂM TRA TỔNG: tự đánh dấu đạt/không đạt cho từng điều kiện và kết luận ĐẠT / KHÔNG ĐẠT
# (dùng để đánh giá cả bản chạy thử lẫn bản chính thức; chỉ đọc file kết quả, không chạy lại gì)
# ==============================================================================
import json, os, csv, re, torch
from common import SEEDS, BDD_CLASSES
W = "/kaggle/working" if os.path.exists("/kaggle/working") else os.environ.get("WORK_DIR", ".")
K = EXPERT_ID
results = []                                    # (đạt?, bắt buộc?, nội dung)
def check(ok, msg, required=True):
    results.append((bool(ok), required, msg))

# 1) Manifest: đúng seed, bản vá loss đã chạy, train hết số bước
mp = os.path.join(W, f"expert_{K}_manifest.json")
m = json.load(open(mp)) if os.path.exists(mp) else {}
check(m, "Có expert_K_manifest.json (train đã chạy xong và lưu mô hình)")
check(m.get("seed") == SEEDS.get(K), f"Seed đúng ({m.get('seed')} = {SEEDS.get(K)})")
check(m.get("loss_fix_calls", 0) > 0, f"Bản vá loss đã chạy trong lúc train ({m.get('loss_fix_calls', 0)} lần)")
check(not m.get("stopped_early", True), f"Train hết số bước đã đặt ({m.get('steps_done')}/{m.get('max_steps')})")
epochs = m.get("epochs_done", 0)
check(epochs >= 3 or K == 9, f"Đủ số epoch ({epochs:.2f}; bản chạy thử không cần)")

# 2) Probe điểm tin cậy: phải giữ ở mức cao, không có lần nào tụt dưới 0.1
pp = os.path.join(W, f"expert_{K}_probe.csv")
rows = list(csv.DictReader(open(pp))) if os.path.exists(pp) else []
scores = [float(r["mean_max_score"]) for r in rows]
check(rows, f"Có dữ liệu probe ({len(rows)} lần đo)")
if scores:
    check(min(scores) >= 0.1, f"Điểm probe không lần nào dưới 0.1 (thấp nhất {min(scores):.3f})")
    check(scores[-1] >= 0.3, f"Điểm probe lần cuối từ 0.3 trở lên ({scores[-1]:.3f})")

# 3) Báo cáo trên val (chỉ để xem): mô hình có đưa ra dự đoán, các lớp phổ biến đã học
rp = os.path.join(W, f"expert_{K}_class_report.txt")
rep = open(rp, encoding="utf-8").read() if os.path.exists(rp) else ""
rec, npred = {}, 0
for line in rep.splitlines():
    mm = re.match(r"^(\S+(?: \S+)?)\s+(\d+)\s+(\d+)\s+([\d.]+|nan)\s", line)
    if mm and mm.group(1) in BDD_CLASSES:
        npred += int(mm.group(3)); rec[mm.group(1)] = float(mm.group(4)) if mm.group(4) != "nan" else 0.0
check(rep, "Có expert_K_class_report.txt")
check(npred > 0, f"Mô hình có đưa ra dự đoán trên val ({npred} box)")
check(rec.get("car", 0) >= (0.3 if K == 9 else 0.5), f"Recall lớp car đủ cao ({rec.get('car', 0):.3f})")
if K != 9:
    check("CHƯA HỌC ĐƯỢC" not in rep, "Không lớp nào bị báo CHƯA HỌC ĐƯỢC")
else:
    check("CHƯA HỌC ĐƯỢC" not in rep, "Không lớp nào bị báo CHƯA HỌC ĐƯỢC (bản chạy thử train ngắn, lớp hiếm có thể chưa học kịp)", required=False)

# 4) Thống kê tiềm ẩn
cp = os.path.join(W, f"expert_{K}_centroids.pt"); lp = os.path.join(W, f"expert_{K}_latent_stats.pt")
check(os.path.exists(cp), f"Có expert_{K}_centroids.pt (định dạng M29)")
check(os.path.exists(lp), f"Có expert_{K}_latent_stats.pt (cho H2)")
if K == 1:
    check(os.path.exists(os.path.join(W, "centroids.pt")), "Có centroids.pt (base model của M29)")
if os.path.exists(lp):
    s = torch.load(lp, map_location="cpu", weights_only=True)
    n_car = int(s["n_vectors"][BDD_CLASSES.index("car")])
    check(n_car >= 1000, f"Đủ vector của lớp car để tính thống kê ({n_car})")

# 5) Thông tin thêm (không quyết định): đường cong loss val
lc = os.path.join(W, f"expert_{K}_loss_curve.csv")
if os.path.exists(lc):
    v = [(int(r["step"]), float(r["val_loss"])) for r in csv.DictReader(open(lc)) if r.get("val_loss")]
    if v:
        print("[i] Val loss (chỉ để xem):", ", ".join(f"bước {a}: {b:.2f}" for a, b in v))

print()
for ok, req, msg in results:
    print(("  [V] " if ok else ("  [X] " if req else "  [!] ")) + msg)
passed = all(ok for ok, req, _ in results if req)
print("\n" + ("=" * 22 + " ĐẠT " + "=" * 22 if passed else "=" * 18 + " KHÔNG ĐẠT " + "=" * 18))
print("Có thể dùng mô hình này." if passed and K != 9 else
      "Bản chạy thử ổn: có thể cho 3 chuyên gia chính thức chạy." if passed else
      "Xem các dòng [X] ở trên trước khi chạy tiếp.")
