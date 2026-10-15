%%writefile latent_stats.py
# ==============================================================================
# THỐNG KÊ KHÔNG GIAN TIỀM ẨN (chạy sau khi train, cho MỌI chuyên gia)
#  Trên một tập con cố định của BDD train (md5(tên ảnh) % 7 == 0, khoảng 10.000 ảnh), theo M29 Mục 4.B:
#   - Max-IoU Assignment: với mỗi GT, chọn query có IoU >= 0.5 cao nhất VÀ đoán đúng lớp;
#     lấy vector ẩn 256D của lớp decoder cuối (decoder_hidden_states[-1]).
#  Đầu ra:
#   - expert_K_centroids.pt  : ĐÚNG định dạng M29 (pca, centroids, inv_covs, class_names, d_in, d_out, jitter_eps),
#                              PCA 256 -> 64, Σ_k^-1 theo lớp (+1e-6 I). Chuyên gia 1 chép thêm thành centroids.pt.
#                              Lớp có quá ít vector (< 192) dùng hiệp phương sai co rút Ledoit-Wolf thay vì
#                              hiệp phương sai mẫu (vốn suy biến), và được ghi trong "shrunk_classes".
#   - expert_K_latent_stats.pt: chỉ chứa tensor (nạp được bằng weights_only=True), cho tín hiệu FCD của H2:
#                              tâm lớp 256D, hiệp phương sai gộp Ledoit-Wolf 256D (precision), trung vị/MAD của
#                              khoảng cách Mahalanobis trên chính các vector TP này (để chuẩn hóa robust).
# Không dùng val: thống kê tham chiếu chỉ lấy từ train.
# ==============================================================================
import os, sys, json, hashlib, collections
import numpy as np, torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from torchvision.ops import box_iou
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from common import *

EXPERT_ID = int(sys.argv[1])
CAP_PER_CLASS = 30000            # car chiếm đa số; giới hạn để vừa RAM, lớp hiếm lấy hết
JITTER = 1e-6
dev = "cuda" if torch.cuda.is_available() else "cpu"
final_dir = os.path.join(WORK, f"expert_{EXPERT_ID}_final")
model = RTDetrForObjectDetection.from_pretrained(final_dir).to(dev).eval()
proc = RTDetrImageProcessor.from_pretrained(final_dir)
with open(COMPACT["train"]) as f:
    items = [r for r in json.load(f) if int(hashlib.md5(r["name"].encode()).hexdigest(), 16) % 7 == 0 and r["boxes"]]
print(f"[*] Chuyên gia {EXPERT_ID}: trích vector ẩn trên {len(items)} ảnh train")

emb = collections.defaultdict(list); n_gt = 0; n_kept = 0
rng = np.random.default_rng(EXPERT_ID)
for k in range(0, len(items), 16):
    batch = items[k:k + 16]
    ims = [Image.open(r["file"]).convert("RGB") for r in batch]
    inp = proc(images=ims, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        out = model(pixel_values=inp["pixel_values"], output_hidden_states=True)
    hid = out.decoder_hidden_states[-1].float()                       # [B, 300, 256]
    pcls = out.logits.float().argmax(-1)                              # [B, 300]
    pb = out.pred_boxes.float()                                       # cx, cy, w, h chuẩn hóa
    pxy = torch.stack([pb[..., 0] - pb[..., 2] / 2, pb[..., 1] - pb[..., 3] / 2,
                       pb[..., 0] + pb[..., 2] / 2, pb[..., 1] + pb[..., 3] / 2], -1)
    for b, (r, im) in enumerate(zip(batch, ims)):
        W, H = im.size
        g = torch.tensor(r["boxes"], dtype=torch.float32, device=dev)
        gxy = g[:, :4] / torch.tensor([W, H, W, H], device=dev); gc = g[:, 4].long()
        iou = box_iou(gxy, pxy[b])                                    # [N_gt, 300]
        ok = (iou >= 0.5) & (pcls[b][None, :] == gc[:, None])
        n_gt += len(gc)
        for i in torch.nonzero(ok.any(1)).flatten().tolist():
            q = int(torch.argmax(iou[i] * ok[i].float()))
            c = int(gc[i])
            if len(emb[c]) < CAP_PER_CLASS:
                emb[c].append(hid[b, q].cpu().numpy())
            n_kept += 1
    if (k // 16) % 100 == 0:
        print(f"    {k + len(batch)}/{len(items)} ảnh | GT khớp được: {n_kept}/{n_gt}", flush=True)

counts = {BDD_CLASSES[c]: len(emb[c]) for c in range(10)}
print(f"[*] Vector theo lớp: {counts} | tỉ lệ GT khớp được (IoU>=0.5, đúng lớp): {n_kept / max(n_gt, 1):.3f}")
X = np.vstack([np.asarray(emb[c]) for c in range(10) if emb[c]])
MIN_VEC = int(os.environ.get("LATENT_MIN_VECTORS", 256))
assert len(X) >= MIN_VEC, f"Quá ít vector ({len(X)}): mô hình có thể chưa học được gì."

# ---- Định dạng M29: PCA 64 + Σ_k^-1 theo lớp ----
pca = PCA(n_components=64, random_state=42).fit(X)
centroids, inv_covs, shrunk = {}, {}, []
for c in range(10):
    if not emb[c]:
        continue
    Z = pca.transform(np.asarray(emb[c]))
    centroids[c] = torch.tensor(Z.mean(0), dtype=torch.float32)
    if len(Z) >= 3 * 64:
        cov = np.cov(Z, rowvar=False)
    else:
        cov = LedoitWolf().fit(Z).covariance_ if len(Z) > 1 else np.eye(64); shrunk.append(BDD_CLASSES[c])
    inv_covs[c] = torch.tensor(np.linalg.inv(cov + JITTER * np.eye(64)), dtype=torch.float32)
m29 = {"pca": pca, "centroids": centroids, "inv_covs": inv_covs, "class_names": BDD_CLASSES, "d_in": 256, "d_out": 64,
       "jitter_eps": JITTER, "shrunk_classes": shrunk, "n_vectors": counts, "source": "BDD train, md5(name) % 7 == 0",
       "expert_id": EXPERT_ID}
p = os.path.join(WORK, f"expert_{EXPERT_ID}_centroids.pt"); torch.save(m29, p)
if EXPERT_ID == 1:
    torch.save(m29, os.path.join(WORK, "centroids.pt"))   # tên M29 dùng cho base model
print(f"[V] Đã lưu {p}" + (" và centroids.pt" if EXPERT_ID == 1 else "") +
      (f" | lớp dùng hiệp phương sai co rút: {shrunk}" if shrunk else ""))

# ---- Cho H2: tâm 256D + hiệp phương sai gộp Ledoit-Wolf + chuẩn hóa robust ----
means = np.zeros((10, 256), np.float32)
within = []
for c in range(10):
    if emb[c]:
        A = np.asarray(emb[c]); means[c] = A.mean(0); within.append(A - means[c])
lw = LedoitWolf().fit(np.vstack(within))
P = lw.precision_.astype(np.float32)
D = []
for c in range(10):
    if emb[c]:
        d = np.asarray(emb[c]) - means[c]
        D.append(np.sqrt(np.einsum("ij,jk,ik->i", d, P, d)))
D = np.concatenate(D); med = float(np.median(D)); mad = float(np.median(np.abs(D - med)) * 1.4826)
torch.save({"class_means": torch.tensor(means), "precision": torch.tensor(P), "shrinkage": torch.tensor(lw.shrinkage_),
            "tp_median_D": torch.tensor(med), "tp_mad_D": torch.tensor(mad),
            "n_vectors": torch.tensor([len(emb[c]) for c in range(10)])},
           os.path.join(WORK, f"expert_{EXPERT_ID}_latent_stats.pt"))
print(f"[V] Đã lưu expert_{EXPERT_ID}_latent_stats.pt | co rút Ledoit-Wolf {lw.shrinkage_:.3f} | "
      f"Mahalanobis TP: trung vị {med:.2f}, MAD {mad:.2f}")
