%%writefile latent_stats.py
# ==============================================================================
# THỐNG KÊ KHÔNG GIAN TIỀM ẨN (chạy sau khi train, cho MỌI chuyên gia, trên TOÀN BỘ BDD train)
#  Theo M29 Mục 4.B, Max-IoU Assignment: với mỗi GT, chọn query có IoU >= 0.5 cao nhất VÀ đoán đúng lớp;
#  lấy vector ẩn 256D của lớp decoder cuối (decoder_hidden_states[-1]).
#  Hai bước:
#   python latent_stats.py extract <K> <phần> <số phần>  : trích vector trên một phần tập train (mỗi GPU một phần)
#   python latent_stats.py merge   <K> <số phần>          : gộp các phần và tính thống kê
#  Đầu ra:
#   - expert_K_centroids.pt  : ĐÚNG định dạng M29 (pca, centroids, inv_covs, class_names, d_in, d_out, jitter_eps),
#                              PCA 256 -> 64, Σ_k^-1 theo lớp (+1e-6 I). Chuyên gia 1 chép thêm thành centroids.pt.
#                              Lớp có dưới 192 vector dùng hiệp phương sai co rút Ledoit-Wolf ("shrunk_classes").
#   - expert_K_latent_stats.pt: chỉ chứa tensor (nạp được bằng weights_only=True), cho tín hiệu FCD của H2:
#                              tâm lớp 256D, precision của hiệp phương sai gộp Ledoit-Wolf 256D, trung vị/MAD của
#                              khoảng cách Mahalanobis trên các vector TP (để chuẩn hóa robust).
# Không dùng val: thống kê tham chiếu chỉ lấy từ train.
# ==============================================================================
import os, sys, json, collections
import numpy as np, torch
from common import *

MODE, EXPERT_ID = sys.argv[1], int(sys.argv[2])
CAP_PER_CLASS = 50000           # giới hạn mỗi lớp (car, traffic sign, traffic light vượt mức này); lớp hiếm lấy hết
JITTER = 1e-6
part_path = lambda i: os.path.join(WORK, f"expert_{EXPERT_ID}_latent_part{i}.npz")

if MODE == "extract":
    from PIL import Image
    from torchvision.ops import box_iou
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    part, nparts = int(sys.argv[3]), int(sys.argv[4])
    dev = f"cuda:{part % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu"
    final_dir = os.path.join(WORK, f"expert_{EXPERT_ID}_final")
    model = RTDetrForObjectDetection.from_pretrained(final_dir).to(dev).eval()
    proc = RTDetrImageProcessor.from_pretrained(final_dir)
    with open(COMPACT["train"]) as f:
        items = [r for i, r in enumerate(json.load(f)) if r["boxes"] and i % nparts == part]
    print(f"[*] Chuyên gia {EXPERT_ID}, phần {part + 1}/{nparts} trên {dev}: {len(items)} ảnh train", flush=True)
    cap = CAP_PER_CLASS // nparts
    emb = collections.defaultdict(list); seen = collections.Counter()
    rng = np.random.default_rng(1000 * EXPERT_ID + part); n_gt = n_kept = 0
    for k in range(0, len(items), 16):
        batch = items[k:k + 16]
        ims = [Image.open(r["file"]).convert("RGB") for r in batch]
        inp = proc(images=ims, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev != "cpu"):
            out = model(pixel_values=inp["pixel_values"], output_hidden_states=True)
        hid = out.decoder_hidden_states[-1].float(); pcls = out.logits.float().argmax(-1); pb = out.pred_boxes.float()
        pxy = torch.stack([pb[..., 0] - pb[..., 2] / 2, pb[..., 1] - pb[..., 3] / 2,
                           pb[..., 0] + pb[..., 2] / 2, pb[..., 1] + pb[..., 3] / 2], -1)
        for b, (r, im) in enumerate(zip(batch, ims)):
            W, H = im.size
            g = torch.tensor(r["boxes"], dtype=torch.float32, device=dev)
            gxy = g[:, :4] / torch.tensor([W, H, W, H], device=dev); gc = g[:, 4].long()
            iou = box_iou(gxy, pxy[b])
            ok = (iou >= 0.5) & (pcls[b][None, :] == gc[:, None])
            n_gt += len(gc)
            for i in torch.nonzero(ok.any(1)).flatten().tolist():
                q = int(torch.argmax(iou[i] * ok[i].float())); c = int(gc[i]); v = hid[b, q].cpu().numpy()
                seen[c] += 1; n_kept += 1
                if len(emb[c]) < cap:                      # reservoir sampling: mẫu đều trên toàn bộ ảnh
                    emb[c].append(v)
                else:
                    j = rng.integers(0, seen[c])
                    if j < cap:
                        emb[c][j] = v
        if (k // 16) % 500 == 0:
            print(f"    phần {part + 1}: {k + len(batch)}/{len(items)} ảnh | GT khớp được {n_kept}/{n_gt}", flush=True)
    np.savez(part_path(part), n_gt=n_gt, n_kept=n_kept,
             **{f"c{c}": np.asarray(emb[c], np.float32).reshape(-1, 256) for c in range(10)},
             **{f"seen{c}": seen[c] for c in range(10)})
    print(f"[V] Phần {part + 1}/{nparts}: {n_kept}/{n_gt} GT khớp được", flush=True)

elif MODE == "merge":
    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf
    nparts = int(sys.argv[3])
    parts = []
    for i in range(nparts):
        with np.load(part_path(i)) as z:                        # đọc hết vào RAM rồi đóng file
            parts.append({k: z[k] for k in z.files})
    emb = {c: np.concatenate([p[f"c{c}"] for p in parts]) for c in range(10)}
    seen = {c: int(sum(p[f"seen{c}"] for p in parts)) for c in range(10)}
    n_gt = int(sum(p["n_gt"] for p in parts)); n_kept = int(sum(p["n_kept"] for p in parts))
    counts = {BDD_CLASSES[c]: len(emb[c]) for c in range(10)}
    print(f"[*] GT khớp được (IoU>=0.5, đúng lớp): {n_kept}/{n_gt} = {n_kept / max(n_gt, 1):.3f}")
    print(f"[*] Vector dùng để tính (sau giới hạn {CAP_PER_CLASS}/lớp): {counts}")
    X = np.vstack([emb[c] for c in range(10) if len(emb[c])])
    MIN_VEC = int(os.environ.get("LATENT_MIN_VECTORS", 256))
    assert len(X) >= MIN_VEC, f"Quá ít vector ({len(X)}): mô hình có thể chưa học được gì."
    # ---- Định dạng M29: PCA 64 + Σ_k^-1 theo lớp ----
    sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 200000), replace=False)]
    pca = PCA(n_components=64, random_state=42).fit(sub)
    centroids, inv_covs, shrunk = {}, {}, []
    for c in range(10):
        if not len(emb[c]):
            continue
        Z = pca.transform(emb[c])
        centroids[c] = torch.tensor(Z.mean(0), dtype=torch.float32)
        if len(Z) >= 3 * 64:
            cov = np.cov(Z, rowvar=False)
        else:
            cov = LedoitWolf().fit(Z).covariance_ if len(Z) > 1 else np.eye(64)
            shrunk.append(BDD_CLASSES[c])
        inv_covs[c] = torch.tensor(np.linalg.inv(cov + JITTER * np.eye(64)), dtype=torch.float32)
    m29 = {"pca": pca, "centroids": centroids, "inv_covs": inv_covs, "class_names": BDD_CLASSES, "d_in": 256,
           "d_out": 64, "jitter_eps": JITTER, "shrunk_classes": shrunk, "n_vectors": counts,
           "n_matched_total": {BDD_CLASSES[c]: seen[c] for c in range(10)},
           "source": "toàn bộ BDD train, Max-IoU (IoU>=0.5, đúng lớp), decoder_hidden_states[-1]", "expert_id": EXPERT_ID}
    torch.save(m29, os.path.join(WORK, f"expert_{EXPERT_ID}_centroids.pt"))
    if EXPERT_ID == 1:
        torch.save(m29, os.path.join(WORK, "centroids.pt"))   # tên M29 dùng cho base model
    print(f"[V] Đã lưu expert_{EXPERT_ID}_centroids.pt" + (" và centroids.pt" if EXPERT_ID == 1 else "") +
          (f" | lớp dùng hiệp phương sai co rút: {shrunk}" if shrunk else ""))
    # ---- Cho H2: tâm 256D + hiệp phương sai gộp Ledoit-Wolf + chuẩn hóa robust ----
    means = np.zeros((10, 256), np.float32); within = []
    for c in range(10):
        if len(emb[c]):
            means[c] = emb[c].mean(0); within.append(emb[c] - means[c])
    W_ = np.vstack(within)
    W_ = W_[np.random.default_rng(1).choice(len(W_), min(len(W_), 200000), replace=False)]
    lw = LedoitWolf().fit(W_, ); P = lw.precision_.astype(np.float32)
    D = np.concatenate([np.sqrt(np.einsum("ij,jk,ik->i", emb[c] - means[c], P, emb[c] - means[c]))
                        for c in range(10) if len(emb[c])])
    med = float(np.median(D)); mad = float(np.median(np.abs(D - med)) * 1.4826)
    torch.save({"class_means": torch.tensor(means), "precision": torch.tensor(P), "shrinkage": torch.tensor(lw.shrinkage_),
                "tp_median_D": torch.tensor(med), "tp_mad_D": torch.tensor(mad),
                "n_vectors": torch.tensor([len(emb[c]) for c in range(10)])},
               os.path.join(WORK, f"expert_{EXPERT_ID}_latent_stats.pt"))
    for i in range(nparts):
        os.remove(part_path(i))                                 # M29: không giữ vector thô
    print(f"[V] Đã lưu expert_{EXPERT_ID}_latent_stats.pt | co rút Ledoit-Wolf {lw.shrinkage_:.3f} | "
          f"Mahalanobis TP: trung vị {med:.2f}, MAD {mad:.2f}")
