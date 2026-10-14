%%writefile prepare_data.py
# ==============================================================================
# CHUẨN BỊ DỮ LIỆU (tự bỏ qua nếu đã có kết quả)
#  - BDD train -> "train"; BDD val -> "val" (chỉ để xem). BDD test không đọc.
#  - Giữ TOÀN BỘ box của 10 lớp; chỉ bỏ box suy biến (rộng hoặc cao <= 0.5 px).
#  - Đếm box gốc và box giữ lại theo lớp; dừng lại nếu có box bị mất vì lý do khác.
#  - Tính hệ số lấy mẫu lặp (Repeat Factor Sampling; Gupta et al., 2019) cho lớp hiếm của tập train.
# ==============================================================================
import os, json, math, collections
from common import *

RFS_T = 0.02  # lớp xuất hiện trong dưới 2% số ảnh train sẽ được lặp lại

if all(os.path.exists(p) for p in list(COMPACT.values()) + [DATA_STATS]):
    print("[V] Dữ liệu đã được chuẩn bị từ trước, bỏ qua bước này.")
    raise SystemExit(0)

P = find_bdd()
for k, v in P.items():
    print(f"[*] {k:10s}: {v}")
missing = [k for k, v in P.items() if not v]
assert not missing, (
    f"Thiếu {missing}. Cần Add Input bộ BDD100K (ví dụ solesensei_bdd100k) có "
    "'bdd100k_labels_images_train.json', 'bdd100k_labels_images_val.json', 'images/100k/train', 'images/100k/val'.")

def image_map(d):
    m = {}
    for r, _, files in os.walk(d):
        for f in files:
            if f.endswith(".jpg"):
                m[f] = os.path.join(r, f)
    return m

excl_path = find_exclude_list()
exclude = set()
if excl_path:
    with open(excl_path, encoding="utf-8") as f:
        exclude = {ln.strip() if ln.strip().endswith(".jpg") else ln.strip() + ".jpg" for ln in f if ln.strip()}
    print(f"[V] Danh sách loại trừ cho tập train: {excl_path} ({len(exclude)} ảnh)")

raw_cnt = collections.Counter(); degen_cnt = collections.Counter(); other_cats = collections.Counter()
kept = {s: collections.Counter() for s in SPLITS}
out = {s: [] for s in SPLITS}
info = collections.Counter(); max_boxes = 0
for source, json_key, img_key in (("bdd_train", "TRAIN_JSON", "TRAIN_IMG"), ("bdd_val", "VAL_JSON", "VAL_IMG")):
    img_map = image_map(P[img_key])
    with open(P[json_key], encoding="utf-8") as f:
        raw = json.load(f)
    labeled = set()
    for item in raw:
        name = item['name'] if item['name'].endswith(".jpg") else item['name'] + ".jpg"
        labeled.add(name)
        path = img_map.get(name)
        if path is None:
            info[f"{source}: nhãn không có ảnh"] += 1; continue
        if source == "bdd_train" and name in exclude:
            info["bdd_train: loại trừ"] += 1; continue
        split = "train" if source == "bdd_train" else "val"
        boxes = []
        for o in raw_objects(item):
            if 'box2d' not in o:
                continue
            cat = CATEGORY_NORMALIZER.get(o.get('category'))
            if cat is None:
                other_cats[o.get('category')] += 1; continue
            raw_cnt[cat] += 1
            b = o['box2d']; x1, y1, x2, y2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
            if x2 - x1 > 0.5 and y2 - y1 > 0.5:
                boxes.append([x1, y1, x2, y2, CLASS2ID[cat]]); kept[split][cat] += 1
            else:
                degen_cnt[cat] += 1
        info[f"{source}: ảnh không có box"] += not boxes
        max_boxes = max(max_boxes, len(boxes))
        out[split].append({"file": path, "name": name, "boxes": boxes, "attributes": item.get("attributes", {}),
                           "monitor": split == "val" and in_val_monitor(name)})
    # Ảnh có trong thư mục nhưng không có mục nào trong file nhãn: bỏ qua, vì mọi object trong ảnh sẽ bị coi là nền.
    unlabeled = sorted(set(img_map) - labeled)
    info[f"{source}: ảnh không có nhãn"] = len(unlabeled)
    with open(os.path.join(WORK, f"unlabeled_images_{source}.txt"), "w") as f:
        f.write("\n".join(unlabeled))
    del raw

img_with = collections.Counter()
for rec in out["train"]:
    img_with.update({BDD_CLASSES[b[4]] for b in rec["boxes"]})
n_train = len(out["train"])
r_c = {c: max(1.0, math.sqrt(RFS_T / (img_with[c] / n_train))) if img_with[c] else 1.0 for c in BDD_CLASSES}
for rec in out["train"]:
    rec["repeat"] = max([r_c[BDD_CLASSES[b[4]]] for b in rec["boxes"]], default=1.0)
for s in SPLITS:
    with open(COMPACT[s], "w") as f:
        json.dump(out[s], f)

kept_all = collections.Counter()
for s in SPLITS:
    kept_all.update(kept[s])
stats = dict(sources={k: v for k, v in P.items()}, split_rule="train = toàn bộ BDD train; val = BDD val, chỉ để xem "
             "(loss theo dõi trên tập con md5(tên ảnh) % 10 == 0, báo cáo từng lớp trên toàn bộ val); BDD test không đọc",
             n_val_monitor=sum(r["monitor"] for r in out["val"]),
             n_images={s: len(out[s]) for s in SPLITS}, info=dict(info), exclude_list=excl_path,
             train_filtered=bool(excl_path), max_boxes_per_image=max_boxes, raw_boxes=dict(raw_cnt),
             kept_boxes={s: dict(kept[s]) for s in SPLITS}, degenerate_boxes=dict(degen_cnt),
             train_images_with_class=dict(img_with), rfs_t=RFS_T, rfs_repeat_per_class=r_c,
             train_images_per_epoch=sum(r["repeat"] for r in out["train"]), box2d_of_other_categories=dict(other_cats))
with open(DATA_STATS, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("\n[*] Số ảnh: " + " | ".join(f"{s}: {len(out[s])}" for s in SPLITS))
for k, v in sorted(info.items()):
    print(f"    {k}: {v}")
print(f"\n{'lớp':14s} {'box gốc':>9s} {'train':>9s} {'val':>9s} {'suy biến':>9s} {'lặp RFS':>8s}")
for c in BDD_CLASSES:
    print(f"{c:14s} {raw_cnt[c]:9d} {kept['train'][c]:9d} {kept['val'][c]:9d} {degen_cnt[c]:9d} {r_c[c]:8.2f}")
assert all(raw_cnt[c] == kept_all[c] + degen_cnt[c] for c in BDD_CLASSES), "Có box bị mất ngoài lý do suy biến!"
print(f"[V] Mọi box của 10 lớp đều được giữ (trừ box suy biến). Nhiều box nhất trong 1 ảnh: {max_boxes}. "
      f"Ảnh train/epoch sau RFS ≈ {round(stats['train_images_per_epoch'])}")
if max_boxes > 300:
    print("[!] Có ảnh nhiều hơn 300 object, vượt số query của RT-DETR (300).")
