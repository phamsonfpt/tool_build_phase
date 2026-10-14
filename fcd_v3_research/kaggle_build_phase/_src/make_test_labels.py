"""Gộp nhãn BDD100K test dạng mỗi ảnh một file JSON thành một file duy nhất
`bdd100k_labels_images_test.json` (cùng định dạng danh sách với file nhãn train), để tải lên Kaggle
thành một dataset riêng.

Dùng:  python make_test_labels.py <thư mục 100k/test chứa *.json> <thư mục đầu ra>
"""
import json, glob, os, sys, collections

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "*.json")))
items, cats, no_box = [], collections.Counter(), 0
for f in files:
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    objs = (d.get("frames") or [{}])[0].get("objects", []) if d.get("frames") else (d.get("labels") or [])
    labels = [o for o in objs if "box2d" in o]
    no_box += not labels
    cats.update(o["category"] for o in labels)
    items.append({"name": d["name"] if d["name"].endswith(".jpg") else d["name"] + ".jpg",
                  "attributes": d.get("attributes", {}), "labels": labels})
out = os.path.join(dst, "bdd100k_labels_images_test.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(items, fh)
meta = {"title": "BDD100K test labels (det, 20k)", "id": "YOUR_KAGGLE_USERNAME/bdd100k-test-labels",
        "licenses": [{"name": "other"}]}
with open(os.path.join(dst, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)
print(f"{len(items)} ảnh, {sum(cats.values())} box, {no_box} ảnh không có box2d -> {out} "
      f"({os.path.getsize(out) / 2**20:.1f} MB)")
print(dict(cats))
