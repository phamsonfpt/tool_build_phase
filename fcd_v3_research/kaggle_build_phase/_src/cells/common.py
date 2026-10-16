%%writefile common.py
# ==============================================================================
# CẤU HÌNH DÙNG CHUNG (chuẩn bị dữ liệu, huấn luyện, kiểm tra)
#
# Cách dùng dữ liệu BDD100K ở build phase:
#   BDD train (69.863 ảnh có nhãn) -> "train" : huấn luyện detector, dùng TOÀN BỘ
#   BDD val   (10.000 ảnh có nhãn) -> "val"   : CHỈ ĐỂ XEM: đường cong loss trong lúc train và báo cáo
#                                               từng lớp sau khi train. Không chọn checkpoint, không chỉnh
#                                               siêu tham số theo val, để val vẫn dùng được cho bài báo.
#   BDD test (20.000 ảnh)          -> KHÔNG ĐỌC : giữ cho bài báo hoặc cho bước hiệu chuẩn sau này
# ==============================================================================
import os, json, glob, hashlib

BDD_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
CLASS2ID = {n: i for i, n in enumerate(BDD_CLASSES)}
ID2CLASS = {i: n for i, n in enumerate(BDD_CLASSES)}
# Nhãn BDD100K bản 2018 dùng 'person', 'bike', 'motor'; bản 2020 dùng tên mới.
CATEGORY_NORMALIZER = {
    'pedestrian': 'pedestrian', 'person': 'pedestrian', 'rider': 'rider', 'car': 'car',
    'truck': 'truck', 'bus': 'bus', 'train': 'train', 'motorcycle': 'motorcycle', 'motor': 'motorcycle',
    'bicycle': 'bicycle', 'bike': 'bicycle', 'traffic light': 'traffic light', 'traffic sign': 'traffic sign',
}

# Seed riêng của từng chuyên gia. Phải truyền vào TrainingArguments(seed=..., data_seed=...),
# vì Trainer gọi lại set_seed(args.seed) khi khởi tạo và sẽ ghi đè mọi seed đặt trước đó.
SEEDS = {1: 42, 2: 142, 3: 242, 9: 999}   # 9 = bản chạy thử (canary), kết quả không dùng

WORK = os.environ.get("WORK_DIR", "/kaggle/working")
SPLITS = ("train", "val")
COMPACT = {s: os.path.join(WORK, f"bdd_det_{s}_compact.json") for s in SPLITS}
DATA_STATS = os.path.join(WORK, "data_stats.json")

# Một tập con cố định 1.000 ảnh val (chọn theo hash tên ảnh) để đo loss trong lúc train cho nhanh;
# báo cáo từng lớp sau khi train dùng toàn bộ 10.000 ảnh val.
def in_val_monitor(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0

# (Tùy chọn) danh sách ảnh train phải loại, một tên ảnh mỗi dòng (dùng cho bài báo, giao thức A2 của H1).
EXCLUDE_LIST_CANDIDATES = ["/kaggle/input/*/train_exclude.txt", "/kaggle/input/*/*/train_exclude.txt",
                           os.path.join(WORK, "train_exclude.txt")]

def find_exclude_list():
    for pat in EXCLUDE_LIST_CANDIDATES:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def find_bdd():
    """Tìm nhãn và ảnh của BDD train và BDD val trong /kaggle/input.
    Có thể chỉ định thẳng bằng biến môi trường TRAIN_JSON, TRAIN_IMG, VAL_JSON, VAL_IMG."""
    found = {k: os.environ.get(k) for k in ("TRAIN_JSON", "TRAIN_IMG", "VAL_JSON", "VAL_IMG")}
    if all(found.values()):
        return found
    for root, dirs, files in os.walk("/kaggle/input"):
        r = root.replace("\\", "/")
        for f in files:
            if f.endswith("labels_images_train.json"):
                found["TRAIN_JSON"] = found["TRAIN_JSON"] or os.path.join(root, f)
            if f.endswith("labels_images_val.json"):
                found["VAL_JSON"] = found["VAL_JSON"] or os.path.join(root, f)
        for split in ("train", "val"):
            key = f"{split.upper()}_IMG"
            # Bản Kaggle để một phần ảnh trong thư mục con (ví dụ test/trainA), nên chấp nhận thư mục
            # images/100k/<split> dù ảnh không nằm trực tiếp ở đó; image_map() quét đệ quy.
            if not found[key] and r.endswith(f"100k/{split}") and (
                    "images" in r or any(x.endswith(".jpg") for x in os.listdir(root)[:50])):
                found[key] = root
    return found

def raw_objects(item):
    """Hỗ trợ cả hai định dạng nhãn BDD: item['labels'] hoặc item['frames'][0]['objects']."""
    if item.get('labels') is not None:
        return item['labels']
    fr = item.get('frames') or []
    return fr[0].get('objects', []) if fr else []
