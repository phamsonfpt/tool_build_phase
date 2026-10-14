%%writefile common.py
# ==============================================================================
# CẤU HÌNH DÙNG CHUNG (chuẩn bị dữ liệu, huấn luyện, kiểm tra)
#
# Cách dùng dữ liệu BDD100K ở build phase (phương án B):
#   BDD train (69.863 ảnh có nhãn) -> "train"       : huấn luyện detector, dùng TOÀN BỘ
#   BDD test  (20.000 ảnh có nhãn) -> "build_cal"   : nửa thứ nhất, hiệu chuẩn LTT của pipeline
#                                  -> "build_check" : nửa thứ hai, kiểm tra từng lớp và edge case
#   BDD val   (10.000 ảnh)         -> KHÔNG ĐỌC     : giữ nguyên làm tập đánh giá cho bài báo (H1/H2)
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
SEEDS = {1: 42, 2: 142, 3: 242}

WORK = os.environ.get("WORK_DIR", "/kaggle/working")
SPLITS = ("train", "build_cal", "build_check")
COMPACT = {s: os.path.join(WORK, f"bdd_det_{s}_compact.json") for s in SPLITS}
DATA_STATS = os.path.join(WORK, "data_stats.json")

def test_half(name):
    """Chia BDD test thành hai nửa cố định theo hash tên ảnh (giống nhau với mọi người, mọi lần chạy)."""
    return "build_cal" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 2 == 0 else "build_check"

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
    """Tìm nhãn và ảnh của BDD train và BDD test trong /kaggle/input (mọi dataset đã Add Input).
    Có thể chỉ định thẳng bằng biến môi trường TRAIN_JSON, TRAIN_IMG, TEST_JSON, TEST_IMG."""
    found = {k: os.environ.get(k) for k in ("TRAIN_JSON", "TRAIN_IMG", "TEST_JSON", "TEST_IMG")}
    if all(found.values()):
        return found
    for root, dirs, files in os.walk("/kaggle/input"):
        r = root.replace("\\", "/")
        for f in files:
            if f.endswith("labels_images_train.json"):
                found["TRAIN_JSON"] = found["TRAIN_JSON"] or os.path.join(root, f)
            if f.endswith("labels_images_test.json"):
                found["TEST_JSON"] = found["TEST_JSON"] or os.path.join(root, f)
        for split in ("train", "test"):
            key = f"{split.upper()}_IMG"
            if not found[key] and r.endswith(f"100k/{split}") and any(x.endswith(".jpg") for x in os.listdir(root)[:50]):
                found[key] = root
    return found

def raw_objects(item):
    """Hỗ trợ cả hai định dạng nhãn BDD: item['labels'] hoặc item['frames'][0]['objects']."""
    if item.get('labels') is not None:
        return item['labels']
    fr = item.get('frames') or []
    return fr[0].get('objects', []) if fr else []
