import os
from pathlib import Path

# Thư mục gốc của dự án
BASE_DIR = Path(r"D:\tool_build_phase\al_router")

# Đường dẫn Data
DATA_DIR = BASE_DIR / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# Cấu hình Huấn luyện Giai đoạn 0 (Baseline)
G0_EXPERIMENT_DIR = BASE_DIR / "experiments" / "G0_baseline"
G0_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# Cấu hình Model
MODEL_NAME = "PekingU/rtdetr_r50vd" # Sử dụng HuggingFace RT-DETR để lấy được Raw Logits
NUM_CLASSES = 3 # car, person, traffic light (trong bản mini)
