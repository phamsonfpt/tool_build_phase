# ==============================================================================
# SCRIPT HUẤN LUYỆN TỰ ĐỘNG ALL-IN-ONE (DÀNH CHO KAGGLE & GOOGLE COLAB)
# ==============================================================================
# HƯỚNG DẪN DÀNH CHO THÀNH VIÊN NHÓM:
# 1. Tạo 1 Notebook mới trên Kaggle (hoặc Colab).
# 2. Bấm "Add Input" -> Tìm bộ dữ liệu "BDD100K" và thêm vào Notebook.
# 3. Chọn GPU (T4 x2 trên Kaggle hoặc T4 trên Colab).
# 4. Copy TOÀN BỘ file này dán vào 1 ô Code (Cell).
# 5. SỬA DUY NHẤT DÒNG "ENSEMBLE_ID" BÊN DƯỚI theo số thứ tự của bạn (1, 2, 3, 4, hoặc 5).
# 6. Bấm RUN. Khi chạy xong, tải file 'expert_X.pth' ở cột Output bên phải về nộp lại.
# ==============================================================================

# >>> [CẤU HÌNH THÀNH VIÊN] ĐỔI SỐ NÀY TỪ 1 ĐẾN 5 THEO PHÂN CÔNG <<<
ENSEMBLE_ID = 1
# ==================================================================

import os
import sys
import json
import glob
import random
import subprocess
import numpy as np
from PIL import Image

# --- 1. TỰ ĐỘNG CÀI ĐẶT THƯ VIỆN CẦN THIẾT NẾU CHƯA CÓ ---
print("[*] Kiem tra va cai dat thu vien...")
try:
    import transformers
    import timm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.38.0", "timm", "accelerate"])
    import transformers

import torch
from torch.utils.data import Dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, TrainingArguments, Trainer

# --- 2. THIẾT LẬP SEED KHOA HỌC (CHUẨN BÀI BÁO) ---
def set_seed(ensemble_id):
    seed = 42 + ((ensemble_id - 1) * 100) # Model 1 = 42, Model 2 = 142, Model 3 = 242...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[*] DA KHOI TAO: Chuyen gia so {ensemble_id} voi SEED = {seed}")
    return seed

CURRENT_SEED = set_seed(ENSEMBLE_ID)

# --- 3. TỰ ĐỘNG TÌM ĐƯỜNG DẪN BDD100K TRÊN KAGGLE / COLAB / LOCAL ---
def auto_detect_bdd_path():
    possible_roots = [
        "/kaggle/input",
        "/content",
        r"D:\tool_build_phase\ai_router\data",
        "."
    ]
    
    print("[*] Dang tu dong tim kiem thu muc BDD100K...")
    for root in possible_roots:
        if os.path.exists(root):
            # Tim thu muc chua cac file anh 100k
            for dirpath, dirnames, filenames in os.walk(root):
                if "train" in dirnames and ("val" in dirnames or "100k" in dirnames):
                    # Kiem tra them su ton tai cua file json hoac jpg
                    return dirpath
    
    # Fallback mac dinh cho Kaggle
    return "/kaggle/input/bdd100k"

DATA_DIR = auto_detect_bdd_path()
print(f"[*] Thu muc du lieu duoc xac dinh tai: {DATA_DIR}")

# --- 4. DATASET LOADER CHUẨN COCO CHO BDD100K ---
TARGET_CLASS = 'car'
COCO_CAR_ID = 3

class BDD100kDataset(Dataset):
    def __init__(self, data_dir, split='train', processor=None, max_samples=None):
        self.processor = processor
        self.split = split
        
        # Ho tro ca cau truc phang lan cau truc nested tren Kaggle
        possible_img_dirs = [
            os.path.join(data_dir, split),
            os.path.join(data_dir, "images", "100k", split),
            os.path.join(data_dir, "bdd100k", "images", "100k", split)
        ]
        self.img_dir = next((d for d in possible_img_dirs if os.path.exists(d)), possible_img_dirs[0])
        
        possible_json_dirs = [
            os.path.join(data_dir, "100k", split),
            os.path.join(data_dir, "labels", "100k", split),
            os.path.join(data_dir, "bdd100k", "labels", "100k", split)
        ]
        self.json_dir = next((d for d in possible_json_dirs if os.path.exists(d)), possible_json_dirs[0])
        
        print(f"[*] Split [{split.upper()}]: Img dir: {self.img_dir} | Label dir: {self.json_dir}")
        
        self.samples = []
        if os.path.exists(self.json_dir):
            json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
            if max_samples:
                json_files = json_files[:max_samples]
                
            for jf in json_files:
                json_path = os.path.join(self.json_dir, jf)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Ten anh thuong co duoi -0000100.jpg hoac truc tiep .jpg
                base_name = data.get('name', jf.replace('.json', ''))
                candidate_names = [f"{base_name}-0000100.jpg", f"{base_name}.jpg"]
                
                found_img = None
                for cname in candidate_names:
                    p = os.path.join(self.img_dir, cname)
                    if os.path.exists(p):
                        found_img = p
                        break
                        
                if found_img:
                    self.samples.append((found_img, data))
                    
        print(f"[*] Loaded {len(self.samples)} valid samples for split [{split.upper()}].")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, data = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        annotations = []
        if 'frames' in data and len(data['frames']) > 0 and 'objects' in data['frames'][0]:
            for obj in data['frames'][0]['objects']:
                if obj.get('category') == TARGET_CLASS and 'box2d' in obj:
                    b = obj['box2d']
                    w = max(0.0, float(b['x2'] - b['x1']))
                    h = max(0.0, float(b['y2'] - b['y1']))
                    if w > 1.0 and h > 1.0:
                        annotations.append({
                            "bbox": [float(b['x1']), float(b['y1']), w, h],
                            "category_id": COCO_CAR_ID,
                            "area": w * h,
                            "iscrowd": 0
                        })
        
        target = {"image_id": idx, "annotations": annotations}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {k: v[0] for k, v in encoding.items()}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}

# --- 5. TIẾN HÀNH HUẤN LUYỆN ---
if __name__ == "__main__":
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Dang khoi tao Processor va Base Model RT-DETR tu Hugging Face...")
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", ignore_mismatched_sizes=True)
    
    # Nap Dataset Train va Val
    train_dataset = BDD100kDataset(DATA_DIR, split='train', processor=processor)
    val_dataset = BDD100kDataset(DATA_DIR, split='val', processor=processor)
    
    # Cau hinh tham so Train toi uu cho GPU T4 (16GB VRAM) tren Kaggle/Colab
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"checkpoints_expert_{ENSEMBLE_ID}"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,    # Tuong duong batch size = 8
        learning_rate=5e-5,
        weight_decay=1e-4,
        save_strategy="epoch",
        logging_steps=50,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )
    
    print("=" * 60)
    print(f"[*] BAT DAU TRAIN CHUYEN GIA {ENSEMBLE_ID} (SEED = {CURRENT_SEED})...")
    print("=" * 60)
    
    trainer.train()
    
    # Luu ket qua cuoi cung vao file .pth san sang download
    save_path = os.path.join(output_dir, f"expert_{ENSEMBLE_ID}.pth")
    torch.save(model.state_dict(), save_path)
    
    print("=" * 60)
    print(f"[*] HOAN THANH XUAT SAC! FILE TRONG SO DA DUOC LUU TAI: {save_path}")
    print(f"[*] HAY TAI FILE 'expert_{ENSEMBLE_ID}.pth' VE DE NOP LAI CHO NHOM TRUONG!")
    print("=" * 60)
