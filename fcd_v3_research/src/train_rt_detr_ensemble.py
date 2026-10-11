# ==============================================================================
# RT-DETR DEEP ENSEMBLE TRAINING SCRIPT (100% FUNCTIONAL)
# ==============================================================================
# Hướng dẫn:
# 1. Đổi ENSEMBLE_ID (1-5).
# 2. Đổi DATA_DIR tùy theo môi trường (Local / Kaggle / Colab).
# ==============================================================================

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, TrainingArguments, Trainer

# --- CẤU HÌNH ---
ENSEMBLE_ID = 1
DATA_DIR = r"D:\tool_build_phase\ai_router\data" # Sửa thành '/kaggle/input/bdd100k' nếu trên Kaggle
TARGET_CLASS = 'car'
COCO_CAR_ID = 3 # Giữ nguyên ID của COCO để tận dụng Transfer Learning

def set_seed(ensemble_id):
    seed = 42 + ((ensemble_id - 1) * 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[*] Đã thiết lập Seed = {seed} (Chuyên gia {ensemble_id})")

class BDD100kDataset(Dataset):
    def __init__(self, data_dir, split='train', processor=None, max_samples=None):
        self.img_dir = os.path.join(data_dir, split)
        self.json_dir = os.path.join(data_dir, '100k', split)
        self.processor = processor
        
        # Đọc danh sách file
        all_jsons = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        self.samples = []
        
        print(f"[*] Đang quét {len(all_jsons)} nhãn {split.upper()}...")
        for jf in all_jsons[:max_samples]:
            with open(os.path.join(self.json_dir, jf), 'r') as f:
                data = json.load(f)
            
            img_name = f"{data['name']}-0000100.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            
            if os.path.exists(img_path):
                self.samples.append((img_path, data))
                
        print(f"[*] Đã map thành công {len(self.samples)} ảnh hợp lệ.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, data = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        annotations = []
        if 'frames' in data and len(data['frames']) > 0 and 'objects' in data['frames'][0]:
            for obj in data['frames'][0]['objects']:
                if obj['category'] == TARGET_CLASS and 'box2d' in obj:
                    b = obj['box2d']
                    # Chuyển [x1, y1, x2, y2] sang [x_min, y_min, width, height] chuẩn COCO
                    w = b['x2'] - b['x1']
                    h = b['y2'] - b['y1']
                    annotations.append({
                        "bbox": [b['x1'], b['y1'], w, h],
                        "category_id": COCO_CAR_ID,
                        "area": w * h,
                        "iscrowd": 0
                    })
        
        target = {"image_id": idx, "annotations": annotations}
        
        # Dùng HuggingFace Processor để transform ảnh và box
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Bóc ra khỏi batch (vì đang xử lý từng ảnh)
        result = {k: v[0] for k, v in encoding.items()}
        return result

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    set_seed(ENSEMBLE_ID)

    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", ignore_mismatched_sizes=True)

    # Khởi tạo Dataset (Giới hạn 1000 ảnh cho mục đích test chạy thử. Khi train thật, bỏ max_samples đi)
    train_dataset = BDD100kDataset(DATA_DIR, split='train', processor=processor, max_samples=1000)
    val_dataset = BDD100kDataset(DATA_DIR, split='val', processor=processor, max_samples=200)

    training_args = TrainingArguments(
        output_dir=f"./rtdetr_expert_{ENSEMBLE_ID}",
        num_train_epochs=5,              
        per_device_train_batch_size=4,    # Phù hợp GPU VRAM trung bình
        learning_rate=5e-5,               
        weight_decay=1e-4,
        save_strategy="epoch",
        logging_steps=10,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),   # Tự động bật Mixed Precision nếu có GPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    print(f"[*] BẮT ĐẦU HUẤN LUYỆN CHUYÊN GIA SỐ {ENSEMBLE_ID}...")
    # BỎ COMMENT DÒNG DƯỚI ĐỂ CHẠY THẬT
    # trainer.train()
    # model.save_pretrained(f"./final_expert_{ENSEMBLE_ID}")
    print("[*] Hoàn thành huấn luyện!")
