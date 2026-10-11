import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class BDD100KMiniDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        """
        Đọc định dạng JSON nguyên khối của BDD100K
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Load file JSON duy nhất chứa toàn bộ nhãn
        with open(labels_file, 'r') as f:
            self.data = json.load(f)
            
        # Mapping Class (giả lập)
        self.class_map = {"person": 0, "traffic light": 1, "car": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['name']
        img_path = os.path.join(self.images_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        
        for label_obj in item.get('labels', []):
            cat = label_obj['category']
            if cat in self.class_map:
                box2d = label_obj['box2d']
                # BDD100K format: x1, y1, x2, y2
                boxes.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
                labels.append(self.class_map[cat])
                
        # Nếu ảnh không có nhãn (Background), khởi tạo tensor rỗng
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        if self.transform:
            # Transform HuggingFace image processor (chưa áp dụng ở đây)
            pass
            
        return image, target
