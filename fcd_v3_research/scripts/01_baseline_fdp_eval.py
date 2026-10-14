import os
import json
import torch
import time
import sys
from PIL import Image
from torchvision.ops import box_iou
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Ép hệ thống dùng utf-8 nếu có thể, nhưng an toàn nhất là in tiếng Anh
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

class BaselineFDP_Evaluator:
    def __init__(self, data_dir, conf_threshold=0.95, iou_threshold=0.75):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.val_img_dir = os.path.join(data_dir, 'val')
        self.val_json_dir = os.path.join(data_dir, '100k', 'val')
        
        print("Loading RT-DETR model from HuggingFace (may take a few minutes for the first time)...")
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model is ready on {self.device}")

        # BDD100k class -> COCO class mapping (Focusing on 'car')
        self.target_class = 'car'
        
    def get_coco_label_for_car(self):
        # Find the ID for 'car' in model config
        for id, label in self.model.config.id2label.items():
            if label == 'car':
                return id
        return 3 # COCO default

    def evaluate_dataset(self, num_samples=50):
        print(f"\nSTARTING GO/NO-GO TEST (Limit: {num_samples} images)")
        json_files = [f for f in os.listdir(self.val_json_dir) if f.endswith('.json')]
        
        total_images = 0
        sum_fdp = 0.0
        total_V = 0
        total_R = 0
        car_id = self.get_coco_label_for_car()
        
        start_time = time.time()
        
        for json_filename in json_files[:num_samples]:
            # 1. Read Ground Truth
            json_path = os.path.join(self.val_json_dir, json_filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            base_name = data['name']
            img_filename = f"{base_name}-0000100.jpg"
            img_path = os.path.join(self.val_img_dir, img_filename)
            
            if not os.path.exists(img_path):
                continue
                
            # Filter GT boxes for cars
            gt_boxes = []
            if 'frames' in data and len(data['frames']) > 0 and 'objects' in data['frames'][0]:
                for obj in data['frames'][0]['objects']:
                    if obj['category'] == self.target_class and 'box2d' in obj:
                        b = obj['box2d']
                        gt_boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
            gt_boxes_tensor = torch.tensor(gt_boxes) if len(gt_boxes) > 0 else torch.zeros((0, 4))
            
            # 2. Run Inference (RT-DETR)
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.conf_threshold)[0]
            
            # 3. Filter predictions for 'car' only
            pred_boxes = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == car_id:
                    pred_boxes.append(box.cpu().tolist())
            pred_boxes_tensor = torch.tensor(pred_boxes) if len(pred_boxes) > 0 else torch.zeros((0, 4))
            
            total_accepted = len(pred_boxes_tensor)
            total_R += total_accepted
            false_positives = 0
            
            # 4. Calculate False Positives
            if total_accepted > 0:
                if len(gt_boxes_tensor) == 0:
                    false_positives = total_accepted # Hallucination
                else:
                    ious = box_iou(pred_boxes_tensor, gt_boxes_tensor)
                    for i in range(total_accepted):
                        best_iou = ious[i].max().item()
                        if best_iou < self.iou_threshold:
                            false_positives += 1
            
            total_V += false_positives
            image_fdp = false_positives / max(total_accepted, 1)
            sum_fdp += image_fdp
            total_images += 1
            
            print(f"Image {total_images}: Predicted {total_accepted} cars -> FPs: {false_positives} (FDP: {image_fdp*100:.1f}%)")

        end_time = time.time()
        
        expected_fdp = sum_fdp / max(total_images, 1)
        
        print("\n=== QUANTITATIVE RESULTS (BASELINE GO/NO-GO TEST) ===")
        print(f"Total evaluated images: {total_images}")
        print(f"Total confident predictions > 0.95 (Coverage R): {total_R}")
        print(f"Total False Positives (Hallucinations/Misaligned V): {total_V}")
        print(f"-> EXPECTED BASELINE FDP: {expected_fdp * 100:.2f}%")
        print(f"Average processing speed: {(end_time - start_time)/max(total_images, 1):.2f} seconds/image")
        
        if expected_fdp > 0.02:
            print("CONCLUSION: Baseline FDP is > 2%. Research on LTT + Uncertainty Signals is MANDATORY and viable!")
        else:
            print("CONCLUSION: Model is too perfect (FDP < 2%). Risk of Null Result Trap.")

if __name__ == "__main__":
    DATA_DIR = r"D:\tool_build_phase\ai_router\data"
    evaluator = BaselineFDP_Evaluator(data_dir=DATA_DIR, conf_threshold=0.95, iou_threshold=0.75)
    evaluator.evaluate_dataset(num_samples=50)
