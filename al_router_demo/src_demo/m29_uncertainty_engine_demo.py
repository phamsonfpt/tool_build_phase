import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from scipy.optimize import linear_sum_assignment
except ImportError:
    os.system("pip install scipy transformers torch torchvision")
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from scipy.optimize import linear_sum_assignment

def calculate_shannon_entropy(probs):
    """Tính Shannon Entropy cho phân phối xác suất"""
    return -np.sum(probs * np.log(probs + 1e-9), axis=-1)

def main():
    print("=== START PHASE 2: M29 UNCERTAINTY ENGINE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model A (Anchor)
    model_A_path = BASE_DIR / "experiments" / "G1_ensemble" / "model_A"
    print("1. Loading Expert A (Anchor Model)...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model_A = AutoModelForObjectDetection.from_pretrained(model_A_path).to(device)
    model_A.eval()

    # Get test images
    test_dir = IMAGES_DIR / "test"
    test_images = os.listdir(test_dir)
    print(f"2. Scanning {len(test_images)} unlabeled Test images...")

    report = []

    for img_name in test_images:
        img_path = test_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_A(**inputs)
            
        logits = outputs.logits.softmax(-1)[0, :3].cpu().numpy()
        boxes = outputs.pred_boxes[0, :3].cpu().numpy()
        
        image_uncertainties = []
        image_confidences = []
        
        # 3. Hungarian Matching & Math (Noise Injection for B, C, D, E)
        for i in range(len(boxes)):
            box_A = boxes[i]
            logit_A = logits[i]
            
            ensemble_boxes = [box_A]
            ensemble_probs = [logit_A]
            
            for _ in range(4):
                noise_box = np.random.normal(0, 0.05, 4)
                noise_prob = np.random.normal(0, 0.1, logit_A.shape)
                
                ensemble_boxes.append(box_A + noise_box)
                
                new_prob = np.clip(logit_A + noise_prob, 0, 1)
                new_prob = new_prob / new_prob.sum()
                ensemble_probs.append(new_prob)
                
            ensemble_boxes = np.array(ensemble_boxes)
            ensemble_probs = np.array(ensemble_probs)
            
            # A. Calculate Spatial Variance
            var_xc = np.var(ensemble_boxes[:, 0])
            var_yc = np.var(ensemble_boxes[:, 1])
            spatial_variance = var_xc + var_yc
            
            # B. Calculate Shannon Entropy
            mean_probs = np.mean(ensemble_probs, axis=0)
            entropy = calculate_shannon_entropy(mean_probs)
            
            # M29 Metric
            obj_uncertainty = float(spatial_variance * 10 + entropy)
            obj_confidence = float(np.max(mean_probs))
            
            image_uncertainties.append(obj_uncertainty)
            image_confidences.append(obj_confidence)
            
        # 4. Image-Level Aggregation (Max-Pooling)
        img_max_uncertainty = max(image_uncertainties) if image_uncertainties else 0
        img_min_confidence = min(image_confidences) if image_confidences else 0
        
        report.append({
            "image": img_name,
            "uncertainty_score": img_max_uncertainty,
            "min_confidence": img_min_confidence
        })

    # 5. Elastic Grey Zone (Funnel)
    print("3. Classifying Elastic Grey Zone (Funnel)...")
    report.sort(key=lambda x: x["uncertainty_score"], reverse=True)
    
    red_zone = []
    green_zone = []
    grey_zone = []
    
    for item in report:
        if item["uncertainty_score"] > 0.8:
            item["zone"] = "RED (CVAT Review)"
            red_zone.append(item)
        elif item["min_confidence"] > 0.95:
            item["zone"] = "GREEN (Pseudo-label)"
            green_zone.append(item)
        else:
            item["zone"] = "GREY (Ignored)"
            grey_zone.append(item)
            
    out_dir = BASE_DIR / "experiments" / "G2_m29_scores"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(out_dir / "m29_funnel_report.json", "w") as f:
        json.dump({"red_zone": red_zone, "green_zone": green_zone, "grey_zone": grey_zone}, f, indent=4)
        
    print(f"-> Analysis of {len(test_images)} Test images complete!")
    print(f"   + RED ZONE (To CVAT): {len(red_zone)} images")
    print(f"   + GREEN ZONE (Pseudo-labels): {len(green_zone)} images")
    print(f"   + GREY ZONE (Ignored): {len(grey_zone)} images")
    print(f"-> Report saved at: {out_dir / 'm29_funnel_report.json'}")
    print("=== END PHASE 2 ===")

if __name__ == "__main__":
    main()
