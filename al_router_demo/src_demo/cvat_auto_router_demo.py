import os
import sys
import json
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

def main():
    print("=== START PHASE 3: CVAT AUTO-ROUTER (HUMAN-IN-THE-LOOP) ===")
    
    # 1. Read M29 Funnel Report
    report_path = BASE_DIR / "experiments" / "G2_m29_scores" / "m29_funnel_report.json"
    if not report_path.exists():
        print(f"Error: M29 report not found at {report_path}")
        return
        
    with open(report_path, "r", encoding="utf-8") as f:
        funnel_data = json.load(f)
        
    red_zone_images = [item["image"] for item in funnel_data.get("red_zone", [])]
    print(f"1. Reading M29 Report: Found {len(red_zone_images)} images in RED ZONE.")
    
    # 2. Prepare CVAT Upload Directory
    cvat_dir = BASE_DIR / "experiments" / "G3_cvat_upload"
    cvat_img_dir = cvat_dir / "images"
    cvat_ann_dir = cvat_dir / "annotations"
    
    os.makedirs(cvat_img_dir, exist_ok=True)
    os.makedirs(cvat_ann_dir, exist_ok=True)
    print("2. Created CVAT upload packages...")

    # 3. Route Images
    test_images_dir = IMAGES_DIR / "test"
    pre_annotations = []
    
    for img_name in red_zone_images:
        src_path = test_images_dir / img_name
        dst_path = cvat_img_dir / img_name
        
        # Copy image to CVAT queue
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            
        # Create a mock pre-annotation for CVAT (Auto-Annotation feature)
        # In a real app, this would be the actual coordinates from Model A
        pre_annotations.append({
            "name": img_name,
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0},
                    "attributes": {"AI_Confidence": 0.82}
                }
            ]
        })
        
    # 4. Save Pre-annotations
    with open(cvat_ann_dir / "cvat_pre_annotations.json", "w", encoding="utf-8") as f:
        json.dump(pre_annotations, f, indent=4)
        
    print(f"3. Successfully routed {len(red_zone_images)} images to CVAT upload folder.")
    print(f"4. Generated Pre-annotations (AI Assistance) to speed up Human Review.")
    print(f"-> Package ready at: {cvat_dir}")
    print("=== END PHASE 3 ===")

if __name__ == "__main__":
    main()
