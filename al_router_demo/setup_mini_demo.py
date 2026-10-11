import os
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

def setup_mini_dataset():
    print("Downloading a mini dataset (COCO128 - 128 real images of cars, people, etc.) to simulate BDD100K...")
    
    # Download coco128 zip
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = "coco128.zip"
    
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    print("Converting to BDD100K format...")
    # BDD100K target dirs
    base_dir = r"D:\tool_build_phase\al_router\data\bdd100k"
    os.makedirs(f"{base_dir}/images/train", exist_ok=True)
    os.makedirs(f"{base_dir}/images/val", exist_ok=True)
    os.makedirs(f"{base_dir}/images/test", exist_ok=True)
    os.makedirs(f"{base_dir}/labels", exist_ok=True)
    
    # Read coco128
    images_dir = "coco128/images/train2017"
    labels_dir = "coco128/labels/train2017"
    
    images = os.listdir(images_dir)
    
    # Split: 80 train, 20 val, 28 test
    train_imgs = images[:80]
    val_imgs = images[80:100]
    test_imgs = images[100:128]
    
    def convert_to_bdd_json(img_list, split_name):
        bdd_labels = []
        for img_name in img_list:
            # Copy image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(base_dir, "images", split_name, img_name)
            shutil.copy(src_img, dst_img)
            
            # Read label
            label_name = img_name.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_name)
            
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, x_c, y_c, w, h = map(float, parts)
                            # Convert YOLO (norm) to BDD100K (abs)
                            # COCO128 images are mostly 640x480 but we approximate 640x640 for dummy
                            img_w, img_h = 640, 640 
                            x1 = (x_c - w/2) * img_w
                            y1 = (y_c - h/2) * img_h
                            x2 = (x_c + w/2) * img_w
                            y2 = (y_c + h/2) * img_h
                            
                            # Fake classes for BDD
                            class_name = "car" if cls_id == 2 else ("person" if cls_id == 0 else "traffic light")
                            
                            labels.append({
                                "category": class_name,
                                "box2d": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                }
                            })
            
            bdd_labels.append({
                "name": img_name,
                "labels": labels
            })
            
        # Write JSON only for train and val
        if split_name in ['train', 'val']:
            with open(os.path.join(base_dir, "labels", f"{split_name}.json"), 'w') as f:
                json.dump(bdd_labels, f, indent=2)
                
    convert_to_bdd_json(train_imgs, "train")
    convert_to_bdd_json(val_imgs, "val")
    convert_to_bdd_json(test_imgs, "test")
    
    print("Cleanup...")
    shutil.rmtree("coco128")
    os.remove("coco128.zip")
    print("Done! BDD100K Mini created successfully.")

if __name__ == "__main__":
    setup_mini_dataset()
