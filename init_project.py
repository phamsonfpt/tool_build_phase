import os
import subprocess

project_dir = r"D:\tool_build_phase\ai_router"

# 1. Tạo các thư mục dự án
directories = [
    project_dir,
    os.path.join(project_dir, "src"),
    os.path.join(project_dir, "data"),
    os.path.join(project_dir, "experiments"),
    os.path.join(project_dir, "docs"),
    os.path.join(project_dir, "scripts")
]

for d in directories:
    os.makedirs(d, exist_ok=True)

# 2. Khởi tạo Git
subprocess.run(["git", "init"], cwd=project_dir)

# 3. Tạo môi trường ảo (venv)
print("Setting up Virtual Environment (venv)...")
subprocess.run(["python", "-m", "venv", "venv"], cwd=project_dir)

# 4. Tạo .gitignore
gitignore_content = """# Environments
venv/
env/
.env

# Python
__pycache__/
*.pyc
*.pyo

# Project Data & Experiments (Không đẩy tệp lớn lên git)
data/
experiments/
.cache/

# IDEs
.vscode/
.idea/
"""
with open(os.path.join(project_dir, ".gitignore"), "w", encoding="utf-8") as f:
    f.write(gitignore_content)

# 5. Tạo README.md
readme_content = """# AI Router (Active Learning Pipeline - M29)

## Tổng quan (Overview)
Dự án triển khai thực tế (Production) của lõi kiến trúc Active Learning M29.
Sử dụng mô hình RT-DETR kết hợp kỹ thuật Deep Ensembles để đo lường Độ bất định (Epistemic Uncertainty).

## Cấu trúc Giai đoạn (Pipeline Phases)
* **Giai đoạn 0:** Base Model Training
* **Giai đoạn 1:** Deep Ensembles Generation (Tạo 5 Chuyên gia)
* **Giai đoạn 2:** M29 Uncertainty Engine (Lọc ảnh khó)
* **Giai đoạn 3:** CVAT Auto-Router (Điều phối Human-in-the-Loop)
* **Giai đoạn 4:** M35 KPI Benchmark (Đánh giá hiệu suất)
"""
with open(os.path.join(project_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

# 6. Tạo requirements.txt
requirements_content = """torch
torchvision
transformers
scipy
numpy
Pillow
tqdm
cvat-sdk
"""
with open(os.path.join(project_dir, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(requirements_content)

print("Successfully initialized ai_router production directory!")
