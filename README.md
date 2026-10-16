# tool_build_phase

Bản đồ thư mục — đọc file này trước để biết thứ gì nằm ở đâu.

```
tool_build_phase/
├── docs/                      Tài liệu chung của dự án
│   ├── architecture/          Kế hoạch kiến trúc (final_architecture_plan, extra_final_architecture_plan)
│   ├── specs/                 Đặc tả kỹ thuật (M29_MASTER_SPECIFICATION)
│   └── reviews/               Bản review / nhận xét
├── scripts/                   Script tiện ích (init_project.py: tạo khung thư mục ai_router)
├── ai_router/                 Dự án chính: data/, experiments/ (weights), src/, venv/
├── al_router_demo/            Bản demo thu nhỏ của pipeline (dữ liệu + code *_demo)
├── fcd_v3_research/           Nghiên cứu FCD v3
│   ├── hypotheses/            Tài liệu giả thuyết: mỗi giả thuyết một thư mục, mỗi phiên bản một
│   │   │                      thư mục con vX.Y/ (xem hypotheses/README.md)
│   │   ├── H1_statistical_guarantee/   v1.2, v1.3, v1.4
│   │   ├── H2_signal/                  v1.2, v1.4
│   │   ├── H3_al_utility/              chưa có tài liệu riêng
│   │   └── shared/                     Đề cương, chuỗi giả thuyết, tài liệu dùng chung
│   ├── h1_real/               Code chạy H1 trên BDD100K thật
│   ├── h2_simulation/         Code mô phỏng thiết kế H2
│   ├── kaggle_build_phase/    Nguồn sinh notebook Kaggle train expert
│   ├── scripts/               Script đánh giá chạy độc lập (01_baseline_fdp_eval.py)
│   └── src/                   Code huấn luyện ensemble RT-DETR, notebook, uncertainty_plugins/
└── test/ train/ val/          Ảnh frame *-0000100.jpg (không đưa vào git)
```

## Lưu ý

- Không đổi tên hay di chuyển `ai_router/`: nhiều script hardcode đường dẫn
  `D:\tool_build_phase\ai_router\data`.
- `ai_router/data/`, `ai_router/experiments/` và các file weights (`*.pth`, `*.safetensors`, ...)
  nằm trong `.gitignore`.
- Khi ra phiên bản mới của một giả thuyết, tạo thư mục con `vX.Y/` mới trong thư mục của giả thuyết đó,
  không sửa đè bản cũ và không thêm số phiên bản vào tên file.
