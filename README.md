# tool_build_phase

Bản đồ thư mục — đọc file này trước để biết thứ gì nằm ở đâu.

```
tool_build_phase/
├── docs/                      Tài liệu chung của dự án
│   ├── architecture/          Kế hoạch kiến trúc (final_architecture_plan, extra_final_architecture_plan)
│   ├── specs/                 Đặc tả kỹ thuật (M29_MASTER_SPECIFICATION)
│   └── reviews/               Bản review / nhận xét (H1_statistical_guarantee_review)
├── scripts/                   Script tiện ích (init_project.py: tạo khung thư mục ai_router)
├── ai_router/                 Dự án chính: data/, experiments/ (weights), src/, venv/
├── al_router_demo/            Bản demo thu nhỏ của pipeline (dữ liệu + code *_demo)
├── fcd_v3_research/           Nghiên cứu FCD v3
│   ├── docs/                  Đề xuất nghiên cứu, giả thuyết H1/H2, báo cáo mô phỏng
│   ├── scripts/               Script đánh giá chạy độc lập (01_baseline_fdp_eval.py)
│   └── src/                   Code huấn luyện ensemble RT-DETR + uncertainty_plugins/
└── test/ train/ val/          Ảnh frame *-0000100.jpg (không đưa vào git)
```

## Lưu ý

- Không đổi tên hay di chuyển `ai_router/`: nhiều script hardcode đường dẫn
  `D:\tool_build_phase\ai_router\data`.
- `ai_router/data/`, `ai_router/experiments/` và các file weights (`*.pth`, `*.safetensors`, ...)
  nằm trong `.gitignore`.
