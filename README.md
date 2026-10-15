# tool_build_phase

Bản đồ thư mục — đọc file này trước để biết thứ gì nằm ở đâu.

```
tool_build_phase/
├── docs/                      Tài liệu chung của dự án
│   ├── architecture/          Kế hoạch kiến trúc (final_architecture_plan, extra_final_architecture_plan)
│   ├── paper/                 Thiết kế paper gộp H1+H2+H3 (blueprint, giải thích toán học,
│   │                          kế hoạch phân tích thống kê; results/ = số liệu mô phỏng)
│   ├── specs/                 Đặc tả kỹ thuật (M29_MASTER_SPECIFICATION)
│   └── reviews/               Bản review / nhận xét (H1_statistical_guarantee_review)
├── scripts/                   Script tiện ích (init_project.py: tạo khung thư mục ai_router)
├── ai_router/                 Dự án chính: data/, experiments/ (weights), src/, venv/
├── al_router_demo/            Bản demo thu nhỏ của pipeline (dữ liệu + code *_demo)
├── fcd_v3_research/           Nghiên cứu FCD v3
│   ├── docs/                  Đề xuất nghiên cứu, giả thuyết H1/H2, báo cáo mô phỏng
│   ├── scripts/               Script chạy độc lập (01_baseline_fdp_eval.py,
│   │                          02_statistical_validation.py: mô phỏng power / sai lầm loại I)
│   ├── src/                   Code huấn luyện ensemble RT-DETR, uncertainty_plugins/,
│   │                          risk_stats/ (p-value HB, LTT, kiểm định, power)
│   └── tests/                 Unit test cho risk_stats (python -m unittest discover -s fcd_v3_research/tests)
└── test/ train/ val/          Ảnh frame *-0000100.jpg (không đưa vào git)
```

## Lưu ý

- Không đổi tên hay di chuyển `ai_router/`: nhiều script hardcode đường dẫn
  `D:\tool_build_phase\ai_router\data`.
- `ai_router/data/`, `ai_router/experiments/` và các file weights (`*.pth`, `*.safetensors`, ...)
  nằm trong `.gitignore`.
