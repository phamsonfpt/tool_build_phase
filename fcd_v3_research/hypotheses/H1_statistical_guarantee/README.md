# H1: The Statistical Guarantee Hypothesis

**Bản hiện hành: [v1.4](v1.4/H1_statistical_guarantee_research.md)** (15/10/2026). Code: `fcd_v3_research/h1_real/`.

| Bản | Ngày | Thư mục | Nội dung |
|---|---|---|---|
| v1.4 | 15/10/2026 | [v1.4/](v1.4/) | Thiết kế, bản trình bày dạng bài báo (`H1_paper_design.md`), kịch bản thảo luận với mentor, báo cáo chạy thật, báo cáo mô phỏng, `figures/` |
| v1.3 | 14/10/2026 | [v1.3/](v1.3/) | Thiết kế, báo cáo chạy thật (bản lúc phát hành v1.3), báo cáo mô phỏng |
| v1.2 | 13/10/2026 | [v1.2/](v1.2/) | Thiết kế, báo cáo mô phỏng (bản 1.1 → 1.2), bản review ngày 13/10 |

Bản 1.0 và 1.1 không còn được lưu.

## Nhật ký thay đổi

**v1.4** (chi tiết ở đầu tài liệu v1.4)
- Ride ID chính thức lấy từ trường `rideID` của `bdd100k_info`; block và super-block dựng từ GPS và `startTime`.
- Ba vai trò dữ liệu tách bạch: Meta-Train (3.000 ride của split test), Pool V (BDD Val, chỉ thăm dò), Pool T (9.033 ride của split test, chưa đọc nhãn, dùng cho kiểm định xác nhận).
- Diễn giải lại Exp B: B-rand chỉ là kiểm tra cài đặt (Mệnh đề 4); nội dung thực nghiệm của H1a nằm ở B-block và C2b.
- Baseline của C3b đổi thành plug-in trên cùng lưới G (τ_base^G).
- Exp C cần detector riêng M_C, train trên tập train đã bỏ các ride của MOT\*.
- Thêm D8 (rò rỉ cấp super-block).

**v1.3** (chạy thật trên BDD100K)
- Ride ID lấy từ tiền tố 8 ký tự của tên video; xác nhận split BDD100K tách theo chuyến đi (train và Val chung đúng 1 ride).
- Thêm quy tắc chạm và quy tắc độ phân giải cho lưới τ_cls.

**v1.2** (dựa trên mô phỏng edge case)
- Thêm C2b (design effect thực nghiệm) để kiểm trực tiếp A1.
- C2 và C5 dùng rủi ro chính xác trên toàn pool.
- C4 đặt trên design effect theo frame thay cho ICC ≥ 0,30.
- Ngưỡng R-unit đổi thành min(2, D\*).
- Gộp các sửa đổi từ bản review (`v1.2/H1_statistical_guarantee_review.txt`).

## Lưu ý về detector

Mọi số liệu thăm dò trong v1.3 và v1.4 được tính bằng các expert cũ: head 80 lớp COCO, chỉ fine-tune lớp car, còn dính lỗi loss denoising. Từ 16/10, `ai_router/experiments/expert_1.pth` là bản mới (10 lớp BDD); bản cũ được giữ ở `expert_1_old.pth`. Khi dùng detector mới cho H1, cần ra một phiên bản mới.
