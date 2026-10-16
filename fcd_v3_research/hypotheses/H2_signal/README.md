# H2: The Signal Hypothesis

**Bản hiện hành: [v1.4](v1.4/H2_the_signal_hypothesis.md)** (14/10/2026, bản nháp đăng ký trước). Code mô phỏng: `fcd_v3_research/h2_simulation/`.

| Bản | Ngày | Thư mục | Nội dung |
|---|---|---|---|
| v1.4 | 14/10/2026 | [v1.4/](v1.4/) | Thiết kế, báo cáo mô phỏng S1–S2 |
| v1.2 | 14/10/2026 | [v1.2/](v1.2/) | Thiết kế (bản đăng ký trước chính thức của v1.2) |

Bản 1.3 không được lưu thành file riêng; v1.4 ghi lại những gì giữ từ 1.3.

## Nhật ký thay đổi

**v1.4** (chi tiết ở đầu tài liệu v1.4)
- Ensemble cố định là 3 detector có sẵn, mỗi detector lần lượt làm anchor; bỏ pool 5 mô hình của v1.3.
- Tín hiệu đề xuất đổi thành dạng lai: bất đồng đầu ra của ensemble cộng độ lệch tiềm ẩn. Mô phỏng S1 cho thấy tín hiệu chỉ dùng không gian tiềm ẩn thua điểm tin cậy ở mọi kịch bản.
- Giữ quy tắc thống kê của v1.3 (bootstrap theo super-block, quy tắc ba vùng).
- Đăng ký trước các dự đoán về kết quả dựa trên mô phỏng.

## Cần đồng bộ với H1 v1.4

v1.4 của H2 được viết trước H1 v1.4, nên còn hai điểm lệch:
- Evaluation pool vẫn là Val, trong khi H1 v1.4 chuyển kiểm định xác nhận sang Pool T.
- Mục 0 khuyến nghị train lại trên `Train_filtered` vì Meta-Train nằm trong tập train. Ở H1 v1.4, Meta-Train lấy từ split test nên lý do này không còn.
