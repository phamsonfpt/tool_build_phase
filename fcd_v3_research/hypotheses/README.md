# Tài liệu giả thuyết

Mỗi giả thuyết có một thư mục riêng. Mỗi phiên bản là một thư mục con `vX.Y/`, chứa tài liệu thiết kế và các báo cáo đi kèm đúng như lúc phát hành phiên bản đó.

| Giả thuyết | Thư mục | Bản hiện hành | Các bản cũ |
|---|---|---|---|
| H1: Statistical Guarantee | [H1_statistical_guarantee/](H1_statistical_guarantee/) | [v1.4](H1_statistical_guarantee/v1.4/H1_statistical_guarantee_research.md) | v1.2, v1.3 |
| H2: Signal | [H2_signal/](H2_signal/) | [v1.4](H2_signal/v1.4/H2_the_signal_hypothesis.md) | v1.2 |
| H3: AL Utility | [H3_al_utility/](H3_al_utility/) | chưa có tài liệu riêng | — |

[shared/](shared/) chứa tài liệu dùng chung cho cả ba giả thuyết:

| File | Nội dung | Trạng thái |
|---|---|---|
| `HYPOTHESIS_CHAIN.md` | Chuỗi logic H1 → H2 → H3, bảng đối chiếu đánh số cũ/mới, yêu cầu dữ liệu cho H3 | Viết ngày 14/10, trước H1 v1.4 |
| `RESEARCH_PROPOSAL.md` | Đề cương ban đầu | Lỗi thời: đánh số cũ (H1 = Signal, H2 = Guarantee), ensemble 5 mô hình |
| `HYPOTHESES_AND_MATHEMATICAL_METHODS.md` | Hệ thống giả thuyết và đặc tả toán ban đầu | Lỗi thời: đánh số cũ |

Code đi kèm không đổi chỗ: `fcd_v3_research/h1_real/` (H1) và `fcd_v3_research/h2_simulation/` (H2).

## Quy ước khi ra phiên bản mới

1. Tạo thư mục con mới, ví dụ `H1_statistical_guarantee/v1.5/`. Không sửa đè thư mục của bản cũ.
2. Đặt tên file không kèm số phiên bản (`H1_statistical_guarantee_research.md`, không dùng `_v1.5.md`, ` 2.md` hay `_backup.md`). Số phiên bản nằm ở tên thư mục.
3. Chép vào thư mục mới mọi báo cáo và hình mà bản đó trích dẫn, để mỗi thư mục tự đủ.
4. Cập nhật README của giả thuyết (bản hiện hành và nhật ký thay đổi) và bảng ở trên.
