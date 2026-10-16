# H3: The Active Learning Utility Hypothesis

Chưa có tài liệu thiết kế riêng. Phát biểu ban đầu nằm trong `../shared/HYPOTHESES_AND_MATHEMATICAL_METHODS.md` (đánh số cũ). Các yêu cầu bắt buộc để giữ tính hợp lệ của H1 nằm ở Mục 3.3 của `../shared/HYPOTHESIS_CHAIN.md`:

- Hiệu chuẩn lại ở mỗi vòng AL bằng một tập hiệu chuẩn mới (H3-Cal, ít nhất 4.500 đơn vị cho 3 vòng).
- Đo mAP trên tập không dùng để chọn tín hiệu U\* (H3-Test, khoảng 5.000 ảnh).
- H3-Cal, H3-Test và H3-Pool được tách ra từ split train theo trọn super-block, **trước khi** huấn luyện detector ban đầu cho H3. Vì vậy H3 cần bộ detector riêng; các expert train trên toàn bộ split train không dùng được.

Bản đầu tiên đặt ở `v1.0/`.
