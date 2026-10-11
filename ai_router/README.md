# AI Router (Active Learning Pipeline - M29)

## Tổng quan (Overview)
Dự án triển khai thực tế (Production) của lõi kiến trúc Active Learning M29.
Sử dụng mô hình RT-DETR kết hợp kỹ thuật Deep Ensembles để đo lường Độ bất định (Epistemic Uncertainty).

## Cấu trúc Giai đoạn (Pipeline Phases)
* **Giai đoạn 0:** Base Model Training
* **Giai đoạn 1:** Deep Ensembles Generation (Tạo 5 Chuyên gia)
* **Giai đoạn 2:** M29 Uncertainty Engine (Lọc ảnh khó)
* **Giai đoạn 3:** CVAT Auto-Router (Điều phối Human-in-the-Loop)
* **Giai đoạn 4:** M35 KPI Benchmark (Đánh giá hiệu suất)
