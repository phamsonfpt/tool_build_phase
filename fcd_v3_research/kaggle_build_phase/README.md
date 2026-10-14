# Build phase: nhóm cùng huấn luyện 3 chuyên gia RT-DETR trên Kaggle

Ba notebook (mỗi notebook một chuyên gia, `EXPERT_ID` điền sẵn) huấn luyện ensemble RT-DETR-R50vd trên **toàn bộ 10 lớp BDD100K**, giữ **mọi object**:

| Notebook (trong `fcd_v3_research/src/`) | Chuyên gia | Seed | Ai chạy |
|---|---|---|---|
| `Train_RTDETR_Ensemble.ipynb` | 1 | 42 | thành viên A |
| `Train_RTDETR_Ensemble_Expert2.ipynb` | 2 | 142 | thành viên B |
| `Train_RTDETR_Ensemble_Expert3.ipynb` | 3 | 242 | thành viên C |
| `Verify_Ensemble.ipynb` | kiểm tra cả 3 | — | ai cũng được, sau cùng |

**Dữ liệu:** BDD train (69.863 ảnh) dùng toàn bộ để train. BDD test (20.000 ảnh) chia đôi theo hash tên ảnh thành `build_cal` (hiệu chuẩn LTT) và `build_check` (kiểm tra từng lớp và edge case). **BDD val (10.000 ảnh) không bao giờ được đọc**, vì nó là tập đánh giá cho bài báo.

Mỗi chuyên gia mất khoảng 8.5 giờ GPU (T4 x2). Ba người chạy song song trên ba tài khoản, mỗi người dùng hạn mức của mình.

---

## A. Người phụ trách: chuẩn bị một lần (khoảng 20 phút)

**1. Tải nhãn BDD test lên Kaggle**

Bộ BDD100K trên Kaggle (`solesensei_bdd100k`) thường chỉ có file nhãn train/val, nên cần tải thêm nhãn test.
* File cần tải: `kaggle_build_phase/test_labels_dataset/bdd100k_labels_images_test.json` (72.5 MB; 20.000 ảnh, 367.728 box). File này được tạo từ thư mục `ai_router/data/100k/test` bằng `_src/make_test_labels.py`.
* Trên Kaggle: **Datasets → New Dataset**, kéo file vào, đặt tên `bdd100k-test-labels`, chọn **Private**, rồi **Create**.
* Mở dataset vừa tạo → **Settings → Sharing**: thêm tài khoản Kaggle của các thành viên.

**2. Tạo 3 notebook trên Kaggle**

Với mỗi file `Train_RTDETR_Ensemble*.ipynb`:
1. **Create → New Notebook → File → Import Notebook** → chọn file.
2. **Add Input**: thêm `solesensei_bdd100k`, rồi thêm `bdd100k-test-labels`. Bộ BDD100K phải có `images/100k/test`; nếu không có, hãy thêm một bộ BDD100K khác có ảnh test.
3. Thanh bên phải: **Accelerator = GPU T4 x2**, **Internet = On**.
4. **Save Version → Quick Save** (chỉ lưu, chưa chạy).
5. **Share**: thêm thành viên được giao chuyên gia này (quyền xem là đủ).

## B. Thành viên: 3 thao tác
1. Mở link notebook được giao → **Copy & Edit**.
2. Nhìn thanh bên phải: GPU T4 x2, Internet On, Input có 2 dataset. Nếu Kaggle không giữ GPU khi copy, chọn lại **GPU T4 x2**.
3. **Save Version → Save** (để mặc định *Save & Run All*). Có thể tắt máy; kết quả nằm ở tab **Output** của phiên bản đó.

Nếu thiếu GPU, Internet hay dữ liệu, notebook **dừng ngay trong phút đầu** với lời nhắc cụ thể, nên không tốn quota. Bật Internet cần tài khoản Kaggle đã xác minh số điện thoại.

## C. Sau khi cả 3 chạy xong
Import `Verify_Ensemble.ipynb` → **Add Input → Notebook Output** của 3 notebook (bản Copy & Edit của các thành viên) → **Run All**. Dòng cuối phải là `[V] ENSEMBLE HỢP LỆ`.

---

## Kết quả của mỗi chuyên gia (tab Output)
| File | Nội dung |
|---|---|
| `expert_K.pth` | `state_dict` (nạp bằng `torch.load(..., weights_only=True)`) |
| `expert_K_final/` | Mô hình và processor định dạng HuggingFace (`from_pretrained`) |
| `expert_K_manifest.json` | seed, số bước và epoch đã chạy, batch, sha256 trọng số, nguồn dữ liệu, phiên bản thư viện |
| `expert_K_class_report.txt` | Recall và precision theo lớp; recall theo thời tiết, thời điểm, loại cảnh (trên `build_check`) |
| `data_stats.json` | Số box gốc và giữ lại theo lớp và theo tập, ảnh không có nhãn, hệ số lấy mẫu lặp |
| `bdd_det_{train,build_cal,build_check}_compact.json` | Danh sách ảnh và box của từng tập |
| `expert_K/checkpoint-*` | Checkpoint để tiếp tục train nếu phiên bị dừng sớm |

## Các quyết định thiết kế
* **Không giới hạn object:** giữ mọi `box2d` của 10 lớp, chỉ bỏ box suy biến (≤ 0.5 px). Notebook dừng nếu tổng số box giữ lại cộng số box suy biến khác số box gốc.
* **Ảnh không có nhãn:** file nhãn train của bản 2018 có 69.863 mục cho 70.000 ảnh. 137 ảnh không có nhãn bị bỏ qua (danh sách ở `unlabeled_images_bdd_train.txt`), vì nếu train trên chúng thì mọi object sẽ bị coi là nền.
* **Lớp hiếm:** lấy mẫu lặp (Repeat Factor Sampling) với ngưỡng 2% số ảnh, chủ yếu tác động tới lớp `train`.
* **Đa dạng giữa các chuyên gia:** seed được truyền vào `TrainingArguments(seed, data_seed)`, khiến thứ tự dữ liệu, augmentation (lật ngang, đổi màu) và làm tròn của lấy mẫu lặp đều khác nhau. Notebook kiểm tra báo lỗi nếu hai chuyên gia trùng trọng số; đây là lỗi đã xảy ra với 5 checkpoint M29.
* **Thời gian:** đo tốc độ trong 40 bước, đặt `max_steps` cho vừa `TRAIN_HOURS` (8 giờ) với lịch learning rate cosine, cộng chốt chặn tự lưu và dừng ở 11 giờ.
* **Siêu tham số:** AdamW, learning rate 1e-4 (backbone 1e-5), warmup, gradient clipping 0.1, fp16, ảnh 640×640, batch hiệu dụng 16. Khởi tạo đầu phân loại từ COCO cho 8/10 lớp.
* **Dữ liệu giống nhau với mọi thành viên:** nếu thiếu nhãn hoặc ảnh test, notebook **dừng** thay vì tự đổi cách chia, để 3 chuyên gia luôn được train và kiểm tra trên cùng dữ liệu.

## Giới hạn đã biết
* Chưa chạy thử trên Kaggle. Các script đã được chạy thử trên GPU cục bộ (1 GPU, vài trăm ảnh, vài bước): chuẩn bị dữ liệu, đo tốc độ, train, chốt chặn thời gian, lưu, kiểm tra theo lớp và theo nhóm điều kiện. **Chế độ 2 GPU (`torchrun` DDP) chưa được chạy thử.**
* Chưa xác minh được nguồn phát hành của bộ nhãn BDD test trong `ai_router/data`; nên ghi lại nguồn tải để trích dẫn.
* Detector của build phase không lọc theo chuyến đi (`train_filtered = False`). Bài báo cần train lại trên `Train_filtered` (`HYPOTHESIS_CHAIN.md`); khi có `train_exclude.txt`, chỉ cần thêm file đó vào Input.

## Sửa notebook
Không sửa trực tiếp file `.ipynb`. Hãy sửa trong `_src/cells/*.py`, rồi tạo lại cả 4 notebook:
```
python fcd_v3_research/kaggle_build_phase/_src/build_nb.py fcd_v3_research/src
```
