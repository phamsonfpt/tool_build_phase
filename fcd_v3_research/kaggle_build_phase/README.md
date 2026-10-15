# Build phase: nhóm cùng huấn luyện 3 chuyên gia RT-DETR trên Kaggle

Ba notebook (mỗi notebook một chuyên gia, `EXPERT_ID` điền sẵn) huấn luyện ensemble RT-DETR-R50vd trên **toàn bộ 10 lớp BDD100K**, giữ **mọi object**:

| Notebook (trong `fcd_v3_research/src/`) | Chuyên gia | Seed | Ai chạy |
|---|---|---|---|
| `Train_RTDETR_Ensemble.ipynb` | 1 | 42 | thành viên A |
| `Train_RTDETR_Ensemble_Expert2.ipynb` | 2 | 142 | thành viên B |
| `Train_RTDETR_Ensemble_Expert3.ipynb` | 3 | 242 | thành viên C |
| `Verify_Ensemble.ipynb` | kiểm tra cả 3 | — | ai cũng được, sau cùng |

**Dữ liệu:** chỉ cần bộ `solesensei_bdd100k` trên Kaggle, không phải tải gì thêm.
* **BDD train** (69.863 ảnh): dùng toàn bộ để train.
* **BDD val** (10.000 ảnh): **chỉ để xem**. Dùng cho đường cong loss trong lúc train (1.000 ảnh cố định) và báo cáo từng lớp sau khi train. **Không** chọn checkpoint, **không** chỉnh epoch hay learning rate theo val. Mô hình lưu ra luôn là bản cuối cùng. Nhờ vậy val vẫn dùng được cho bài báo; cách dùng này phải được công bố (`HYPOTHESIS_CHAIN.md`, Mục 4).
* **BDD test** (20.000 ảnh): không đọc. Giữ cho bài báo hoặc cho bước hiệu chuẩn LTT sau này, tùy kết quả kiểm tra gói Info.

Mỗi chuyên gia mất khoảng 9 giờ GPU (T4 x2): khoảng 8 giờ train, phần còn lại cho chuẩn bị dữ liệu, kiểm tra trên val và thống kê tiềm ẩn trên toàn bộ tập train. Ba chuyên gia tổng cộng khoảng 27 giờ, vừa hạn mức khoảng 30 giờ/tuần của một tài khoản; nếu mỗi người chạy trên tài khoản của mình thì không lo hết quota. Ba người chạy song song trên ba tài khoản, mỗi người dùng hạn mức của mình.

---

## A. Người phụ trách: chuẩn bị một lần (khoảng 15 phút)

**Tạo 3 notebook trên Kaggle**

Với mỗi file `Train_RTDETR_Ensemble*.ipynb`:
1. **Create → New Notebook → File → Import Notebook** → chọn file.
2. **Add Input**: thêm `solesensei_bdd100k`.
3. Thanh bên phải: **Accelerator = GPU T4 x2**, **Internet = On**.
4. **Save Version → Quick Save** (chỉ lưu, chưa chạy).
5. **Share**: thêm thành viên được giao chuyên gia này (quyền xem là đủ).

## B. Thành viên: 3 thao tác
1. Mở link notebook được giao → **Copy & Edit**.
2. Nhìn thanh bên phải: GPU T4 x2, Internet On, Input có `solesensei_bdd100k`. Nếu Kaggle không giữ GPU khi copy, chọn lại **GPU T4 x2**.
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
| `expert_K_class_report.txt` | Recall và precision theo lớp; recall theo thời tiết, thời điểm, loại cảnh (trên val, chỉ để xem) |
| `expert_K_centroids.pt` (chuyên gia 1: thêm `centroids.pt`) | Định dạng M29 Mục 4.B: `pca` (sklearn, 256→64), `centroids` μ_k, `inv_covs` Σ_k⁻¹ (+1e-6 I) theo lớp, tính bằng Max-IoU (IoU ≥ 0.5, đúng lớp) trên **toàn bộ** tập train; mỗi GPU xử lý một nửa, mất khoảng 20–25 phút. Mỗi lớp giữ tối đa 50.000 vector, lấy mẫu đều (reservoir). Lớp có dưới 192 vector dùng hiệp phương sai co rút (`shrunk_classes`). Chứa object sklearn nên phải nạp với `weights_only=False` |
| `expert_K_latent_stats.pt` | Cho tín hiệu FCD của H2 (chỉ tensor, nạp với `weights_only=True`): tâm lớp 256D, precision của hiệp phương sai gộp Ledoit-Wolf, trung vị/MAD của khoảng cách Mahalanobis trên các vector TP |
| `expert_K_probe.csv` | Điểm tin cậy cao nhất trung bình và số box > 0.3 trên 32 ảnh train cố định, mỗi 2.000 bước (kiểm tra sớm) |
| `expert_K_loss_curve.csv` | Loss train (mỗi 50 bước) và val (mỗi 2.000 bước, trên 1.000 ảnh val) |
| `data_stats.json` | Số box gốc và giữ lại theo lớp và theo tập, ảnh không có nhãn, hệ số lấy mẫu lặp |
| `bdd_det_{train,val}_compact.json` | Danh sách ảnh và box của từng tập |
| `expert_K/checkpoint-*` | Checkpoint để tiếp tục train nếu phiên bị dừng sớm |

## Lỗi đã sửa (15/10/2026): mô hình không đưa ra dự đoán nào
Lần chạy đầu của chuyên gia 2 cho **0 dự đoán** trên val, dù loss vẫn giảm. Nguyên nhân là một lỗi của `transformers` (có ở cả 4.57 và 5.16):
* Trong `RTDetrForObjectDetectionLoss`, query khử nhiễu (denoising) chỉ được tách khỏi các lớp decoder phụ.
* Loss của **lớp decoder cuối**, chính là lớp dùng khi suy luận, vẫn ghép GT trên cả 498 query (300 query thật cộng 198 query khử nhiễu). GT vì thế bị ghép vào query khử nhiễu, vốn nằm sát GT.
* 300 query thật của lớp cuối chỉ nhận mục tiêu "nền", nên điểm tin cậy khi suy luận bị ép về khoảng 0.01–0.04, trong khi box vẫn học tốt (IoU trung vị 0.77).

`train.py` vá lỗi này bằng cách tách query khử nhiễu trước khi tính loss của lớp cuối, giống code gốc của RT-DETR. Bản vá đã được kiểm trên cả hai phiên bản: `loss_vfl` báo ra khớp với VFL tính trên đúng 300 query thật. Notebook cũng có thêm:
* **ScoreProbe:** dừng train nếu điểm tin cậy tụt dưới 0.1 từ bước 4.000.
* **NanGuard:** dừng khi loss là NaN hoặc Inf.
* **Cố định `transformers==5.16.1`.**

**Mọi checkpoint train trước bản vá đều phải train lại.**

## Các quyết định thiết kế
* **Không giới hạn object:** giữ mọi `box2d` của 10 lớp, chỉ bỏ box suy biến (≤ 0.5 px). Notebook dừng nếu tổng số box giữ lại cộng số box suy biến khác số box gốc.
* **Ảnh không có nhãn:** file nhãn train của bản 2018 có 69.863 mục cho 70.000 ảnh. 137 ảnh không có nhãn bị bỏ qua (danh sách ở `unlabeled_images_bdd_train.txt`), vì nếu train trên chúng thì mọi object sẽ bị coi là nền.
* **Lớp hiếm:** lấy mẫu lặp (Repeat Factor Sampling) với ngưỡng 2% số ảnh, chủ yếu tác động tới lớp `train`.
* **Đa dạng giữa các chuyên gia:** seed được truyền vào `TrainingArguments(seed, data_seed)`, khiến thứ tự dữ liệu, augmentation (lật ngang, đổi màu) và làm tròn của lấy mẫu lặp đều khác nhau. Notebook kiểm tra báo lỗi nếu hai chuyên gia trùng trọng số; đây là lỗi đã xảy ra với 5 checkpoint M29.
* **Thời gian:** đo tốc độ trong 40 bước, đặt `max_steps` cho vừa `TRAIN_HOURS` (8 giờ) với lịch learning rate cosine, cộng chốt chặn tự lưu và dừng ở 11 giờ.
* **Siêu tham số:** AdamW, learning rate 1e-4 (backbone 1e-5), warmup, gradient clipping 0.1, fp16, ảnh 640×640, batch hiệu dụng 16. Khởi tạo đầu phân loại từ COCO cho 8/10 lớp.
* **Dữ liệu giống nhau với mọi thành viên:** nếu thiếu nhãn hoặc ảnh của train/val, notebook **dừng** ngay, để 3 chuyên gia luôn được train và kiểm tra trên cùng dữ liệu.

## Giới hạn đã biết
* Chưa chạy thử trên Kaggle. Các script đã được chạy thử trên GPU cục bộ (1 GPU, vài trăm ảnh, vài bước): chuẩn bị dữ liệu, đo tốc độ, train, chốt chặn thời gian, lưu, kiểm tra theo lớp và theo nhóm điều kiện. **Chế độ 2 GPU (`torchrun` DDP) chưa được chạy thử.**
* Detector của build phase không lọc theo chuyến đi (`train_filtered = False`). Bài báo cần train lại trên `Train_filtered` (`HYPOTHESIS_CHAIN.md`); khi có `train_exclude.txt`, chỉ cần thêm file đó vào Input.

## Nhãn BDD test (chưa dùng)
`test_labels_dataset/` và `_src/make_test_labels.py` được giữ lại phòng khi cần nhãn test sau này (ví dụ hiệu chuẩn LTT hoặc bài báo). Nhãn lấy từ gói **100K Images** chính thức trên bdd-data.berkeley.edu; bộ Kaggle không có nhãn test.

## Sửa notebook
Không sửa trực tiếp file `.ipynb`. Hãy sửa trong `_src/cells/*.py`, rồi tạo lại cả 4 notebook:
```
python fcd_v3_research/kaggle_build_phase/_src/build_nb.py fcd_v3_research/src
```
