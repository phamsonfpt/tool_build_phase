# Chuỗi Logic H1 → H2 → H3

**Ngày:** 14/10/2026. **Mục đích:** một chỗ duy nhất ghi lại (1) mỗi giả thuyết nhận gì và giao gì cho giả thuyết sau, (2) kết quả của giả thuyết trước ràng buộc cách diễn giải giả thuyết sau như thế nào, (3) các tham số dùng chung và giá trị chuẩn của chúng, (4) các mâu thuẫn hiện có giữa tài liệu, code và detector đã huấn luyện.

---

## 1. Tài liệu nào là chuẩn

| Tài liệu | Trạng thái | Ghi chú |
|---|---|---|
| `H1_statistical_guarantee_research 2.md` (bản 1.2) | **Chuẩn cho H1** | Cần sửa một số điểm để khớp với detector thực tế (Mục 5) |
| `H2_the_signal_hypothesis.md` (bản 1.4) | **Chuẩn cho H2** | — |
| `H2_simulation_report.md`, `h2_simulation/` | Căn cứ thiết kế của H2 | — |
| H3 | **Chưa có tài liệu riêng** | Mục 3 dưới đây là giao diện bắt buộc cho tài liệu H3 |
| `RESEARCH_PROPOSAL.md`, `HYPOTHESES_AND_MATHEMATICAL_METHODS.md` | **Lỗi thời** | Đánh số cũ (H1 = Signal, H2 = Guarantee), ensemble 5 mô hình, softmax, tín hiệu phương sai của 5 khoảng cách, Wilcoxon, $\delta = 0.1$. Không dùng để trích dẫn; cần cập nhật hoặc ghi chú "đã thay thế" |

**Bảng đối chiếu đánh số:** cũ H1 (Signal) = mới **H2**; cũ H2 (Guarantee) = mới **H1**; H3 giữ nguyên.

---

## 2. Sơ đồ phụ thuộc

```
                ┌───────────────────────── DỮ LIỆU & DETECTOR DÙNG CHUNG ──────────────────────────┐
                │ BDD100K; ride proxy, block, super-block (H1 P1); Train_filtered (H1 A2, P2);     │
                │ Meta-Train (10% super-block) → Meta-Fit / Meta-Path (H2 P-H2.0);                 │
                │ 3 detector RT-DETR; θ_IoU, C_eval, đơn vị, N (H1 P3)                             │
                └────────────────────────────────────────┬─────────────────────────────────────────┘
                                                         │
   ┌─────────────────────────────────────────────────────▼───────────────────────────────────────────┐
   │ H1  Thước đo có hợp lệ không?                                                                   │
   │     H1a bảo chứng (C2, C2b) · H1b sụp đổ khi chia theo frame (C4, C5) · H1c không rỗng (C3, C3b) │
   │ Giao cho H2: đơn vị, N, θ_IoU, lưới rủi ro, thuật toán đường đi, các phân hoạch B-block, quy tắc C2│
   └─────────────────────────────────────────────────────┬───────────────────────────────────────────┘
                                         điều kiện: H1a không Bác bỏ
   ┌─────────────────────────────────────────────────────▼───────────────────────────────────────────┐
   │ H2  Ở cùng rủi ro được chứng nhận, tín hiệu nào giữ được nhiều nhãn đúng nhất?                  │
   │     H2a thông tin tiềm ẩn · H2b ensemble · H2c vượt mọi baseline · H2d hợp lệ rủi ro            │
   │ Giao cho H3: U* (quy tắc H2 Mục 9.2), siêu tham số, thống kê tham chiếu, quy trình hiệu chuẩn   │
   └─────────────────────────────────────────────────────┬───────────────────────────────────────────┘
                                         điều kiện: H2d không Bác bỏ cho U*
   ┌─────────────────────────────────────────────────────▼───────────────────────────────────────────┐
   │ H3  Định tuyến bằng U* trong vòng AL có tăng mAP hơn Random / Least-Confidence không?            │
   │     Hiệu chuẩn lại LTT trên dữ liệu mới ở mỗi vòng; mAP đo trên tập không dùng để chọn U*        │
   └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Lan truyền kết quả: kết quả trước ràng buộc diễn giải sau

### 3.1. H1 → H2

| Kết quả H1 | Hệ quả cho H2 |
|---|---|
| H1a **Chấp nhận** | H2 chạy như đăng ký |
| H1a **Chấp nhận nhờ biên chịu đựng** (C2 đạt, C2b bác bỏ A1) | H2 chạy, nhưng mọi Coverage được báo cáo là "dưới bảo chứng xấp xỉ". C-H2d vẫn kiểm VR của từng tín hiệu |
| H1a **Không kết luận** | H2 chạy; kết luận H2 ghi rõ "có điều kiện trên tính hợp lệ chưa được xác nhận của thước đo" |
| H1a **Bác bỏ** | H2 **không** được kết luận xác nhận. Các so sánh Coverage chỉ là mô tả, vì Coverage có thể được mua bằng vi phạm rủi ro |
| H1c **Không đạt** (Yield < 85% hoặc $\rho_{\text{cov}} < 0.5$) | Không chặn H2, nhưng nếu Yield thấp thì Coverage của nhiều vòng bằng 0, phương sai lớn, và H2 dễ rơi vào Không kết luận. P-H2.5 phải ước lượng lại năng lực |
| H1b (bất kỳ) | Không ảnh hưởng H2, vì H2 dùng đơn vị theo R-unit chứ không dùng frame |
| Quy tắc R-unit chọn **super-block** và giảm $N$ | H2 dùng cùng $N$. Lưới rủi ro và mức chứng nhận tối đa ($k_{\max}/N$) thay đổi theo bảng 5.4 của H1 |

**Phạm vi của bảo chứng:** H1 kiểm chứng bảo chứng trên **đường đi của H1**. H2 dùng đường đi riêng cho từng tín hiệu. Định lý của LTT áp dụng cho mọi đường đi cố định trước, nên không cần H1 kiểm lại từng tín hiệu. Tuy vậy, C-H2d vẫn kiểm VR cho từng tín hiệu trên chính các vòng B-block, như một phép kiểm đầu-cuối.

### 3.2. H2 → H3

| Kết quả H2 | $U^\star$ giao cho H3 (H2 Mục 9.2) | Câu hỏi H3 trả lời |
|---|---|---|
| C-H2-Final **Chấp nhận** | $U_{\text{fcd}}$ (lai ensemble) | Định tuyến bằng tín hiệu tốt nhất có tăng mAP không |
| C-H2a Chấp nhận, C-H2b không Chấp nhận | B8 (lai đơn mô hình), nếu B8 Đạt so với B1 | Thông tin tiềm ẩn có chuyển thành lợi ích mAP không, với chi phí 1 forward |
| C-H2a không Chấp nhận | Tín hiệu theo quy tắc 9.2, thường là B1 hoặc B2 | H3 trở thành phép thử của LTT + điểm tin cậy; H3 **không** được trình bày như bằng chứng cho FCD |
| C-H2d Bác bỏ cho $U^\star$ | Chọn tín hiệu kế tiếp theo 9.2 mà không vi phạm | — |

**Nguyên tắc:** H3 kiểm chứng chuỗi "LTT + $U^\star$". Tên của tín hiệu trong phát biểu H3 phải lấy từ kết quả H2, không viết cứng "LTT + FCD" như trong `HYPOTHESES_AND_MATHEMATICAL_METHODS.md` cũ.

### 3.3. Yêu cầu bắt buộc cho tài liệu H3 (để giữ tính hợp lệ của H1)
1. **Hiệu chuẩn lại ở mỗi vòng AL.** Sau mỗi vòng, detector được huấn luyện lại trên nhãn giả do $\hat\tau$ chọn, mà $\hat\tau$ phụ thuộc tập hiệu chuẩn. Nếu dùng lại cùng tập hiệu chuẩn ở vòng sau, tập đó không còn độc lập với mô hình, và giả định A2 của H1 bị vi phạm. Vì vậy mỗi vòng cần một tập hiệu chuẩn mới gồm $N$ đơn vị có nhãn, rời với mọi dữ liệu huấn luyện. Với $T = 3$ vòng và $N = 1.500$, cần **4.500 đơn vị có nhãn** chỉ cho hiệu chuẩn.
2. **mAP đo trên tập không dùng để chọn $U^\star$.** H2 chọn $U^\star$ trên Val. Nếu H3 đo mAP trên Val, phép chọn và phép đo dùng chung dữ liệu. *(Cập nhật 14/10: gói "100K Images" chính thức trên bdd-data.berkeley.edu có nhãn cho cả 20.000 ảnh `test`, nhưng build phase đã dùng tập này làm `build_cal` / `build_check`.)* Vì vậy tập đo mAP của H3 phải được tách ra từ `train` theo trọn super-block, và tách **trước khi** huấn luyện lại detector cho bài báo.
3. **Đơn vị và $\theta_{\text{IoU}}$** dùng đúng giá trị của H1.
4. **So sánh ở cùng ngân sách gán nhãn** ($B$ ảnh mỗi vòng), với ít nhất 3 seed cho mỗi nhánh. Tiêu chí quyết định dùng quy tắc ba vùng với biên SESOI như H2, thay cho kiểm định t ghép cặp với $p < 0.05$ trên 3 seed (bản cũ), vì 3 điểm dữ liệu không đủ để ước lượng phương sai một cách tin cậy.

**Ngân sách dữ liệu đề xuất** (phân chia theo trọn super-block, thực hiện ở bước P2 của H1):

| Phần | Nguồn | Dùng cho | Ước tính |
|---|---|---|---|
| Evaluation pool | Toàn bộ Val | H1 Exp B, H2 | khoảng 7.000 đơn vị |
| Meta-Train | 10% super-block của train | H1 P3, H2 Meta-Fit/Meta-Path | khoảng 7.000 ảnh |
| H3-Cal | Phần của train, rời với detector | 3 tập hiệu chuẩn mới cho 3 vòng AL | ít nhất 4.500 đơn vị |
| H3-Test | Phần của train, rời với detector | Đo mAP của H3 | khoảng 5.000 ảnh |
| H3-Pool | Phần còn lại của train (ẩn nhãn) | Nguồn ảnh chưa gán nhãn của AL | — |
| Huấn luyện detector ban đầu | Phần còn lại của `Train_filtered` | 3 detector | — |

Cần kiểm tra lại các con số sau bước P1 của H1 (số ride proxy thật).

---

## 4. Tham số dùng chung (giá trị chuẩn)

| Tham số | Giá trị chuẩn | Nơi định nghĩa | Mâu thuẫn hiện có |
|---|---|---|---|
| $\alpha$ | 0.02 | H1 | — |
| $\delta$ | 0.05 | H1 | Tài liệu cũ dùng 0.1 |
| Đơn vị thống kê | ride, block hoặc super-block theo R-unit | H1 Mục 8.1 | Tài liệu cũ: "video sequence" |
| $N$ | 1.500 (giảm nếu R-unit yêu cầu) | H1 | — |
| $\theta_{\text{IoU}}$ | Chọn trong $\{0.75, 0.5\}$ theo quy tắc của H1 | H1 Mục 2.1 | H2 bản 1.2 khóa cứng 0.75 (đã sửa) |
| $\mathcal{C}_{\text{eval}}$ | **Cần chốt:** 8 lớp (H1) hoặc chỉ `car` (detector hiện tại) | H1 Mục 2.1 | **Mâu thuẫn**: 3 detector chỉ huấn luyện lớp `car` |
| Số detector $M$ | 3 | H2 Mục 0 | Tài liệu cũ và README của `ai_router`: 5 |
| Đầu phân loại | Sigmoid (varifocal) | H2 | Tài liệu cũ, `min_conf.py`: softmax |
| Tín hiệu tiềm ẩn | $\bar\zeta + \lambda s_\zeta$ trên đủ 256 chiều, chuẩn hóa robust theo từng mô hình | H2 Mục 5.9 | H1 ký hiệu $\tau_{\text{fcd}}$ nhưng không định nghĩa lại; `fcd_variance.py` cài đặt khác hẳn |
| Đường đi của H1 | 3 chiều $(\tau_{\text{cls}}, \tau_{\text{loc}}, \tau_{\text{fcd}})$, dựng Pareto | H1 Mục 6.2 | Cần định nghĩa $\tau_{\text{loc}}$ và $\tau_{\text{fcd}}$ theo tín hiệu của H2 (Mục 5) |
| Biên SESOI | H2: $\gamma = 5$ pp Coverage; H3: chưa có | H2 | H3 cũ: 1.5% mAP với kiểm định t |

---

## 5. Danh sách việc cần làm để chuỗi nhất quán

**Ưu tiên 1, chặn đăng ký trước:**
1. **Chốt tập lớp** (8 lớp hay chỉ `car`). Lựa chọn này quyết định $\mathcal{C}_{\text{eval}}$ trong cả ba giả thuyết.
2. **Huấn luyện lại 3 detector**, hoặc ghi rõ các giới hạn nếu không huấn luyện lại (H2 Mục 0). Nếu huấn luyện lại:
   * dùng `Train_filtered` của H1, đồng thời loại Meta-Train, H3-Cal và H3-Test;
   * đặt `TrainingArguments(seed=s, data_seed=s)` khác nhau cho từng mô hình (hiện tại `Trainer` ghi đè seed thành 42);
   * nếu được, thêm nguồn đa dạng thật (augmentation khác nhau, hoặc khởi tạo lại ngẫu nhiên đầu phân loại), vì mô phỏng S1 cho thấy hiệu ứng ensemble gần như biến mất khi tương quan giữa các thành viên trên 0.95.
3. **H1 cần sửa:** (a) $\mathcal{C}_{\text{eval}}$ theo quyết định 1; (b) định nghĩa $\tau_{\text{fcd}}$ dẫn tới tín hiệu tiềm ẩn của H2 (Mục 5.9), và $\tau_{\text{loc}}$ dẫn tới thành phần lệch hộp của B2; (c) ghi rõ detector là ensemble 3 mô hình và mô hình nào là anchor của đường đi H1; (d) thêm H3-Cal và H3-Test vào phân chia dữ liệu ở bước P2.

**Ưu tiên 2:**
4. Viết tài liệu H3 theo các yêu cầu ở Mục 3.3.
5. Đánh dấu `RESEARCH_PROPOSAL.md` và `HYPOTHESES_AND_MATHEMATICAL_METHODS.md` là đã thay thế, hoặc cập nhật theo bảng đối chiếu đánh số.
6. Code: viết lại `fcd_variance.py` theo H2 Mục 5.9 (đã có task riêng); sửa `min_conf.py` sang sigmoid; sửa seed trong `kaggle_all_in_one_train.py`; cập nhật `02_signal_superiority_benchmark.py` theo thiết kế 2×2 mới.

**Ưu tiên 3:**
7. Sau bước P1 của H1, kiểm lại ngân sách dữ liệu ở Mục 3.3 với số ride proxy thật.
