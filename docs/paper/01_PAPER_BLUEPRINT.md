# BẢN THIẾT KẾ PAPER HỢP NHẤT H1 + H2 + H3

**Phiên bản:** 0.1 (bản thiết kế, 15/10/2026). **Đi kèm:** [02_MATH_EXPLAINED.md](02_MATH_EXPLAINED.md), phần giải thích toán học chi tiết từ đầu, có ví dụ số và trích dẫn.

**Nguồn đã đọc để viết bản này:** `fcd_v3_research/docs/H1_statistical_guarantee_research 2.md` (v1.2), `H2_the_signal_hypothesis.md` (v1.2), `HYPOTHESES_AND_MATHEMATICAL_METHODS.md`, `RESEARCH_PROPOSAL.md`, `H1_simulation_report.md`, `docs/reviews/H1_statistical_guarantee_review.txt`, `docs/specs/M29_MASTER_SPECIFICATION.md`, hai file kiến trúc trong `docs/architecture/`, và code trong `fcd_v3_research/src/uncertainty_plugins/`.

> **Quy ước:** mọi con số kết quả trong file này đều là **chỗ trống** dạng `[xx.x]`. Các tài liệu cũ có những bảng kết quả "đăng ký trước" ghi sẵn số (Coverage 72.8%, mAP +4.4%, ...). Những con số đó chưa được đo, nên không được đưa vào paper hay slide dưới dạng kết quả (xem Mục 1, vấn đề P6).

---

## 0. Trả lời ngắn: gộp cả 3 giả thuyết vào một paper được không?

**Được, và nên gộp**, vì ba giả thuyết là ba bước nối tiếp của cùng một lập luận, bước sau đứng trên bước trước:

```
H1 (thước đo hợp lệ)  ──►  H2 (tín hiệu nào tốt nhất trên thước đo đó)  ──►  H3 (dùng kết quả để huấn luyện có lợi không)
"LTT trên đơn vị        "Ở cùng mức rủi ro được chứng nhận,          "Định tuyến mẫu bị từ chối cho người,
 chuyến đi cho bảo        tín hiệu latent ensemble thu được            nhãn giả được chứng nhận cho máy:
 chứng đúng; chia         nhiều nhãn giả đúng nhất"                    mAP tăng nhanh hơn Random/Least-Conf"
 theo frame thì không"
```

* Nếu tách riêng, paper H2 sẽ bị hỏi "thước đo có đáng tin không?" và phải viết lại toàn bộ H1. Paper H3 sẽ bị hỏi "tại sao dùng tín hiệu này?" và phải viết lại H2. Một paper gộp trả lời cả ba câu hỏi trong cùng một khung ký hiệu.
* Đóng góp mới nằm ở **sự kết hợp**: LTT đã được chứng minh (Angelopoulos et al., 2021), Mahalanobis đã có (Lee et al., 2018), AL kết hợp nhãn giả đã có (Wang et al., 2017; Elezi et al., 2022). Chưa có công trình đặt cả ba lên dữ liệu video lái xe với đơn vị thống kê đúng.

**Cấu trúc khuyến nghị:** H1 và H2 là phần lõi của bài chính. H3 là thực nghiệm đầu-cuối cho thấy kết quả có giá trị thực tế. Chứng minh, bảng biên HB đầy đủ và mô phỏng edge case của H1 đưa vào phụ lục.

**Rủi ro khi gộp và phương án dự phòng:**

| Rủi ro | Hệ quả | Phương án |
|---|---|---|
| Quá dài: riêng tài liệu H1 đã đủ cho một paper | Vượt giới hạn trang (NeurIPS: khoảng 9 trang nội dung chính) | Bài chính giữ Theorem 1, Exp B và Exp C. Exp A, bảng 5.4–5.5, Lemma A4 và mô phỏng đưa vào phụ lục |
| H3 tốn compute (huấn luyện lại RT-DETR nhiều vòng × nhiều arm × nhiều seed) | Không kịp hạn nộp | H3 chạy trên tập con (ví dụ 20K ảnh) và lịch huấn luyện ngắn. Nếu vẫn không kịp, nộp H1+H2 trước và để H3 thành paper thứ hai |
| H2 thất bại (FCD không vượt baseline) | Mất "phương pháp đề xuất" | Paper vẫn đứng được như một **benchmark** (NeurIPS D&B rất hợp): "dưới rủi ro được chứng nhận, tín hiệu nào thực sự tốt?". Kết quả âm có thiết kế đăng ký trước vẫn công bố được |
| H1b không tái hiện (frame-split không sụp đổ trên dữ liệu thật) | Mất một đóng góp | Báo cáo design effect thực đo. Kết luận "bảo chứng bền hơn dự đoán" cũng là kết quả có giá trị |

**Đề xuất venue:** NeurIPS Datasets & Benchmarks hợp với khung "benchmark tín hiệu có bảo chứng". IEEE T-IV hợp nếu nhấn vào ứng dụng xe tự hành và H3. Kiểm tra giới hạn trang và chính sách phụ lục của venue chọn trước khi chốt bố cục.

---

## 1. Những vấn đề phải sửa trước khi viết paper

Đây là phần quan trọng nhất của bản thiết kế. Reviewer sẽ tìm ra các điểm này, nên sửa trước.

### P1. Số hiệu giả thuyết không thống nhất
* `HYPOTHESES_AND_MATHEMATICAL_METHODS.md` đặt H1 = Signal Superiority, H2 = Statistical Validity.
* Hai tài liệu v1.2 mới hơn (13–14/10) đặt **H1 = Statistical Guarantee (LTT), H2 = Signal (FCD)**, H3 = AL Utility.
* **Quyết định:** dùng thứ tự của bản v1.2 (thước đo trước, tín hiệu sau). Cập nhật lại file HYPOTHESES cho khớp.

### P2. Mức δ không thống nhất
* Bản cũ: $\delta = 0.1$. Bản v1.2: $\delta = 0.05$. **Dùng $\delta = 0.05$** ở mọi nơi. Mọi bảng $k_{\max}$ đã được tính với $0.05$.

### P3. "FCD" đang có ba định nghĩa khác nhau
| Nơi | Tên | Công thức thực tế |
|---|---|---|
| `M29_MASTER_SPECIFICATION.md` | Feature-Confidence Dissonance | Mahalanobis của **một** mô hình (anchor) tới tâm lớp, so với phân vị 95%. Đây thực chất là Baseline 4 |
| `H2_the_signal_hypothesis.md` | Relative Latent Discrepancy Variance | Phương sai của $(D_1, D_2, D_3)$, mỗi $D_m$ là Mahalanobis trong không gian của mô hình $m$ |
| `fcd_variance.py` | (không tên) | Mahalanobis của 3 vector latent **của 3 mô hình khác nhau** so với trung bình của chính 3 vector đó, với ma trận hiệp phương sai ước lượng từ 3 điểm trong $\mathbb{R}^{256}$ |

Định nghĩa trong code **không dùng được**: ma trận hiệp phương sai từ 3 điểm có hạng tối đa 2 trong 256 chiều, nên nghịch đảo của nó gần như chỉ còn jitter $\epsilon^{-1}$. Thêm nữa, so sánh tọa độ latent của các mô hình khác seed là vô nghĩa, vì mỗi mô hình có hệ trục riêng (Li et al., 2016; Kornblith et al., 2019). Paper phải chọn **một** định nghĩa và code phải khớp (xem P4).

### P4. Tín hiệu phương sai thuần $U = \operatorname{Var}(D_1, D_2, D_3)$ có điểm mù nghiêm trọng
Khi cả ba mô hình **cùng** đặt một mẫu lạ ở xa tâm lớp (ví dụ $D = (14.0, 14.6, 13.8)$), phương sai gần bằng 0 và mẫu được chấp nhận làm nhãn giả. Đây lại chính là loại mẫu lạ dễ thấy nhất. Mô phỏng trong [02, Mục D5.4](02_MATH_EXPLAINED.md) cho thấy: khi lệch đồng thuận, tín hiệu phương sai bắt được đúng 5%, bằng tỉ lệ báo động nhầm, tức không bắt được gì. Trong khi đó trung bình, max hoặc trung bình + độ lệch chuẩn bắt được gần 100%.
* **Đề xuất (bạn cần quyết định):** tín hiệu chính là $U_{\text{fcd}} = \bar D + \lambda\, s_D$ (hoặc $\max_m D_m$), với $\lambda$ chọn trên Meta-Train. Phương sai thuần giữ lại làm **ablation** để trả lời câu hỏi "bất đồng giữa các mô hình có thêm thông tin ngoài khoảng cách trung bình không?".
* Ví dụ minh họa trong tài liệu cũ ($D_1 = 2.1$, $D_2 = 6.8$, $D_3 = 5.2$) sai thang đo. Trong 64 chiều, Mahalanobis của mẫu **bình thường** tập trung quanh $\sqrt{64} \approx 8$ (độ lệch chuẩn khoảng 0.7). Cả ba số trên đều **nhỏ hơn** mức của mẫu bình thường. Xem [02, Mục D4.5](02_MATH_EXPLAINED.md).

### P5. "Theorem 2" trong tài liệu H2 chưa phải là định lý
Các bước chứng minh dựa trên những khẳng định chưa được chứng minh:
1. "Trọng số khởi tạo trực giao nên biểu diễn OOD của các mô hình độc lập": khởi tạo gần trực giao không suy ra được ánh xạ sau huấn luyện độc lập. Thực nghiệm còn cho thấy biểu diễn của các seed khác nhau **rất giống nhau** sau một phép biến đổi tuyến tính (Kornblith et al., 2019, chỉ số CKA).
2. Công thức $\mathbb{E}[s^2] = \frac{1}{M}\sum_m \operatorname{Var}(D_m)$ thiếu số hạng độ phân tán của các kỳ vọng $\frac{1}{M-1}\sum_m(\mu_m - \bar\mu)^2$, và cần giả định độc lập không có căn cứ.
3. "$\operatorname{Var}(D_m) = O(\Delta^2)$" được khẳng định chứ không được suy ra.
4. Phát biểu "Strict Pareto Dominance với mọi $\alpha$" là một kết luận thực nghiệm, không suy ra được từ giả định hình học.

**Thay bằng:** một **Proposition** phát biểu trong mô hình đồ chơi có giả định rõ ràng (đã viết sẵn trong [02, Mục D5.5](02_MATH_EXPLAINED.md)), kèm câu "ưu thế trên dữ liệu thật là một giả thuyết thực nghiệm, được kiểm ở H2". Bỏ các câu kiểu "bắt trúng 100%", "triệt tiêu hoàn toàn", "vô địch".

### P6. Các con số kết quả viết sẵn
* `H2_the_signal_hypothesis.md` Mục 9.3, `HYPOTHESES...md` Phần IV, `extra_final_architecture_plan.md` (mAP 62.4 → 66.8, Night +13.4, ...) có các con số trông như kết quả nhưng chưa được đo.
* Trong bản đăng ký trước, chỉ được ghi **dạng** bảng và **tiêu chí quyết định**. Thay mọi con số bằng `[—]`. Nếu muốn có kỳ vọng định lượng, ghi rõ "giá trị dự kiến dùng để tính cỡ mẫu" và tách khỏi bảng kết quả.

### P7. RT-DETR dùng sigmoid, không dùng softmax
* RT-DETR huấn luyện đầu phân loại bằng varifocal loss trên từng lớp với **sigmoid** (Zhao et al., 2024), không có lớp "no-object" như DETR gốc. Logit đầu ra không phải phân phối trên simplex.
* Hệ quả: $\hat p_j = \max_c \sigma(l_{j,c})$. Lập luận "chuẩn hóa ép tổng bằng 1" của Baseline 1 không áp dụng. Entropy và JS của Baseline 2 phải tính trên điểm sigmoid đã chuẩn hóa $\tilde p_{j,c} = \sigma(l_{j,c}) / \sum_k \sigma(l_{j,k})$, hoặc dùng tổng entropy nhị phân theo lớp. Phải đăng ký trước cách nào.
* Checkpoint `PekingU/rtdetr_r50vd` có 80 lớp COCO. Detector phải được **fine-tune lại đầu phân loại về 8 lớp BDD100K** trên `Train_filtered`. COCO không có lớp `rider`, nên không thể ánh xạ trực tiếp.

### P8. Các thành phần của $U_{\text{out}}$ lệch thang đo
* $\sigma^2_{\text{loc}}$ tính trên tọa độ chuẩn hóa $[0,1]$: lệch 0.02 cho phương sai khoảng $4\times 10^{-4}$. $D_{\text{JS}}$ nằm trong $[0, \ln C] \approx [0, 2.08]$. Với $\beta = 1$, thành phần vị trí gần như không đóng góp. Spec M29 dùng hệ số 10, cũng tùy ý.
* **Sửa:** (i) đo độ lệch hộp theo kích thước hộp, giống tham số hóa hộp của Fast R-CNN (Girshick, 2015): $\Delta x / w$, $\Delta y / h$, $\Delta \log w$, $\Delta \log h$; (ii) đưa mỗi thành phần về thang phân vị bằng hàm phân phối thực nghiệm (ECDF) trên Meta-Train, rồi lấy max hoặc trung bình. Xem [02, Mục D2.4](02_MATH_EXPLAINED.md).

### P9. So sánh công bằng giữa 5 tín hiệu
* **Cùng tập ứng viên:** mọi tín hiệu lọc **cùng một** tập box ứng viên (các query của anchor có $\hat p \ge 0.3$, sau khi bỏ lớp ngoài $\mathcal{C}_{\text{eval}}$). Nếu không, Coverage khác nhau một phần vì tập ứng viên khác nhau.
* **EDL là một detector khác:** nó cần huấn luyện lại với loss riêng, nên tập ứng viên và mAP của nó khác. Báo cáo mAP của detector EDL, và thêm một chỉ số chuẩn hóa: Coverage chia cho recall của tập ứng viên.
* **Chi phí:** báo cáo chi phí GPU (1× hay 3×), và thêm baseline "ensemble 3 mô hình nhưng chỉ dùng confidence trung bình" để tách hiệu ứng "nhiều mô hình" khỏi hiệu ứng "không gian latent".

### P10. Kiểm định của H3 thiếu năng lực thống kê
* Paired t-test trên 3 seed chỉ có 2 bậc tự do. Với độ lệch chuẩn giữa các seed $\sigma_d = 1$ điểm, công suất phát hiện chênh 1.5 điểm chỉ là 0.53 ([03, Mục 5.2](03_STATISTICAL_ANALYSIS_PLAN.md)). Cần **ít nhất 8 seed** nếu $\sigma_d \le 1.5$ (Bouthillier et al., 2021), kèm khoảng tin cậy bootstrap theo ảnh test trong từng seed.
* Phát biểu "$\forall t \in \{1..T\}$" là một kiểm định giao-hợp (intersection–union test, Berger, 1982): bác bỏ $H_0$ khi **từng** vòng đều có ý nghĩa ở mức $\alpha$, và không cần hiệu chỉnh Bonferroni. Đăng ký trước endpoint chính là vòng cuối $T$, và diện tích dưới đường học (ALC) là endpoint phụ.

### P11. Tính hợp lệ của LTT trong vòng AL (H3)
* Ở vòng $t$, ngưỡng được hiệu chuẩn trên $\mathcal{D}^{(t)}_{\text{cal}}$ quyết định ảnh nào thành nhãn giả, và nhãn giả đó dùng để huấn luyện mô hình vòng $t+1$. Vì vậy mô hình $\theta_{t+1}$ **phụ thuộc** vào $\mathcal{D}^{(t)}_{\text{cal}}$. Dùng lại cùng tập hiệu chuẩn ở vòng $t+1$ vi phạm A2 (detector phải độc lập với tập hiệu chuẩn).
* **Sửa:** mỗi vòng dùng một tập hiệu chuẩn **mới, rời rạc** $\mathcal{D}^{(t)}_{\text{cal}}$ (đúng tinh thần "Adaptive LTT" trong RESEARCH_PROPOSAL). Nếu muốn bảo chứng đồng thời cho cả $T$ vòng, dùng $\delta / T$ mỗi vòng (bất đẳng thức hợp). Chi phí gán nhãn của các tập hiệu chuẩn phải được tính vào ngân sách khi so sánh với Random.

### P12. Phân vị 95% của FCD gate trong M29 và báo động nhầm theo ảnh
* Spec M29 viết: "$D_M > \tau_{\text{fcd}}$ ⟹ p-value < 0.05 ⟹ kết luận toán học: ảo giác". Câu đúng phải là: **"mẫu này nằm ngoài 95% mẫu bình thường"**. Mỗi box bình thường vẫn có 5% khả năng bị gắn cờ.
* Ảnh có 10 box bình thường độc lập bị gắn cờ với xác suất $1 - 0.95^{10} \approx 40\%$, ảnh có 20 box là khoảng 64%. Nếu cổng hoạt động theo ảnh (một box bị cờ thì cả ảnh sang Red), cần hiệu chuẩn trực tiếp ngưỡng của $\max_j D_j$ theo ảnh, hoặc hiệu chỉnh theo số box. Xem [02, Mục D6](02_MATH_EXPLAINED.md).

### P13. Lỗi nhỏ trong code cần sửa trước khi chạy
* `fcd_variance.py` và `output_variance.py` gọi `linear_sum_assignment` **bên trong vòng lặp từng query**, tức 300 lần cho mỗi ảnh. Chỉ cần gọi một lần cho mỗi cặp (anchor, member).
* `output_variance.py` chỉ tính entropy của phân phối trung bình (tổng bất định), không tính JS/mutual information như tài liệu mô tả.
* Ma trận chi phí trong code chỉ dùng L1, trong khi tài liệu dùng L1 + GIoU + JS. Chọn một và ghi vào paper.
* Box của member không khớp được gán `0`. Điều này làm phương sai tọa độ lớn một cách tùy ý. Nên tách thành một biến riêng "số member không tìm thấy" (Miller et al., 2019 bàn về các chiến lược gộp này).

---

## 2. Tên paper, câu chuyện và đóng góp

### 2.1. Gợi ý tên (chọn một)
1. *Certified Pseudo-Labels for Driving Video: Risk-Controlled Selection with Ensemble Latent Uncertainty*
2. *How Much Can We Safely Auto-Label? A Risk-Certified Benchmark of Uncertainty Signals for Object Detection in Driving Video*
3. *From Guarantees to Gains: Ride-Level Risk Control, Latent Ensemble Uncertainty, and Human Routing for Active Learning in Autonomous Driving*

Tên 2 hợp NeurIPS D&B (nhấn benchmark). Tên 3 hợp T-IV (nhấn hệ thống đầu-cuối).

### 2.2. Câu chuyện một đoạn (dùng cho Introduction)
Tự động gán nhãn giả giúp giảm chi phí gán nhãn cho xe tự hành, nhưng mỗi nhãn sai đưa vào tập huấn luyện sẽ bị mô hình học lại và khuếch đại (confirmation bias; Arazo et al., 2020). Cách làm phổ biến là chọn một ngưỡng confidence (ví dụ 0.9) mà không có bảo chứng nào về tỉ lệ nhãn sai. Learn-then-Test (LTT) cho phép chọn ngưỡng với bảo chứng hữu hạn mẫu, nhưng bảo chứng chỉ đúng khi các đơn vị hiệu chuẩn độc lập. Trong video lái xe, các frame liền kề gần như trùng nhau. Chúng tôi chỉ ra rằng (i) hiệu chuẩn trên đơn vị chuyến đi giữ đúng bảo chứng, còn chia theo frame thì phá vỡ nó; (ii) khi bảo chứng đã đúng, sức mạnh tách biệt của tín hiệu bất định quyết định lượng nhãn giả thu được, và tín hiệu latent của ensemble cho nhiều nhãn hơn tín hiệu ở đầu ra; (iii) đưa những mẫu bị từ chối cho người gán nhãn, kết hợp nhãn giả đã được chứng nhận, giúp mAP tăng nhanh hơn các chiến lược chọn mẫu thông thường.

### 2.3. Đóng góp (viết ở cuối Introduction)
1. **Giao thức hiệu chuẩn có bảo chứng cho video lái xe** (H1): LTT + fixed-sequence trên đơn vị chuyến đi/super-block, quy tắc chọn đơn vị thống kê (R-unit), và bằng chứng thực nghiệm về sự sụp đổ của bảo chứng khi chia theo frame.
2. **Benchmark tín hiệu bất định dưới rủi ro được chứng nhận** (H2): so sánh 5 tín hiệu theo thiết kế giai thừa 2×2 (đầu ra/latent × một mô hình/ensemble) bằng chỉ số *Coverage tại rủi ro được chứng nhận*, thay vì AUROC.
3. **Tín hiệu latent của ensemble** (FCD) cho object detection dựa trên DETR, kèm phân tích khi nào bất đồng giữa các mô hình mang thêm thông tin.
4. **Vòng AL khép kín** (H3): định tuyến mẫu bị từ chối cho người, và nhãn giả đã được chứng nhận cho máy, với tập hiệu chuẩn mới ở mỗi vòng.
5. **Tài nguyên mở**: ride proxy cho BDD100K, các phân chia dữ liệu, code và bản đăng ký trước.

---

## 3. Hệ ký hiệu thống nhất cho cả paper

| Ký hiệu | Nghĩa |
|---|---|
| $U_i$ | Đơn vị thống kê thứ $i$ (chuyến đi, hoặc super-block nếu quy tắc R-unit yêu cầu) |
| $\mathcal{D}_{\text{meta}}, \mathcal{D}_{\text{cal}}, \mathcal{D}_{\text{test}}$ | Meta-Train (dựng đường đi), tập hiệu chuẩn, tập kiểm thử. Rời rạc theo super-block |
| $f_{\theta_m}$, $m = 1..M$ | Detector RT-DETR thứ $m$ ($M = 3$); $m = 1$ là anchor |
| $j$ | Chỉ số box ứng viên (query của anchor, $\hat p_j \ge 0.3$) |
| $\mathbf z^{(m)}_j \in \mathbb R^{256}$, $\tilde{\mathbf z}^{(m)}_j \in \mathbb R^{64}$ | Hidden state cuối của decoder, trước và sau PCA |
| $S(j) \in \mathbb R$ | Điểm bất định (lớn = nghi ngờ). **Dùng $S$, không dùng $U$**, để tránh trùng với đơn vị $U_i$ |
| $\lambda \in \Lambda_{1D}$ | Ngưỡng; box được chấp nhận khi $S(j) \le \lambda$. **Dùng $\lambda$ cho ngưỡng**, như bài LTT gốc |
| $N^{\text{acc}}_i(\lambda)$, $V_i(\lambda)$ | Số box được chấp nhận và số FP trong đơn vị $i$ |
| $\text{FDP}_i(\lambda) = V_i / \max(N^{\text{acc}}_i, 1)$ | Tỉ lệ phát hiện sai của đơn vị $i$ |
| $\mathcal R(\lambda) = \mathbb E[\text{FDP}_i(\lambda)]$ | Rủi ro quần thể (Macro-FDP) |
| $\alpha = 0.02$, $\delta = 0.05$ | Mức rủi ro và xác suất thất bại cho phép |
| $\operatorname{Cov}(\lambda) = \sum_i \text{TP}_i / \sum_i M^{\text{gt}}_i$ | Coverage: tỉ lệ đối tượng GT được nhãn giả đúng bao phủ |

> Tài liệu cũ dùng $U$ cho cả đơn vị lẫn điểm bất định, và $\tau$ cho ngưỡng. Đổi sang $S$ và $\lambda$ sẽ khớp với ký hiệu của Angelopoulos et al. (2021) và tránh nhầm lẫn.

---

## 4. Ba giả thuyết chính thức (bản hợp nhất)

### H1: Bảo chứng thống kê (giữ nguyên nội dung v1.2)
* **H1a** (bảo chứng PAC): với đơn vị chuyến đi/super-block, LTT + fixed-sequence cho $\mathbb P(\mathcal R(\hat\lambda) \le \alpha) \ge 1 - \delta$ trên các super-block chưa thấy. Kiểm bằng C2 (tỉ lệ vi phạm VR $\le \delta$) và C2b (design effect thực nghiệm).
* **H1b** (chia theo frame sụp đổ): hiệu chuẩn trên frame cho VR $> \delta$ rõ rệt trên các clip mới. Kiểm ở $\alpha_C = 0.10$ trên BDD100K MOT.
* **H1c** (không tầm thường): $\mathbb P(\hat\lambda \ne \emptyset) \ge 0.85$, Coverage tương đối so với baseline plug-in $\ge 0.5$, **và** mật độ nhãn giả $\mathbb E[N^{\text{acc}}_i] \ge 1$ box/đơn vị (bổ sung theo bản review).

### H2: Tín hiệu (sửa theo P4, P5, P8, P9)
Mọi tín hiệu dùng cùng tập ứng viên, cùng quy trình LTT của H1 (cùng $\alpha, \delta, N$), đường đi riêng cho từng tín hiệu dựng trên Meta-Train.
* **H2a** (latent hơn đầu ra): $\operatorname{Cov}(S_{\text{maha}}) - \operatorname{Cov}(S_{\text{conf}}) \ge 5$ điểm **và** $\operatorname{Cov}(S_{\text{fcd}}) - \operatorname{Cov}(S_{\text{out}}) \ge 5$ điểm.
* **H2b** (ensemble latent hơn single latent): $\operatorname{Cov}(S_{\text{fcd}}) - \operatorname{Cov}(S_{\text{maha}}) \ge 5$ điểm. **Ablation đăng ký trước:** $\bar D$ ($\lambda = 0$), $s^2_D$ (phương sai thuần), $\max_m D_m$, $\bar D + \lambda s_D$.
* **H2c** (vượt baseline tốt nhất, an toàn): $S_{\text{fcd}}$ vượt từng baseline với Wilcoxon một phía, hiệu chỉnh Holm cho 4 so sánh ở mức $0.01$, và rủi ro thực đo trên pool test $\le 0.020$.
* **Quy tắc chấp nhận chung của H2a/b/c** (thay cho "$\widehat\Delta \ge 5$ điểm", xem [03, Mục 4.2](03_STATISTICAL_ANALYSIS_PLAN.md)): Wilcoxon có ý nghĩa sau Holm **và** bác bỏ $H_0: \Delta\text{Cov} \le 2$ điểm. Mốc 5 điểm là hiệu ứng mục tiêu dùng để chọn cỡ mẫu.

### H3: Lợi ích cho AL (thiết kế lại theo P10, P11)
Thiết kế giai thừa để tách hai thành phần:
* **H3a** (định tuyến cho người): với cùng chính sách nhãn giả, chọn $B$ ảnh cho người theo luật Red-zone (bị LTT/FCD từ chối, lọc đa dạng bằng K-means) cho $\Delta\text{mAP}_{50:95}$ ở vòng $T$ cao hơn Random và Least-Confidence. Hiệu ứng mục tiêu 1.5 điểm. Quy tắc chấp nhận: t có cặp theo ≥ 8 seed có ý nghĩa sau Holm, và cận dưới CI 95% $> 0.5$ điểm ([03, Mục 5](03_STATISTICAL_ANALYSIS_PLAN.md)).
* **H3b** (nhãn giả có chứng nhận): với cùng chính sách chọn cho người, nhãn giả từ ngưỡng LTT cho mAP không thấp hơn, và tỉ lệ nhãn giả sai thấp hơn, so với ngưỡng heuristic cố định ($\hat p \ge 0.9$).

---

## 5. Phân chia dữ liệu chung cho cả ba giả thuyết

Một sơ đồ phân chia duy nhất để H1, H2, H3 không rò rỉ sang nhau. Mọi phân chia đều **theo chuyến đi** (ride proxy, bước P1 của H1) hoặc theo super-block.

```
BDD100K Detection Train 70K ── loại mọi chuyến đi dính Val/MOT/Meta ──► Train_filtered
   │
   ├── L0: tập có nhãn ban đầu (ví dụ 10%) ─────────► huấn luyện 3 detector (3 seed) + detector EDL
   ├── Pool AL không nhãn (nhãn ẩn, chỉ "mở" khi người gán) ──► H3
   └── Cal_AL^(1..T): T tập hiệu chuẩn rời rạc cho từng vòng AL ──► H3 (P11)

BDD100K Detection Val 10K (theo super-block)
   ├── Meta-Train (10% super-block) ──► dựng đường đi Λ_1D cho từng tín hiệu, chọn λ của FCD, chọn d_pca
   ├── Cal pool  ──► Exp B (H1a), hiệu chuẩn 5 tín hiệu (H2)
   └── Test pool ──► Exp B (H1a), Table H2, đánh giá mAP của H3

BDD100K MOT (1.400 + 200 video) ──► Exp C (H1b, chia frame vs chia clip)
```

* Nhãn test 20K của BDD100K theo hiểu biết của chúng tôi không công khai (đánh giá qua server). Cần xác minh. Vì vậy mọi đánh giá dùng Val.
* Tập thời tiết khó (Night/Rain/Snow/Fog) lấy theo thuộc tính có sẵn trong nhãn BDD100K, chỉ dùng cho stress test (H2-D), không thuộc phạm vi bảo chứng (A3).

---

## 6. Bố cục paper theo từng section

Số trong ngoặc là ước lượng độ dài cho bản 9 trang. Bản T-IV có thể nới.

### Abstract (khoảng 200 từ, khung mẫu)
> Automatic pseudo-labeling can cut annotation cost for autonomous driving, but a single wrong label is amplified by retraining. We study selective pseudo-labeling for object detection with a finite-sample guarantee on the false discovery proportion (FDP). First, we show that Learn-then-Test calibration is valid when the statistical unit is the driving ride, and that frame-level calibration, the common practice, violates the guarantee in `[xx]%` of held-out evaluations at nominal 5%. Second, holding the certified risk fixed at 2%, we compare five uncertainty signals in a 2×2 design (output vs. latent space, single model vs. ensemble). An ensemble latent discrepancy signal certifies `[xx]%` of ground-truth objects as correct pseudo-labels versus `[xx]%` for softmax confidence. Third, routing rejected detections to human review while training on certified pseudo-labels improves mAP by `[xx]` points over random selection after three rounds on BDD100K. Code, ride-level splits and the pre-registration are released.

### 1. Introduction (1 trang)
Đoạn 1: bài toán và chi phí gán nhãn. Đoạn 2: confirmation bias và vì sao ngưỡng heuristic nguy hiểm. Đoạn 3: LTT và vấn đề phụ thuộc trong video. Đoạn 4: khi bảo chứng đúng, tín hiệu quyết định lượng nhãn thu được. Đoạn 5: đóng góp (Mục 2.3). **Hình 1**: sơ đồ ba hồi.

### 2. Related Work (0.75 trang)
* *Kiểm soát rủi ro không phụ thuộc phân phối:* conformal prediction (Angelopoulos & Bates, 2023), RCPS (Bates et al., 2021), LTT (Angelopoulos et al., 2021), CRC (Angelopoulos et al., 2024), selective classification có bảo chứng (Geifman & El-Yaniv, 2017), chọn mẫu bằng conformal p-value (Jin & Candès, 2023), áp dụng cho object detection (Andéol et al., 2023; Timans et al., 2024).
* *Dữ liệu không trao đổi được:* Barber et al. (2023), Oliveira et al. (2024), giả lặp (Hurlbert, 1984), lấy mẫu cụm (Kish, 1965).
* *Bất định trong object detection:* Miller et al. (2019), Feng et al. (2022), VOS (Du et al., 2022).
* *Tín hiệu bất định:* MSP (Hendrycks & Gimpel, 2017), Deep Ensembles (Lakshminarayanan et al., 2017), EDL (Sensoy et al., 2018), Mahalanobis (Lee et al., 2018), DDU (Mukhoti et al., 2023).
* *Nhãn giả và AL cho detection:* Pseudo-Label (Lee, 2013), STAC (Sohn et al., 2020), Unbiased Teacher (Liu et al., 2021), UPS (Rizve et al., 2021), CEAL (Wang et al., 2017), Core-set (Sener & Savarese, 2018), Haussmann et al. (2020), Choi et al. (2021), Elezi et al. (2022).
* **Câu định vị:** chưa công trình nào trong số trên kết hợp bảo chứng hữu hạn mẫu theo đơn vị đúng của video với so sánh tín hiệu có kiểm soát và vòng AL.

### 3. Problem Setup (0.5 trang)
Định nghĩa ứng viên, TP/FP (ghép tham lam kiểu COCO, $\theta_{\text{IoU}}$ chọn theo quy tắc của H1), FDP, Macro- và Micro-FDP, Coverage, mục tiêu $\max_\lambda \operatorname{Cov}(\lambda)$ s.t. $\mathbb P(\mathcal R(\hat\lambda) \le \alpha) \ge 1 - \delta$. Nội dung lấy từ H1 Mục II.

### 4. Method (2 trang)
* **4.1 Hiệu chuẩn có bảo chứng trên đơn vị chuyến đi:** p-value HB, fixed-sequence, đường đi Pareto trên Meta-Train, Theorem 1 (chứng minh đưa vào phụ lục), quy tắc R-unit. Giải thích chi tiết: [02, Phần B](02_MATH_EXPLAINED.md).
* **4.2 Năm tín hiệu bất định:** bảng 2×2 cùng một dòng công thức cho mỗi tín hiệu. Hungarian alignment. Chi tiết: [02, Phần C–D](02_MATH_EXPLAINED.md).
* **4.3 Tín hiệu latent của ensemble (FCD):** định nghĩa, Proposition về hai kiểu lệch (lệch một phần và lệch đồng thuận), lý do chọn $\bar D + \lambda s_D$.
* **4.4 Định tuyến và vòng AL:** ba vùng Red/Green/Grey, K-means đa dạng, WBF tạo box mồi, tập hiệu chuẩn mới mỗi vòng. Chi tiết: [02, Phần F](02_MATH_EXPLAINED.md).

### 5. Experiments (3 trang)
* **5.1 Thiết lập chung:** dữ liệu (Mục 5 ở trên), detector, compute, các mức đăng ký trước.
* **5.2 H1:** Exp B (VR, Yield, Coverage theo super-block, B-rand và B-block), Exp C (VR theo $N_{\text{cluster}}$ của 3 arm). **Bảng 1, Hình 2.**
* **5.3 H2:** H2-A Bảng chính, H2-B giai thừa 2×2, H2-C giải phẫu FP (ảo giác hay lệch vị trí), H2-D stress test thời tiết. **Bảng 2, Hình 3 (đường risk–coverage), Hình 4.**
* **5.4 H3:** đường học theo vòng, 4 arm × ≥5 seed. **Bảng 3, Hình 5.**

### 6. Discussion & Limitations (0.5 trang)
Ride proxy không có device ID (rò rỉ dư 20–25% theo mô phỏng), bảo chứng là bảo chứng biên (không có điều kiện theo từng tập hiệu chuẩn), không bảo chứng dưới dịch chuyển phân phối, chi phí 3× của ensemble, giới hạn của mô hình đồ chơi trong Proposition.

### 7. Conclusion (0.25 trang)

### Phụ lục
A. Chứng minh Theorem 1, Lemma A4, Proposition 3 (folded splitting). B. Bảng $k_{\max}$ theo $N$ (5.4, 5.5) và giải thích mốc $N \ge 149$. C. Exp A (ICC, Moran's I, quy tắc R-unit). D. Mô phỏng edge case (tóm tắt `H1_simulation_report.md`). E. Chi tiết 5 tín hiệu (toàn bộ công thức). F. Siêu tham số, lưới, phần cứng. G. Bản đăng ký trước và nhật ký thay đổi.

---

## 7. Kế hoạch hình và bảng (khuôn, chưa có số)

**Bảng 1 (H1):**

| Arm | Đơn vị | $N$ | VR [CI 95%] | Yield | Coverage | Kết luận |
|---|---|---|---|---|---|---|
| B-rand | chuyến đi | 1.500 | [—] | [—] | [—] | C2 |
| B-block | super-block | 1.500 | [—] | [—] | [—] | C2, C2b |
| C-Arm1 | clip | {20..800} | [—] | | | |
| C-Arm2 | frame (bỏ tương quan) | {4K..160K} | [—] | | | C5 |
| C-Arm3 | frame, test rò rỉ | | [—] | | | |

**Bảng 2 (H2):**

| Không gian | Tín hiệu | Số mô hình | GPU | $\widehat{\text{FDP}}_{\text{test}}$ | Coverage | $\Delta$ so với FCD | $p$ (Holm) |
|---|---|---|---|---|---|---|---|
| Đầu ra | MSP min-conf | 1 | 1× | [—] | [—] | [—] | [—] |
| Đầu ra | Ensemble disagreement | 3 | 3× | [—] | [—] | [—] | [—] |
| Đầu ra | EDL vacuity | 1 | 1× | [—] | [—] | [—] | [—] |
| Latent | Mahalanobis (anchor) | 1 | 1× | [—] | [—] | [—] | [—] |
| Latent | **FCD ($\bar D + \lambda s_D$)** | 3 | 3× | [—] | [—] | — | — |
| *Ablation* | $\bar D$, $s^2_D$, $\max_m D_m$, ensemble mean-conf | 3 | 3× | [—] | [—] | [—] | |

**Bảng 3 (H3):** mAP$_{50:95}$ ở vòng 0..T cho Random, Least-Conf, Ensemble-entropy, Proposed (trung bình ± độ lệch chuẩn qua seed), cùng số nhãn người, số nhãn giả, và tỉ lệ nhãn giả sai thực đo.

**Hình:** (1) sơ đồ ba hồi; (2) VR theo $N_{\text{cluster}}$ cho 3 arm, đường ngang tại $\delta$; (3) đường risk–coverage của 5 tín hiệu trên test, đánh dấu điểm $\hat\lambda$ được chứng nhận; (4) cột giai thừa 2×2; (5) đường học AL; (6) ví dụ định tính: box FCD bắt được và box bị lọt.

---

## 8. Kế hoạch phân tích thống kê (tóm tắt)

> **Bản đầy đủ, có số liệu công suất và cỡ mẫu tính sẵn, cùng ba tầng kiểm thử:** [03_STATISTICAL_ANALYSIS_PLAN.md](03_STATISTICAL_ANALYSIS_PLAN.md). Hai quy tắc quyết định ở Mục 4 đã được sửa theo bản đó: (i) H2 dùng kiểm định có biên $H_0: \Delta\text{Cov} \le 2$ điểm thay cho "$\widehat\Delta \ge 5$ và $p < 0.0025$" (quy tắc cũ chỉ có công suất ≈ 0.5 tại đúng hiệu ứng 5 điểm); (ii) H3 dùng ≥ 8 seed, kiểm vòng cuối cộng cận dưới CI $> 0.5$ điểm.

| Câu hỏi | Phép kiểm | Mức | Ghi chú |
|---|---|---|---|
| H1a | VR với Clopper–Pearson hoặc bootstrap theo phân hoạch | $\delta = 0.05$ | B-block dùng bootstrap vì các vòng chồng lấn |
| H1b | VR của Arm 2 so với $\delta$, kèm $D^{\text{frame}}_{\text{eff}}$ | | Tiêu chí C4, C5 của H1 |
| H2 | Wilcoxon signed-rank một phía trên $\Delta\text{TP}_i$ theo đơn vị test | Holm, 4 so sánh, $0.01$ | Phân tích phụ: phân phối $\Delta\operatorname{Cov}$ qua 100 lần chia lại cal/test, vì ngưỡng là ngẫu nhiên |
| H3a | Paired t-test trên $\Delta\text{mAP}$ theo seed (≥8 seed), và bootstrap theo ảnh test | $0.05$, Holm cho 3 so sánh | IUT nếu khẳng định cho mọi vòng |
| H3b | Không thua kém (non-inferiority) về mAP, biên 0.5 điểm, và tỉ lệ nhãn giả sai | $0.05$ | |

Toàn bộ ngưỡng, lưới, $\lambda$ của FCD, $d_{\text{pca}}$, $\theta_{\text{IoU}}$ được chốt trên Meta-Train **trước** khi mở dữ liệu cal/test, và ghi vào bản đăng ký trước.

---

## 9. Danh sách việc cần làm theo thứ tự

1. Sửa P1–P3 trong các tài liệu cũ (số hiệu, δ, một định nghĩa FCD duy nhất).
2. Quyết định công thức FCD chính (P4). Sửa `fcd_variance.py` cho khớp: Mahalanobis riêng cho từng mô hình, so với tâm lớp của chính mô hình đó.
3. Fine-tune 3 detector RT-DETR 8 lớp trên `Train_filtered` (P7). Huấn luyện detector EDL.
4. Dựng ride proxy, super-block, Meta-Train (bước P1–P2 của H1).
5. Chạy pilot trên Meta-Train: chọn $\theta_{\text{IoU}}$, $d_{\text{pca}}$, $\lambda$, đường đi. Đóng băng và đăng ký.
6. Chạy H1 (Exp A, B, C), rồi H2, rồi H3.
7. Viết paper theo Mục 6, điền số vào các bảng ở Mục 7.

Tài liệu tham khảo đầy đủ nằm cuối [02_MATH_EXPLAINED.md](02_MATH_EXPLAINED.md#tài-liệu-tham-khảo).
