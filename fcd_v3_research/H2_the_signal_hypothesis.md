# 🏛️ GIẢ THUYẾT 2 (H2): THE SIGNAL HYPOTHESIS
## Khung Lý Thuyết, Phân Tích Động Cơ & Thiết Kế Thực Chứng Cho Bài Báo Khoa Học

**Phiên bản 1.4 — Bản nháp đăng ký trước (Pre-registration Draft), 14/10/2026.**  
*Tài liệu nghiên cứu thuộc dự án FCD Autonomous Driving Active Learning Benchmark.*  
*Hội nghị / Tạp chí Mục tiêu: IEEE Transactions on Intelligent Vehicles (T-IV) / NeurIPS (Datasets and Benchmarks Track).*  
*Tài liệu liên quan:* `HYPOTHESIS_CHAIN.md` (chuỗi logic H1 → H2 → H3 và các điểm cần đồng bộ), `H2_simulation_report.md` (mô phỏng so sánh các phương án thiết kế), `H2_the_signal_hypothesis_v1.2_backup.md` (bản 1.2).

*Thay đổi ở bản 1.4 (dựa trên mô phỏng S1–S2 và cấu hình detector thực tế):*
1. **Ensemble cố định là 3 detector có sẵn** (`ai_router/experiments/expert_{1,2,3}.pth`). Bỏ pool 5 mô hình của bản 1.3; còn 3 cấu hình, mỗi cấu hình dùng một mô hình làm anchor. Mục 0 ghi lại cấu hình huấn luyện thực tế và các hệ quả: chỉ có lớp `car`, seed bị ghi đè, huấn luyện trên toàn bộ tập train.
2. **Tín hiệu đề xuất đổi thành dạng lai (hybrid):** điểm bất đồng đầu ra của ensemble cộng với độ lệch tiềm ẩn của ensemble. Trong mô phỏng S1, ở cả 18 kịch bản, mọi tín hiệu **chỉ dùng không gian tiềm ẩn** (Mahalanobis đơn, Ensemble-Mahalanobis, $\bar\zeta + \lambda s_\zeta$, và tín hiệu phương sai của bản 1.2) đều có Coverage thấp hơn nhiều so với điểm tin cậy. Nguyên nhân: tại mức FDP khoảng 1%, lỗi lệch vị trí gần như không lộ ra trong không gian tiềm ẩn. Ngược lại, thêm thông tin tiềm ẩn **vào** điểm tin cậy cho Coverage cao hơn điểm tin cậy đơn thuần ở mọi kịch bản. Thiết kế 2×2 vì vậy được đặt lại thành: *có hoặc không thêm thông tin tiềm ẩn* × *đơn mô hình hoặc ensemble*.
3. **Giữ quy tắc thống kê của bản 1.3** (bootstrap theo super-block, quy tắc ba vùng). Trong mô phỏng S2, khi Δ thật bằng 0, Wilcoxon theo đơn vị của bản 1.2 cho $p < 0.0025$ ở 33% số lần, trong khi quy tắc của bản 1.3 không lần nào cho $\text{LCB} > 0$ (H2_simulation_report.md, S2).
4. **Đăng ký trước các dự đoán về kết quả** dựa trên mô phỏng: hiệu ứng ensemble giảm nhanh khi tương quan giữa các thành viên tăng (Mục 8.0, P-H2.2). Dự đoán này được kiểm trên Meta-Path trước khi mở dữ liệu đánh giá.

---

## 0. CẤU HÌNH DETECTOR THỰC TẾ & HỆ QUẢ CHO THIẾT KẾ

Kiểm tra trực tiếp 3 file trọng số và script huấn luyện `fcd_v3_research/src/kaggle_all_in_one_train.py` cho thấy:

| Đặc điểm | Thực tế | Hệ quả cho H2 | Hệ quả cho H1 |
|---|---|---|---|
| Số mô hình | 3 (RT-DETR-R50vd, fine-tune từ `PekingU/rtdetr_r50vd`, 5 epoch) | $M = 3$, chỉ có **một** bộ ba. Không ước lượng được biến thiên giữa các ensemble khác nhau | — |
| Lớp được huấn luyện | Chỉ `car` (`TARGET_CLASS = 'car'`, nhãn COCO id 3); đầu phân loại vẫn có 80 đầu ra COCO | $\mathcal{C}_{\text{eval}} = \{\text{car}\}$ với bộ detector này. Mọi thống kê theo lớp suy biến thành một lớp; EDL phải dùng 2 phạm trù (car / nền) | $\mathcal{C}_{\text{eval}}$ của H1 (8 lớp) không khớp; cần chọn: huấn luyện lại 8 lớp, hoặc thu phạm vi H1–H3 về lớp `car` |
| Seed | Script gọi `set_seed(42 + 100(k−1))`, nhưng `transformers.Trainer` (bản 4.57) gọi lại `set_seed(args.seed)` với `args.seed = 42` mặc định khi khởi tạo. Cả 3 mô hình xuất phát từ cùng trọng số pretrained | Nhiều khả năng cả 3 mô hình có cùng thứ tự dữ liệu; khác biệt chủ yếu đến từ tính không tất định của GPU. Trọng số khác nhau thật (lệch tương đối khoảng 16% ở backbone, 0.8% ở đầu phân loại) nhưng **độ đa dạng có thể thấp**, tức tương quan giữa thành viên cao | — |
| Dữ liệu huấn luyện | Toàn bộ split `train` của BDD100K | Meta-Train nằm **trong** dữ liệu huấn luyện: thống kê tham chiếu và đường đi dựng trên Meta-Train sẽ lạc quan. LTT vẫn hợp lệ (A5 của H1 chỉ yêu cầu đường đi độc lập với tập hiệu chuẩn và tập kiểm thử) | Vi phạm giao thức A2 của H1 (`Train_filtered`): không loại các chuyến anh em của Val, không giữ riêng Meta-Train |

**Khuyến nghị (cần nhóm quyết định trước khi đăng ký):** huấn luyện lại 3 detector trên `Train_filtered` của H1, đặt `TrainingArguments(seed=s, data_seed=s)` khác nhau cho từng mô hình, và chốt tập lớp (8 lớp hoặc chỉ `car`). Chi phí ước tính khoảng 40 phút mỗi mô hình trên Kaggle, theo ghi chú trong script. Nếu giữ nguyên 3 detector hiện tại, H2 vẫn chạy được theo tài liệu này, nhưng phải công bố ba giới hạn ở bảng trên. Mô phỏng S1 dự đoán rằng khi độ đa dạng thấp, H2b gần như chắc chắn không đạt (Mục 8.0, P-H2.2).

Tài liệu dưới đây viết cho trường hợp tổng quát $C = |\mathcal{C}_{\text{eval}}|$. Các chỗ khác biệt khi $C = 1$ được ghi rõ.

---

### BẢN ĐỒ QUAN HỆ VỚI H1, H3 (chi tiết ở `HYPOTHESIS_CHAIN.md`)
* **Nhận từ H1:** định nghĩa đơn vị (quy tắc R-unit), $N$, $\theta_{\text{IoU}}$, $\mathcal{C}_{\text{eval}}$, lưới rủi ro và thuật toán dựng đường đi, các phân hoạch B-block của Exp B, quy tắc C2. **Điều kiện tiên quyết:** H1a không bị Bác bỏ. Nếu H1a bị Bác bỏ, H2 chỉ được báo cáo như phân tích mô tả.
* **Giao cho H3:** tín hiệu $U^\star$ dùng để định tuyến (chọn theo quy tắc đăng ký trước ở Mục 9.2), cùng quy trình hiệu chuẩn. H3 không dùng lại tập đánh giá của H2 để đo mAP.

---

## I. ĐẶT VẤN ĐỀ & PHÁT BIỂU CHÍNH THỨC

### 1.1. Bối cảnh & Động lực
Trong hệ thống gán nhãn giả có chọn lọc, tập nhãn giả phải có FDP thấp ($\alpha = 0.02$) để tránh thiên kiến xác nhận tích lũy khi huấn luyện lại. Câu hỏi thực tế là ở mức rủi ro được chứng nhận đó, giữ được bao nhiêu nhãn đúng:
$$\max_{\tau} \operatorname{Cov}(\tau) \quad \text{với ràng buộc} \quad \mathcal{R}(\tau) \le \alpha.$$
Dưới LTT, Coverage tại $\hat\tau$ phản ánh năng lực phân tách TP/FP của tín hiệu **ở vùng đuôi rủi ro thấp**. Tại $N = 1.500$, HB chỉ chứng nhận được các điểm có rủi ro thực nghiệm không quá khoảng 1.2% (H1, bảng 5.4).

**Bản chất bài toán:** đây là bài toán phát hiện lỗi (failure detection) trên các phát hiện của detector, không phải phát hiện OOD. FP gồm lỗi lệch vị trí (đúng vật thể, IoU dưới $\theta_{\text{IoU}}$), lỗi trùng lặp, và ảo giác. Jaeger et al. (2023) cho thấy với bài toán phát hiện lỗi, điểm tin cậy của chính mô hình thường thắng các phương pháp thiết kế cho OOD. Mô phỏng S1 lặp lại kết luận này cho bài toán của chúng ta: tín hiệu chỉ dùng không gian tiềm ẩn thua điểm tin cậy ở mọi kịch bản. Vì vậy, câu hỏi khoa học được đặt lại như sau:

> *Không gian tiềm ẩn có mang thông tin về lỗi mà điểm tin cậy **không có** hay không, và ensemble có khuếch đại thông tin đó hay không?*

### 1.2. Phát biểu Giả thuyết H2
Mọi so sánh dùng cùng quy trình: LTT fixed-sequence với HB, $\alpha = 0.02$, $\delta = 0.05$, trên các vòng B-block của H1 Exp B. Coverage đo trên tập kiểm thử của từng vòng (Mục 8.1). $\Delta$ ký hiệu hiệu Coverage tuyệt đối (pp), trung bình qua các vòng và 3 cấu hình anchor. Biên $\gamma = 5.0$ pp là hiệu ứng nhỏ nhất có ý nghĩa thực tiễn (SESOI), không được điều chỉnh theo kết quả pilot.

Bốn ô chính của thiết kế (Mục III): $U_{\text{conf}}$ (B1), $U_{\text{ens-out}}$ (B2), $U_{\text{hyb-s}}$ (B8, lai đơn mô hình), $U_{\text{fcd}}$ (đề xuất, lai ensemble).

> **H2a (Thông tin tiềm ẩn bổ sung cho điểm tin cậy):** *Ở cả hàng đơn mô hình và hàng ensemble, thêm thành phần tiềm ẩn vào tín hiệu đầu ra làm tăng Coverage:*
> $$\Delta_{\text{lat}}^{\text{single}} = \operatorname{Cov}(U_{\text{hyb-s}}) - \operatorname{Cov}(U_{\text{conf}}) \ge \gamma, \qquad \Delta_{\text{lat}}^{\text{ens}} = \operatorname{Cov}(U_{\text{fcd}}) - \operatorname{Cov}(U_{\text{ens-out}}) \ge \gamma.$$

> **H2b (Hiệu ứng ensemble trên tín hiệu lai):**
> $$\Delta_{\text{ens}}^{\text{hyb}} = \operatorname{Cov}(U_{\text{fcd}}) - \operatorname{Cov}(U_{\text{hyb-s}}) \ge \gamma.$$
> *Phân tích phụ H2b′ (không vào kết luận gộp): nếu $\lambda^* > 0$, thành phần độ phân tán $s_\zeta$ đóng góp thêm, tức $\operatorname{Cov}(U_{\text{fcd}}) - \operatorname{Cov}(U_{\text{fcd}}|_{\lambda = 0}) > 0$.*

> **H2c (Vượt trội so với mọi baseline):** *Tín hiệu đề xuất vượt từng baseline trong $\mathcal{B} = \{B1, \dots, B8\}$ với biên $\gamma$ (phép kiểm intersection-union; Berger, 1982):*
> $$\min_{k \in \mathcal{B}} \big[\operatorname{Cov}(U_{\text{fcd}}) - \operatorname{Cov}(U_k)\big] \ge \gamma.$$
> *Vì $\mathcal{B}$ chứa B8 (lai đơn mô hình) và B7 (lai TTA, cân bằng compute), H2c chỉ đạt khi ensemble thực sự có giá trị vượt qua cả phương án một mô hình tốn cùng compute.*

> **H2d (Hợp lệ rủi ro):** *Mọi tín hiệu dùng trong so sánh giữ được bảo chứng của H1 trên Evaluation pool, theo quy tắc ba vùng của C2 ở H1.*

*Ghi chú về giả thuyết cũ "không gian tiềm ẩn vượt không gian đầu ra":* so sánh tín hiệu chỉ dùng không gian tiềm ẩn (B4, B6, A2) với B1 vẫn được báo cáo ở Exp H2-B như phân tích mô tả, kèm dự đoán đăng ký trước là chúng **thua** B1 (theo S1 và Jaeger et al., 2023). So sánh này không còn là giả thuyết xác nhận, vì dự đoán thuận chiều của bản 1.2 mâu thuẫn với cả tài liệu lẫn mô phỏng.

### 1.3. Định vị Đóng góp
1. **Thiết kế 2×2 tách hai câu hỏi:** không gian tiềm ẩn có bổ sung thông tin cho điểm tin cậy không, và ensemble có khuếch đại thông tin đó không. Ngân sách tinh chỉnh được giới hạn cho mọi phương pháp, và mọi tín hiệu dùng một tập ứng viên chung.
2. **Thước đo dựa trên rủi ro được chứng nhận:** theo hiểu biết của chúng tôi, các so sánh tín hiệu bất định cho phát hiện vật thể trên dữ liệu lái xe thường dùng AUROC hoặc ngưỡng chọn thủ công. Ở đây điểm so sánh là ngưỡng do LTT chọn, tức điểm vận hành thực tế của hệ thống gán nhãn giả.
3. **Đối chứng cân bằng compute (B7, TTA-3)** để tách giá trị của ensemble khỏi giá trị của việc tốn gấp 3 lần compute.

---

## II. HỆ THỐNG KÝ HIỆU

| Ký hiệu | Ý nghĩa | Miền giá trị |
|---|---|---|
| $\mathcal{F} = \{f_1, f_2, f_3\}$ | 3 detector RT-DETR (Mục 0) | Trọng số đóng băng |
| $a \in \{1, 2, 3\}$ | Mô hình anchor của một cấu hình. Có 3 cấu hình | — |
| $M = 3$, $Q = 300$ | Số thành viên; số object query | — |
| $\mathcal{A}$ | Tập ứng viên của anchor trên một ảnh (Mục 2.2) | $|\mathcal{A}| \le 100$ |
| $s_{j,c}^{(m)} = \sigma(l_{j,c}^{(m)})$ | Điểm sigmoid theo lớp của query khớp với $j$ ở mô hình $m$ (varifocal loss, không có softmax) | $[0, 1]$ |
| $s_j$, $\hat y_j$ | Điểm tin cậy và lớp dự đoán của anchor, lấy trên $\mathcal{C}_{\text{eval}}$ | — |
| $\mathbf{z}_j^{(m)} \in \mathbb{R}^{256}$ | Đặc trưng ra của tầng decoder cuối, trước đầu phân loại | — |
| $\pi_m$ | Phép gán Hungarian từ ứng viên của anchor sang query của $m$ (Mục 2.3) | — |
| $D_m(j)$, $\zeta_m(j)$ | Khoảng cách Mahalanobis và bản chuẩn hóa về thang ID của mô hình $m$ (Mục 5.4) | — |
| $\operatorname{rz}(x)$ | Chuẩn hóa robust: $(x - \operatorname{med})/(1.4826\,\operatorname{MAD})$, với med và MAD tính trên các ứng viên TP của Meta-Fit | — |
| $U(j)$ | Điểm bất định; càng lớn càng bất định; chấp nhận khi $U(j) \le \tau$ | — |
| $\text{FDP}_i$, $\mathcal{R}$, $\operatorname{Cov}$ | Như H1 Mục II; $\operatorname{Cov} = \sum_i \text{TP}_i / \sum_i M_i^{\text{gt}}$ | $[0, 1]$ |
| $\Lambda_{1D}(U)$ | Đường đi $K_\Lambda = 33$ ngưỡng của $U$, dựng trên Meta-Path | — |

### 2.1. Đối sánh nhãn chuẩn
Dùng nguyên văn H1 Mục 2.1: $\mathcal{C}_{\text{eval}}$, vùng ignore, ghép tham lam theo $s_j$ giảm dần với quy tắc phá hòa cố định, $\theta_{\text{IoU}}$ do quy tắc của H1 chọn, và phân rã $V_i = V_i^{\text{halluc}} + V_i^{\text{loc}}$. Mọi tín hiệu dùng chung thứ tự ghép, nên nhãn TP/FP của từng ứng viên không phụ thuộc vào tín hiệu.

### 2.2. Tập ứng viên chung
Tập ứng viên gồm tối đa 100 query của anchor có $s_j$ lớn nhất, với $s_j \ge 0.05$. Mọi tín hiệu lọc trên cùng tập này, cùng hộp và cùng lớp của anchor, nên trần Coverage $\operatorname{Cov}_{\max}$ giống nhau cho mọi tín hiệu trong một cấu hình.

### 2.3. Gióng hàng ensemble
Thuật toán Hungarian trên ma trận chi phí $|\mathcal{A}| \times Q$:
$$C_{j,v}^{(m)} = 5\,\big\|\hat{\mathbf{b}}_j^{(a)} - \hat{\mathbf{b}}_v^{(m)}\big\|_1 + 2\,\mathcal{L}_{\text{giou}}\big(\hat{\mathbf{b}}_j^{(a)}, \hat{\mathbf{b}}_v^{(m)}\big) + 2\,\big(1 - s_{v,\hat y_j}^{(m)}\big).$$
Nếu $\operatorname{IoU} < 0.2$, ứng viên được coi là **không có đối ứng** ở mô hình $m$. Tỷ lệ không có đối ứng được báo cáo, tách theo TP và FP.

---

## III. THIẾT KẾ 2×2 & CÁC BASELINE

```
                           CHỈ ĐẦU RA                               ĐẦU RA + TIỀM ẨN (LAI)
               ┌──────────────────────────────────────┬──────────────────────────────────────────┐
 ĐƠN MÔ HÌNH   │ B1  U_conf = 1 − s_j                 │ B8  U_hyb-s = rz(U_conf) + κ·rz(ζ_a)     │
 (1× forward)  ├──────────────────────────────────────┼──────────────────────────────────────────┤
 ENSEMBLE      │ B2  U_ens-out = (1 − s̄) +            │ ĐỀ XUẤT  U_fcd = rz(U_ens-out)           │
 (3× forward)  │       λ_out·(1 − IoU̅)                │            + κ·rz(ζ̄ + λ·s_ζ)            │
               └──────────────────────────────────────┴──────────────────────────────────────────┘
 Baseline khác:  B3 EDL (2 phạm trù khi C = 1) · B4 Mahalanobis đơn (chỉ tiềm ẩn) · B5 JointEnergy
                 B6 KNN tiềm ẩn · B7 TTA-3 lai (1 mô hình × 3 phép biến đổi ảnh, cùng công thức với đề xuất)
 Ablation:       A1 đề xuất với λ = 0 · A2 chỉ tiềm ẩn ensemble ζ̄ + λ·s_ζ · A3 phương sai của bản 1.2
                 A4 PCA-64 · A5 cực đại max_m ζ_m thay cho ζ̄ + λ·s_ζ
```

Hai cột khác nhau **đúng một thành phần**: số hạng tiềm ẩn $\kappa \cdot \operatorname{rz}(\cdot)$. Hai hàng khác nhau đúng ở chỗ dùng một mô hình hay ba mô hình. Nhờ vậy:
* $\Delta_{\text{lat}}^{\text{single}}$ và $\Delta_{\text{lat}}^{\text{ens}}$ đo giá trị của thông tin tiềm ẩn;
* $\Delta_{\text{ens}}^{\text{hyb}}$ và $\Delta_{\text{ens}}^{\text{out}} = \operatorname{Cov}(B2) - \operatorname{Cov}(B1)$ đo giá trị của ensemble;
* B7 so với tín hiệu đề xuất cho biết lợi ích của ensemble có tái tạo được bằng TTA với cùng compute hay không.

---

## IV. CƠ CHẾ (ĐỘNG CƠ, KHÔNG PHẢI CHỨNG MINH)

### 4.1. Không gian null của đầu phân loại
$\mathbf{l} = \mathbf{W}_{\text{cls}}\mathbf{z} + \mathbf{b}$ với $\mathbf{W}_{\text{cls}} \in \mathbb{R}^{C \times 256}$. Mọi độ lệch của $\mathbf{z}$ trong không gian null (ít nhất $256 - C$ chiều) không làm đổi logit nào, nên tín hiệu chỉ dùng logit không thấy được các độ lệch đó. Đây là lý do để **thêm** thông tin tiềm ẩn vào điểm tin cậy. Phép chiếu PCA lên các thành phần phương sai lớn có thể cắt bỏ đúng các hướng này (Kamoi & Kobayashi, 2020), nên bản 1.4 dùng đủ 256 chiều; PCA-64 chỉ là ablation A4.

### 4.2. Vì sao tín hiệu chỉ dùng không gian tiềm ẩn không đủ
RT-DETR được huấn luyện bằng varifocal loss (Zhang et al., 2021), trong đó mục tiêu điểm của mẫu dương là IoU với nhãn chuẩn. Vì thế $s_j$ mã hóa chất lượng định vị. Đặc trưng của một hộp lệch vị trí vẫn là đặc trưng của một chiếc xe thật, nên nằm gần đa tạp ID. Tại mức FDP khoảng 1%, một tín hiệu không thấy được lỗi lệch vị trí phải dùng ngưỡng rất chặt, và Coverage giảm mạnh. Mô phỏng S1 cho Coverage của các tín hiệu chỉ dùng tiềm ẩn từ 1% đến 31%, trong khi điểm tin cậy đạt 36–48% ở cùng kịch bản.

### 4.3. Thiên kiến chung của các thành viên
Ba thành viên chung kiến trúc, chung trọng số pretrained, chung dữ liệu, và với bộ detector hiện tại có thể chung cả thứ tự dữ liệu (Mục 0). Biểu diễn của các mạng như vậy rất giống nhau (Kornblith et al., 2019). Thiên kiến chung làm giảm cả độ bất đồng đầu ra lẫn độ phân tán tiềm ẩn. Mục VII định lượng hiệu ứng này qua tham số $\rho$.

### 4.4. EDL
Bằng chứng $\mathbf{e} = g(\cdot) \ge 0$ vẫn là hàm của đặc trưng, nên có cùng giới hạn như 4.1 nếu $g$ chỉ nhìn vào các hướng phân loại. Với $C = 1$, phân phối Dirichlet trên một phạm trù không có ý nghĩa; B3 dùng 2 phạm trù (car / nền), tức phân phối Beta.

---

## V. ĐẶC TẢ CÁC TÍN HIỆU

Thống kê tham chiếu (tâm lớp, hiệp phương sai, med/MAD, hàng xóm KNN, trọng số đầu EDL) được ước lượng trên **Meta-Fit** hoặc dữ liệu huấn luyện. Siêu tham số được chọn trên **Meta-Path** (Mục 8.0). Không có bước nào dùng tập hiệu chuẩn hay tập kiểm thử.

### 5.1. B1 — Độ tin cậy sigmoid
$U_{\text{conf}}(j) = 1 - s_j$ (biến thể sigmoid của MSP; Hendrycks & Gimpel, 2017).

### 5.2. B2 — Bất đồng đầu ra của ensemble (Lakshminarayanan et al., 2017)
Với $s_j^{(m)}$ là điểm của mô hình $m$ cho lớp $\hat y_j$ (bằng 0 nếu không có đối ứng), $\bar s_j = \frac{1}{M}\sum_m s_j^{(m)}$, và $\overline{\text{IoU}}_j$ là IoU trung bình giữa hộp của anchor và hộp của các thành viên (bằng 0 nếu không có đối ứng):
$$U_{\text{ens-out}}(j) = (1 - \bar s_j) + \lambda_{\text{out}}(1 - \overline{\text{IoU}}_j), \qquad \lambda_{\text{out}} \in \{0, 0.25, 0.5, 1, 2\}.$$

### 5.3. B3 — EDL trên đặc trưng đóng băng (Sensoy et al., 2018)
Một đầu phụ $g: \mathbb{R}^{256} \to \mathbb{R}^{C'}_{\ge 0}$ (MLP 2 tầng, softplus) huấn luyện trên đặc trưng đóng băng của anchor, với $C' = C + 1$ phạm trù (các lớp của $\mathcal{C}_{\text{eval}}$ và "nền"). Ứng viên khớp TP có nhãn là lớp của nó; ứng viên còn lại có nhãn "nền". $\boldsymbol\alpha_j = g(\mathbf{z}_j^{(a)}) + \mathbf{1}$, $S_j = \sum \alpha_{j,c}$. Tín hiệu:
$$U_{\text{edl}}(j) = \frac{C'}{S_j} + \frac{\alpha_{j,\text{nền}}}{S_j}.$$
Tức là độ rỗng bằng chứng cộng với niềm tin vào phạm trù nền. Hàm mất mát theo Sensoy et al. (2018): bình phương sai số kỳ vọng cộng KL với hệ số làm nóng $\min(1, t/10)$. Siêu tham số chọn trên Meta-Path là số epoch dừng sớm (tối đa 5 giá trị).

### 5.4. B4 — Mahalanobis đơn mô hình (chỉ tiềm ẩn; Lee et al., 2018)
Trên Meta-Fit, với mô hình $m$: tâm lớp $\boldsymbol\mu_{c,m}$ từ các ứng viên TP (lớp có dưới 50 mẫu dùng tâm tính trên 20.000 ảnh huấn luyện), và hiệp phương sai gộp $\boldsymbol\Sigma_m$ ước lượng co rút Ledoit-Wolf trên đủ 256 chiều.
$$D_m(j) = \big\|\mathbf{z}_j^{(m)} - \boldsymbol\mu_{\hat y_j, m}\big\|_{\boldsymbol\Sigma_m^{-1}}, \qquad \zeta_m(j) = \operatorname{rz}_m\big(D_m(j)\big), \qquad U_{\text{maha}}(j) = \zeta_a(j).$$
Với $C = 1$, chỉ có một tâm. Với dữ liệu ID xấp xỉ Gauss, $D \approx \chi_{256}$ có trung vị khoảng 16, nên mọi ngưỡng chỉ được đọc trên thang $\zeta$.

### 5.5. B5 — JointEnergy (Wang et al., 2021; Liu et al., 2020)
$U_{\text{energy}}(j) = -\sum_{c} \log(1 + e^{l_{j,c}^{(a)}})$, lấy tổng trên **toàn bộ** các đầu ra của đầu phân loại (80 đầu ra COCO với bộ detector hiện tại). Khi chỉ có một đầu ra có nghĩa, B5 gần như đơn điệu theo $s_j$; khi đó nó được báo cáo nhưng ghi rõ là gần trùng với B1.

### 5.6. B6 — KNN tiềm ẩn (Sun et al., 2022)
$U_{\text{knn}}(j) = \|\hat{\mathbf{z}}_j^{(a)} - \hat{\mathbf{z}}_{(k)}\|_2$ trên đặc trưng đã chuẩn hóa độ dài, với hàng xóm lấy từ tập đặc trưng TP của Meta-Fit (tối đa 200.000 vector). $k \in \{10, 50, 200\}$.

### 5.7. B7 — TTA-3 lai (cân bằng compute)
Anchor chạy trên ảnh gốc, ảnh lật ngang, và ảnh thu nhỏ còn 0.8 lần. Hai phiên bản biến đổi đóng vai trò hai "thành viên", được gióng hàng như 2.3. Tín hiệu có **cùng công thức** với tín hiệu đề xuất (5.9), cùng lưới siêu tham số, nhưng thống kê tham chiếu chỉ của mô hình anchor.

### 5.8. B8 — Lai đơn mô hình
$$U_{\text{hyb-s}}(j) = \operatorname{rz}\big(U_{\text{conf}}(j)\big) + \kappa\, \operatorname{rz}\big(\zeta_a(j)\big), \qquad \kappa \in \{0.25, 0.5, 1, 2, 4\}.$$

### 5.9. Tín hiệu đề xuất — Lai ensemble ($U_{\text{fcd}}$)
Với $\zeta_m(j)$ của từng thành viên (ứng viên không có đối ứng ở $m$ nhận $\zeta_m^{\text{cap}}$, là phân vị 99.9% của $\zeta_m$ trên Meta-Fit):
$$\bar\zeta(j) = \frac{1}{M}\sum_m \zeta_m(j), \qquad s_\zeta(j) = \sqrt{\frac{1}{M-1}\sum_m\big(\zeta_m(j) - \bar\zeta(j)\big)^2},$$
$$\boxed{\,U_{\text{fcd}}(j) = \operatorname{rz}\big(U_{\text{ens-out}}(j)\big) + \kappa\, \operatorname{rz}\big(\bar\zeta(j) + \lambda\, s_\zeta(j)\big)\,}$$
với $\lambda_{\text{out}}$ lấy giá trị đã chọn cho B2, $\lambda \in \{0, 0.25, 0.5, 1, 2\}$, $\kappa \in \{0.25, 0.5, 1, 2, 4\}$. Thứ tự chọn: $\lambda$ trước (trên tín hiệu tiềm ẩn A2), rồi $\kappa$.

**Lựa chọn thiết kế và căn cứ (mô phỏng S1, 18 kịch bản):**
* **Lai thay vì chỉ tiềm ẩn:** tín hiệu lai ensemble có Coverage cao nhất ở cả 18 kịch bản. Tín hiệu chỉ tiềm ẩn tốt nhất (A2) đạt 1.5–31%, trong khi B1 đạt 36–48%.
* **$\bar\zeta + \lambda s_\zeta$ thay vì phương sai (bản 1.2):** tín hiệu phương sai cho Coverage thấp nhất ở mọi kịch bản (0.7–4%).
* **Giữ $\lambda$ trong lưới:** $\lambda^*$ được chọn bằng 2 ở các kịch bản ảo giác chiếm đa số và các thành viên bất đồng trên ảo giác, nhưng có thể bằng 0 khi lỗi lệch vị trí chiếm đa số. Vì vậy H2b′ là câu hỏi thực nghiệm mở.

---

## VI. KHUNG LTT DÙNG CHUNG
1. **Đường đi $\Lambda_{1D}(U)$:** dựng trên Meta-Path bằng thuật toán truy vết biên Pareto của H1 (Mục 6.2 của H1), cùng lưới rủi ro ($K_\Lambda = 33$ tại $N = 1.500$) và cùng định nghĩa đơn vị.
2. **Hiệu chuẩn:** fixed-sequence với p-value HB của H1; $\hat\tau = \emptyset$ nếu điểm đầu tiên không đạt.
3. **Hệ quả:** Macro-FDP tại $\hat\tau$ dự kiến quanh hoặc dưới 1.2%. Trong S1, tỷ lệ vi phạm của mọi tín hiệu ở mọi kịch bản không vượt 0.4%, nhất quán với độ bảo thủ của HB đã thấy ở H1.

---

## VII. PHÂN TÍCH ĐỘNG CƠ: MÔ HÌNH ĐỒ CHƠI CHO HIỆU ỨNG ENSEMBLE

**Mô hình.** Với một ứng viên cố định, $\zeta_m = \mu + b\,\xi_m$, $m = 1, \dots, M$. Trong đó $\mu$ là độ lệch chung mọi thành viên cùng thấy, $b > 0$ là cường độ dao động riêng của mô hình, và $\xi_m$ hoán đổi được với kỳ vọng 0, phương sai 1, tương quan $\rho \in [0, 1)$ giữa các thành viên. Tham số $\rho$ biểu diễn thiên kiến chung (4.3).

> **Mệnh đề 1.** $\mathbb{E}[\bar\zeta] = \mu$; $\operatorname{Var}(\zeta_a) = b^2$; $\operatorname{Var}(\bar\zeta) = b^2\big(\rho + \frac{1-\rho}{M}\big)$; $\mathbb{E}[s_\zeta^2] = b^2(1 - \rho)$.

*Chứng minh.* $\operatorname{Var}(\bar\zeta) = \frac{b^2}{M^2}\big(M + M(M-1)\rho\big)$. Từ $\sum_m(\zeta_m - \bar\zeta)^2 = \sum_m \zeta_m^2 - M\bar\zeta^2$, và giả sử $\mu = 0$ vì $\mu$ triệt tiêu trong độ lệch: $\mathbb{E}\big[\sum_m(\zeta_m - \bar\zeta)^2\big] = Mb^2 - M\operatorname{Var}(\bar\zeta) = (M-1)b^2(1-\rho)$. $\blacksquare$ *(Đã kiểm bằng Monte Carlo 400.000 mẫu, sai khác dưới 0.3%.)*

**Hệ quả kiểm được.**
* **P1 (ensemble giảm nhiễu, không thêm tín hiệu về mức):** lợi ích của $\bar\zeta$ so với $\zeta_a$ đến từ việc giảm phương sai từ $b^2$ xuống $b^2(\rho + (1-\rho)/M)$, và lợi ích này biến mất khi $\rho \to 1$. Trong S1, hiệu ứng ensemble trên tín hiệu lai ($\Delta_{\text{ens}}^{\text{hyb}}$) bằng +6.3 đến +8.4 pp khi $\rho = 0.3$, +2.8 đến +6.0 pp khi $\rho = 0.7$, và +0.3 đến +2.4 pp khi $\rho = 0.95$.
* **P2 (độ phân tán chỉ thấy phần không chung):** $\mathbb{E}[s_\zeta^2] = b^2(1-\rho)$ không phụ thuộc $\mu$. $s_\zeta$ chỉ giúp nếu FP có $b^2(1-\rho)$ lớn hơn TP. Trong S1, khi ảo giác không làm các thành viên bất đồng hơn ($b_h = b$), $\Delta_{\text{ens}}^{\text{hyb}}$ chỉ còn +0.5 đến +2.0 pp.
* **Đại lượng đo được tương ứng với $\rho$:** trên dữ liệu thật không tách được $\mu$ khỏi phần dao động chung, nhưng đo được **tỷ trọng phần không chung** $\phi = \mathbb{E}[s_\zeta^2] / \operatorname{Var}(\zeta_m)$, tương đương $1 - \bar r$ với $\bar r$ là tương quan trung bình giữa $\zeta_m$ và $\zeta_{m'}$ trên các ứng viên TP. Trong tham số của S1, $\rho = 0.3$, $0.7$, $0.95$ tương ứng với $\bar r \approx 0.55$, $0.81$, $0.97$.
* **Mệnh đề không nói gì trực tiếp về Coverage dưới LTT.** Liên hệ giữa $\rho$ và Coverage ở trên đến từ mô phỏng, phụ thuộc mô hình sinh dữ liệu, và chỉ dùng để đăng ký **chiều** của dự đoán.

---

## VIII. THIẾT KẾ THỰC CHỨNG

```
[3 detector có sẵn hoặc huấn luyện lại theo Mục 0]  →  3 cấu hình anchor
[Meta-Train của H1] tách theo trọn super-block:
   ├── Meta-Fit (50%):  tâm lớp, hiệp phương sai, med/MAD, ζ_cap, hàng xóm KNN, đầu EDL
   └── Meta-Path (50%): pilot P-H2, chọn siêu tham số, dựng Λ_1D(U)
[Đăng ký trước P-H2.6]
[Evaluation pool = toàn bộ Val, các vòng B-block của H1 Exp B]
   ├── H2-A: Coverage và rủi ro tại τ̂ (C-H2a, C-H2b, C-H2c, C-H2d)
   ├── H2-B: Ablation, so sánh chỉ-tiềm-ẩn với B1 (mô tả), Track C
   ├── H2-C: Phân rã FP tại τ̂
   └── H2-D: Phân tầng theo điều kiện cảnh (mô tả)
```

### 8.0. Pilot tiên nghiệm P-H2 (chỉ dùng Meta-Train)
Chạy sau bước P3 của H1 và trước bước P4 của H1. Không chạm vào nhãn hộp của Val.

**P-H2.0. Tách Meta-Train** thành Meta-Fit và Meta-Path (mỗi phần 50% số super-block, hạt giống cố định). *Nếu dùng 3 detector hiện tại:* Meta-Train nằm trong dữ liệu huấn luyện, nên ghi rõ trong bản đăng ký rằng các thống kê trên Meta-Fit và Meta-Path là lạc quan.

**P-H2.1. Phân rã FP.** Trên Meta-Path, tại điểm $\alpha_k = 0.008$ của $\Lambda_{1D}(U_{\text{conf}})$, tính tỷ trọng ảo giác $\pi_{\text{halluc}} = \sum V^{\text{halluc}} / \sum V$ kèm CI bootstrap theo super-block. Dự đoán được đăng ký theo S1: $\Delta_{\text{lat}}^{\text{single}}$ lớn (từ +12 pp trở lên) khi ảo giác chiếm từ một nửa số FP trở lên, và nhỏ (dưới +5 pp) khi lỗi lệch vị trí chiếm đa số **và** không lộ ra trong không gian tiềm ẩn. Nếu lỗi lệch vị trí có lộ ra trong không gian tiềm ẩn (kịch bản $\delta_{\text{loc}} = 1$ của S1), hiệu ứng vẫn có thể lớn (+13.7 pp). Vì vậy P-H2.1 báo cáo thêm AUROC giữa FP lệch vị trí và TP của $\zeta_a$ trên Meta-Path. Tiêu chí xác nhận ở Mục IX không thay đổi theo kết quả này.

**P-H2.2. Đo độ đa dạng của ensemble.** Trên các ứng viên TP và FP của Meta-Path, cho từng cặp thành viên:
* $\bar r_\zeta$: tương quan Pearson trung bình giữa $\zeta_m$ và $\zeta_{m'}$;
* $\bar r_s$: tương quan giữa logit tin cậy của các thành viên;
* CKA tuyến tính giữa đặc trưng decoder (Kornblith et al., 2019);
* $\mathbb{E}[s_\zeta^2 \mid \text{FP}] / \mathbb{E}[s_\zeta^2 \mid \text{TP}]$.

**Dự đoán đăng ký trước (từ S1):** nếu $\bar r_\zeta$ trên TP từ 0.95 trở lên, H2b và H2c được dự đoán là **Không đạt**. Đây là kết quả có ý nghĩa khoa học: ensemble fine-tune từ cùng một trọng số pretrained không đủ đa dạng. Nếu $\bar r_\zeta \le 0.8$ và ảo giác chiếm đa số FP, H2b được dự đoán là có thể đạt. Dự đoán chỉ được ghi lại để so với kết quả thật; nó không làm thay đổi tiêu chí.

**P-H2.3. Chọn siêu tham số.** Mỗi siêu tham số ($\lambda_{\text{out}}$, $\lambda$, $\kappa$, $k$, epoch của B3) được chọn trên Meta-Path theo cùng một quy tắc: giá trị cho Coverage lớn nhất tại ngưỡng plug-in có Macro-FDP $\le r^*$ (công thức của H1, khoảng 0.009–0.011 tại $N = 1.500$). Mỗi thành phần có tối đa 5 giá trị lưới. Chọn riêng cho từng cấu hình anchor. Khi hòa, chọn giá trị nhỏ hơn.

**P-H2.4. Dựng đường đi** cho mọi tín hiệu và mọi cấu hình. Báo cáo Coverage trên Meta-Path dọc đường đi và $\operatorname{Cov}_{\max}$.

**P-H2.5. Năng lực thống kê.** Dựng 50 pool tổng hợp bằng cách rút có hoàn lại các super-block của Meta-Path cho đến khi bằng kích thước Evaluation pool. Ước lượng $\widehat{\text{SE}}_k$ của $\hat\Delta_k$ và năng lực khi hiệu ứng thật bằng $\gamma$: $\Phi(\gamma/\widehat{\text{SE}}_k - z_{1-\alpha_{\text{each}}})$. Yêu cầu ít nhất 0.80; nếu không đạt, ghi rõ trong bản đăng ký. Tham khảo từ S2: SD của ước lượng điểm qua các pool vào khoảng 0.7–2.3 pp. Khi Δ thật bằng 8.8 pp, "Đạt" xảy ra ở 100% số lần; khi Δ thật bằng 4.5 pp, kết quả là "Không kết luận" ở 80% số lần.

**P-H2.6. Đăng ký trước:** cách tách Meta-Train, thống kê tham chiếu (file băm), siêu tham số đã chọn, các đường đi, kết quả P-H2.1 đến P-H2.5, các dự đoán, $\gamma$, $\alpha_{\text{each}}$ và Mục IX. Gộp vào bước P4 của H1.

### 8.1. H2-A: Coverage và rủi ro tại $\hat\tau$ (phép kiểm xác nhận)
**Vòng đánh giá:** dùng chính các vòng B-block của H1 Exp B (334 phân hoạch × 3, khoảng 1.000 vòng; tập kiểm thử khoảng 5.500 đơn vị). Với mỗi vòng $b$, cấu hình anchor $a$ và tín hiệu $U$: chạy LTT trên tập hiệu chuẩn để có $\hat\tau_{U,b}$; tính $\operatorname{Cov}_b(U)$ trên tập kiểm thử (bằng 0 nếu $\hat\tau = \emptyset$); ghi nhận vi phạm theo rủi ro trên toàn pool (định nghĩa của H1).

**Đại lượng ước lượng:** $\Delta_k = \frac{1}{3}\sum_a \mathbb{E}_b\big[\operatorname{Cov}_b(U_{\text{fcd}}) - \operatorname{Cov}_b(U_k)\big]$, tính ghép cặp (mọi tín hiệu dùng cùng vòng và cùng anchor). Đại lượng này **có điều kiện trên bộ ba detector đã huấn luyện** và trên Evaluation pool.

**Khoảng tin cậy:** bootstrap theo cụm super-block của Evaluation pool, $R = 4.000$ lần lặp. Mỗi lần lặp tạo 30 vòng B-block trên pool bootstrap, mỗi vòng rút ngẫu nhiên một anchor, rồi tính trung bình hiệu Coverage ghép cặp. $\text{LCB}_k$ và $\text{UCB}_k$ là các phân vị $\alpha_{\text{each}}$ và $1 - \alpha_{\text{each}}$. Trong S2, SD bootstrap lệch không quá khoảng 20% so với SD thật của ước lượng điểm qua các pool độc lập (H2_simulation_report.md).

**Mức ý nghĩa:** họ xác nhận gồm 4 phép so sánh ($\Delta_{\text{lat}}^{\text{single}}$, $\Delta_{\text{lat}}^{\text{ens}}$, $\Delta_{\text{ens}}^{\text{hyb}}$, H2c), mức một phía cho cả họ là 0.025, nên $\alpha_{\text{each}} = 0.025/4 = 0.00625$.

**Chỉ số báo cáo cho mọi tín hiệu:** Cov, $\operatorname{Cov}/\operatorname{Cov}_{\max}$, Yield, VR, Macro-FDP trên toàn pool (trung vị và phân vị 95%), Micro-FDP, tỷ lệ đơn vị rỗng, tỷ lệ không có đối ứng.

### 8.2. H2-B: Ablation và phân tích mô tả
* $\Delta_{\text{ens}}^{\text{out}} = \operatorname{Cov}(B2) - \operatorname{Cov}(B1)$, và tương tác $\Delta_{\text{ens}}^{\text{hyb}} - \Delta_{\text{ens}}^{\text{out}}$.
* **Chỉ tiềm ẩn so với B1:** B4, B6 và A2 so với B1. Dự đoán đăng ký trước: hiệu âm.
* Thành phần của tín hiệu đề xuất: A1 ($\lambda = 0$, tức H2b′), A3, A4, A5.
* Cân bằng compute: tín hiệu đề xuất so với B7, và B2 so với B7.
* Kiểm dự đoán P1 và P2 bằng $\bar r_\zeta$ và tỷ số độ phân tán FP/TP đo trên Evaluation pool.
* **Track C:** đường đi 2 chiều $(\tau_{\text{conf}}, \tau_{\text{latent}})$ theo thuật toán Pareto của H1. Track này kiểm tra xem phép kết hợp tuyến tính trong $U_{\text{fcd}}$ có làm mất Coverage so với một ranh giới 2 chiều tổng quát hơn hay không.
* **Đường đi 3 chiều của H1** $(\tau_{\text{cls}}, \tau_{\text{loc}}, \tau_{\text{fcd}})$, tức vùng chấp nhận dạng giao của ba ngưỡng, với $\tau_{\text{loc}}$ đặt trên $1 - \overline{\text{IoU}}$ và $\tau_{\text{fcd}}$ đặt trên $\bar\zeta + \lambda s_\zeta$ (`HYPOTHESIS_CHAIN.md`, Mục 4). Phép so sánh này nối kết quả H2 với đường đi mà H1 dùng để kiểm chứng bảo chứng.

### 8.3. H2-C: Phân rã FP tại $\hat\tau$
Cho từng tín hiệu: tỷ lệ ảo giác HR, số FP mỗi loại trên 1.000 TP, và chỉ số Jaccard giữa tập FP được chấp nhận của tín hiệu đề xuất và của B1. Dự đoán (chỉ về chiều): thành phần tiềm ẩn làm giảm số FP ảo giác trên 1.000 TP nhiều hơn so với số FP lệch vị trí.

### 8.4. H2-D: Phân tầng theo điều kiện cảnh (mô tả)
Đêm, mưa, tuyết và sương mù đều có trong tập huấn luyện của BDD100K, nên các tầng này không phải OOD. H2-D báo cáo Coverage và Macro-FDP theo tầng tại $\hat\tau$ hiệu chuẩn trên toàn pool. Bảo chứng của LTT là bảo chứng biên, nên không có bảo chứng cho từng tầng.

---

## IX. MA TRẬN QUYẾT ĐỊNH TIÊN NGHIỆM

**Quy tắc ba vùng cho một phép so sánh $k$:**
* **Đạt:** $\hat\Delta_k \ge \gamma$ và $\text{LCB}_k > 0$.
* **Không đạt:** $\text{UCB}_k < \gamma$.
* **Không kết luận:** các trường hợp còn lại.

*Đặc tính đã kiểm bằng mô phỏng S2:* khi Δ thật bằng 0, quy tắc không bao giờ cho "Đạt"; khi Δ thật bằng 1.5–2.6 pp, cũng không bao giờ cho "Đạt". Tuy nhiên, khi Δ thật **ngay dưới** γ (4.5 pp), quy tắc cho "Đạt" ở 20% số lần, vì "Đạt" chỉ đòi ước lượng điểm $\ge \gamma$. Phương án chặt hơn là $\text{LCB}_k > \gamma$. Phương án này cần Δ thật khoảng 8 pp để có năng lực 50%; nhóm cần chọn một trong hai trước khi đăng ký.

| Mã | Nội dung | CHẤP NHẬN | BÁC BỎ | KHÔNG KẾT LUẬN |
|:---:|---|---|---|---|
| **C-H2a** | Thông tin tiềm ẩn: $\Delta_{\text{lat}}^{\text{single}}$ và $\Delta_{\text{lat}}^{\text{ens}}$ | Cả hai Đạt | Ít nhất một Không đạt | Còn lại |
| **C-H2b** | Hiệu ứng ensemble: $\Delta_{\text{ens}}^{\text{hyb}}$ | Đạt | Không đạt | Còn lại |
| **C-H2c** | Vượt từng baseline B1–B8 | Mọi $k$ Đạt | Tồn tại $k$ Không đạt | Còn lại |
| **C-H2d** | Hợp lệ rủi ro theo C2 của H1 | Mọi tín hiệu trong C-H2a–c ở vùng Chấp nhận của C2 | Tín hiệu đề xuất ở vùng Bác bỏ của C2 | Còn lại |
| **C-H2-anchor** | Độ bền theo anchor | Với mỗi tiêu chí Chấp nhận: $\Delta^{(a)} > 0$ ở cả 3 cấu hình anchor | — | Không thỏa thì tiêu chí đó hạ xuống Không kết luận |
| **C-H2-Final** | Kết luận gộp | C-H2a, C-H2b, C-H2c, C-H2d Chấp nhận, sau khi áp cổng anchor | Ít nhất một Bác bỏ | Còn lại |

*Ghi chú:*
1. **C-H2d đứng trước.** Nếu tín hiệu đề xuất vi phạm bảo chứng, H2 được ghi là Bác bỏ. Nếu một baseline vi phạm, phép so sánh với baseline đó được gắn cờ.
2. **Phạm vi suy diễn:** với một bộ ba detector duy nhất, mọi kết luận về ensemble chỉ áp dụng cho bộ ba này. Cổng anchor không thay thế được việc lặp lại với nhiều ensemble độc lập; paper phải ghi rõ điều này.
3. **Kết quả kết hợp có thể xảy ra:** C-H2a Chấp nhận và C-H2b Bác bỏ có nghĩa là thông tin tiềm ẩn có giá trị nhưng ensemble hiện tại không khuếch đại được nó. Đây là kết quả được S1 dự đoán khi độ đa dạng thấp. Khi đó H3 dùng B8 (xem 9.2).

### 9.1. Khuôn Bảng 1 (điền sau khi chạy)

| Ô | Phương pháp | Forward | Siêu tham số | Yield | VR | Macro-FDP pool | Micro-FDP | Cov | Cov/Cov$_{\max}$ | $\Delta$ so với đề xuất [CI] | Anchor có $\Delta > 0$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Đơn, đầu ra | B1 Độ tin cậy | 1× | — | — | — | — | — | — | — | — | — |
| Đơn, lai | B8 Lai đơn mô hình | 1× | — | — | — | — | — | — | — | — | — |
| Ensemble, đầu ra | B2 Bất đồng đầu ra | 3× | — | — | — | — | — | — | — | — | — |
| Cân bằng compute | B7 TTA-3 lai | 3× | — | — | — | — | — | — | — | — | — |
| Đơn, đầu ra | B3 EDL | 1× | — | — | — | — | — | — | — | — | — |
| Đơn, đầu ra | B5 JointEnergy | 1× | — | — | — | — | — | — | — | — | — |
| Đơn, tiềm ẩn | B4 Mahalanobis | 1× | — | — | — | — | — | — | — | — | — |
| Đơn, tiềm ẩn | B6 KNN | 1× | — | — | — | — | — | — | — | — | — |
| Ensemble, lai | **Đề xuất $U_{\text{fcd}}$** | 3× | — | — | — | — | — | — | — | tham chiếu | — |

### 9.2. Quy tắc chọn tín hiệu giao cho H3 (đăng ký trước)
1. Nếu C-H2-Final là Chấp nhận: $U^\star = U_{\text{fcd}}$.
2. Ngược lại, $U^\star$ là tín hiệu có $\hat\Delta$ lớn nhất so với B1 trong tập $\{B1, B2, B7, B8, U_{\text{fcd}}\}$, **với điều kiện** tín hiệu đó Đạt so với mọi tín hiệu rẻ hơn trong tập. Thứ tự chi phí: B1 = B8 < B7 = B2 = $U_{\text{fcd}}$ về số lượt forward; B8 đắt hơn B1 một phép tính khoảng cách. Nếu không tín hiệu nào thỏa, $U^\star = U_{\text{conf}}$.
3. Việc chọn $U^\star$ dùng dữ liệu Val của H2. Vì vậy H3 **không** được đo mAP trên Val, và phải hiệu chuẩn lại LTT trên dữ liệu mới ở mỗi vòng AL (`HYPOTHESIS_CHAIN.md`, Mục 3).

---

## X. DANH MỤC THAM KHẢO

1. **Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L.** Learn then test: Calibrating predictive algorithms to achieve risk control. Preprint arXiv:2110.01052. *(Cần xác minh thông tin xuất bản cuối cùng.)*
2. **Bentkus, V.** (2004). On Hoeffding's inequalities. *The Annals of Probability*, 32(2), 1650–1673.
3. **Berger, R. L.** (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics*, 24(4), 295–300.
4. **Geirhos, R., et al.** (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR 2019*.
5. **Hendrycks, D., & Gimpel, K.** (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR 2017*.
6. **Hoeffding, W.** (1963). Probability inequalities for sums of bounded random variables. *JASA*, 58(301), 13–30.
7. **Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J.** (2023). A call to reflect on evaluation practices for failure detection in image classification. *ICLR 2023*.
8. **Kamoi, R., & Kobayashi, K.** (2020). Why is the Mahalanobis distance effective for anomaly detection? Preprint arXiv:2003.00402.
9. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.** (2019). Similarity of neural network representations revisited. *ICML 2019*.
10. **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
11. **Ledoit, O., & Wolf, M.** (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
12. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS 2018*.
13. **Liu, W., Wang, X., Owens, J., & Li, Y.** (2020). Energy-based out-of-distribution detection. *NeurIPS 2020*.
14. **Miller, D., Nicholson, L., Dayoub, F., & Sünderhauf, N.** (2018). Dropout sampling for robust object detection in open-set conditions. *ICRA 2018*.
15. **Sensoy, M., Kaplan, L., & Kandemir, M.** (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS 2018*.
16. **Sun, Y., Ming, Y., Zhu, X., & Li, Y.** (2022). Out-of-distribution detection with deep nearest neighbors. *ICML 2022*.
17. **Wang, H., Liu, W., Bocchieri, A., & Li, Y.** (2021). Can multi-label classification networks know what they don't know? *NeurIPS 2021*.
18. **Westfall, P. H., & Krishen, A.** (2001). Optimally weighted, fixed sequence and gatekeeping multiple testing procedures. *Journal of Statistical Planning and Inference*, 99(1), 25–40.
19. **Yu, F., et al.** (2020). BDD100K: A diverse driving dataset for heterogeneous multitask learning. *CVPR 2020*.
20. **Zhang, H., Wang, Y., Dayoub, F., & Sünderhauf, N.** (2021). VarifocalNet: An IoU-aware dense object detector. *CVPR 2021*.
21. **Zhao, Y., et al.** (2024). DETRs beat YOLOs on real-time object detection. *CVPR 2024*.
