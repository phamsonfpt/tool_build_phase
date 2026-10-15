# KẾ HOẠCH PHÂN TÍCH THỐNG KÊ & KIỂM THỬ (SAP) CHO PAPER H1 + H2 + H3

**Phiên bản:** 0.1, 15/10/2026. **Đi kèm:** [01_PAPER_BLUEPRINT.md](01_PAPER_BLUEPRINT.md), [02_MATH_EXPLAINED.md](02_MATH_EXPLAINED.md).

**Vì sao có file này:** mentor nhận xét phần thuật toán đã ổn nhưng còn thiếu **kiểm thử** và **giá trị thống kê**. File này bổ sung ba thứ:
1. **Kế hoạch phân tích thống kê đầy đủ** cho từng giả thuyết: đại lượng cần ước lượng (estimand), phép kiểm, mức ý nghĩa, hiệu chỉnh đa so sánh, khoảng tin cậy, cỡ hiệu ứng (effect size) và quy tắc quyết định.
2. **Phân tích công suất (power) và cỡ mẫu bằng số cụ thể**, tính bằng mô phỏng Monte Carlo và công thức giải tích. Kết quả cho thấy hai quy tắc quyết định cũ (của H2 và H3) có lỗi thiết kế cần sửa.
3. **Ba tầng kiểm thử:** unit test cho code thống kê (19 test, đều pass), kiểm chứng bằng mô phỏng (sai lầm loại I, công suất), và các phép kiểm đối chứng trên dữ liệu thật để phát hiện lỗi pipeline.

**Cách tái tạo mọi con số trong file này:**
```bash
python -m unittest discover -s fcd_v3_research/tests -v
```
```bash
python fcd_v3_research/scripts/02_statistical_validation.py
```
Lệnh thứ hai chạy khoảng 4 phút (seed `20261015`) và ghi kết quả vào [results/statistical_validation.json](results/statistical_validation.json). Code nằm ở `fcd_v3_research/src/risk_stats/`.

> **Giới hạn quan trọng:** chưa có output của detector đã fine-tune trên BDD100K, nên **mọi con số dưới đây là kết quả mô phỏng dưới các giả định đầu vào được nêu rõ**, không phải kết quả thực nghiệm. Chúng dùng để (i) chọn cỡ mẫu, (ii) chứng minh các quy tắc quyết định có công suất đủ, (iii) kiểm tra code đúng. Mục 7 mô tả cách thay các giả định bằng ước lượng từ pilot trên Meta-Train trước khi đăng ký trước.

---

## 1. Bảng tổng hợp

| Giả thuyết | Đại lượng (estimand) | Phép kiểm chính | Mức | Đa so sánh | Khoảng tin cậy | Effect size | Cỡ mẫu chọn | Công suất (mô phỏng) |
|---|---|---|---|---|---|---|---|---|
| **H1a** | VR = $\mathbb P(\mathcal R(\hat\lambda) > \alpha)$ | Cận trên Clopper–Pearson một phía của VR $\le \delta$ | $\delta = 0.05$ | Không (một endpoint) | CP 95% (B-rand), bootstrap theo phân hoạch (B-block) | VR, $D_{\text{eff}}$ | $N = 1.500$ đơn vị, $B = 1.000$ vòng | Phát hiện VR = 0.10 với xác suất ≈1.00, VR = 0.08 với 0.98 |
| **H1b** | VR của Arm 2 (frame) | Binomial chính xác một phía, $H_0$: VR $\le \delta$ | 0.05 | Không | CP 95% | $D^{\text{frame}}_{\text{eff}}$, CI bootstrap | $B \ge 500$ mỗi ô | Phụ thuộc $D_{\text{eff}}$ (Mục 3.2) |
| **H1c** | $\mathbb P(\hat\lambda \ne \emptyset)$, $\rho_{\text{cov}}$, mật độ box | CP một phía cho Yield; bootstrap cho $\rho_{\text{cov}}$ | 0.05 | Không | CP, bootstrap | | như H1a | |
| **H2a/b/c** | $\Delta\text{Cov} = \text{Cov}(\text{FCD}) - \text{Cov}(\text{baseline})$ | Wilcoxon một phía trên $\Delta\text{TP}_i$ **và** kiểm định có biên $H_0: \Delta\text{Cov} \le 2$ điểm | 0.01 họ, Holm | Holm cho 4 baseline | Bootstrap theo đơn vị, có cặp | $\Delta\text{Cov}$ (điểm %), rank-biserial $r$ | $N_{\text{test}} \approx 4.800$ đơn vị | MDE = 3 điểm (công suất ≥ 0.98) |
| **H3a** | $\Delta\text{mAP}_T$ = Proposed − Random ở vòng cuối | t có cặp một phía theo seed | 0.05 | Holm cho 3 so sánh | t 95%, bootstrap theo ảnh | $\Delta\text{mAP}$, Cohen $d_z$ | ≥ 8 seed (nếu $\sigma_d \le 1.5$) | 0.82 tại $\Delta = 1.5$, $\sigma_d = 1.5$ |
| **H3b** | $\Delta\text{mAP}$ LTT − heuristic, tỉ lệ nhãn giả sai | Không thua kém, biên −0.5 điểm | 0.05 | Không | t 95% | | như H3a | |

---

## 2. Nguyên tắc chung (áp dụng cho mọi giả thuyết)

1. **Mỗi con số trong bảng kết quả có khoảng tin cậy 95%**, cỡ mẫu $N$ (số đơn vị, số seed), và phương pháp tính CI.
2. **Báo cáo p-value chính xác** (ví dụ $p = 0.0031$), không viết "$p < 0.05$". Kèm **effect size**, vì với $N = 4.800$ đơn vị, chênh lệch rất nhỏ cũng có ý nghĩa thống kê.
3. **Đơn vị lấy mẫu lại (resampling) là đơn vị thống kê** (chuyến đi hoặc super-block), không bao giờ là frame hay box. Đây là cùng lý do với H1: box và frame trong một chuyến đi tương quan với nhau.
4. **Tách rõ phân tích đăng ký trước và phân tích khám phá.** Các ablation của FCD ($\bar D$, $s^2$, max) là khám phá: báo cáo CI nhưng không dùng để kết luận giả thuyết.
5. **Mọi tham số được chốt trên Meta-Train** (đường đi, $\lambda$ của FCD, $d_{\text{pca}}$, $\theta_{\text{IoU}}$) trước khi mở cal/test.
6. **Quy tắc quyết định phải có công suất được tính trước.** Quy tắc nào chỉ đạt khoảng 50% công suất ở đúng hiệu ứng mục tiêu thì phải sửa (Mục 4.2, 5.2).

---

## 3. H1: Bảo chứng thống kê

### 3.1. H1a: tính hợp lệ của bảo chứng
* **Estimand:** VR, xác suất ngưỡng được chọn có rủi ro thật vượt $\alpha$, trên phân phối của tập hiệu chuẩn.
* **Ước lượng:** lặp $B$ vòng: rút $\mathcal D_{\text{cal}}$ ($N = 1.500$ đơn vị) từ pool, chạy LTT, đo rủi ro của $\hat\lambda$ **trên toàn pool test** (tài liệu H1 v1.2 dùng rủi ro trên toàn pool thay cho phần bù). $\widehat{\text{VR}} = \#\text{vi phạm} / B$.
* **Quy tắc:** chấp nhận C2 khi cận trên Clopper–Pearson một phía 95% của VR $\le \delta = 0.05$.
* **Chọn $B$: tính bằng giá trị thống kê.** Cận trên CP 95% một phía:

| $B$ | VR quan sát 0 | 0.01 | 0.02 | 0.03 |
|---|---|---|---|---|
| 100 | 0.030 | 0.047 | **0.062** | 0.076 |
| 300 | 0.010 | 0.026 | 0.039 | 0.052 |
| 500 | 0.006 | 0.021 | 0.034 | 0.046 |
| **1.000** | 0.003 | 0.017 | 0.029 | 0.040 |
| 3.000 | 0.001 | 0.014 | 0.025 | 0.036 |

  Với $B = 100$, ngay cả một quy trình đúng có VR quan sát 2% cũng **không chứng minh được** VR $\le 5\%$ (cận trên 6.2%). Với $B = 1.000$, VR quan sát tới khoảng 3% vẫn cho cận trên dưới 5%. **Chọn $B = 1.000$.**

* **Công suất phát hiện một quy trình sai** (kiểm binomial chính xác $H_0$: VR $\le 0.05$):

| $B$ | VR thật = 0.08 | VR thật = 0.10 |
|---|---|---|
| 100 | 0.28 | 0.55 |
| 200 | 0.54 | 0.86 |
| 500 | 0.86 | 0.995 |
| 1.000 | 0.98 | ≈1.00 |

* **Lưu ý về tính phụ thuộc giữa các vòng:** 1.500 trên pool khoảng 4.200 đơn vị nghĩa là các vòng chồng lấn (khoảng 36%). CP giả định các vòng độc lập nên CI sẽ quá hẹp. Nhánh B-block vì vậy dùng **bootstrap theo phân hoạch** (tài liệu H1, Mục 8.2) và báo cáo cả hai CI.
* **C2b (kiểm trực tiếp giả định A1):** $\hat D_{\text{eff}} = \operatorname{Var}_{\text{B-block}}(\widehat{\mathcal R}) / \operatorname{Var}_{\text{B-rand}}(\widehat{\mathcal R})$, CI 95% bằng bootstrap theo super-block. Chấp nhận khi cận trên $\le \min(2, \hat D^*)$.

### 3.2. H1b: chia theo frame làm sụp đổ bảo chứng (kết quả mô phỏng)
**Mô hình mô phỏng:** $\alpha_C = 0.10$, đường đi 38 điểm của Exp C (có phần kéo dài tới $2\alpha_C$), 50 frame mỗi clip, mỗi frame có $1 + \text{Poisson}(3)$ box. Box là FP với xác suất $\text{expit}(\beta_k + b_{\text{clip}} + e_{\text{frame}})$, với $b \sim \mathcal N(0, \sigma_b^2)$ là hiệu ứng của clip và $e$ là AR(1) theo thời gian ($\phi$). 400 vòng mỗi ô.

| Kịch bản | $\sigma_b$ | $\phi$ | $\hat D^{\text{frame}}_{\text{eff}}$ | $C$ (clip) | VR Arm 1 (clip) [CI 95%] | VR Arm 2 (frame) [CI 95%] | Rủi ro thật TB của ngưỡng chọn (Arm 1 / Arm 2) |
|---|---|---|---|---|---|---|---|
| Frame độc lập (đối chứng âm) | 0 | 0 | 1.0 | 100 | 0.000 [0, 0.009] | 0.000 [0, 0.009] | 0.033 / 0.089 |
| Chỉ AR(1) | 0 | 0.9 | 2.3 | 100 | 0.000 [0, 0.009] | 0.010 [0.003, 0.025] | 0.033 / 0.089 |
| Clip + AR(1) | 0.8 | 0.9 | 6.4 | 100 | 0.000 [0, 0.009] | **0.068 [0.045, 0.097]** | 0.033 / 0.089 |
| Clip mạnh + AR(1) | 1.2 | 0.9 | 11.7 | 100 | 0.000 [0, 0.009] | **0.138 [0.105, 0.175]** | 0.033 / 0.089 |
| Frame độc lập | 0 | 0 | 1.0 | 400 | 0.000 [0, 0.009] | 0.000 [0, 0.009] | 0.065 / 0.094 |
| Chỉ AR(1) | 0 | 0.9 | 2.3 | 400 | 0.000 [0, 0.009] | 0.000 [0, 0.009] | 0.065 / 0.094 |
| Clip + AR(1) | 0.8 | 0.9 | 6.4 | 400 | 0.000 [0, 0.009] | 0.018 [0.007, 0.036] | 0.065 / 0.094 |
| Clip mạnh + AR(1) | 1.2 | 0.9 | 12.0 | 400 | 0.000 [0, 0.009] | **0.090 [0.064, 0.122]** | 0.065 / 0.094 |

**Đọc bảng:**
* **Arm 1 hợp lệ ở mọi kịch bản** (VR = 0, cận trên 0.9%). Cái giá là sự bảo thủ: với 100 clip, ngưỡng được chọn có rủi ro thật trung bình 3.3%, thấp xa mức trần 10%. Tăng lên 400 clip thì nới ra được 6.5%. Đây là đánh đổi H1c phải báo cáo.
* **Đối chứng âm đạt:** khi frame thật sự độc lập, Arm 2 cũng hợp lệ. Vậy vi phạm ở các dòng khác do tương quan gây ra, không phải lỗi code.
* **Arm 2 chỉ vi phạm rõ khi $D_{\text{eff}} \gtrsim 6$.** Với $D_{\text{eff}} \approx 2.3$, HB đủ bảo thủ để "chịu đựng" tương quan. Kết quả này khớp với "biên chịu đựng tương quan" $D^*$ ở Mục 5.6 của tài liệu H1. **Hệ quả cho paper:** H1b chỉ được chấp nhận khi $D^{\text{frame}}_{\text{eff}}$ đo trên dữ liệu thật đủ lớn (tiêu chí C4). Không được viết "chia theo frame luôn sụp đổ".
* **Vì sao VR của Arm 2 giảm khi $C$ tăng:** mức chứng nhận của Arm 2 đã sát trần $\alpha_C$ ngay từ $C = 100$ (bảng 5.5 của H1), trong khi nhiễu của $\widehat{\mathcal R}$ giảm theo $C$. Mô phỏng dùng 50 frame mỗi clip. BDD100K MOT có khoảng 200 frame mỗi clip, nên $D_{\text{eff}}$ thật dự kiến **lớn hơn** (Kish: $1 + 199\rho$ so với $1 + 49\rho$).
* **Quy tắc C5 (đăng ký trước):** chấp nhận H1b khi cận dưới CP 95% của VR Arm 2 $> \delta$ ở **ít nhất một** $N_{\text{cluster}}$, **và** cận dưới CI của $D^{\text{frame}}_{\text{eff}} \ge \max(5, \hat D^*_C)$.

### 3.3. H1c: không tầm thường
* **Yield** $= \mathbb P(\hat\lambda \ne \emptyset)$: CP 95% một phía, chấp nhận khi cận dưới $\ge 0.85$.
* **Độ phủ tương đối** $\rho_{\text{cov}} = \text{Cov}(\hat\lambda) / \text{Cov}(\lambda_{\text{plug-in}})$: trung vị qua $B$ vòng, CI bootstrap theo vòng. Chấp nhận khi trung vị $\ge 0.5$.
* **Mật độ nhãn giả** $\mathbb E[N^{\text{acc}}_i(\hat\lambda)]$: trung bình kèm CI bootstrap theo đơn vị. Chấp nhận khi $\ge 1$ box/đơn vị (bổ sung từ bản review).
* **Cỡ mẫu tối thiểu theo HB** (để giải thích vì sao chọn $N = 1.500$), $k_{\max}$ tại $\alpha = 0.02$, $\delta = 0.05$:

| $N$ | 150 | 400 | 800 | 1.000 | **1.500** | 2.000 | 3.000 | 5.000 |
|---|---|---|---|---|---|---|---|---|
| $k_{\max}$ | 0 | 2 | 7 | 10 | **18** | 26 | 44 | 79 |
| FDP cal tối đa được chứng nhận | 0 | 0.50% | 0.88% | 1.00% | **1.20%** | 1.30% | 1.47% | 1.58% |

  Từ 1.500 lên 5.000 đơn vị chỉ nới được thêm 0.38 điểm FDP, nhưng tốn hơn ba lần nhãn. $N = 1.500$ là điểm cân bằng hợp lý.

---

## 4. H2: Tín hiệu bất định

### 4.1. Estimand, phép kiểm và effect size
* **Estimand:** $\Delta\text{Cov} = \text{Cov}(\text{FCD}; \hat\lambda_{\text{FCD}}) - \text{Cov}(\text{baseline}; \hat\lambda_{\text{base}})$ trên tập test, đơn vị điểm phần trăm.
* **Thống kê theo đơn vị:** $d_i = \text{TP}_i(\text{FCD}) - \text{TP}_i(\text{baseline})$. Hai tín hiệu được đánh giá trên **cùng** các đối tượng GT, nên đây là so sánh có cặp.
* **Phép kiểm 1:** Wilcoxon signed-rank một phía trên $\{d_i\}$, effect size là rank-biserial $r = (W^+ - W^-)/(W^+ + W^-)$.
* **Phép kiểm 2 (có biên):** $H_0: \Delta\text{Cov} \le \gamma_0 = 2$ điểm, thống kê $z = (\widehat{\Delta\text{Cov}} - \gamma_0)/\widehat{\text{SE}}$. SE theo công thức ước lượng tỉ số (ratio estimator) với đơn vị là cụm: $\widehat{\text{SE}} = \sqrt{\tfrac{N}{N-1}\sum_i (d_i - \widehat{\Delta} g_i)^2} / \sum_i g_i$, với $g_i$ là số GT của đơn vị $i$. Báo cáo thêm CI bootstrap 95% có cặp (`cluster_bootstrap_diff_ci`).
* **Đa so sánh:** 4 cặp (FCD so với từng baseline). Holm ở mức họ 0.01. Trường hợp xấu nhất, p nhỏ nhất được so với $0.01/4 = 0.0025$; mô phỏng dùng mức này.

### 4.2. Phát hiện: quy tắc quyết định cũ chỉ có khoảng 50% công suất
Quy tắc cũ (tài liệu H2, C-H2b): chấp nhận khi $\widehat{\Delta\text{Cov}} \ge 5$ điểm **và** $p < 0.0025$.

**Vấn đề:** nếu FCD thật sự tốt hơn đúng 5 điểm, ước lượng $\widehat{\Delta\text{Cov}}$ dao động quanh 5. Nó lớn hơn 5 khoảng một nửa số lần, bất kể $N$ lớn đến đâu. **Tăng dữ liệu không cứu được quy tắc này.**

**Mô phỏng** (300 lần lặp mỗi ô). Giả định: Coverage baseline 55%; mỗi đơn vị có $1 + \text{Poisson}(15)$ đối tượng GT; độ khó giữa các đơn vị $\sigma_u = 0.8$ trên thang logit; tương quan "được phủ" giữa hai tín hiệu trên cùng đối tượng $\rho_{\text{obj}} = 0.5$. Mỗi ô ghi xác suất chấp nhận theo dạng **Wilcoxon / quy tắc cũ / quy tắc có biên**:

| $N_{\text{test}}$ | SD của $\widehat\Delta$ | $\Delta$ = 0 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 500 | 0.61 | 0.00/0.00/0.00 | 0.66/0.00/0.00 | 0.99/0.00/0.13 | 1.00/0.04/0.71 | 1.00/**0.51**/0.99 | 1.00/0.98/1.00 | 1.00/1.00/1.00 |
| 1.000 | 0.41 | 0.01/0.00/0.00 | 0.97/0.00/0.00 | 1.00/0.00/0.31 | 1.00/0.03/0.95 | 1.00/**0.55**/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| 2.000 | 0.31 | 0.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.69 | 1.00/0.00/1.00 | 1.00/**0.47**/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| 4.800 | 0.19 | 0.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/**0.98** | 1.00/0.00/1.00 | 1.00/**0.52**/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |

Với $\rho_{\text{obj}} = 0.8$ (hai tín hiệu đồng ý nhiều hơn), SD của $\widehat\Delta$ nhỏ hơn (0.16 tại $N = 4.800$). Quy tắc cũ vẫn khoảng 0.48–0.52 tại $\Delta = 5$; quy tắc có biên đạt 1.00 tại $\Delta = 3$.

**Đọc bảng:**
* **Sai lầm loại I được kiểm soát:** tại $\Delta = 0$, cả ba quy tắc chấp nhận với tỉ lệ $\le 0.01$.
* **Quy tắc cũ:** công suất ≈ 0.50 tại đúng hiệu ứng mục tiêu 5 điểm, ở **mọi** $N$.
* **Wilcoxon đơn thuần quá nhạy:** với $N = 4.800$, nó phát hiện cả chênh lệch 2 điểm, tức có ý nghĩa thống kê nhưng có thể không có ý nghĩa thực tế.
* **Quy tắc có biên** (Wilcoxon $p < 0.0025$ **và** bác bỏ $H_0: \Delta \le 2$ điểm ở mức 0.0025): không chấp nhận chênh lệch tầm thường ($\le 2$ điểm), và có **hiệu ứng tối thiểu phát hiện được (MDE, công suất ≥ 0.8) = 3 điểm** tại $N = 4.800$, hoặc 4 điểm tại $N = 1.000$.

**Quy tắc mới đề xuất cho C-H2a, C-H2b, C-H2c:**
> Chấp nhận khi (i) Wilcoxon một phía có p đã hiệu chỉnh Holm $< 0.01$, **và** (ii) kiểm định có biên bác bỏ $H_0: \Delta\text{Cov} \le 2$ điểm ở cùng mức. Báo cáo $\widehat{\Delta\text{Cov}}$ kèm CI 95%. Mốc 5 điểm được giữ làm **hiệu ứng mục tiêu** (cỡ mẫu được chọn để có công suất ≈1 tại 5 điểm), không còn là ngưỡng cứng cho ước lượng điểm.
> **Bác bỏ** khi cận trên CI 95% của $\Delta\text{Cov} < 2$ điểm. **Không kết luận** trong các trường hợp còn lại.

### 4.3. Ngưỡng là ngẫu nhiên: phân tích bổ sung
Phép kiểm ở 4.2 coi $\hat\lambda$ là cố định. Thực tế $\hat\lambda$ phụ thuộc vào tập hiệu chuẩn cụ thể. **Phân tích phụ đăng ký trước:** lặp 100 lần chia cal/test theo super-block, mỗi lần tính lại $\hat\lambda$ cho cả hai tín hiệu và $\widehat{\Delta\text{Cov}}$. Báo cáo phân vị 2.5% và 97.5%, và tỉ lệ số lần FCD thắng. Nếu phân vị 2.5% < 0, ghi rõ trong Discussion rằng ưu thế phụ thuộc vào lần chia.

### 4.4. Các chỉ số phụ (khám phá, không hiệu chỉnh)
* AUROC (TP so với FP) với CI DeLong; AURC (diện tích dưới đường risk–coverage).
* H2-C: tỉ lệ ảo giác thuần HR $= \sum V^{\text{halluc}} / \sum V$, CI bootstrap theo đơn vị.
* H2-D: $\Delta\text{Cov}_{\text{OOD}}$ và FDP thực đo ở Night/Rain/Snow/Fog. Báo cáo CP 95% cho FDP. **Không** gọi đây là bảo chứng (A3).
* Ablation của FCD: $\bar D$, $s^2_D$, $\max_m D_m$, $\bar D + \lambda s_D$.

### 4.5. Công suất của các cách gộp FCD (mô phỏng, tái tạo bảng ở [02, D5.4](02_MATH_EXPLAINED.md))
200.000 lần lặp; mẫu bình thường có $D_m \sim \mathcal N(7.97, 0.71^2)$ độc lập; ngưỡng của mỗi cách gộp ở phân vị 95% của mẫu bình thường (anchor 9.13, $\bar D$ 8.64, $s^2$ 1.50, max 9.47, $\bar D + s$ 9.48).

| Kiểu lệch | $\Delta$ | Anchor | $\bar D$ | $s^2$ | max | $\bar D + s$ |
|---|---|---|---|---|---|---|
| Một phần | 2.0 | 0.33 | 0.50 | 0.53 | **0.77** | 0.75 |
| Một phần | 3.0 | 0.36 | 0.79 | 0.88 | **0.98** | **0.98** |
| Đồng thuận | 1.5 | 0.68 | **0.98** | **0.05** | 0.87 | 0.88 |
| Đồng thuận | 3.0 | 1.00 | 1.00 | **0.05** | 1.00 | 1.00 |

Phương sai thuần có công suất 0.05 với lệch đồng thuận, đúng bằng tỉ lệ báo động nhầm, tức không phát hiện được gì.

---

## 5. H3: Lợi ích cho Active Learning

### 5.1. Estimand và phép kiểm
* **Estimand chính:** $\Delta_T = \mathbb E_{\text{seed}}[\text{mAP}_{50:95}(\text{Proposed}, T) - \text{mAP}_{50:95}(\text{Random}, T)]$, tức chênh lệch ở **vòng cuối**.
* **Ghép cặp theo seed:** cùng seed dùng cùng tập $L_0$, cùng khởi tạo và cùng thứ tự dữ liệu cho mọi arm. Hiệu $d_s$ loại bỏ nhiễu chung.
* **Phép kiểm:** t có cặp một phía trên $\{d_s\}$, kèm Cohen $d_z = \bar d / s_d$, CI t 95%, và CI bootstrap theo ảnh test trong từng seed (để tách nhiễu do tập test khỏi nhiễu do seed).
* **Đa so sánh:** Proposed so với Random, Least-Conf, Ensemble-entropy: Holm ở mức 0.05.
* **Endpoint phụ:** diện tích dưới đường học (ALC) qua các vòng; IUT "vượt ở mọi vòng" (Berger, 1982).

### 5.2. Công suất theo số seed (t có cặp một phía, $\alpha = 0.05$, công thức noncentral t)
$\sigma_d$ = độ lệch chuẩn của $d_s$ giữa các seed (chưa biết, Mục 7 mô tả cách ước lượng):

| $\sigma_d$ | $\Delta$ | n = 3 | 5 | 8 | 10 | 15 | **n cho công suất 0.8** |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.5 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 3 |
| 1.0 | 1.5 | **0.53** | 0.86 | 0.98 | 1.00 | 1.00 | **5** |
| 1.5 | 1.5 | 0.32 | 0.58 | 0.82 | 0.90 | 0.98 | **8** |
| 2.0 | 1.5 | 0.23 | 0.40 | 0.61 | 0.71 | 0.87 | **13** |
| 1.5 | 2.0 | 0.46 | 0.79 | 0.96 | 0.99 | 1.00 | 6 |
| 2.0 | 3.0 | 0.53 | 0.86 | 0.98 | 1.00 | 1.00 | 5 |

**Kết luận:** thiết kế cũ với 3 seed chỉ có công suất 0.53 ngay cả khi nhiễu giữa các seed vừa phải ($\sigma_d = 1$). **Đề xuất:** ít nhất 8 seed. Nếu pilot cho $\sigma_d \le 1$ thì có thể giảm xuống 5.

### 5.3. Quy tắc cũ và IUT (mô phỏng, 20.000 lần lặp, tương quan giữa các vòng 0.5, $T = 3$)
| $\sigma_d$ | n | $\Delta$ | Chỉ kiểm vòng cuối | Quy tắc cũ ($\bar d \ge 1.5$ **và** $p < 0.05$) | IUT mọi vòng |
|---|---|---|---|---|---|
| 1.0 | 3 | 1.5 | 0.54 | 0.37 | 0.23 |
| 1.0 | 5 | 1.5 | 0.86 | **0.50** | 0.70 |
| 1.0 | 10 | 1.5 | 1.00 | **0.51** | 0.99 |
| 1.5 | 10 | 1.5 | 0.90 | **0.50** | 0.77 |
| 1.0 | 5 | 2.5 | 1.00 | 0.99 | 0.99 |

* Quy tắc cũ có cùng lỗi với H2: khoảng 50% công suất tại đúng hiệu ứng mục tiêu, dù thêm bao nhiêu seed.
* IUT đòi hỏi mọi vòng đều có ý nghĩa nên cần nhiều seed hơn (10 seed cho 0.99 khi $\sigma_d = 1$).
* **Quy tắc mới:** chấp nhận H3a khi t có cặp ở vòng $T$ có p đã hiệu chỉnh Holm $< 0.05$, **và** cận dưới CI 95% của $\Delta_T > 0.5$ điểm (biên tầm thường). IUT báo cáo là endpoint phụ.

### 5.4. H3b: không thua kém (non-inferiority)
* $H_0: \text{mAP}(\text{LTT}) - \text{mAP}(\text{heuristic}) \le -0.5$ điểm, t có cặp theo seed (`paired_t_greater(d, margin=-0.5)`).
* Thêm: tỉ lệ nhãn giả sai thực đo (FDP pool của nhãn Green, tính được vì nhãn GT của pool AL có sẵn nhưng bị ẩn), so sánh bằng Wilcoxon theo seed × vòng.

### 5.5. Chi phí nhãn
Nhãn của các tập hiệu chuẩn theo vòng $\mathcal D^{(t)}_{\text{cal}}$ phải được cộng vào ngân sách của arm Proposed. Báo cáo hai phiên bản của Bảng 3: (a) cùng số nhãn **chọn** cho người, (b) cùng **tổng** số nhãn người (Random được thêm đúng số nhãn hiệu chuẩn).

---

## 6. Kiểm thử

### 6.1. Tầng 1: unit test cho code thống kê (`fcd_v3_research/tests/test_risk_stats.py`)
Kết quả chạy ngày 15/10/2026: **19/19 test pass** (khoảng 2 giây).

| Nhóm | Test | Kiểm tra điều gì |
|---|---|---|
| HB | `test_table_5_3_values` | $p_{\text{Hoeff}}$, $p_{\text{Bentkus}}$, $p_{\text{HB}}$ khớp bảng 5.3 của H1 tới 4 chữ số |
| HB | `test_table_5_4_kmax_alpha_002`, `test_table_5_5_kmax_alpha_010` | $k_{\max}(N)$ khớp **toàn bộ** bảng 5.4 và 5.5 của tài liệu H1 (ví dụ 18 tại $N$ = 1.500, 1.517 tại 80.000) |
| HB | `test_minimum_n` | Mốc $N \ge 149$ ($\alpha = 0.02$) và $N \ge 29$ ($\alpha = 0.10$) |
| HB | `test_ucb_duality` | UCB $\le \alpha \iff p(\alpha) \le \delta$ (Mục 6.1 của H1) |
| HB | `test_type_one_error_binary_worst_case` | Sai lầm loại I $\le \delta$ với $Z$ nhị phân (trường hợp xấu nhất), 20.000 lần lặp |
| LTT | `test_toy_path_from_doc_b44` | Ví dụ đồ chơi ở [02, B4.4](02_MATH_EXPLAINED.md) chọn đúng bước 4 |
| LTT | `test_empty_set` | Trả về tập rỗng khi điểm đầu tiên không bác bỏ được |
| LTT | `test_fwer_with_unsafe_points_on_path` | Đầu-cuối: đường đi có điểm nguy hiểm, $\mathbb P(\text{chọn điểm nguy hiểm}) \le \delta$ |
| LTT | `test_naive_rule_violates` | **Đối chứng âm:** quy tắc plug-in vi phạm > 20%, tức test đủ nhạy để bắt quy trình sai |
| Kiểm định | `test_wilcoxon_hand_example` | $W^+ = 19$, $p = 3/64$ đúng như ví dụ tính tay ở [02, E2](02_MATH_EXPLAINED.md) |
| Kiểm định | `test_holm_known`, `test_holm_dominates_bonferroni` | p hiệu chỉnh Holm đúng; Holm luôn bác bỏ ít nhất bằng Bonferroni |
| Kiểm định | `test_paired_t_and_power`, `test_iut` | t có cặp, công suất tại $\Delta = 0$ bằng $\alpha$, n cho công suất 0.8, IUT |
| CI | `test_clopper_pearson_coverage`, `test_cluster_bootstrap_ratio_coverage` | Độ phủ thực tế của CI $\ge$ mức danh nghĩa |
| Cụm | `test_icc_recovers_truth` | Ước lượng ICC(1) khôi phục $\rho$ thật; công thức Kish |

### 6.2. Tầng 2: kiểm chứng bằng mô phỏng (`fcd_v3_research/scripts/02_statistical_validation.py`)
* Tính hợp lệ của LTT theo clip, và mức vi phạm theo frame dưới các cấu trúc tương quan (Mục 3.2).
* Công suất và sai lầm loại I của các quy tắc quyết định H2 (Mục 4.2) và H3 (Mục 5.2–5.3).
* Công suất của các cách gộp FCD (Mục 4.5).
* Báo động nhầm theo ảnh của cổng phân vị 95% (P12 trong blueprint): 1 box 5%, 5 box 23%, 10 box 40%, 20 box 64%, 40 box 87%. Muốn giữ 5% theo ảnh với $n$ box, mức cho mỗi box phải là $0.05/n$ (Bonferroni), ví dụ 0.0025 với 20 box.

### 6.3. Tầng 3: phép kiểm đối chứng trên dữ liệu thật (đăng ký trước, chạy cùng thực nghiệm chính)
Mục đích: phát hiện lỗi của pipeline (ghép nhãn, alignment, rò rỉ) trước khi tin vào kết quả. Mỗi phép kiểm có **kết quả mong đợi** được ghi trước. Nếu không đạt thì dừng và sửa, không báo cáo kết quả chính.

| # | Phép kiểm | Cách làm | Kết quả mong đợi |
|---|---|---|---|
| T1 | **Tín hiệu ngẫu nhiên** (đối chứng âm) | $S(j) \sim \text{Uniform}(0, 1)$, chạy nguyên quy trình LTT | Rủi ro trên test $\le \alpha$ (bảo chứng không phụ thuộc tín hiệu), Coverage thấp nhất trong các tín hiệu |
| T2 | **Tín hiệu oracle** (đối chứng dương) | $S(j) = 0$ nếu box là TP theo GT, 1 nếu FP | Coverage cao nhất; cho biết mức trần mà tín hiệu thật có thể đạt |
| T3 | **Hoán vị tín hiệu** | Xáo trộn $S$ giữa các box trong cùng đơn vị | $\Delta\text{Cov}$ của FCD so với bản xáo trộn có ý nghĩa; nếu không, tín hiệu không mang thông tin |
| T4 | **Bơm FP đã biết** | Thêm $k$ box giả ở vị trí không có GT vào tập ứng viên | Số FP đo được tăng đúng $k$ (kiểm code ghép nhãn) |
| T5 | **Xóa GT** (Lemma A4) | Xóa ngẫu nhiên 5% nhãn GT | FDP quan sát tăng, không giảm (tính đơn điệu của ghép tham lam) |
| T6 | **Tự gióng hàng** | Dùng cùng một mô hình làm cả 3 thành viên ensemble | $\sigma^2_{\text{loc}} = 0$, $D_{\text{JS}} = 0$, $s^2_D = 0$ cho mọi box; Hungarian trả về hoán vị đồng nhất |
| T7 | **Hệ trục latent** | Xoay latent của member bằng một ma trận trực giao ngẫu nhiên trước PCA | $D_m$ không đổi (Mahalanobis bất biến với biến đổi tuyến tính khả nghịch khi fit lại thống kê); bắt lỗi kiểu `fcd_variance.py` cũ |
| T8 | **Không rò rỉ** | Đếm giao của ride proxy giữa Train_filtered, Meta-Train, cal, test, pool AL | Bằng 0 (assert trong code) |
| T9 | **Tái lập** | Chạy lại với cùng seed | $\hat\lambda$, Coverage, p-value giống hệt |
| T10 | **Đối chiếu COCO** | So TP/FP của code ghép nhãn với `pycocotools` trên 200 ảnh tại cùng IoU | Khớp 100% số TP |

---

## 7. Thay giả định bằng ước lượng pilot (trước khi đăng ký trước)
Các con số ở trên phụ thuộc vào giả định đầu vào. Trước khi đăng ký trước, ước lượng chúng trên **Meta-Train** (dùng được mà không vi phạm A5) rồi chạy lại script:

| Giả định trong mô phỏng | Giá trị đã dùng | Ước lượng thay thế từ đâu | Ảnh hưởng tới |
|---|---|---|---|
| Số GT mỗi đơn vị | $1 + \text{Poisson}(15)$ | Đếm trên nhãn Meta-Train | SE của $\Delta\text{Cov}$ (H2) |
| Độ khó giữa các đơn vị $\sigma_u$ | 0.8 (logit) | Phương sai của Coverage theo đơn vị ở ngưỡng tham chiếu | SE của $\Delta\text{Cov}$ |
| Tương quan giữa hai tín hiệu $\rho_{\text{obj}}$ | 0.5 và 0.8 | Tỉ lệ đồng ý chấp nhận/từ chối giữa FCD và baseline | SE của $\Delta\text{Cov}$ |
| Coverage baseline | 55% | Coverage plug-in trên Meta-Train | Vị trí trên đường cong |
| $\sigma_b, \phi$ (frame) | xem 3.2 | ICC và tự tương quan theo độ trễ của FDP trên MOT (Exp A) | VR Arm 2, $D_{\text{eff}}$ |
| $\sigma_d$ (seed, H3) | 0.5–2.0 | Huấn luyện vòng 0 với 3 seed, rồi 1 vòng AL với 2 arm × 3 seed | Số seed cần cho H3 |
| Phân phối $D_m$ của mẫu bình thường | $\mathcal N(7.97, 0.71^2)$ độc lập | Phân phối thực nghiệm của $D_m$ trên box TP của Meta-Train, cả tương quan giữa các mô hình | Công suất của các cách gộp FCD |

**Lưu ý:** không dùng 200 ảnh val trong `ai_router/data/100k/val` làm pilot nếu chúng thuộc BDD100K Val, vì Val là nguồn của tập cal/test (P1–P2 của H1). Pilot chỉ chạy trên Meta-Train sau khi đã tách theo super-block.

---

## 8. Mẫu báo cáo kết quả (để mentor/reviewer thấy đủ giá trị thống kê)

**H1 (một dòng trong Bảng 1):**
> B-block, $N = 1.500$ đơn vị, $B = 1.000$ vòng: $\widehat{\text{VR}} = [x.xxx]$ (CP 95%: [a, b]; bootstrap phân hoạch: [c, d]); Yield = [x.xx] (cận dưới CP 95% [a]); $\hat D_{\text{eff}} = [x.xx]$ (CI 95% [a, b]).

**H2 (một dòng trong Bảng 2):**
> FCD so với Mahalanobis: $\widehat{\Delta\text{Cov}} = [+x.x]$ điểm (CI 95% bootstrap có cặp [a, b]), Wilcoxon $W^+ = [\cdot]$, $p_{\text{Holm}} = [\cdot]$, rank-biserial $r = [\cdot]$; kiểm định có biên $H_0: \Delta \le 2$: $z = [\cdot]$, $p = [\cdot]$; FDP test của FCD = [x.xx]% (CP 95% [a, b]).

**H3:**
> Sau $T = 3$ vòng, $n = [8]$ seed: $\bar d = [+x.x]$ mAP (CI 95% [a, b]), $t_{(7)} = [\cdot]$, $p_{\text{Holm}} = [\cdot]$, $d_z = [\cdot]$; IUT qua 3 vòng: [đạt/không đạt].

---

## 9. Thay đổi so với các tài liệu giả thuyết cũ (cần cập nhật bản đăng ký trước)
1. **H2:** thay quy tắc "$\widehat\Delta \ge 5$ và $p < 0.0025$" bằng quy tắc có biên ở Mục 4.2 (quy tắc cũ có công suất ≈ 0.5 tại hiệu ứng mục tiêu). Dùng Holm thay Bonferroni.
2. **H3:** tối thiểu 8 seed thay cho 3. Endpoint chính là vòng cuối. Quy tắc mới ở Mục 5.3. IUT là endpoint phụ.
3. **H1:** $B = 1.000$ vòng cho Exp B (con số được biện minh ở Mục 3.1). H1b chỉ kết luận khi đo được $D^{\text{frame}}_{\text{eff}}$ đủ lớn (Mục 3.2).
4. **Thêm** tầng kiểm thử 3 (T1–T10) vào phần Experiments của paper hoặc vào phụ lục.
