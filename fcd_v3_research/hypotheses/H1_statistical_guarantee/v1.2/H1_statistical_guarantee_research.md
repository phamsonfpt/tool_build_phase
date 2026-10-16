# 🏛️ GIẢ THUYẾT 1 (H1): THE STATISTICAL GUARANTEE HYPOTHESIS
## Khung Lý Thuyết, Chứng Minh Toán Học & Thiết Kế Thực Chứng Hoàn Chỉnh Cho Bài Báo Khoa Học (Final Manuscript Standard)

**Phiên bản 1.2 — bản đăng ký trước (pre-registration release), 13/10/2026.** Các giá trị đánh dấu "tham khảo" được chốt theo quy tắc đã nêu sau bước P1 (Mục 8.0), trước khi mở bất kỳ dữ liệu đánh giá nào.

*Thay đổi so với bản trước:* gộp các sửa đổi từ bản đánh giá v1.0 (loại rò rỉ cấp chuyến đi, tách Meta-Train theo super-block, chạy Exp A tầng 2 trong P3, quy tắc $r^*$, phá hòa cố định khi ghép cặp), và thêm ba sửa đổi: (1) kéo dài hai đường đi qua mức trần rủi ro kèm điều kiện kiểm định được (Mục 6.2, IX); (2) quy tắc liên kết ride proxy dạng chuỗi kèm vùng chắn rò rỉ riêng (bước P1); (3) mô phỏng năng lực phát hiện của C2 (bước P3.7).

*Thay đổi ở bản 1.2 (dựa trên mô phỏng edge case, xem file `H1_simulation_report.md`):* (1) thêm phép kiểm trực tiếp A1 bằng design effect thực nghiệm (C2b), vì HB bảo thủ tới mức C2 không phát hiện được vi phạm A1 có design effect dưới khoảng 5 (Mục 5.6); bước MDV của bản 1.1 được thay bằng bước ước lượng biên chịu đựng $D^*$ và năng lực của C2b; (2) tiêu chí vi phạm của C2 và C5 dùng rủi ro chính xác trên toàn pool thay cho phần bù và $e_b^{\text{cons}}$, vốn làm mất phần lớn tín hiệu; (3) C4 đặt trên design effect theo frame (so với biên chịu đựng $\hat D^*_C$) thay cho ICC $\ge 0.30$, vì ICC của FDP theo frame nhỏ do nhiễu nhị thức dù design effect lớn; (4) ngưỡng của quy tắc R-unit đổi từ 1.2 thành $\min(2, \hat D^*)$ và Meta-Train tăng lên 10% số super-block, vì ngưỡng cũ gộp đơn vị thô sai trong tới 85% số trường hợp khi A1 đúng; (5) nêu rõ rò rỉ dư của ride proxy và thêm phân tích độ nhạy D8.

---

## I. ĐẶT VẤN ĐỀ & PHÁT BIỂU CHÍNH THỨC (FORMAL PRE-REGISTRATION)

### 1.1. Bối cảnh & Động lực nghiên cứu
Trong quy trình gán nhãn chủ động (Active Learning) và tự động gán nhãn giả có chọn lọc (Selective Pseudo-Labeling) cho bài toán Nhận diện Vật thể (Object Detection) trên xe tự hành, việc đưa các dự đoán sai (False Positives / Hallucinations) vào tập huấn luyện mới sinh ra **Thiên kiến Xác nhận (Confirmation Bias)** mang tính tích lũy.

Để thay thế các ngưỡng heuristics cảm tính (ví dụ: cố định Softmax Confidence $> 0.95$), nghiên cứu này áp dụng khung kiểm soát rủi ro thống kê có bảo chứng hữu hạn mẫu (Finite-Sample Risk Control). Tuy nhiên, **trước khi đem khung đo lường đi so sánh các tín hiệu bất định nhận thức (H2) hoặc đánh giá hiệu quả Active Learning (H3)**, bản thân khung đo lường phải được chứng minh là có giá trị toán học và kiểm soát đúng cam kết rủi ro trên dữ liệu thực tế.

### 1.2. Phát biểu Giả thuyết H1 (Tách bạch H1a, H1b và H1c)
Để đảm bảo tính khả bác (falsifiability) và tính độc lập của các mệnh đề khoa học, Giả thuyết 1 được phân tách thành ba thành phần: tính hợp lệ của bảo chứng (H1a), cơ chế sụp đổ của việc chia theo Frame (H1b), và tính không tầm thường của quy trình trên một detector cụ thể (H1c).

> **Hypothesis 1a (The PAC Risk Control Guarantee):**
> *Khi đơn vị thống kê được thiết lập tại cấp độ cụm độc lập (Ride-level Cluster, hoặc Block / Super-block nếu quy tắc R-unit ở Experiment A yêu cầu gộp; xem Assumption A1), quy trình Learn-then-Test (LTT) kết hợp Fixed-Sequence Testing trên một đường đi tham số cố định tiên nghiệm thiết lập ngưỡng lọc $\hat{\tau}$ kiểm soát kỳ vọng rủi ro quần thể $\mathcal{R}(\hat{\tau}) \le \alpha = 0.02$ với độ tin cậy hữu hạn mẫu $1 - \delta \ge 0.95$ trên phân phối kiểm thử cùng phân phối ($\mathcal{P}$):*
> $$\mathbb{P}_{\mathcal{D}_{\text{cal}} \sim \mathcal{P}^N}\left(\mathcal{R}(\hat{\tau}) \le \alpha\right) \ge 1 - \delta$$
> *Bảo chứng được kiểm chứng trên các nhóm dữ liệu **chưa từng xuất hiện** trong tập hiệu chuẩn ở cấp super-block, tức ô 5 km × ngày lịch (held-out theo super-block, Mục 8.2), vì đây là kịch bản triển khai thật: ngưỡng được dùng trên các chuyến đi mới.*
> *Kiểm chứng H1a gồm hai phần tách biệt: (i) bảo chứng không bị vi phạm trên dữ liệu thật (C2); (ii) tiền đề A1 đúng ở cấp super-block, đo trực tiếp bằng design effect thực nghiệm (C2b). Phần (ii) là cần thiết vì HB đủ bảo thủ để C2 vẫn đạt ngay cả khi A1 bị vi phạm vừa phải (Mục 5.6); khi đó bảo chứng quan sát được là xấp xỉ chứ không còn là hệ quả của Theorem 1.*

> **Hypothesis 1b (The Frame-Level Structural Breakdown):**
> *Việc phân chia dữ liệu ngẫu nhiên theo Frame khi huấn luyện/đánh giá chuỗi video làm sụp đổ cỡ mẫu hiệu dụng ($N_{\text{eff}} \ll N_{\text{nominal}}$) do tương quan nội cụm cao ($\rho_{\text{intra}} > 0$) và gây rò rỉ ngữ cảnh, khiến các bất đẳng thức tập trung tính trên số lượng frame danh nghĩa trở nên quá lạc quan (under-conservative), dẫn đến xác suất vi phạm $\mathbb{P}(\mathcal{R}(\hat{\tau}) > \alpha)$ vượt đáng kể ngân sách $\delta$ trên các đơn vị kiểm thử độc lập mới. Vì ngưỡng tính theo $N_{\text{nominal}}$ thường tiến sát $\alpha$ (Mục 5.4, 5.5), mức vi phạm kỳ vọng ở cỡ vài chục phần trăm chứ không nhất thiết là vượt trần với độ lớn lớn.*
> *H1b là mệnh đề về cơ chế, không gắn với mức vận hành $\alpha = 0.02$ của H1a, nên được kiểm chứng ở mức $\alpha_C = 0.10$ (lý do ở Mục 8.3).*

> **Hypothesis 1c (Non-Vacuity):**
> *Với detector $\mathcal{M}$ và đường đi $\Lambda_{1D}$ đã chọn, quy trình không rỗng với xác suất cao tại cỡ mẫu hiệu chuẩn đã đăng ký $N$ (mặc định $1.500$; nhỏ hơn nếu quy tắc R-unit buộc giảm, Mục 8.1):*
> $$\mathbb{P}_{\mathcal{D}_{\text{cal}} \sim \mathcal{P}^N}\left(\hat{\tau} \ne \emptyset\right) \ge 1 - \beta, \quad \beta = 0.15$$
> *và khi $\hat\tau \ne \emptyset$, lượng nhãn giả đúng mà $\hat\tau$ tạo ra không tầm thường so với cách chọn ngưỡng thông thường không có bảo chứng (baseline plug-in, bước P3, Mục 8.0): tỉ lệ độ phủ tương đối $\rho_{\text{cov}}(\hat\tau) \ge \gamma_{\text{cov}}$ (định nghĩa và ngưỡng ở Mục 8.2, chỉ số C3b). Điều kiện thứ hai loại trừ trường hợp $\hat\tau \ne \emptyset$ nhưng chỉ chấp nhận một lượng box không đáng kể.*
> *Đây là tính chất của cặp $(\mathcal{M}, \Lambda_{1D})$, không phải hệ quả của lý thuyết LTT. Thất bại của H1c không bác bỏ H1a: bảo chứng vẫn hợp lệ nhưng không mang thông tin thực tiễn.*

### 1.3. Định vị Đóng góp (Positioning)
Tính hợp lệ của LTT dưới giả định i.i.d. là kết quả đã được chứng minh (Angelopoulos et al., 2025); H1a vì vậy không tự nó là đóng góp mới mà là điều kiện tiên quyết để các giả thuyết sau (H2, H3) đứng vững. Đóng góp thực chứng chính của H1 nằm ở hai điểm:
1. **H1b:** định lượng mức sụp đổ của bảo chứng khi bỏ qua cấu trúc phụ thuộc theo thời gian của dữ liệu video lái xe, trên cùng một detector và cùng dữ liệu.
2. **Giao thức chọn đơn vị thống kê** (quy tắc R-unit, Mục 8.1) kèm phép kiểm đầu-cuối held-out theo super-block (Mục 8.2), áp dụng được cho các bài toán hiệu chuẩn trên dữ liệu video khác.
3. **Biên chịu đựng tương quan của HB** (Mục 5.6): đo trên dữ liệu thật mức design effect mà LTT dùng bất đẳng thức Hoeffding-Bentkus còn chịu được trước khi bảo chứng bị vi phạm, cùng với design effect thực nghiệm ở cấp chuyến đi (C2b) và cấp frame (C4).

Nghiên cứu này thuộc dòng công trình về suy diễn phi tham số khi dữ liệu không trao đổi được (non-exchangeable). Barber et al. (2023) chứng minh rằng độ phủ của conformal prediction suy giảm theo mức độ vi phạm tính trao đổi được, và Oliveira et al. (2024) phân tích split conformal trên dữ liệu phụ thuộc (chuỗi β-mixing). Khác với các công trình này, H1 không đề xuất bất đẳng thức mới cho dữ liệu phụ thuộc mà chọn lại đơn vị thống kê để phục hồi giả định i.i.d., rồi kiểm chứng thực nghiệm lựa chọn đó.

---

## II. HỆ THỐNG KÝ HIỆU & KHUNG KHÁI NIỆM CHUẨN XÁC

| Ký hiệu | Ý nghĩa toán học | Miền giá trị |
|---|---|---|
| $U_i = (X_i, Y_i)$ | Đơn vị thống kê độc lập thứ $i$ (Ride-level Cluster / Driving Session; hoặc Block / Super-block nếu quy tắc R-unit yêu cầu gộp) | Không gian $\mathcal{U}$ |
| $\mathcal{D}_{\text{cal}} = \{U_1, \dots, U_N\}$ | Tập hiệu chuẩn gồm $N$ đơn vị độc lập | $|\mathcal{D}_{\text{cal}}| = N$ |
| $\mathcal{M}$ | Mô hình Object Detector đóng băng (RT-DETR, Zhao et al., 2024; bản Base / Anchor) | Cố định trọng số |
| $\mathcal{C}_{\text{eval}}$ | Tập lớp đánh giá chung cho mọi thực nghiệm (Mục 2.1) | Tập con các lớp BDD100K |
| $\tau = (\tau_{\text{cls}}, \tau_{\text{loc}}, \tau_{\text{fcd}})$ | Vector tham số ngưỡng lọc kiểm định | $\tau \in \Lambda \subset \mathbb{R}^3$ |
| $N_i^{\text{acc}}(\tau)$ | Tổng số bounding box dự đoán trong đơn vị $U_i$ được chấp nhận tại ngưỡng $\tau$ | $\mathbb{N}_0$ |
| $V_i(\tau)$ | Số lượng False Positives trong đơn vị $U_i$ tại ngưỡng $\tau$ | $0 \le V_i(\tau) \le N_i^{\text{acc}}(\tau)$ |
| $\text{FDP}_i(\tau)$ | Tỷ lệ khám phá sai cục bộ của đơn vị $U_i$: $\text{FDP}_i(\tau) \triangleq \frac{V_i(\tau)}{\max(N_i^{\text{acc}}(\tau),\, 1)}$ | $[0, 1]$ |
| $Z_i(\tau)$ | Thống kê rủi ro của đơn vị $i$; mặc định $Z_i = \text{FDP}_i$ (Exp B), trung bình FDP theo frame ở Exp C (Mục 2.3) | $[0, 1]$ |
| $\mathcal{R}(\tau)$ | Rủi ro kỳ vọng quần thể: $\mathcal{R}(\tau) \triangleq \mathbb{E}_{U \sim \mathcal{P}}[Z(U;\, \tau)]$ | $[0, 1]$ |
| $\widehat{\mathcal{R}}_N(\tau)$ | Rủi ro thực nghiệm trên tập hiệu chuẩn: $\widehat{\mathcal{R}}_N(\tau) \triangleq \frac{1}{N}\sum_{i=1}^N Z_i(\tau)$ | $[0, 1]$ |
| $\alpha$ | Mức trần rủi ro danh nghĩa (Nominal Risk Bound, mặc định $0.02 \equiv 2\%$; dùng cho H1a, H1c) | $(0, 1)$ |
| $\alpha_C$ | Mức trần rủi ro của Experiment C (kiểm chứng H1b), $\alpha_C = 0.10$ | $(0, 1)$ |
| $\delta$ | Ngân sách sai số kiểm định (Significance Level / Failure Probability, mặc định $0.05$) | $(0, 1)$ |

### 2.1. Quy tắc Khớp Cặp Song Ánh (Bipartite Ground Truth Matching)
Định nghĩa một phát hiện $\hat{b} \in \mathcal{B}_i^{\text{pred}}$ là **False Positive (FP)** đòi hỏi quy tắc đối sánh rõ ràng với tập nhãn chuẩn $\mathcal{Y}_i = \{b_k^{\text{gt}}\}_{k=1}^{M_i}$:
0. **Lọc tiền xử lý (cố định tiên nghiệm):**
   * *Tập lớp đánh giá:* $\mathcal{C}_{\text{eval}}$ là giao của tập lớp bộ Detection và tập lớp bộ MOT của BDD100K (được xác minh ở bước P1, Mục 8.0). Theo hiểu biết của chúng tôi, bộ MOT gồm 8 lớp (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle) và không có traffic light / traffic sign; nếu đúng, $\mathcal{C}_{\text{eval}}$ gồm 8 lớp này. Dự đoán và nhãn chuẩn thuộc lớp ngoài $\mathcal{C}_{\text{eval}}$ bị loại trước khi lọc ngưỡng (không tính vào $N_i^{\text{acc}}$ hay $V_i$). Cùng một $\mathcal{C}_{\text{eval}}$ được dùng cho Meta-Train, Exp A, B và C.
   * *Vùng bỏ qua:* nếu nhãn có vùng ignore/crowd, dự đoán có IoA (diện tích giao chia diện tích dự đoán) $\ge 0.5$ với một vùng như vậy bị loại khỏi cả $N_i^{\text{acc}}$ và $V_i$. Số dự đoán bị loại được báo cáo.
1. **Ma trận chi phí:** Với mỗi cặp $(\hat{b}_j, b_k^{\text{gt}})$, tính $\text{IoU}(\hat{b}_j, b_k^{\text{gt}})$.
2. **Quy tắc ghép cặp tham lam:** Sắp xếp các dự đoán theo điểm tin cậy giảm dần. Mỗi dự đoán lần lượt được ghép với nhãn chuẩn chưa ghép có IoU lớn nhất (khi IoU bằng nhau, chọn nhãn đứng trước theo thứ tự cố định trong file nhãn, để phép chọn luôn xác định duy nhất) thỏa mãn
   $$\text{class}(\hat{b}_j) = \text{class}(b_k^{\text{gt}}) \quad \land \quad \text{IoU}(\hat{b}_j, b_k^{\text{gt}}) \ge \theta_{\text{IoU}}$$
   và khi đó được công nhận là True Positive (TP); mỗi ground truth $b_k^{\text{gt}}$ chỉ được ghép cặp tối đa một lần.
3. **Định nghĩa False Positive:**
   $$\hat{b}_j \text{ là FP} \iff \hat{b}_j \text{ không được ghép với bất kỳ } b_k^{\text{gt}} \text{ nào thỏa mãn đồng thời hai điều kiện trên.}$$
4. **Phân rã FP (chỉ để báo cáo):** mỗi FP ở ngưỡng $\theta_{\text{IoU}}$ đang dùng được xếp vào đúng một loại, dựa trên IoU lớn nhất của nó với các nhãn chuẩn cùng lớp (kể cả nhãn đã bị dự đoán khác ghép): $V_i^{\text{halluc}}$ đếm các FP có IoU lớn nhất $< 0.5$ (ảo giác hoặc sai lớp); $V_i^{\text{loc}}$ đếm các FP có IoU lớn nhất $\ge 0.5$ (đúng đối tượng nhưng lệch vị trí, hoặc phát hiện trùng lặp). Theo định nghĩa $V_i = V_i^{\text{halluc}} + V_i^{\text{loc}}$. Phân rã này cho biết rủi ro đến từ ảo giác hay từ độ chính xác định vị.

> **Khóa cứng tiên nghiệm (A priori Protocol):** Ngưỡng ghép mặc định là $\theta_{\text{IoU}} = 0.75$ (COCO strict regime). Vì ngưỡng này gộp ảo giác với lệch vị trí nhẹ và có thể khiến H1c thất bại vì lý do không liên quan đến khung LTT, giá trị cuối cùng $\theta_{\text{IoU}} \in \{0.75, 0.5\}$ được chọn bởi **một quy tắc cố định ngay bây giờ** và chỉ áp dụng trên Meta-Train (bước P3, Mục 8.0), trước khi mở bất kỳ dữ liệu hiệu chuẩn/kiểm thử nào:
> * Giữ $\theta_{\text{IoU}} = 0.75$ nếu trên Meta-Train tồn tại $\tau \in G$ với $\widehat{\text{FDP}}_{\text{meta}}(\tau) \le r^*(\tau)$ và $\rho_{\text{cov,meta}}(\tau) \ge 0.6$, với $\rho_{\text{cov,meta}}$ là độ phủ trên Meta-Train chia cho độ phủ của baseline plug-in $\tau_{\text{base}}$ ở cùng $\theta$ (bước P3, Mục 8.0), và
>   $$r^*(\tau) = \frac{k_{\max}(N)}{N} - 1.645\,\frac{\hat\sigma(\tau)}{\sqrt{N}}$$
>   trong đó $k_{\max}(N)/N$ là mức rủi ro thực nghiệm tối đa HB chứng nhận được tại cỡ mẫu đã đăng ký (0.012 tại $N = 1.500$, bảng 5.4) và $\hat\sigma(\tau)$ là độ lệch chuẩn của $\text{FDP}_i(\tau)$ giữa các đơn vị của Meta-Train;
> * Ngược lại, dùng $\theta_{\text{IoU}} = 0.5$.
>
> *Lý do của $r^*$:* một điểm có rủi ro thật $r$ qua kiểm định HB với xác suất xấp xỉ $\Phi\big((k_{\max}/N - r)/(\sigma/\sqrt{N})\big)$, nên $r \le r^*$ bảo đảm điểm đó qua được trong khoảng 95% số vòng. Với phương sai FDP giữa các đơn vị cỡ 0.001–0.005 và $N = 1.500$, $r^*$ nằm trong khoảng 0.0090–0.0107. Mức 95% cao hơn mức C3b thực sự cần (C3b dùng trung vị); phần dư là biên cho sai số ước lượng trên Meta-Train, khác biệt giữa Meta-Train và Val, và việc fixed-sequence dừng sớm. Công thức tự điều chỉnh khi $N$ giảm.
>
> *Lý do của mốc 0.6:* ngưỡng LTT chọn ở mỗi vòng thường dừng sớm hơn điểm đạt $r^*$ một hai bước, nên có độ phủ thấp hơn; biên 0.1 so với $\gamma_{\text{cov}} = 0.5$ bù cho phần hụt này. Mốc 0.5 không có biên; mốc 0.7 quá khắt khe và dễ chuyển sang IoU 0.5 dù 0.75 khả thi.
>
> Kết quả ở ngưỡng không được chọn vẫn được báo cáo như phân tích phụ. Quy tắc này không tạo cơ hội P-hacking vì không chạm vào dữ liệu đánh giá.

### 2.2. Kết Nối: Macro-FDP vs. Micro-FDP & Hàm Mất Mát Phụ Trợ
* **Macro-FDP (Mục tiêu của LTT trong H1):** $\mathcal{R}(\tau) = \mathbb{E}\left[\frac{V_i(\tau)}{\max(N_i^{\text{acc}}(\tau), 1)}\right]$. Đây là kỳ vọng của tỷ lệ lỗi tính theo từng đơn vị thống kê. Do $\text{FDP}_i(\tau) \in [0, 1]$ bị chặn cứng, nó thỏa mãn trực tiếp các định lý tập trung mẫu hữu hạn.
* **Micro-FDP (Tỷ lệ nhiễu gộp của pool nhãn giả):** $\text{FDP}_{\text{pool}} = \frac{\sum_i V_i(\tau)}{\max(\sum_i N_i^{\text{acc}}(\tau), 1)}$.
* **Hạn chế của Macro-FDP (được giám sát bằng chỉ số mô tả):** Đơn vị có $N_i^{\text{acc}} = 0$ được tính $\text{FDP}_i = 0$, và đơn vị có 1 box nặng ngang đơn vị có hàng trăm box. Ngưỡng càng chặt thì càng nhiều đơn vị rỗng, kéo Macro-FDP xuống mà không phản ánh chất lượng pool nhãn giả. Nếu các cảnh dày đặc vừa có nhiều box được chấp nhận vừa có FDP cao, Micro-FDP có thể vượt $\alpha$ trong khi Macro-FDP vẫn đạt. Vì vậy Exp B báo cáo kèm Micro-FDP, tỉ lệ đơn vị rỗng và tương quan Spearman giữa $N_i^{\text{acc}}$ và $\text{FDP}_i$ tại $\hat\tau$ (chỉ số D3, Mục 8.2). Các chỉ số này không thuộc bảo chứng của Theorem 1.
* **Cầu nối toán học qua Hằng số Tiên nghiệm:**
  Nếu muốn kiểm soát tỷ lệ lỗi của toàn bộ pool dữ liệu nhãn giả $\mathbb{E}\left[\sum V_i\right] / \mathbb{E}\left[\sum N_i^{\text{acc}}\right] \le \alpha$, ta có thể định nghĩa hàm mất mát phụ trợ bị chặn:
  $$L_i(\tau) \triangleq \frac{V_i(\tau) - \alpha N_i^{\text{acc}}(\tau)}{C_{\text{max}}} + \alpha$$
  trong đó $C_{\text{max}} = Q_{\text{max}} \times K_{\text{max}}^{\text{frames}}$ là **hằng số vật lý tiên nghiệm** ($Q_{\text{max}} = 300$ là số lượng object queries tối đa của kiến trúc RT-DETR, $K_{\text{max}}^{\text{frames}}$ là số keyframe tối đa trong một cụm).
  Khi đó $\mathbb{E}[L_i(\tau)] \le \alpha \iff \mathbb{E}[V_i(\tau)] \le \alpha \mathbb{E}[N_i^{\text{acc}}(\tau)] \iff \text{Micro-FDP} \le \alpha$ ở mức kỳ vọng.
  *(Ghi chú phương pháp luận: Do $L_i$ dồn rất mạnh quanh $\alpha$ và có phương sai nhỏ, các bất đẳng thức Hoeffding/Bentkus không khai thác phương sai sẽ bị bảo thủ đáng kể trên hàm loss này. Để kiểm soát chặt chẽ Micro-FDP với năng lực thống kê cao, các bất đẳng thức thích ứng phương sai dựa trên lý thuyết đánh cược (testing-by-betting martingales; Waudby-Smith & Ramdas, 2024) sẽ được xem xét ở các công trình mở rộng. Trong phạm vi H1, Macro-FDP được chọn làm chỉ số mục tiêu chính xác).*

### 2.3. Thống kê Rủi ro Tổng quát $Z_i$
Mọi kết quả ở Mục V–VII chỉ cần thống kê của đơn vị $Z_i(\tau) \in [0, 1]$ và các $Z_i$ là i.i.d.; chúng không phụ thuộc vào công thức cụ thể của $Z_i$. Tài liệu dùng hai dạng:
* **Exp A tầng 2 và Exp B:** $Z_i = \text{FDP}_i$ (tỉ số gộp trên toàn đơn vị, như bảng ký hiệu).
* **Exp C:** $Z_c = \frac{1}{M}\sum_{t=1}^{M} \text{FDP}_{c,t}$, trung bình FDP của $M = 200$ frame trong clip $c$. Dạng này được chọn để cả ba arm của Exp C cùng kiểm soát **một** đại lượng: rủi ro trung bình theo frame $\mathcal{R}_{\text{frame}}(\tau) = \mathbb{E}_c[Z_c(\tau)]$, bằng trung bình FDP trên mọi frame vì mỗi clip có đúng $M$ frame (Mục 8.3).

---

## III. HỆ THỐNG CÁC GIẢ ĐỊNH KHOA HỌC (FORMAL ASSUMPTIONS)

Để Theorem 1 và các kết quả toán học đứng vững, toàn bộ các giả định được quy định tường minh như sau:

* **Assumption A1 (Independent Statistical Units):**
  Đơn vị thống kê cơ bản $U_i \in \mathcal{U}$ là một **chuyến đi độc lập (Ride)**, hoặc một **nhóm không gian-thời gian (Block hoặc Super-block, định nghĩa ở bước P1, Mục 8.0)** nếu chẩn đoán ở Experiment A (Mục 8.1, quy tắc R-unit) cho thấy cần gộp các chuyến đi cùng nhóm. Các đơn vị $U_1, \dots, U_N$ được giả định rút độc lập từ phân phối $\mathcal{P}$ trên **đơn vị đã chọn**:
  $$U_i \overset{\text{i.i.d.}}{\sim} \mathcal{P}, \quad \forall i \in \{1, \dots, N\}$$
  *(Nguồn dữ liệu & giới hạn xác minh): Theo mô tả của Yu et al. (2020), BDD100K gồm 100.000 video clips từ hơn 50.000 chuyến đi. Theo hiểu biết của chúng tôi, BDD100K không công bố Ride ID hay Driver ID, vì vậy "chuyến đi" được xác định gần đúng bằng ride proxy dựng từ metadata GPS và thời gian trên **toàn bộ** dữ liệu (train, val và MOT), theo quy tắc liên kết dạng chuỗi ở bước P1 (Mục 8.0). Số lượng đơn vị thực tế trong tập Val là một **ước lượng** $\hat n$ (kế hoạch tham khảo $\approx 7.000$) và phải được đếm xong (bước P1, Mục 8.0) trước mọi phân tích có dùng nhãn hộp.*
  *(Clip thiếu metadata): Clip không có GPS hoặc thời gian bị loại khỏi pool đánh giá; $\mathcal{P}$ khi đó là phân phối của các clip có metadata. Tỉ lệ bị loại và phân phối các thuộc tính cảnh (thời tiết, loại cảnh, thời điểm trong ngày) của nhóm bị loại so với nhóm còn lại được báo cáo. Nếu tỉ lệ bị loại vượt 10%, chạy thêm phân tích độ nhạy coi mỗi clip thiếu metadata là một đơn vị riêng.*
  *(Cấu trúc phụ thuộc còn lại): Các chuyến đi có thể tương quan theo vị trí và thời gian. Hai cấp nhóm lồng nhau được định nghĩa ở bước P1 (Mục 8.0): block (ô 1 km × khung 2 giờ) và super-block (ô 5 km × ngày lịch). Experiment A đo ICC ở hai cấp này và Moran's $I$ giữa các super-block để quyết định mức gộp đơn vị (quy tắc R-unit, Mục 8.1). Tương quan **giữa** các super-block kề nhau không thể loại bỏ bằng thiết kế lấy mẫu; nó được xem là rủi ro dư của A1 và được kiểm tra đầu-cuối bằng nhánh **B-block** của Experiment B (held-out theo super-block, Mục 8.2).*
  *(Giới hạn không kiểm soát được): BDD100K được thu từ người dùng ứng dụng camera hành trình, nên một người có thể đóng góp nhiều chuyến đi. Tương quan theo cùng tài xế/đội xe không quan sát được, không được A1 kiểm soát, và không phát hiện được bằng B-block vì các chuyến của cùng một người nằm rải rác ở nhiều super-block. Đây là giới hạn được nêu rõ của nghiên cứu.*
  *(Tính nhất quán của $\mathcal{P}$): Cal và test luôn dùng **cùng định nghĩa đơn vị và cùng cơ chế lấy mẫu/trọng số** (không áp dụng phân tầng hay trọng số riêng cho một bên), để $\mathcal{P}$ là phân phối chung của cả hai (xem A3).*

* **Assumption A2 (Detector Independence & Non-Leakage Protocol):**
  Mô hình Object Detector $\mathcal{M}$ được huấn luyện hoàn tất và đóng băng trọng số trước khi thực hiện hiệu chuẩn. Tập huấn luyện $\mathcal{D}_{\text{train}}$ và tập hiệu chuẩn/kiểm thử $\mathcal{D}_{\text{cal}} \cup \mathcal{D}_{\text{test}}$ là hoàn toàn rời rạc ở cấp độ Ride/Video clip:
  $$\text{Rides}(\mathcal{D}_{\text{train}}) \cap \text{Rides}(\mathcal{D}_{\text{cal}} \cup \mathcal{D}_{\text{test}}) = \emptyset$$
  Không có bất kỳ frame nào (kể cả keyframe tại giây thứ 10 hay các frame lân cận trong cùng video clip) thuộc tập huấn luyện xuất hiện trong tập hiệu chuẩn hoặc tập kiểm thử.
  *(Giao thức thực thi):* Các clip MOT của BDD100K được giả định là trích từ cùng kho video với bộ Detection 100K, nên keyframe của một clip MOT có thể nằm trong $\text{Train}_{\text{det}}^{70\text{K}}$. Giả định này **chưa được xác minh** và phải được kiểm bằng cách đối chiếu video ID (tên video) giữa hai bộ trước khi huấn luyện.
  *(Rò rỉ cấp chuyến đi):* theo hiểu biết của chúng tôi, BDD100K chia train/val theo video chứ không theo chuyến đi (cần xác minh ở bước P1). Với trung bình khoảng 2 clip mỗi chuyến, phần lớn chuyến đi trong Val nhiều khả năng có clip anh em trong tập train: cùng tuyến đường, cùng ánh sáng, cùng thời tiết. Nếu không xử lý, detector đã "thấy" bối cảnh của chính các chuyến đi dùng để hiệu chuẩn và kiểm thử, và Exp B **không phát hiện được** rò rỉ này vì tập hiệu chuẩn và tập kiểm thử cùng hưởng lợi như nhau. Vì vậy detector được huấn luyện trên
  $$\text{Train}_{\text{filtered}} = \text{Train}_{\text{det}}^{70\text{K}} \setminus \left(\text{Rides}_{\text{Val}} \cup \text{Rides}_{\text{MOT}} \cup \text{Meta-Train}\right)$$
  trong đó $\text{Rides}_{\text{Val}}$ và $\text{Rides}_{\text{MOT}}$ là mọi clip train thuộc cùng ride proxy với ít nhất một clip Val hoặc MOT, hoặc nằm trong vùng chắn rò rỉ một bước của ride proxy đó (bước P1), kể cả chính các clip MOT trùng video ID, và Meta-Train là tập dùng để dựng các đường đi (Mục 8.0). Việc loại được thực hiện **ở cấp chuyến đi, không ở cấp super-block**: quần thể mục tiêu của H1a là các chuyến đi mới trong cùng các khu vực vận hành, nên huấn luyện trên cùng thành phố là điều bình thường khi triển khai thật, còn huấn luyện trên chính chuyến đi đó thì không. Không đặt trần cho tỉ lệ bị loại, vì tính hợp lệ được ưu tiên hơn chất lượng detector (vốn chỉ ảnh hưởng H1c). Mô phỏng ở mật độ 20–200 chuyến mỗi ngày trên một thành phố cho thấy phần bị loại khoảng 13–26% tập train (file mô phỏng, E6); con số thực tế (tách riêng phần do ride proxy và phần do vùng chắn rò rỉ) và chất lượng detector trên Meta-Train được báo cáo trong paper.
  *(Rò rỉ dư không loại được):* nếu các clip của cùng một chuyến cách nhau hơn 10 phút, không quy tắc nào dựa trên thời gian và GPS nối được chúng khi không có device ID. Trong mô phỏng, khi clip được lấy ở vị trí ngẫu nhiên trong chuyến, khoảng 20–25% số clip train dính rò rỉ vẫn lọt qua sau khi lọc. Rò rỉ dư này ảnh hưởng như nhau đến tập hiệu chuẩn và tập kiểm thử nên Exp B không phát hiện được; nó không làm sai bảo chứng trên quần thể Val nhưng có thể làm Val "dễ" hơn các chuyến hoàn toàn mới. Mức ảnh hưởng được định lượng bằng phân tích độ nhạy D8 (Mục 8.2).

* **Assumption A3 (Distributional Invariance for PAC Evaluation):**
  Tập hiệu chuẩn $\mathcal{D}_{\text{cal}}$ và tập kiểm thử $\mathcal{D}_{\text{test}}$ được rút từ cùng một phân phối $\mathcal{P}$ **trên cùng một định nghĩa đơn vị thống kê và cùng cơ chế lấy mẫu/trọng số** (xem A1). Trong Exp B và C, các nhóm dữ liệu được gán ngẫu nhiên vào phía hiệu chuẩn hoặc phía kiểm thử ở mỗi vòng (Mục 8.2–8.3), nên hai phía có cùng phân phối theo phép ngẫu nhiên hóa này. Các kịch bản kiểm thử ngoài phân phối (Out-of-Distribution - OOD: thời tiết cực đoan, thành phố mới) được xếp vào mục **Stress-test riêng biệt**, không thuộc phạm vi đảm bảo toán học của Theorem 1.

* **Assumption A4 (Robustness to Missed Ground Truth Annotations):**
  Tập nhãn chuẩn $\mathcal{Y}_i$ có thể tồn tại sai số do con người. Tác động của sai số nhãn lên rủi ro quan sát được phụ thuộc vào loại sai số:
  * **Nhãn bị bỏ sót (bảo thủ):** nhãn bị bỏ sót biến một dự đoán đúng thành một False Positive quan sát được. Tính chất này được chứng minh chặt chẽ cho quy tắc ghép tham lam ở Mục 2.1:

    > **Lemma A4 (Tính đơn điệu của quy tắc ghép tham lam).** *Với cùng một tập dự đoán (đã sắp theo confidence) và hai tập nhãn chuẩn $S' \subseteq S$, số FP khi ghép với $S'$ không nhỏ hơn số FP khi ghép với $S$.*
    >
    > *Chứng minh.* Chỉ cần xét $S' = S \setminus \{g\}$, rồi lặp lại cho từng nhãn bị bỏ. Chạy quy tắc tham lam song song trên $S$ và $S'$ với cùng thứ tự dự đoán (thứ tự theo confidence không phụ thuộc nhãn chuẩn). Gọi $A_t, A'_t$ là tập nhãn chuẩn còn trống trước dự đoán thứ $t$. Ta chứng minh bằng quy nạp bất biến
    > $$A'_t \subseteq A_t \quad \text{và} \quad |A_t \setminus A'_t| \le 1.$$
    > Ban đầu $A'_0 = A_0 \setminus \{g\}$ nên bất biến đúng. Tại bước $t$, tính hợp lệ của một cặp (cùng lớp, IoU $\ge \theta_{\text{IoU}}$) không phụ thuộc vào lần chạy. Gọi $x$ là nhãn hợp lệ được chọn trong $A_t$ và $x'$ là nhãn tương ứng trong $A'_t$ (có thể không tồn tại). Nhờ quy tắc phá hòa cố định ở Mục 2.1, phép chọn là "phần tử đứng đầu theo một thứ tự toàn phần cố định", nên được xác định duy nhất.
    > * Nếu $x \in A'_t$: vì $A'_t \subseteq A_t$ nên $x' = x$; cả hai lần chạy cùng bỏ $x$, bất biến giữ nguyên.
    > * Nếu $x \notin A'_t$: $x$ là phần tử duy nhất của $A_t \setminus A'_t$, nên $A'_t \subseteq A_t \setminus \{x\} = A_{t+1}$. Do đó $A'_{t+1} = A'_t \setminus \{x'\} \subseteq A_{t+1}$, và $A_{t+1} \setminus A'_{t+1} \subseteq \{x'\}$ có tối đa một phần tử.
    > * Nếu không có nhãn hợp lệ trong $A_t$ thì cũng không có trong $A'_t \subseteq A_t$; không lần chạy nào ghép.
    >
    > Vì $A'_t \subseteq A_t$ ở mọi bước, mỗi khi lần chạy trên $S'$ ghép được dự đoán thứ $t$ thì lần chạy trên $S$ cũng ghép được dự đoán đó. Vậy $\text{TP}(S') \le \text{TP}(S)$, tương đương $\text{FP}(S') \ge \text{FP}(S)$. $\blacksquare$

    Áp dụng với $S$ là tập nhãn đúng và $S'$ là tập nhãn quan sát được (bị bỏ sót), ta có $V_i^{\text{obs}}(\tau) \ge V_i^{\text{true}}(\tau)$ với mọi đơn vị và mọi $\tau$, trong khi $N_i^{\text{acc}}(\tau)$ không đổi. Do đó
    $$\mathcal{R}_{\text{observed}}(\tau) \ge \mathcal{R}_{\text{true}}(\tau), \quad \forall \tau \in \Lambda$$
    và kiểm soát $\mathcal{R}_{\text{observed}}(\hat{\tau}) \le \alpha$ kéo theo $\mathcal{R}_{\text{true}}(\hat{\tau}) \le \alpha$.
  * **Nhãn thừa hoặc gán sai lớp (phản bảo thủ):** một hộp nhãn chuẩn vẽ ở chỗ không có vật thể, hoặc một nhãn sai lớp trùng với dự đoán sai lớp, có thể biến một FP thật thành TP. Chiều sai số này làm $\mathcal{R}_{\text{observed}}$ thấp hơn $\mathcal{R}_{\text{true}}$. Chúng tôi giả định loại sai số này hiếm trong BDD100K. *(Khuyến nghị, không bắt buộc):* kiểm tra thủ công một mẫu ngẫu nhiên đăng ký trước (ví dụ 300 box được tính là TP tại $\tau_{\text{ref}}$ trên Meta-Train) và báo cáo tỉ lệ TP giả.
  
  Vì vậy kết luận chính xác là: nhãn bị bỏ sót chắc chắn làm hệ thống bảo thủ hơn (Lemma A4); toàn bộ sai số nhãn **chủ yếu** làm hệ thống bảo thủ hơn, với điều kiện sai số loại thứ hai hiếm.

* **Assumption A5 (Pre-registered Fixed Candidate Path):**
  Dãy các tham số kiểm định ứng viên $\Lambda_{1D} = (\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(K)})$ là một thứ tự cố định nghiêm ngặt, được xây dựng hoàn toàn độc lập với $\mathcal{D}_{\text{cal}}$ và $\mathcal{D}_{\text{test}}$, bằng thuật toán Truy vết Biên Pareto trên tập Meta-Train riêng biệt (Mục 6.2). Hai đường đi được dựng theo cùng thuật toán: $\Lambda_{1D}$ cho $\alpha = 0.02$ (Exp B; một đường đi tạm $\Lambda_{1D}^{(0)}$ dựng cùng cách phục vụ Exp A tầng 2) và $\Lambda_{1D}^{C}$ cho $\alpha_C = 0.10$ (Exp A tầng 1, Exp C). Mỗi đường đi gồm một phần lõi nằm dưới mức trần rủi ro và một phần kéo dài vượt qua mức trần (Mục 6.2). Thuật toán, các lưới rủi ro và số điểm $K$, $K_C$ được đăng ký sau bước pilot P3 trên Meta-Train và trước khi mở dữ liệu đánh giá (Mục 8.0).

---

## IV. BẢN CHẤT LỖI CỦA FRAME-LEVEL SPLITTING

Việc phân chia ngẫu nhiên dữ liệu video ở cấp độ từng frame làm sụp đổ các bảo chứng thống kê qua 2 cơ chế định lượng:

```
CƠ CHẾ LÀM MÉO MÓ THỐNG KÊ KHI CHIA THEO FRAME:

[Video Clip / Driving Session] 
  Frame 1 ──(corr*)──► Frame 2 ──(corr*)──► Frame 3 ...
       │                   │
  (Random Split)      (Random Split)
       ▼                   ▼
[Calibration Set]     [Test Set]
       │                   │
       └── RÒ RỈ NGỮ CẢNH ─┘
 (Ngưỡng cal thấy trước bối cảnh test)
(*Lưu ý: tương quan liên frame ρ_intra trong ví dụ ở Mục 4.1 (0.60) là giá trị
 giả định minh họa; giá trị thực tế được đo trong Experiment A)
```

### 4.1. Suy giảm Cỡ mẫu Hiệu dụng (Effective Sample Size Collapse)
Khi gom $M_{\text{intra}}$ frames liền kề từ cùng các chuỗi video vào tập hiệu chuẩn, các quan sát có tương quan dương mạnh ($\rho_{\text{intra}} > 0$).

Theo lý thuyết lấy mẫu theo cụm (Kish, 1965):
$$\text{Design Effect: } D_{\text{eff}} = 1 + (M_{\text{intra}} - 1)\rho_{\text{intra}}$$
**Cỡ mẫu hiệu dụng thực tế:**
$$N_{\text{eff}} = \frac{N_{\text{nominal}}}{D_{\text{eff}}} = \frac{N_{\text{nominal}}}{1 + (M_{\text{intra}} - 1)\rho_{\text{intra}}}$$

* **Hậu quả đối với Bất đẳng thức Tập trung:** Khi một nghiên cứu tính toán trên $N_{\text{nominal}} = 4.000$ frames từ 20 video clips ($M_{\text{intra}} = 200$) với giả định minh họa $\rho_{\text{intra}} \approx 0.60$ (sẽ được thay thế bằng số liệu thực nghiệm đo từ Exp A):
  $$D_{\text{eff}} \approx 1 + (199 \times 0.60) \approx 120.4 \implies N_{\text{eff}} \approx \frac{4.000}{120.4} \approx 33 \text{ mẫu độc lập!}$$
  Nếu áp dụng Hoeffding/Bentkus trên $N_{\text{nominal}} = 4.000$ thay vì $N_{\text{eff}} \approx 33$, biên an toàn UCB bị thu hẹp nhân tạo tới $\sqrt{120.4} \approx 11$ lần. Thuật toán ảo tưởng rằng mình có độ tin cậy cực cao và chọn một ngưỡng $\tau^*$ quá lỏng. Hệ quả dự đoán là xác suất vi phạm trên các video mới vượt xa $\delta$; độ lớn của mức vượt trần chưa được biết và sẽ được đo ở Experiment C (xem thêm Mục 5.4).
* **Giới hạn của công thức Kish:** Công thức trên giả định tương quan đều giữa mọi cặp frame trong cụm. Tương quan thực tế giữa các frame giảm dần theo khoảng cách thời gian, nên $D_{\text{eff}}$ tính theo ICC chỉ là xấp xỉ; Exp A báo cáo kèm hàm tự tương quan theo độ trễ (Mục 8.1).
* **ICC nhỏ không có nghĩa là design effect nhỏ:** khi mỗi frame chỉ có vài box, FDP theo frame chứa nhiễu nhị thức lớn nên ICC của nó thường nhỏ (mô phỏng: 0.04–0.21), trong khi $D_{\text{eff}} = 1 + (M-1)\,\text{ICC}$ vẫn lớn vì $M = 200$ (mô phỏng: 9–42) và đủ làm bảo chứng sụp đổ. Vì vậy tiêu chí C4 đặt trên $D_{\text{eff}}^{\text{frame}}$, không đặt trên ICC.

### 4.2. Rò rỉ Ngữ cảnh qua Ngưỡng Hiệu Chuẩn (Context Leakage via Threshold Selection)
Ngay cả khi mô hình $\mathcal{M}$ bị đóng băng hoàn toàn (không rò rỉ trọng số theo Assumption A2), việc phân chia ngẫu nhiên Frame vẫn dẫn đến hiện tượng: Frame $t$ nằm ở $\mathcal{D}_{\text{cal}}$ trong khi Frame $t+1$ nằm ở $\mathcal{D}_{\text{test}}$. Đây là dạng giả lặp (pseudoreplication; Hurlbert, 1984): các quan sát phụ thuộc nhau bị xử lý như các lần lặp độc lập.
Khi đó, ngưỡng $\hat{\tau}$ được hiệu chuẩn trực tiếp trên các khung cảnh cụ thể (điều kiện chiếu sáng, góc camera, các phương tiện đồng hành) mà tập test sở hữu. Ngưỡng này "nhìn thấy trước" bối cảnh của chính các video test, khiến sai số đo được trên tập test thấp hơn mức thực tế (xấp xỉ $\widehat{\mathcal{R}}_{\text{cal}}$). Khi đem chính ngưỡng $\hat{\tau}$ đó áp dụng lên một chuyến đi hoàn toàn mới (Unseen Ride), rủi ro được dự đoán cao hơn mức đo trên tập test rò rỉ và có thể vượt trần $\alpha$. Đây là giả thuyết cơ chế; độ lớn chênh lệch được đo ở Arm 3 (Mục 8.3).

Cơ chế này không chỉ xảy ra ở cấp frame. Nếu các chuyến đi trong cùng một khối không gian-thời gian tương quan với nhau, việc chia ngẫu nhiên ở cấp chuyến đi cũng tạo ra rò rỉ tương tự ở cấp khối. Đây là lý do Exp B kiểm chứng H1a bằng phép chia held-out theo super-block (Mục 8.2).

---

## V. CÁC BẤT ĐẲNG THỨC TẬP TRUNG & TÍNH TOÁN BIÊN HỮU HẠN MẪU

Vì các biến ngẫu nhiên $Z_i(\tau) \in [0, 1]$ là **i.i.d.** (theo Assumption A1) và bị chặn trong $[0, 1]$, ta áp dụng các bất đẳng thức tập trung mẫu hữu hạn để xây dựng Upper Confidence Bound (UCB).

### 5.1. Dạng Hoeffding-Bennett (Kullback-Leibler Form)
Thay vì dùng dạng sub-Gaussian lỏng $\exp(-2Nt^2)$, khung LTT (Angelopoulos et al., 2021/2025) sử dụng dạng phân kỳ Kullback-Leibler, là dạng chặt nhất trong bất đẳng thức gốc của Hoeffding (1963, Theorem 1):
$$p_{\text{Hoeff}}(\tau) = \exp\left(-N \cdot d_{\text{KL}}\left(\widehat{\mathcal{R}}_N(\tau) \wedge \alpha,\, \alpha\right)\right)$$
với $d_{\text{KL}}(a, b) = a \ln\frac{a}{b} + (1-a)\ln\frac{1-a}{1-b}$.

### 5.2. Bất đẳng thức Bentkus (2004) & Dạng Hoeffding-Bentkus (HB)
**Theorem (Bentkus Comparison Theorem — Bentkus, 2004, Theorem 1.1).** Cho $X_1, \dots, X_N$ i.i.d. trong $[0, 1]$ với kỳ vọng $\mu = \mathbb{E}[X_i]$. 
Với mọi ngưỡng rủi ro $r \in [0, 1]$ và mọi phân phối có rủi ro thực tế $\mathcal{R} > r$, tại bất kỳ mức sai số thực nghiệm $s \in [0, 1]$, xác suất đuôi dưới của trung bình mẫu $\widehat{\mathcal{R}}_N = \frac{1}{N}\sum_{i=1}^N X_i$ được chặn trên bởi đuôi của phân phối nhị thức (thông qua phép đối xứng trên các biến phản chiếu $1 - X_i$):
$$\mathbb{P}_{\mathcal{R} > r}\left(\widehat{\mathcal{R}}_N \le s\right) \le c_1 \cdot \mathbb{P}\left(\text{Bin}(N, r) \le \lceil N s \rceil\right)$$
với hằng số $c_1 = e \approx 2.71828$ như được chuẩn hóa và sử dụng trong khung LTT (Angelopoulos et al., 2021/2025).
*(Bước nối từ $\mathcal{R}$ sang $r$):* Định lý gốc, áp dụng cho các biến phản chiếu, so sánh với $\text{Bin}(N, \mathcal{R})$, trong đó $\mathcal{R}$ là rủi ro thật. Vì hàm $p \mapsto \mathbb{P}(\text{Bin}(N, p) \le k)$ giảm theo $p$ và $\mathcal{R} > r$, ta có $\mathbb{P}(\text{Bin}(N, \mathcal{R}) \le k) \le \mathbb{P}(\text{Bin}(N, r) \le k)$, cho ra dạng trên.
*(Ghi chú làm tròn):* Định lý gốc của Bentkus so sánh với đuôi nhị thức được nội suy log-tuyến tính giữa các số nguyên. Vì $Z_i$ không nhận giá trị nguyên nên $Ns$ thường không nguyên; chặn trên an toàn của đuôi nội suy là đuôi nhị thức tại $\lceil Ns \rceil$, đúng như dạng dùng trong LTT. Dùng $\lfloor Ns \rfloor$ sẽ cho p-value nhỏ hơn mức hợp lệ (anti-conservative). Khi $Ns$ nguyên hai dạng trùng nhau, nên các bảng 5.3 và 5.4 (tính tại $\widehat{\mathcal{R}}_N = k/N$) không đổi. Trong mô phỏng (E1), hai dạng cho xác suất bác bỏ dưới $H_0$ gần như trùng nhau và đều $\le \delta$, vì nhánh Hoeffding-KL và độ bảo thủ chung của HB chiếm ưu thế; dạng $\lceil Ns \rceil$ được giữ vì là dạng hợp lệ về lý thuyết.

Thay $r = \alpha$ và đánh giá tại giá trị thống kê quan sát được $s = \widehat{\mathcal{R}}_N(\tau)$, ta thu được p-value Bentkus kiểm định $H_0: \mathcal{R}(\tau) > \alpha$:
$$p_{\text{Bentkus}}(\tau) = \min\left(1,\; e \cdot \sum_{k=0}^{\lceil N \widehat{\mathcal{R}}_N(\tau) \rceil} \binom{N}{k} \alpha^k (1-\alpha)^{N-k}\right)$$

**Bất đẳng thức Hoeffding-Bentkus kết hợp (HB Bound):**
$$p_{\text{HB}}(\tau) \triangleq \min\left(p_{\text{Hoeff}}(\tau),\, p_{\text{Bentkus}}(\tau)\right)$$
*Cơ sở toán học của tính hợp lệ:* Giá trị cực tiểu $p_{\text{HB}}$ là một p-value hợp lệ không phải vì "lấy min của hai p-value bất kỳ", mà bởi vì cả $p_{\text{Hoeff}}$ và $p_{\text{Bentkus}}$ đều là các cận trên hợp lệ của **cùng một hàm xác suất đuôi dưới** $\mathbb{P}_{\mathcal{R} > \alpha}(\widehat{\mathcal{R}}_N \le s)$, và cả hai đều đơn điệu tăng theo thống kê thực nghiệm $s = \widehat{\mathcal{R}}_N(\tau)$. Do đó, chặn trên nhỏ hơn giữa hai cận vẫn kiểm soát đúng xác suất sai lầm loại I, đảm bảo tính super-uniform dưới $H_0$.

### 5.3. Bảng Tính Toán Biên An Toàn Thực Tế ($N = 1.500, \alpha = 0.02, \delta = 0.05$)

| Bất đẳng thức | Dạng giải tích | Điều kiện $\widehat{\mathcal{R}}_N(\tau)$ để Reject $H_0$ | Nhận định về Năng lực Thống kê |
|---|---|---|---|
| **Hoeffding Sub-Gaussian** | $t = \sqrt{\frac{\ln(20)}{3000}} \approx 0.0316$ | $\widehat{\mathcal{R}}_N \le 0.02 - 0.0316 < 0$ | ❌ Không đủ power (do giả định sub-Gaussian đối xứng không khai thác được $\mu \approx 0$). |
| **Hoeffding KL / Bennett** | $\exp(-N \cdot d_{\text{KL}}(\widehat{\mathcal{R}}, \alpha)) \le 0.05$ | $\widehat{\mathcal{R}}_N \le 0.0113$ ($k \le 17 / 1500$) | ✅ Đủ power tốt. |
| **Hoeffding-Bentkus (HB)** | $\min(p_{\text{Hoeff}},\, e \cdot \text{BinTail}) \le 0.05$ | $\widehat{\mathcal{R}}_N \le 0.0120$ ($k \le 18 / 1500$) | ✅ **Đủ power tối ưu**: Chỉ cần FDP trên tập cal $\le 1.20\%$, hệ thống chứng nhận toán học $R \le 2\%$. |
| **Clopper-Pearson (Exact Binomial)** | $\text{Beta}^{-1}(1-\delta;\, k+1,\, N-k) \le \alpha$ | $k \le 20 / 1500$ ($\widehat{p} \le 0.0133$) | ✅ Khả thi cho kiểm soát sự kiện nhị phân $\mathbb{1}[\text{FDP}_i > 0]$. |

### 5.4. Ngưỡng Chứng Nhận Tối Đa Theo Cỡ Mẫu (HB, $\alpha = 0.02$, $\delta = 0.05$)

Bảng sau (tính bằng HB, điều kiện $N\widehat{\mathcal{R}}_N \le k_{\max}$) làm cơ sở để hiệu chỉnh kỳ vọng ở Experiment B và C:

| $N$ | $k_{\max}$ | $\widehat{\mathcal{R}}_N$ tối đa được chứng nhận | Ghi chú |
|---|---|---|---|
| 50, 100 | — | Không chứng nhận được | $N$ quá nhỏ: HB luôn trả về $\hat\tau = \emptyset$ |
| 150, 200 | 0 | $0$ | Chỉ khi mọi đơn vị có $Z_i = 0$; $N \ge 149$ là cỡ mẫu tối thiểu (nhánh Hoeffding-KL, xem chú thích †) |
| 400 | 2 | 0.0050 | |
| 800 | 7 | 0.0088 | |
| 1.500 | 18 | 0.0120 | Khớp bảng 5.3 |
| 4.000 (danh nghĩa) | 61 | 0.0153 | $N_{\text{nominal}}$ của Arm 2 khi $N_{\text{cluster}} = 20$ (lần chạy bổ sung ở $\alpha = 0.02$) |
| 80.000 (danh nghĩa) | 1.517 | 0.0190 | $N_{\text{nominal}}$ của Arm 2 khi $N_{\text{cluster}} = 400$ (lần chạy bổ sung ở $\alpha = 0.02$) |

† *Nguồn gốc mốc $N \ge 149$:* Tại $\widehat{\mathcal{R}}_N = 0$, nhánh Hoeffding-KL cho $p_{\text{Hoeff}} = \exp(-N\, d_{\text{KL}}(0, \alpha)) = (1-\alpha)^N = 0.98^N$, không có hằng số $e$, và $0.98^N \le 0.05 \iff N \ge 149$. Nhánh Bentkus thuần cho $p_{\text{Bentkus}} = e \cdot 0.98^N$ và cần $N \ge 198$. Vì $p_{\text{HB}} = \min(p_{\text{Hoeff}}, p_{\text{Bentkus}})$, mốc tối thiểu do nhánh Hoeffding-KL quyết định. (Các giá trị $k_{\max}$ trong bảng đã được kiểm lại bằng tính toán số.)

Hai dòng cuối cho thấy khi bỏ qua tương quan nội cụm, ngưỡng được chứng nhận tiến sát $\alpha$ (0.0153 và 0.0190). Rủi ro thật của $\hat\tau$ khi đó được dự đoán dao động quanh mức sát trần, nên xác suất vi phạm có thể lên tới cỡ vài chục phần trăm thay vì vượt trần một cách hệ thống. Đây là dự đoán từ cơ chế, chưa phải kết quả; Experiment C sẽ đo trực tiếp. Hiện tượng này chỉ quan sát được nếu đường đi có các điểm với rủi ro thật vượt mức trần, vì vậy các đường đi được kéo dài qua mức trần (Mục 6.2).

Bảng này cũng cho thấy vì sao Exp C không dùng $\alpha = 0.02$: với $N_{\text{cluster}} \le 200$ clip, hiệu chuẩn đúng theo cụm gần như luôn trả về $\emptyset$, nên so sánh giữa các arm mất ý nghĩa (Mục 8.3).

### 5.5. Ngưỡng Chứng Nhận Tối Đa Cho Experiment C (HB, $\alpha_C = 0.10$, $\delta = 0.05$)

| Cỡ mẫu | Vai trò trong Exp C | $k_{\max}$ | $\widehat{\mathcal{R}}_N$ tối đa được chứng nhận |
|---|---|---|---|
| $N < 29$ | Arm 1, $N_{\text{cluster}} = 20$ | — | Không chứng nhận được |
| 100 | Arm 1, $N_{\text{cluster}} = 100$ | 3 | 0.0300 |
| 200 | Arm 1, $N_{\text{cluster}} = 200$ | 11 | 0.0550 |
| 400 | Arm 1, $N_{\text{cluster}} = 400$ | 27 | 0.0675 |
| 800 | Arm 1, $N_{\text{cluster}} = 800$ | 62 | 0.0775 |
| 2.000 / 4.000 | Arm 3 / Arm 2, $N_{\text{cluster}} = 20$ | 171 / 360 | 0.0855 / 0.0900 |
| 10.000 / 20.000 | Arm 3 / Arm 2, $N_{\text{cluster}} = 100$ | 937 / 1.911 | 0.0937 / 0.0955 |
| 20.000 / 40.000 | Arm 3 / Arm 2, $N_{\text{cluster}} = 200$ | 1.911 / 3.874 | 0.0955 / 0.0969 |
| 40.000 / 80.000 | Arm 3 / Arm 2, $N_{\text{cluster}} = 400$ | 3.874 / 7.822 | 0.0969 / 0.0978 |
| 80.000 / 160.000 | Arm 3 / Arm 2, $N_{\text{cluster}} = 800$ | 7.822 / 15.749 | 0.0978 / 0.0984 |

*(Mốc $N \ge 29$: tại $\widehat{\mathcal{R}}_N = 0$, nhánh Hoeffding-KL cho $0.9^N \le 0.05 \iff N \ge 29$. Mọi giá trị trong bảng được tính bằng tính toán số với cùng công thức HB ở Mục 5.2.)*

Ở $\alpha_C = 0.10$, Arm 1 có vùng chứng nhận không tầm thường từ $N_{\text{cluster}} = 100$ trở lên, trong khi Arm 2 và Arm 3 tiến sát $\alpha_C$ giống hệt cách chúng tiến sát $\alpha$ ở bảng 5.4. Cơ chế được kiểm chứng ở H1b vì vậy không phụ thuộc vào việc chọn $\alpha_C$. Ngưỡng chứng nhận sát $\alpha_C$ chỉ biến thành vi phạm khi $\Lambda_{1D}^{C}$ có điểm với rủi ro thật vượt $\alpha_C$, nên đường đi này được kéo dài tới $2\alpha_C$ (Mục 6.2).

### 5.6. Biên Chịu Đựng Tương Quan của HB (Dependence Tolerance)

HB được thiết kế cho trường hợp xấu nhất là $Z_i$ nhị phân. Với FDP thật, mỗi đơn vị có nhiều box nên $\text{Var}(Z)$ nhỏ hơn nhiều so với $\alpha(1-\alpha)$ (mô phỏng: 0.2–0.7 lần). Phần dư này tạo ra một biên chịu đựng tương quan, ước lượng bằng lập luận xấp xỉ chuẩn: tại cỡ mẫu $N$, HB chỉ chứng nhận khi $\widehat{\mathcal{R}}_N \le k_{\max}(N)/N < \alpha$, nên một điểm có rủi ro thật vừa vượt $\alpha$ chỉ bị bác bỏ nhầm khi $\widehat{\mathcal{R}}_N$ lệch xuống ít nhất $\alpha - k_{\max}(N)/N$. Nếu $\text{Var}(\widehat{\mathcal{R}}_N) = D_{\text{eff}}\,\sigma_\alpha^2 / N$ thì xác suất đó không quá $\delta$ khi
$$D_{\text{eff}} \le D^* \triangleq \frac{N\,\big(\alpha - k_{\max}(N)/N\big)^2}{z_{1-\delta}^2\,\sigma_\alpha^2},$$
với $\sigma_\alpha^2$ là phương sai của $Z$ tại điểm đầu tiên của đường đi có rủi ro vượt $\alpha$. Tại $N = 1.500$: $\alpha - k_{\max}/N = 0.008$.

* **Đây là xấp xỉ, không phải định lý.** Bảo chứng chính xác của Theorem 1 chỉ có dưới A1. $D^*$ dùng để (i) đặt ngưỡng cho quy tắc R-unit (Mục 8.1) và (ii) diễn giải kết quả khi C2b phát hiện vi phạm A1 (Mục IX).
* **Kiểm chứng bằng mô phỏng:** ở cấp chuyến đi (E7), với $D_{\text{eff}} \approx 1.4\,D^*$, VR = 0.025 ($\le \delta$); với $D_{\text{eff}} \approx 1.9\,D^*$, VR = 0.056; với $D_{\text{eff}} \approx 3.9\,D^*$, VR = 0.16. Nhưng ở cấp frame (E3), Arm 2 đã vi phạm 14–19% khi $D_{\text{eff}}$ chỉ khoảng $1.3\,D^*_C$. Vì vậy $D^*$ chỉ là thước đo **cỡ độ lớn**, không bảo thủ đồng đều; mọi quyết định dùng nó đều kèm biên an toàn (ngưỡng $\min(2, \hat D^*)$, Mục 8.1 và IX).
* **Hệ quả cho phép kiểm H1a:** khi $D_{\text{eff}} < D^*$, VR vẫn dưới $\delta$ nên C2 đạt dù A1 bị vi phạm. Mô phỏng E4: C2 chấp nhận H1a ở mọi kịch bản có $D_{\text{eff}}$ tới khoảng 3, và chỉ rơi vào Không kết luận ở $D_{\text{eff}} \approx 4.8$. Kể cả quy trình lỗi phóng đại cỡ mẫu 32 lần cũng chỉ đạt VR = 0.033 (E2). C2 vì vậy là phép kiểm "bảo chứng không sụp đổ", không phải phép kiểm A1; A1 được kiểm trực tiếp bằng C2b.
* **Hệ quả cho H1b:** ở cấp frame, $D_{\text{eff}}$ lên tới hàng chục (Mục 4.1), vượt $D^*_C$ của Arm 2 (4.8–6.5 trong mô phỏng ở $\alpha_C = 0.10$, gần như không đổi theo $N_{\text{cluster}}$), nên bảo chứng sụp đổ đúng như H1b khẳng định.

---

## VI. KHUNG KIỂM ĐỊNH TUẦN TỰ LTT & HIỆU CHUẨN ĐA THAM SỐ

### 6.1. Định nghĩa p-value qua UCB Inversion & Tính Đối Ngẫu
Xét bài toán kiểm định giả thuyết: $H_0(\tau): \mathcal{R}(\tau) > \alpha$ vs. $H_1(\tau): \mathcal{R}(\tau) \le \alpha$.

**Định nghĩa p-value qua UCB Inversion:**
$$p(\tau) \triangleq \inf \left\{ \delta' \in (0, 1] :\; \text{UCB}_{\delta'}(\tau) \le \alpha \right\} \quad (\text{với quy ước } \inf \emptyset = 1)$$

Giả định $\text{UCB}_{\delta'}(\tau)$ là hàm không tăng và liên tục phải theo $\delta'$, giá trị infimum được xác định chặt chẽ.

**Tính Đối Ngẫu Toán Học Giữa p-value Trực Tiếp và UCB:**
Vì p-value kiểm định $p_{\text{HB}}(r;\, \widehat{\mathcal{R}}_N(\tau))$ là hàm không tăng theo ngưỡng rủi ro giả định $r$, vùng bác bỏ $H_0: \mathcal{R} > r$ tại mức ý nghĩa $\delta$ là tập $\{r \in [0, 1] : p_{\text{HB}}(r;\, \widehat{\mathcal{R}}_N(\tau)) \le \delta\}$.
Upper Confidence Bound chính là mức chặn rủi ro nhỏ nhất mà ta có thể chứng nhận an toàn tại độ tin cậy $1 - \delta$:
$$\text{UCB}_\delta^{\text{HB}}(\tau) \triangleq \inf \left\{ r \in [0, 1] :\; p_{\text{HB}}(r;\, \widehat{\mathcal{R}}_N(\tau)) \le \delta \right\}$$
Do tính đơn điệu của hàm xác suất đuôi, phép đối ngẫu nghịch đảo kiểm định (test inversion duality) được bảo toàn chuẩn xác:
$$\text{UCB}_\delta^{\text{HB}}(\tau) \le \alpha \iff p_{\text{HB}}(\alpha;\, \widehat{\mathcal{R}}_N(\tau)) \le \delta$$

**Chứng minh tính hợp lệ (Super-uniform under Null):**
Dưới $H_0(\tau)$, ta có $\mathcal{R}(\tau) > \alpha$. Với mọi $u \in (0, 1)$:
$$\mathbb{P}\left(p(\tau) \le u\right) = \mathbb{P}\left(\inf\{\delta' : \text{UCB}_{\delta'}(\tau) \le \alpha\} \le u\right) \le \mathbb{P}\left(\text{UCB}_u(\tau) \le \alpha\right)$$
Vì $\mathcal{R}(\tau) > \alpha$, biến cố $\{\text{UCB}_u(\tau) \le \alpha\}$ kéo theo biến cố $\{\text{UCB}_u(\tau) < \mathcal{R}(\tau)\}$. Theo định nghĩa của UCB mức $u$:
$$\mathbb{P}\left(\text{UCB}_u(\tau) < \mathcal{R}(\tau)\right) \le u \implies \mathbb{P}_{H_0}\left(p(\tau) \le u\right) \le u \quad \forall u \in (0, 1) \quad \blacksquare$$

### 6.2. Hiệu Chuẩn Đa Tham số Hợp Lệ (Valid Multi-Parameter Calibration)
Để tối ưu hóa đồng thời 3 ngưỡng $\tau = (\tau_{\text{cls}}, \tau_{\text{loc}}, \tau_{\text{fcd}})$, chúng tôi thiết lập:

#### Phương án Khuyên dùng: Tuyến đường Đơn điệu Tiên nghiệm (Pre-registered 1D Pareto Path)
1. **Thiết lập đường đi 1D (Assumption A5):** Trên tập Meta-Train riêng biệt, sắp xếp không gian tham số thành một dãy 1D hữu hạn gồm $K$ điểm:
   $$\Lambda_{1D} = \{\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(K)}\}$$
   theo thứ tự giảm dần về độ bảo thủ (Coverage tăng dần, FDP dự kiến tăng dần).

   **Thuật toán dựng $\Lambda_{1D}$: Truy vết Biên Pareto (Pareto Frontier Tracing) trên Meta-Train.**
   * Rời rạc hóa không gian tham số thành lưới hữu hạn $G \subset \mathbb{R}^3$ (đăng ký trước).
   * Với mỗi $\tau \in G$, ước lượng trên Meta-Train (đơn vị ride, cùng $\mathcal{C}_{\text{eval}}$ và $\theta_{\text{IoU}}$ đã chọn ở Mục 2.1): $\widehat{\text{FDP}}_{\text{meta}}(\tau)$ là Macro-FDP, và $\widehat{\text{Cov}}_{\text{meta}}(\tau)$ là tỉ lệ đối tượng GT được phủ bởi box chấp nhận đúng (cùng định nghĩa với C3b, Mục 8.2).
   * Với lưới rủi ro tăng dần $0 < \alpha_1 < \dots < \alpha_K$ (phần lõi nằm dưới $\alpha$, phần kéo dài vượt qua $\alpha$, xem bên dưới):
     $$\tau^{(k)} = \arg\max_{\tau \in G} \widehat{\text{Cov}}_{\text{meta}}(\tau) \quad \text{s.t.} \quad \widehat{\text{FDP}}_{\text{meta}}(\tau) \le \alpha_k$$
     Phá hòa bằng quy tắc cố định (ví dụ: chọn $\tau$ có $\widehat{\text{FDP}}_{\text{meta}}$ nhỏ hơn); bỏ các bước không khả thi và loại các điểm trùng nhau liên tiếp.
   * **Lưới rủi ro, gồm phần lõi và phần kéo dài (đăng ký cùng nhau):**
     * *Phần lõi (nơi kỳ vọng đạt chứng nhận):* vì tại $N = 1.500$ HB chỉ chứng nhận được $\widehat{\mathcal{R}}_N \lesssim 0.012$ (bảng 5.4), lõi được đặt dày từ $0.002$ đến $u(N) = \min\big(0.015,\; k_{\max}(N)/N + 0.003\big)$ (làm tròn xuống bội của $0.001$) với bước $0.001$; tại $N = 1.500$ là $0.002$–$0.015$, $K_{\text{lõi}} = 14$. Cận trên tự co lại khi quy tắc R-unit buộc giảm $N$.
     * *Phần kéo dài (vượt qua mức trần):* từ $u(N) + 0.001$ đến $1.5\alpha = 0.030$ với bước $0.001$, rồi đến $2\alpha = 0.040$ với bước $0.0025$; tại $N = 1.500$ là 19 điểm, tổng $K = 33$.
     * *Vì sao cần phần kéo dài:* nếu mọi điểm của đường đi đều có rủi ro thật $\le \alpha$ thì $\mathcal{H}_0 = \emptyset$ và vi phạm là bất khả thi theo cấu trúc, bất kể giả định A1 có đúng hay không (bước 3 trong chứng minh Theorem 1). Khi đó Exp B "chấp nhận" H1a mà không kiểm tra gì, và Arm 2 của Exp C không thể vi phạm dù bỏ qua tương quan. Phần kéo dài đặt các null đúng lên đường đi để một quy trình sai (do vi phạm giả định hoặc do bỏ qua tương quan) có chỗ để chọn ngưỡng không an toàn; một quy trình hợp lệ chỉ chọn chúng với xác suất $\le \delta$.
     * *Phần kéo dài không làm giảm Yield hay độ phủ:* fixed-sequence dừng ở điểm thất bại đầu tiên, nên mọi vòng dừng trong phần lõi cho ra đúng $\hat\tau$ như khi không có phần kéo dài; chỉ các vòng vượt qua hết phần lõi mới được kiểm tiếp, và chỉ đi xa hơn khi kiểm định cho phép. Tính hợp lệ (Theorem 1) không đổi vì đường đi vẫn cố định tiên nghiệm (A5).
     * $G$, hai phần của lưới và $K$ được chốt ở bước pilot P3 (Mục 8.0) và đăng ký trước khi mở dữ liệu đánh giá. Việc đường đi thực sự chứa null đúng trên dữ liệu đánh giá được xác nhận bằng điều kiện kiểm định được ở Mục IX.
   * **Đường đi cho Experiment C:** $\Lambda_{1D}^{C}$ được dựng bằng cùng thuật toán và cùng lưới $G$, với $\alpha_C = 0.10$ và lưới rủi ro ba đoạn: $0.005$–$0.090$ (bước $0.005$, 18 điểm), $0.0925$–$0.120$ (bước $0.0025$, 12 điểm) và $0.13$–$0.20$ (bước $0.01$, 8 điểm); tổng $K_C = 38$.
     * Đoạn đầu phủ vùng Arm 1 chứng nhận được ở cỡ mẫu nhỏ (từ $0.030$ tại $N_{\text{cluster}} = 100$, bảng 5.5).
     * Đoạn giữa dày quanh $\alpha_C$, nơi Arm 2 và Arm 3 chứng nhận tới ($0.0855$–$0.0984$, bảng 5.5), để ngưỡng được chọn bám theo dao động của $\widehat{\mathcal{R}}_N$ thay vì bị khóa ở một điểm duy nhất.
     * Đoạn cuối kéo dài tới $2\alpha_C$ vì cùng lý do với phần kéo dài của $\Lambda_{1D}$. Nếu thiếu hai đoạn sau, vi phạm của Arm 2 chỉ phụ thuộc vào việc điểm cuối của đường đi tình cờ có rủi ro thật trên MOT vượt $\alpha_C$ hay không, tức phụ thuộc vào đường đi chứ không vào cơ chế mà H1b khẳng định.
   * **Điểm tham chiếu:** các chỉ số mô tả và Exp A dùng điểm tham chiếu chọn theo **mức rủi ro lưới** trong phần lõi, không theo chỉ số vị trí, để việc thêm phần kéo dài không làm dịch điểm tham chiếu (Mục 8.1).
   * **Đơn vị khi dựng:** $\Lambda_{1D}$ được dựng với đơn vị do quy tắc R-unit chọn và với $N$ cuối cùng; $\Lambda_{1D}^{C}$ dùng đơn vị ride (thứ tự các bước ở P3, Mục 8.0).
   * **Phạm vi của tính đơn điệu:** Cách dựng này chỉ bảo đảm $\widehat{\text{FDP}}_{\text{meta}}$ tăng dần theo $k$ (bị chặn bởi $\alpha_k$); rủi ro thật $\mathcal{R}(\tau^{(k)})$ không được bảo đảm đơn điệu. Tính hợp lệ của Theorem 1 **không** cần đơn điệu (chỉ cần A5); đơn điệu kém chỉ làm dừng sớm và giảm power (H1c).
   * **Nguồn Meta-Train:** tách theo trọn super-block (bước P2), rời rạc với $\mathcal{D}_{\text{cal}} \cup \mathcal{D}_{\text{test}}$ và nằm ngoài tập huấn luyện của detector (A2, Mục 8.0). Nếu Meta-Train trùng tập huấn luyện, điểm tin cậy trên đó lạc quan và đường đi trở nên quá lỏng; điều này không phá vỡ tính hợp lệ nhưng làm giảm Yield.
2. **Quy trình Fixed-Sequence Testing (Maurer et al., 1995; Westfall & Krishen, 2001; là trường hợp riêng của closed testing theo Marcus et al., 1976):**
   * Với $j = 1, 2, \dots, K$:
     * Tính $p(\tau^{(j)})$.
     * Nếu $p(\tau^{(j)}) \le \delta$: **Bác bỏ $H_0(\tau^{(j)})$**, ghi nhận $\hat{\tau} \leftarrow \tau^{(j)}$ an toàn.
     * Nếu $p(\tau^{(j)}) > \delta$: **DỪNG LẠI NGAY LẬP TỨC (STOP)**.
3. **Quy ước Tập Bác bỏ Rỗng (Empty Rejection Fallback):**
   Nếu ngay tại điểm đầu tiên $j = 1$ mà $p(\tau^{(1)}) > \delta$, thuật toán không bác bỏ được giả thuyết nào. Khi đó, hệ thống kích hoạt chế độ Fail-safe:
   $$\hat{\tau} = \emptyset \implies \text{Không tự động gán nhãn bất kỳ box nào } (N_i^{\text{acc}} = 0 \implies \text{FDP} = 0 \le \alpha)$$
   đảm bảo rủi ro không bao giờ bị vượt ngưỡng ngay cả trong kịch bản xấu nhất.

#### Phương án Phụ: Chia Fold Hiệu Chuẩn (Folded Sample Splitting)
**Proposition 3 (Validity of Folded Sample Splitting).**
*Giả sử tập hiệu chuẩn được phân hoạch thành 3 fold rời rạc độc lập $\mathcal{D}_{\text{cal}} = \mathcal{D}_{\text{cal}}^{(1)} \cup \mathcal{D}_{\text{cal}}^{(2)} \cup \mathcal{D}_{\text{cal}}^{(3)}$ với kích thước mỗi fold là $N/3$. Giả sử Fold 1 dùng để chọn $\hat{\tau}_{\text{cls}}$, Fold 2 dùng để chọn $\hat{\tau}_{\text{loc}}$, và Fold 3 dùng để kiểm định đường đi 1D cố định tiên nghiệm của tham số còn lại $\Lambda_{\text{fcd}} = (\tau_{\text{fcd}}^{(1)}, \dots, \tau_{\text{fcd}}^{(M)})$ thông qua Fixed-Sequence LTT tại mức sai số $\delta$.*

*Khi đó, ngưỡng kết hợp $\hat{\tau} = (\hat{\tau}_{\text{cls}}, \hat{\tau}_{\text{loc}}, \hat{\tau}_{\text{fcd}})$ kiểm soát chặt chẽ kỳ vọng rủi ro quần thể:*
$$\mathbb{P}\left(\mathcal{R}(\hat{\tau}) \le \alpha\right) \ge 1 - \delta$$

*Chứng minh:*
Điều kiện hóa trên dữ liệu của Fold 1 và Fold 2, các tham số nuisance $(\hat{\tau}_{\text{cls}}, \hat{\tau}_{\text{loc}})$ trở thành các hằng số cố định tất định. Do tính độc lập giữa các fold ($\mathcal{D}_{\text{cal}}^{(3)} \perp \mathcal{D}_{\text{cal}}^{(1)}, \mathcal{D}_{\text{cal}}^{(2)}$), dãy giả thuyết kiểm định trên Fold 3 là hoàn toàn cố định có điều kiện. Áp dụng Theorem 1 trên tập dữ liệu Fold 3, ta có:
$$\mathbb{P}\left(\mathcal{R}(\hat{\tau}) > \alpha \;\middle|\; \mathcal{D}_{\text{cal}}^{(1)},\, \mathcal{D}_{\text{cal}}^{(2)}\right) \le \delta$$
Lấy kỳ vọng toàn phần theo phân phối của Fold 1 và Fold 2 thu được kết quả vô điều kiện: $\mathbb{P}(\mathcal{R}(\hat{\tau}) > \alpha) \le \delta \implies \mathbb{P}(\mathcal{R}(\hat{\tau}) \le \alpha) \ge 1 - \delta$. $\blacksquare$

*(Đánh đổi về Hiệu suất Mẫu - Sample Efficiency Trade-off):* Mặc dù Folded Splitting giải quyết triệt để sự phụ thuộc tham số mà không cần đường đi 3D tiên nghiệm, kích thước hiệu chuẩn trên fold kiểm định bị giảm xuống $N_{\text{test-fold}} = N / 3 = 500$. Tại $N = 500, \alpha = 0.02, \delta = 0.05$, bất đẳng thức HB đòi hỏi rủi ro thực nghiệm khắt khe $\widehat{\mathcal{R}}_{500} \le 0.006$ ($k \le 3 / 500$), khiến năng lực thống kê (Yield) bị suy giảm đáng kể so với việc chạy Fixed-Sequence 1D trên toàn bộ $N = 1.500$.

---

## VII. ĐỊNH LÝ CHÍNH & CHỨNG MINH TOÀN DIỆN (MAIN THEOREM & PROOF)

> **Theorem 1 (Finite-Sample PAC Guarantee on Population Risk).**
> *Giả sử các Giả định A1, A2 và A5 thỏa mãn, với thống kê đơn vị $Z_i(\tau) \in [0, 1]$ bất kỳ (Mục 2.3). Gọi $\hat{\tau} \in \Lambda_{1D} \cup \{\emptyset\}$ là ngưỡng được lựa chọn bởi thuật toán Fixed-Sequence LTT chạy trên đường đi $\Lambda_{1D}$ tại mức trần rủi ro danh nghĩa $\alpha$ và ngân sách sai số $\delta$.*
>
> *Khi đó, kỳ vọng rủi ro quần thể trên phân phối $\mathcal{P}$ được kiểm soát chặt chẽ với xác suất ít nhất $1 - \delta$:*
> $$\boxed{\mathbb{P}_{\mathcal{D}_{\text{cal}} \sim \mathcal{P}^N}\left(\mathcal{R}(\hat{\tau}) \le \alpha\right) \ge 1 - \delta}$$
> *với $\mathcal{R}(\hat{\tau}) \triangleq \mathbb{E}_{U \sim \mathcal{P}}[Z(U;\, \hat{\tau})]$.*

*Vai trò của các giả định:* Chứng minh chỉ dùng A1 (các $Z_i$ i.i.d.), A2 (detector cố định nên $Z_i(\tau)$ là hàm tất định của $U_i$) và A5 (đường đi không phụ thuộc $\mathcal{D}_{\text{cal}}$). A3 không cần cho định lý mà cần để kết quả đo trên tập kiểm thử phản ánh đúng $\mathcal{R}$ trên $\mathcal{P}$. A4 chỉ phục vụ việc diễn giải rủi ro quan sát được so với rủi ro thật. Định lý áp dụng nguyên dạng cho Experiment C khi thay $\Lambda_{1D}$ bằng $\Lambda_{1D}^{C}$ và $\alpha$ bằng $\alpha_C$. Bảo chứng là xác suất **biên** theo phân phối của $\mathcal{D}_{\text{cal}}$, không phải bảo chứng có điều kiện trên một tập hiệu chuẩn cụ thể (về sự phân biệt này, xem Vovk, 2012); Exp B ước lượng đúng đại lượng biên này bằng cách lặp lại phép rút tập hiệu chuẩn.

### Chứng minh Đầy đủ:
1. **Xét biến cố vi phạm rủi ro:** Định nghĩa biến cố sai lầm loại I của toàn bộ quy trình: $\mathcal{E}_{\text{violate}} = \{\mathcal{R}(\hat{\tau}) > \alpha\}$.
2. **Trường hợp tập bác bỏ rỗng:** Nếu $\hat{\tau} = \emptyset$, theo quy ước mục 6.2, $N^{\text{acc}} = 0 \implies \text{FDP} = 0 \le \alpha$. Biến cố $\mathcal{E}_{\text{violate}}$ không xảy ra.
3. **Trường hợp chọn được ngưỡng $\hat{\tau} \in \Lambda_{1D}$:**
   * Gọi $\mathcal{H}_0 = \{j \in \{1, \dots, K\} : \mathcal{R}(\tau^{(j)}) > \alpha\}$ là tập các ngưỡng không an toàn.
   * Do A5, $\Lambda_{1D}$ là dãy tất định không phụ thuộc vào $\mathcal{D}_{\text{cal}}$, nên tập chỉ số $\mathcal{H}_0$ là một tập cố định (non-random).
   * Nếu $\mathcal{H}_0 = \emptyset$, mọi ngưỡng đều an toàn $\implies \mathbb{P}(\mathcal{E}_{\text{violate}}) = 0 \le \delta$.
   * Nếu $\mathcal{H}_0 \ne \emptyset$, đặt $j^* = \min \mathcal{H}_0$ là chỉ số vi phạm cố định đầu tiên trên đường đi $\Lambda_{1D}$.
4. **Cơ chế Dừng của Fixed-Sequence:**
   * Để thuật toán chọn bất kỳ ngưỡng không an toàn nào ($\hat{\tau} \in \mathcal{H}_0$), điều kiện cần bắt buộc là nó phải bác bỏ thành công giả thuyết null tại điểm $j^*$.
   * Thật vậy, nếu $H_0(\tau^{(j^*)})$ không bị bác bỏ, thuật toán dừng ngay tại bước $j^*$ và chỉ trả về các ngưỡng có chỉ số $j < j^*$ (vốn dĩ an toàn vì $j^* = \min \mathcal{H}_0$).
5. **Đánh giá Xác suất Vi phạm:**
   $$\mathbb{P}(\mathcal{E}_{\text{violate}}) \le \mathbb{P}\left(\text{Reject } H_0(\tau^{(j^*)})\right) = \mathbb{P}\left(p(\tau^{(j^*)}) \le \delta\right)$$
   Vì $j^* \in \mathcal{H}_0$, $H_0(\tau^{(j^*)})$ là một null đúng. Theo Section 6.1, $p(\tau^{(j^*)})$ là super-uniform:
   $$\mathbb{P}_{H_0}\left(p(\tau^{(j^*)}) \le \delta\right) \le \delta \implies \mathbb{P}(\mathcal{R}(\hat{\tau}) > \alpha) \le \delta \quad \blacksquare$$

---

## VIII. THIẾT KẾ THỰC CHỨNG TÁI CẤU TRÚC (REVISED EMPIRICAL PROTOCOL)

```
TỔNG THỂ KIỂM ĐỊNH THỰC CHỨNG H1:

[Bước tiên nghiệm (8.0) — không chạm dữ liệu đánh giá]
  P1  Kiểm tra dữ liệu: đối chiếu video ID, xác minh tập lớp C_eval, dựng ride proxy dạng chuỗi và vùng
      chắn rò rỉ trên toàn bộ dữ liệu, block, super-block; đếm đơn vị và kích thước cụm (chỉ dùng metadata)
  P2  Tách Meta-Train theo super-block; huấn luyện detector trên Train_det \ (Rides_Val ∪ Rides_MOT ∪ Meta-Train)
  P3  Pilot trên Meta-Train: baseline plug-in τ_base; vòng tạm thời (θ_IoU, Λ_1D^(0));
      Exp A tầng 2 + quy tắc R-unit; vòng cuối cùng: θ_IoU, Λ_1D (α = 0.02) và Λ_1D^C (α_C = 0.10),
      cả hai kéo dài qua mức trần; ước lượng biên chịu đựng D* và năng lực của C2b
  P4  Đăng ký trước toàn bộ tham số; toàn bộ Val là Evaluation pool

[Experiment A: Chẩn Đoán Cấu Trúc Phụ Thuộc]
  Tầng 2 (Meta-Train, chạy trong P3): ICC ở cấp block và super-block, Moran's I liên super-block;
                            m̃ tính trên metadata Val  ->  quy tắc R-unit: đơn vị là ride, block hay super-block
  Tầng 1 (MOT, chạy sau đăng ký): ICC của FDP theo frame tại τ_ref^C, hàm tự tương quan

[Experiment B: Kiểm Chứng PAC, α = 0.02]
  B-block (chính): tập cal rút theo trọn super-block; test = phần còn lại của Evaluation pool
  B-rand (đối chứng): tập cal rút theo đơn vị; test = phần còn lại
  P = 334 phân hoạch × K = 3 tập cal rời nhau ≈ 1.000 vòng mỗi nhánh
  Vi phạm đo bằng rủi ro chính xác trên toàn pool
  H1a: C2 (bảo chứng, kèm điều kiện kiểm định được) + C2b (A1, design effect thực nghiệm)
  H1c: C3, C3b; mô tả D1–D4, D7, D8

[Experiment C: Đối Chứng Quét Cụm, α_C = 0.10]
  N_cluster ∈ {20, 100, 200, 400, 800} clip MOT (M = 200 frame/clip), cùng đại lượng R_frame
  Arm 1: Cluster-LTT  |  Arm 2: Dense Frame-LTT (N_nom = 200·N_cluster)  |  Arm 3: Frame Context-Leakage (paired)
  Λ_1D^C kéo dài tới 2α_C = 0.20; C5 chỉ dùng các N_cluster ∈ {100, 200, 400} mà Arm 1 có Yield ≥ 50%
  Chạy bổ sung ở α = 0.02 (chỉ mô tả); Stress-test riêng: OOD (ngoài phạm vi A3)
```

### 8.0. Phân Chia Dữ Liệu & Pilot Tiên Nghiệm (Pre-registered Data Partition & Pilot)
Các bước dưới đây được thực hiện **theo đúng thứ tự**. Mỗi bước chỉ dùng phần dữ liệu được nêu, và không bước nào chạm vào nhãn hộp của Evaluation pool (Val) hay của bộ MOT dùng cho Exp C. Các con số là kế hoạch tham khảo, sẽ được chốt sau bước P1.

**P1. Kiểm tra dữ liệu (chỉ dùng metadata và lược đồ nhãn, không dùng nhãn hộp):**
1. Đối chiếu video ID giữa $\text{Train}_{\text{det}}^{70\text{K}}$ và 1.600 clip MOT; báo cáo số video trùng (A2).
2. Xác minh tập lớp của bộ Detection và bộ MOT trên file nhãn; cố định $\mathcal{C}_{\text{eval}}$ và quy tắc vùng ignore (Mục 2.1).
3. **Dựng ride proxy trên toàn bộ dữ liệu** (train, val, MOT) theo quy tắc **dạng chuỗi**, mô phỏng việc một thiết bị ghi hình liên tục:
   * *Cặp khả thi:* clip $b$ có thể là clip kế tiếp của clip $a$ nếu khoảng nghỉ $\Delta t$ từ cuối $a$ đến đầu $b$ thỏa $0 \le \Delta t \le 10$ phút **và** khoảng cách $d(a, b)$ giữa điểm GPS cuối của $a$ và điểm GPS đầu của $b$ không quá $v_{\max}\Delta t + 0{,}5\,\text{km}$, với $v_{\max} = 30\,\text{m/s}$ (108 km/h). Ràng buộc vận tốc bắt được các chuyến có quãng dừng dài hoặc chạy nhanh mà một ngưỡng khoảng cách cố định (ví dụ 1 km) sẽ bỏ sót.
   * *Ghép một-một:* mỗi clip có tối đa **một** clip kế tiếp và **một** clip liền trước. Các cặp khả thi được xét theo điểm $s(a, b) = \Delta t / (600\,\text{s}) + d(a, b) / \big(v_{\max}\Delta t + 0{,}5\,\text{km}\big)$ tăng dần (phá hòa theo thứ tự video ID); một cặp được chấp nhận nếu $a$ chưa có clip kế tiếp và $b$ chưa có clip liền trước. Ride proxy là các chuỗi thu được.
   * *Vì sao không dùng thành phần liên thông:* ngưỡng khoảng cách lên tới $18{,}5\,\text{km}$ khi $\Delta t = 10$ phút, và BDD100K không có device ID, nên xe của hai người khác nhau ở gần nhau trong cùng khoảng thời gian thường tạo thành cặp khả thi. Lấy thành phần liên thông thì các liên kết bắc cầu có thể gộp cả một thành phố trong một ngày thành một "chuyến đi", làm $N$ sụp và loại phần lớn tập train. Ràng buộc một-một giữ mỗi ride proxy là một chuỗi theo thời gian, nên một liên kết sai chỉ nối hai đoạn chứ không lan ra.
   * *Vùng chắn rò rỉ (chỉ dùng cho A2, không dùng để định nghĩa đơn vị):* quy tắc chuỗi có thể tách nhầm một chuyến thật thành hai đoạn. Vì vậy, khi loại dữ liệu train ở P2, ta dùng thêm một quy tắc lỏng: loại mọi clip train tạo thành cặp khả thi (theo cả hai chiều thời gian, không cần ghép một-một) với **bất kỳ** clip nào thuộc một ride proxy của Val hoặc MOT. Quy tắc này chỉ đi một bước và không bắc cầu, nên lượng dữ liệu bị loại có giới hạn. Ở đây gộp thừa là an toàn (chỉ mất thêm dữ liệu train), còn gộp thiếu mới nguy hiểm (để lọt rò rỉ).
   * *Clip thiếu metadata:* clip thiếu GPS không được liên kết khi định nghĩa đơn vị, vì liên kết chỉ theo thời gian sẽ nối các thiết bị khác nhau ghi cùng lúc; vì lý do tương tự, chúng cũng không được xét trong vùng chắn rò rỉ. Clip train thiếu GPS được giữ lại và số lượng được báo cáo như rủi ro dư. Clip Val thiếu metadata xử lý theo A1. Vị trí và thời điểm của một ride lấy theo clip đầu tiên của ride đó.
   * *Kiểm tra hợp lý (chỉ dùng metadata):* báo cáo phân bố số clip mỗi ride proxy và tổng số ride proxy, so với con số "100.000 clip từ hơn 50.000 chuyến" của Yu et al. (2020), tức trung bình không quá khoảng 2 clip mỗi chuyến. Nếu trung bình vượt 3 clip mỗi ride hoặc có ride dài hơn 30 clip, chạy lại với bộ tham số chặt hơn đã đăng ký ($v_{\max} = 20\,\text{m/s}$, $\Delta t \le 5$ phút) và dùng bộ cho trung bình gần 2 hơn. Bước này không dùng nhãn nên không tạo cơ hội P-hacking.
4. Dựng hai cấp nhóm **lồng nhau**, trên một lưới không gian chung gốc sao cho mỗi ô 5 km chứa trọn 25 ô 1 km:
   * **Block:** ô $1\,\text{km} \times 1\,\text{km}$ $\times$ khung 2 giờ tuyệt đối (cùng ngày lịch và cùng khoảng 2 giờ theo giờ địa phương, các khoảng tính từ 0 giờ).
   * **Super-block:** ô $5\,\text{km} \times 5\,\text{km}$ $\times$ ngày lịch.
   
   Vì hai cấp lồng nhau, mỗi block nằm trọn trong một super-block. Block nhắm tương quan "cùng cảnh, cùng lúc" (cùng dãy phố, dòng xe, góc nắng); super-block nhắm tương quan "cùng khu vực, cùng ngày" (thời tiết đồng nhất ở thang vài km và một ngày). Các phương án đã loại: block nhỏ hơn (100 m × 15 phút) gần như trùng ride proxy; super-block theo thành phố × ngày tạo nhóm quá lớn vì BDD100K tập trung ở vài vùng; super-block theo tuần không bắt thêm nguồn tương quan nào.
   **Quy tắc tách:** nếu một super-block chứa hơn 5% số ride của Val (toàn bộ Val là Evaluation pool, bước P4), tách nó theo khung 6 giờ, rồi 2 giờ nếu vẫn quá lớn (mỗi lần tách vẫn giữ trọn các block). Mốc 5% (khoảng 350 ride) bảo đảm điều kiện khả thi ở cuối mục này còn dư biên, và mỗi tập hiệu chuẩn gồm ít nhất vài super-block.
5. Đếm số đơn vị $\hat n$ của Val, số block, số super-block, và báo cáo phân bố số ride mỗi nhóm cùng **kích thước cụm hiệu dụng** $\tilde m = \sum_g m_g^2 / \sum_g m_g$ ở từng cấp ($m_g$ là số ride của nhóm $g$). $\tilde m$ là số ride trung bình của nhóm chứa một ride ngẫu nhiên, là đại lượng đúng cho design effect khi các cụm có kích thước không đều; trung bình thường $\bar m$ bị kéo về 1 bởi các nhóm đơn lẻ.
6. Đếm số clip train thuộc $\text{Rides}_{\text{Val}} \cup \text{Rides}_{\text{MOT}}$ (sẽ bị loại ở P2) và báo cáo tỉ lệ so với 70K, tách riêng phần do cùng ride proxy và phần do vùng chắn rò rỉ.
7. Với MOT, cố định $M = 200$ frame có nhãn đầu tiên của mỗi clip; clip có ít hơn 200 frame có nhãn bị loại và được báo cáo. Báo cáo tỉ lệ clip MOT nằm một mình trong super-block của nó.

**P2. Tách Meta-Train và huấn luyện detector:**
1. Loại khỏi tập train mọi clip thuộc $\text{Rides}_{\text{Val}} \cup \text{Rides}_{\text{MOT}}$ (A2), tức mọi clip train cùng ride proxy hoặc nằm trong vùng chắn rò rỉ của một ride Val/MOT (bước P1).
2. Từ phần còn lại, tách Meta-Train bằng cách chọn ngẫu nhiên 10% số **super-block** và lấy trọn mọi clip train trong đó. Tỉ lệ 10% (thay vì 5%) giúp cận trên của ICC đủ hẹp để quy tắc R-unit không gộp đơn vị thô sai (Mục 8.1; mô phỏng E5, E10). Lấy trọn super-block (thay vì 5% số video) giữ nguyên mật độ chuyến đi trong mỗi nhóm, nhờ vậy Meta-Train có đủ cặp chuyến đi cùng nhóm để ước lượng ICC chính xác ở Exp A tầng 2.
3. Huấn luyện detector trên $\text{Train}_{\text{filtered}}$ (A2), rồi đóng băng trọng số.

**P3. Pilot trên Meta-Train (được dùng nhãn hộp của Meta-Train; mọi tính toán dùng $\mathcal{C}_{\text{eval}}$; đơn vị ride ở vòng tạm thời, đơn vị do R-unit chọn ở vòng cuối cùng):**
1. **Baseline plug-in $\tau_{\text{base}}$ (dùng cho C3b và quy tắc IoU), tính riêng cho từng $\theta \in \{0.75, 0.5\}$:** chỉ lọc theo confidence lớp, hai thành phần $\tau_{\text{loc}}, \tau_{\text{fcd}}$ đặt ở mức không lọc. Ngưỡng confidence là giá trị nhỏ nhất trên lưới $\{0.05, 0.10, \dots, 0.95, 0.99\}$ sao cho Macro-FDP trên Meta-Train $\le \alpha = 0.02$. Đây là cách làm thông thường không có bảo chứng: chọn ngưỡng "vừa đủ" trên một tập phát triển. Nếu không ngưỡng nào đạt, $\tau_{\text{base}}$ là ngưỡng chặt nhất của lưới và báo cáo ghi rõ C3b không có baseline có ý nghĩa.
   *(Lý do không dùng ngưỡng cố định kiểu "confidence $\ge 0.99$"):* điểm sigmoid của họ DETR có thể hiếm khi vượt 0.99. Khi đó baseline gần như không chấp nhận box nào, $\text{Cov}(\tau_{\text{base}}) \approx 0$ và $\rho_{\text{cov}}$ tăng vọt, khiến C3b và quy tắc chọn IoU đạt một cách tầm thường. Baseline plug-in đo đúng **cái giá của bảo chứng**: ở cùng mức rủi ro danh nghĩa, LTT giữ được bao nhiêu độ phủ so với cách chọn ngưỡng thông thường.
2. **Vòng tạm thời (đơn vị ride, $N = 1.500$):** tính đường precision–coverage (Macro-FDP và $\rho_{\text{cov,meta}}$ so với $\tau_{\text{base}}$ trên lưới $G$) ở cả $\theta = 0.75$ và $\theta = 0.5$, áp dụng quy tắc chọn $\theta_{\text{IoU}}$ (Mục 2.1) và dựng một đường đi tạm $\Lambda_{1D}^{(0)}$. Đường đi tạm chỉ dùng để lấy $\tau_{\text{ref}}$ cho bước kế tiếp.
3. **Chạy Exp A tầng 2 và quy tắc R-unit** (Mục 8.1) tại $\tau_{\text{ref}}$ của $\Lambda_{1D}^{(0)}$. Tầng 2 chỉ dùng Meta-Train và metadata của Val, nên chạy được trước khi đăng ký mà không chạm vào dữ liệu đánh giá. Kết quả: định nghĩa đơn vị cuối cùng và cỡ mẫu $N$ cuối cùng.
4. **Vòng cuối cùng (đơn vị và $N$ đã chốt):** tính lại $r^*$ và áp dụng lại quy tắc chọn $\theta_{\text{IoU}}$ với $N$ cuối cùng, chốt $G$ và dựng **hai** đường đi bằng thuật toán ở Mục 6.2:
   * $\Lambda_{1D}$ cho Exp B, với $\alpha = 0.02$ và lưới $\{\alpha_k\}$ gồm phần lõi và phần kéo dài theo quy tắc ở Mục 6.2 (tại $N = 1.500$: lõi $0.002$–$0.015$ bước 0.001; kéo dài $0.016$–$0.030$ bước 0.001 và $0.0325$–$0.040$ bước 0.0025; $K = 33$).
   * $\Lambda_{1D}^{C}$ cho Exp A tầng 1 và Exp C, với $\alpha_C = 0.10$ và lưới ba đoạn ở Mục 6.2 ($K_C = 38$).
5. Báo cáo tỉ lệ đơn vị rỗng và phân rã FP (Mục 2.1, bước 4) dọc hai đường đi trên Meta-Train, cùng kết quả của cả vòng tạm thời lẫn vòng cuối cùng.
6. **Đơn vị khi dựng đường đi:** $\Lambda_{1D}$ được dựng với đơn vị do quy tắc R-unit chọn ở bước 3, nên khớp với đơn vị dùng khi hiệu chuẩn ở Exp B. Meta-Train được tách theo trọn super-block (P2) nên chứa trọn mọi block và super-block, đủ để tính FDP ở cả ba định nghĩa đơn vị. $\Lambda_{1D}^{C}$ dùng đơn vị ride trên Meta-Train và được áp lên clip MOT ở Exp C; khác biệt về quy trình gán nhãn giữa keyframe Detection và frame MOT chỉ ảnh hưởng power, không ảnh hưởng tính hợp lệ (A5 vẫn đúng vì Meta-Train độc lập với dữ liệu đánh giá).
7. **Ước lượng biên chịu đựng $\hat D^*$ và năng lực của C2b (chỉ dùng Meta-Train).**
   * *$\hat D^*$:* tính theo công thức ở Mục 5.6 với $N$ cuối cùng, trong đó $\hat\sigma_\alpha^2$ là phương sai của $Z$ trên Meta-Train tại điểm đầu tiên của $\Lambda_{1D}$ có mức rủi ro lưới vượt $\alpha$. Tương tự, tính $\hat D^*_C$ cho Arm 2 của Exp C ở $\alpha_C$ (với $\hat\sigma^2$ là phương sai của FDP theo frame trên Meta-Train).
   * *Năng lực của C2b:* dựng pool tổng hợp có cùng số đơn vị và cùng phân bố kích thước super-block như Evaluation pool bằng cách rút có hoàn lại các đơn vị của Meta-Train. Tạo tương quan có kiểm soát bằng cách xếp các đơn vị vào super-block theo hạng của $Z$ tại điểm $\alpha_k = \alpha$ cộng nhiễu Gauss, chỉnh độ lớn nhiễu để đạt $D_{\text{eff}} \in \{1, 1.5, 2, 3\}$. Với mỗi mức, chạy phép tính C2b (Mục 8.2) trên 50 pool tổng hợp và báo cáo tỉ lệ C2b rơi vào từng vùng. Yêu cầu tối thiểu được đăng ký: tỉ lệ Bác bỏ khi $D_{\text{eff}} = 1$ dưới 5% (báo động giả), và khi $D_{\text{eff}} = 2$ trên 80%.
   * *Vì sao thay bước MDV của bản 1.1:* mô phỏng E2 cho thấy quy trình lỗi phóng đại cỡ mẫu tới 32 lần vẫn chỉ đạt VR thật 0.033, nên MDV của C2 không xác định được; nguyên nhân là độ bảo thủ của HB chứ không phải độ dài đường đi, nên kéo dài đường đi thêm không giúp gì.

**P4. Đăng ký trước:**
1. Đăng ký trước toàn bộ: $\mathcal{C}_{\text{eval}}$, $\theta_{\text{IoU}}$ (cùng giá trị $r^*$ đã dùng), $\tau_{\text{base}}$, $\Lambda_{1D}$ và $\Lambda_{1D}^{C}$ (gồm cả phần kéo dài), $\alpha_C$, quy tắc liên kết ride proxy và vùng chắn rò rỉ (cùng bộ tham số được chọn ở kiểm tra hợp lý), $\hat D^*$, $\hat D^*_C$ và năng lực của C2b (bước P3.7), định nghĩa block/super-block và quy tắc tách, tỉ lệ train bị loại, $\gamma_{\text{cov}} = 0.5$, quy tắc R-unit và các tiêu chí ở Mục IX.
2. **BDD100K Val:** toàn bộ là Evaluation pool ($\hat n_{\text{eval}} \approx 7.000$ đơn vị nếu đơn vị là ride). Không cần tách Diagnostic pool vì ICC được ước lượng trên Meta-Train (Mục 8.1).
3. **BDD100K MOT (1.600 clip):** toàn bộ dùng cho Exp A tầng 1 và Exp C. Exp A tầng 1 chỉ đo ICC và không chọn tham số nào cho Exp C, nên việc dùng chung không ảnh hưởng tính hợp lệ của Exp C.

**Điều kiện khả thi:** $n_{\text{eval}} \ge 3N + 3\,m_{\max}$ đơn vị theo định nghĩa đơn vị cuối cùng, với $m_{\max}$ là kích thước super-block lớn nhất (tính theo số đơn vị), để luôn cắt được 3 tập hiệu chuẩn giữ trọn super-block (Mục 8.2). Nếu không đạt, giảm $N$ theo bảng 5.4 và đánh giá lại khả năng đạt C3.

### 8.1. Experiment A: Chẩn Đoán Cấu Trúc Phụ Thuộc (Cluster Diagnostic)
* **Thời điểm chạy:** tầng 2 chạy trong bước P3, trước khi đăng ký, vì kết quả của nó (đơn vị và $N$) quyết định cách dựng đường đi cuối cùng; tầng 1 dùng nhãn MOT nên chạy sau khi đăng ký.
* **Điểm tham chiếu (chọn theo mức rủi ro lưới, không theo chỉ số vị trí):** $\tau_{\text{ref}}$ là điểm của đường đi tạm $\Lambda_{1D}^{(0)}$ có $\alpha_k = 0.008$, tức điểm giữa phần lõi tại $N = 1.500$ ($\Lambda_{1D}^{(0)}$ chỉ cần phần lõi); dùng cho tầng 2, và báo cáo thêm ICC tại điểm giữa phần lõi của $\Lambda_{1D}$ cuối cùng. $\tau_{\text{ref}}^{C}$ là điểm của $\Lambda_{1D}^{C}$ có $\alpha_k^C = 0.050$; dùng cho tầng 1, vì tầng 1 phục vụ H1b và Exp C. Nếu bước lưới đó bị loại khi dựng đường đi (không khả thi hoặc trùng điểm), dùng điểm có mức rủi ro lưới gần nhất. Báo cáo thêm tại các điểm phần tư của phần lõi mỗi đường đi. Chọn theo mức rủi ro giữ nguyên vị trí điểm tham chiếu như trước khi thêm phần kéo dài. Tương quan được đo trên **chính đại lượng mà LTT kiểm soát** (FDP tại các ngưỡng trên đường đi), không phải trên điểm tin cậy chung chung.
* **Tầng 1 (Within-Video, dữ liệu MOT):** Với mỗi clip và mỗi frame, tính $\text{FDP}_{c,t}(\tau_{\text{ref}}^{C})$. Tính ICC bằng ANOVA:
  $$\text{ICC} = \frac{\text{MSB} - \text{MSW}}{\text{MSB} + (M - 1)\text{MSW}}$$
  với khoảng tin cậy 95% bằng bootstrap theo clip, và $D_{\text{eff}}^{\text{frame}} = 1 + (M - 1)\widehat{\text{ICC}}_{\text{frame}}$. Báo cáo kèm hàm tự tương quan trung bình theo độ trễ (1–50 frame) vì tương quan thực tế giảm dần theo khoảng cách thời gian (Mục 4.1). ICC tại $\tau_{\text{ref}}$ của $\Lambda_{1D}$ và ICC của điểm tin cậy dự đoán được báo cáo như chỉ số mô tả phụ.
* **Tầng 2 (Between-Rides, Meta-Train):** ICC được ước lượng trên Meta-Train thay vì trên một phần của Val. Meta-Train độc lập với dữ liệu đánh giá, detector không được huấn luyện trên nó, và **gần** cùng phân phối với Val vì BDD100K chia train/val ngẫu nhiên. Phân phối không hoàn toàn trùng nhau: việc loại các chuyến liên kết với Val/MOT ở P2 lấy bớt dữ liệu train ở đúng các khu vực và thời điểm mà Val dày đặc. Vì vậy báo cáo so sánh phân phối thuộc tính cảnh (vùng địa lý theo ô 5 km, thời tiết, loại cảnh, khung giờ) giữa Meta-Train và Val (D6). Khác biệt lớn được nêu như giới hạn của quy tắc R-unit và của vị trí rủi ro dọc đường đi; trường hợp xấu nhất (đường đi không còn null đúng nào trên Val) bị bắt bởi điều kiện kiểm định được ở Mục IX. ICC là tương quan giữa từng cặp nên không phụ thuộc mật độ chuyến đi, còn kích thước cụm hiệu dụng $\tilde m$ vẫn tính trên Evaluation pool (bước P1). Meta-Train vừa dùng để dựng đường đi vừa dùng để đo ICC; điều này không ảnh hưởng tính hợp lệ vì R-unit chỉ chọn đơn vị và nhánh B-block kiểm tra đầu-cuối.
  1. **ICC theo cấp nhóm.** Với $\ell \in \{\text{block}, \text{super-block}\}$, ước lượng trên Meta-Train $\widehat{\text{ICC}}_{\ell}$ của $\text{FDP}_i(\tau_{\text{ref}})$ (đơn vị là ride, nhóm theo cấp $\ell$) bằng ước lượng ANOVA cho cụm kích thước không đều (hệ số $n_0$), với cận trên 95% $\widehat{\text{ICC}}_{\ell}^{\,\text{upper}}$ từ bootstrap theo super-block. Đặt
     $$D_{\text{eff}}^{\ell} = 1 + (\tilde m_{\ell} - 1)\,\widehat{\text{ICC}}_{\ell}^{\,\text{upper}}$$
     với $\tilde m_{\ell}$ là kích thước cụm hiệu dụng của cấp $\ell$ trong Evaluation pool (bước P1). Nếu phần lớn nhóm ở cấp $\ell$ chỉ chứa một ride thì $\tilde m_\ell \approx 1$ và $D_{\text{eff}}^{\ell} \approx 1$; điều này phản ánh đúng thực tế là tương quan trong nhóm đơn lẻ không ảnh hưởng đến phương sai, chứ không phải điều kiện bị thỏa mãn một cách tầm thường.
  2. **ICC cấp block trong super-block.** Coi block là đơn vị (FDP gộp trên toàn block), ước lượng tương tự $\widehat{\text{ICC}}_{\text{sb}|\text{b}}$ và $D_{\text{eff}}^{\text{sb}|\text{b}}$ với $\tilde m$ tính theo số block mỗi super-block.
  3. **Moran's $I$ liên super-block.** Đo tương quan không gian còn lại *giữa* các super-block, loại mà phép chia held-out theo super-block không loại bỏ được. Ma trận trọng số: $k$ láng giềng gần nhất theo tọa độ GPS của đơn vị ($k = 8$), chuẩn hóa theo hàng, và **chỉ giữ các cặp láng giềng thuộc hai super-block khác nhau**. Khoảng tin cậy xấp xỉ 95%: $\hat I \pm 1.96\,\widehat{\text{SD}}(I)$, với $\widehat{\text{SD}}(I)$ tính theo phương sai dưới giả thuyết ngẫu nhiên hóa (Cliff & Ord, 1981). Đây là xấp xỉ thông dụng; cận trên chỉ dùng để gắn cờ rủi ro dư, không quyết định đơn vị.
  4. **Quy tắc R-unit (chọn đơn vị thống kê, thực thi trong bước P3, trước khi đăng ký):** đặt ngưỡng $D_{\text{thr}} = \min(2,\; \hat D^*)$, với $\hat D^*$ là biên chịu đựng ước lượng trên Meta-Train (Mục 5.6, bước P3.7).
     * Nếu $D_{\text{eff}}^{\text{block}} \le D_{\text{thr}}$ và $D_{\text{eff}}^{\text{super-block}} \le D_{\text{thr}}$: đơn vị là **ride**.
     * Ngược lại, nếu số block trong Evaluation pool thỏa điều kiện khả thi (Mục 8.0) và $D_{\text{eff}}^{\text{sb}|\text{b}} \le D_{\text{thr}}$: đơn vị là **block** ($\text{FDP}_i$ tính gộp trên toàn block).
     * Ngược lại: đơn vị là **super-block**, và giảm $N$ theo bảng 5.4 cho đến khi thỏa điều kiện khả thi.
     * Cal và test **cùng** áp dụng định nghĩa đơn vị đã chọn để thỏa A3. Không dùng lấy mẫu "mỗi nhóm tối đa một ride", vì cách này đổi phân phối $\mathcal{P}$ thành phân phối theo nhóm trong khi tập kiểm thử lấy theo ride, làm vi phạm A3.
     * **Hướng sai số của quy tắc:** cận trên của ICC rộng khi số cặp chuyến đi cùng nhóm ít, nên quy tắc có thể chọn đơn vị thô hơn mức cần thiết. Sai số này chỉ làm giảm $N$ và power, không làm mất tính hợp lệ.
     * **Vì sao ngưỡng là $\min(2, \hat D^*)$ chứ không phải 1.2:** mô phỏng E5 và E10 cho thấy khi A1 hoàn toàn đúng, ngưỡng 1.2 vẫn khiến quy tắc từ bỏ đơn vị ride trong 27–85% số lần với $\tilde m_{\text{Val}} \ge 3$ và Meta-Train 1.000–2.000 chuyến, vì cận trên của ICC quá rộng so với khoảng 0.2 mà ngưỡng cho phép. Ngưỡng 2 giảm tỉ lệ này xuống 0–28% (tối đa 15% khi Meta-Train từ 2.000 chuyến), trong khi mô phỏng E2 cho thấy $D_{\text{eff}} \le 2$ chỉ cho VR $\le 0.001$. Cận $\hat D^*$ giữ ngưỡng an toàn khi dữ liệu thật có phương sai gần trường hợp nhị phân.
  5. **Rủi ro dư:** Nếu cận trên của Moran's $I$ liên super-block $\ge 0.05$, tương quan không gian giữa các super-block kề nhau được báo cáo như một rủi ro dư của A1. Quy tắc không gộp thêm vì super-block đã là cấp thô nhất. Tương quan theo cùng tài xế/đội xe không đo được bằng bất kỳ chỉ số nào ở đây (xem A1).
* **Tiêu chí kiểm định:**
  * **Cấp Frame (C4):** xem Mục IX. Tiêu chí đặt trên $D_{\text{eff}}^{\text{frame}} = 1 + (M-1)\widehat{\text{ICC}}_{\text{frame}}$ với CI bootstrap theo clip, không đặt trên ICC, vì ICC của FDP theo frame nhỏ do nhiễu nhị thức dù design effect lớn (Mục 4.1). ICC vẫn được báo cáo.
  * **Cấp Ride/Block:** Báo cáo mọi ước lượng và cận trên ở tầng 2. "Không có ý nghĩa thống kê ($p>0.05$)" **không** được dùng làm bằng chứng cho tính độc lập; chỉ cận trên của khoảng tin cậy nhỏ mới có ý nghĩa.

### 8.2. Experiment B: Kiểm Chứng PAC Trên Phân Phối Chuẩn ($\alpha = 0.02$, $B \approx 1.000$ Rounds Mỗi Nhánh)

**Vì sao cần nhánh held-out theo super-block.** Nếu tập hiệu chuẩn và tập kiểm thử được chia ngẫu nhiên ở cấp đơn vị từ cùng một pool, các đơn vị cùng nhóm sẽ nằm ở cả hai phía. Khi các đơn vị cùng nhóm tương quan, rủi ro trên tập kiểm thử tương quan dương với rủi ro trên tập hiệu chuẩn, và tỉ lệ vi phạm đo được bị kéo xuống. Đây là cùng cơ chế rò rỉ ở Mục 4.2, chỉ dịch lên một tầng. Phép chia như vậy gần như không thể phát hiện vi phạm A1, nên không dùng được làm phép thử chính cho H1a. Nhánh B-block đặt mọi super-block trọn vẹn vào một phía, tái hiện kịch bản triển khai: ngưỡng hiệu chuẩn trên một số khu vực và ngày rồi dùng ở nơi khác, ngày khác. Cấp held-out luôn là super-block (cấp thô nhất) bất kể đơn vị được chọn, để phép thử nghiêm ngặt nhất có thể. Nếu quy tắc R-unit chọn super-block làm đơn vị, B-block trùng với B-rand và tương quan giữa các super-block chỉ được mô tả qua Moran's $I$; trường hợp này được nêu rõ như một giới hạn.

* **Hai nhánh** (đơn vị theo kết quả R-unit; số liệu theo 8.0):
  * **B-block (chính, dùng cho C2, C3, C3b):** Mỗi phân hoạch $p$ hoán vị ngẫu nhiên các **super-block** của Evaluation pool rồi cắt tuần tự thành $K = 3$ tập hiệu chuẩn rời nhau. Mỗi tập gồm các super-block liên tiếp cho đến khi đạt ít nhất $N$ đơn vị ($N$ chốt ở bước P3, mặc định $1.500$); super-block cuối được giữ trọn để không nhóm nào bị chia đôi, nên $N_b \in [N,\, N + m_{\max})$ (báo cáo $N_b$ thực tế; điều kiện khả thi ở 8.0 bảo đảm luôn cắt đủ 3 tập). Tập kiểm thử của vòng $b = (p, k)$ là **toàn bộ Evaluation pool trừ** $\mathcal{D}_{\text{cal}}^{(b)}$ (khoảng $5.500$ đơn vị).
  * **B-rand (đối chứng, chỉ mô tả):** Giống B-block nhưng hoán vị và cắt ở cấp **đơn vị**, mỗi tập đúng $N$ đơn vị; tập kiểm thử là các đơn vị còn lại.
  * Mỗi nhánh: $P = 334$ phân hoạch, $B = P \cdot K = 1.002$ vòng.
* **Mỗi vòng $b$:**
  1. Chạy Fixed-Sequence LTT trên $\mathcal{D}_{\text{cal}}^{(b)}$ với $\Lambda_{1D}$, thu $\hat{\tau}_b$ (có thể $\emptyset$).
  2. Ghi nhận biến cố vi phạm $e_b = \mathbb{1}\big[\mathcal{R}_{\text{pool}}(\hat\tau_b) > \alpha\big]$, với $\mathcal{R}_{\text{pool}}(\tau)$ là trung bình $\text{FDP}_j(\tau)$ trên **toàn bộ** Evaluation pool (quy ước $e_b = 0$ nếu $\hat\tau_b = \emptyset$).
  3. Đo kèm $\mathcal{R}_b$, trung bình $\text{FDP}_j(\hat\tau_b)$ trên tập kiểm thử (phần bù của tập hiệu chuẩn), như chỉ số mô tả cho kịch bản triển khai (D1).
* **Vì sao tiêu chí vi phạm dùng rủi ro trên toàn pool:** Exp B coi phân phối thực nghiệm của Evaluation pool là quần thể (xem đoạn kế tiếp), nên $\mathcal{R}_{\text{pool}}(\tau)$ chính là rủi ro thật của mọi ngưỡng trong khung suy diễn này và tính được chính xác, không có sai số đo. Bản 1.1 đo vi phạm trên tập phần bù rồi trừ sai số của nó ($e_b^{\text{cons}}$), cách này vừa lệch đại lượng (phần bù lệch ngược chiều tập hiệu chuẩn với hệ số $N/(n_{\text{eval}} - N)$, làm $e_b$ bị thổi phồng) vừa làm mất phần lớn tín hiệu: trong mô phỏng E3, một quy trình vi phạm thật 14% chỉ còn 1.4% khi đếm bằng $e_b^{\text{cons}}$. Tính hợp lệ của HB khi lấy mẫu không hoàn lại từ một pool hữu hạn được bảo đảm cho nhánh Hoeffding-KL bởi Hoeffding (1963, Mục 6) và được kiểm bằng mô phỏng cho toàn bộ HB, kể cả trường hợp xấu nhất $Z$ nhị phân (E1c, E1d: xác suất bác bỏ dưới $H_0$ thấp hơn trường hợp i.i.d.).
* **Đại lượng ước lượng và CI cho Violation Rate:** Mỗi tập hiệu chuẩn được rút không hoàn lại từ Evaluation pool, nên Exp B ước lượng xác suất vi phạm **có điều kiện trên Evaluation pool**, coi phân phối thực nghiệm của pool là quần thể (xem phạm vi suy diễn bên dưới). Khi cố định pool, các phân hoạch được ngẫu nhiên hóa độc lập nên **i.i.d.**; chồng lấn giữa các phân hoạch không tạo ra phụ thuộc. Phụ thuộc chỉ nằm **trong** một phân hoạch: ba vòng của cùng phân hoạch có tập hiệu chuẩn rời nhau, nên phụ thuộc lẫn nhau vì chúng chia nhau cùng một pool. Vì vậy CI chính thức dùng **bootstrap theo phân hoạch** (resample $P$ phân hoạch có hoàn lại, percentile 95% hai phía), và Clopper-Pearson trên $B$ vòng (coi như độc lập) chỉ được báo cáo để tham chiếu.
* **Rủi ro dọc đường đi trên Evaluation pool (cho điều kiện kiểm định được T1):** sau khi chạy xong, tính $\mathcal{R}_{\text{pool}}(\tau^{(k)})$, trung bình $\text{FDP}_j(\tau^{(k)})$ trên toàn Evaluation pool, cho mọi điểm của $\Lambda_{1D}$. Vì Exp B coi phân phối thực nghiệm của pool là quần thể (xem đại lượng ước lượng ở trên), đây chính là rủi ro thật của từng điểm trong khung suy diễn của Exp B. Đặt $\mathcal{H}_0^{\text{pool}} = \{k : \mathcal{R}_{\text{pool}}(\tau^{(k)}) > \alpha\}$ và $j^*_{\text{pool}} = \min \mathcal{H}_0^{\text{pool}}$. Với mỗi vòng, ghi nhận vòng đó có kiểm định tới $j^*_{\text{pool}}$ hay không (tức mọi điểm trước đó đều bị bác bỏ). Tỉ lệ vòng kiểm định tới $j^*_{\text{pool}}$ cho biết phép thử có thực sự đặt quy trình trước một null đúng hay không (D7).
* **Design effect thực nghiệm (cho C2b, phép kiểm trực tiếp A1 ở cấp super-block):** tại điểm $\tau_{A1}$ của $\Lambda_{1D}$ có mức rủi ro lưới $\alpha_k = \alpha$ (điểm nằm sát ranh giới vi phạm), tính $\widehat{\mathcal{R}}^{(b)}_{\text{cal}}(\tau_{A1})$ ở mọi vòng của cả hai nhánh, rồi đặt
  $$\hat D_{\text{eff}}^{\text{emp}} = \frac{\widehat{\text{Var}}_{\text{B-block}}\big(\widehat{\mathcal{R}}_{\text{cal}}(\tau_{A1})\big)}{\widehat{\text{Var}}_{\text{B-rand}}\big(\widehat{\mathcal{R}}_{\text{cal}}(\tau_{A1})\big)}$$
  với CI 95% bằng bootstrap theo phân hoạch (resample độc lập các phân hoạch của mỗi nhánh). Khi A1 đúng, lấy mẫu theo trọn super-block không làm tăng phương sai nên tỉ số xấp xỉ 1; khi các đơn vị cùng super-block tương quan, tỉ số xấp xỉ design effect ở cấp super-block. Báo cáo thêm tỉ số tại $\tau_{\text{ref}}$. *Kiểm bằng mô phỏng (E10b):* khi A1 đúng, tỉ số tại $\alpha_k = \alpha$ nằm trong khoảng 0.89–1.12 qua 16 pool; khi A1 sai với design effect lý thuyết 1.5–3, tỉ số là 1.9–4.5. Tại $\tau_{\text{ref}}$ (E4, E5b, E10b), tỉ số khi A1 đúng nằm trong khoảng 0.85–1.18 qua 42 pool, và phép kiểm kém nhạy hơn. Phép kiểm này nhạy hơn nhiều so với C2 và so với chênh lệch VR giữa hai nhánh (D2).
* **Chỉ số quyết định:**
  1. **Violation Rate (C2, H1a):** $\widehat{\text{VR}} = \frac{1}{B}\sum e_b$ của nhánh B-block, kèm CI. Quy tắc ba vùng ở Mục IX.
  2. **Design effect thực nghiệm (C2b, A1):** $\hat D_{\text{eff}}^{\text{emp}}$ kèm CI. Quy tắc ba vùng ở Mục IX.
  3. **Tỷ lệ Không Rỗng (C3, H1c):** $\text{Yield} = \frac{1}{B}\sum \mathbb{1}[\hat{\tau}_b \ne \emptyset]$ của B-block. Tiêu chí: $\ge 85\%$.
  4. **Độ phủ Nhãn giả Tương đối (C3b, H1c):** đo trên tập kiểm thử của từng vòng B-block tại $\hat\tau_b$, chỉ với các vòng có $\hat\tau_b \ne \emptyset$:
     $$\text{Cov}(\tau) = \frac{\sum_j \text{TP}_j(\tau)}{\sum_j M_j}, \qquad \rho_{\text{cov}}(\hat\tau_b) = \frac{\text{Cov}(\hat\tau_b)}{\text{Cov}(\tau_{\text{base}})}$$
     với $\text{TP}_j(\tau)$ là số box chấp nhận được ghép đúng (quy tắc 2.1), $M_j$ là số đối tượng GT thuộc $\mathcal{C}_{\text{eval}}$ của đơn vị $j$, và $\tau_{\text{base}}$ là baseline plug-in cố định ở bước P3 (không có bảo chứng). Macro-FDP của $\tau_{\text{base}}$ trên tập kiểm thử được báo cáo kèm, cho biết baseline thông thường có thực sự giữ được mức 2% trên dữ liệu mới hay không. Tiêu chí: $\text{Median}_b\, \rho_{\text{cov}}(\hat\tau_b) \ge \gamma_{\text{cov}} = 0.5$, tức LTT giữ được ít nhất một nửa độ phủ của cách chọn ngưỡng thông thường ở cùng mức rủi ro danh nghĩa. Chỉ số tương đối được dùng thay cho ngưỡng tuyệt đối kiểu "$\ge 1$ box/đơn vị", vì ảnh BDD100K thường có nhiều đối tượng nên ngưỡng tuyệt đối gần như luôn đạt và không mang thông tin.
* **Chỉ số mô tả (báo cáo, không dùng để quyết định):**
  * **D1:** Phân vị 95% và trung vị của $\mathcal{R}_{\text{pool}}(\hat\tau_b)$ và của rủi ro trên tập kiểm thử $\mathcal{R}_b$ (B-block); khoảng bảo thủ $\Delta_{\text{gap}} = \alpha - \text{Median}\,\mathcal{R}_{\text{pool}}(\hat\tau_b)$; tỉ lệ vòng có $\mathcal{R}_b > \alpha$ (mô tả kịch bản triển khai, chịu hiệu ứng phần bù).
  * **D2:** $\widehat{\text{VR}}$ của B-rand và chênh lệch $\Delta\text{VR} = \widehat{\text{VR}}_{\text{block}} - \widehat{\text{VR}}_{\text{rand}}$ kèm CI bootstrap ghép cặp theo phân hoạch. Vì HB bảo thủ, cả hai VR thường gần 0 nên chỉ số này có năng lực thấp; C2b thay thế nó trong việc phát hiện tương quan ở cấp super-block.
  * **D3:** Tại $\hat\tau_b$ trên tập kiểm thử: Micro-FDP (trung vị và phân vị 95% qua các vòng), tỉ lệ đơn vị có $N_j^{\text{acc}} = 0$, và tương quan Spearman giữa $N_j^{\text{acc}}$ và $\text{FDP}_j$ trên các đơn vị không rỗng (Mục 2.2).
  * **D4:** Phân rã FP thành $V^{\text{halluc}}$ và $V^{\text{loc}}$ tại $\hat\tau_b$ (Mục 2.1, bước 4).
  * **D7:** $\mathcal{R}_{\text{pool}}(\tau^{(k)})$ dọc $\Lambda_{1D}$, vị trí $j^*_{\text{pool}}$, và tỉ lệ vòng kiểm định tới $j^*_{\text{pool}}$ ở cả B-block lẫn B-rand.
  * **D8 (độ nhạy với rò rỉ dư của ride proxy):** chia các đơn vị Val thành hai nhóm theo việc có hay không một clip train còn giữ lại (sau lọc ở P2) nằm trong vòng 60 phút và 20 km; báo cáo Macro-FDP tại $\hat\tau_b$ (trung vị qua các vòng) của từng nhóm kèm CI. Nhóm có clip train lân cận có FDP thấp hơn rõ rệt là dấu hiệu rò rỉ dư ảnh hưởng đến kết quả.
* **Phạm vi suy diễn (giới hạn quần thể hữu hạn):** Exp B kiểm chứng bảo chứng PAC trên **phân phối thực nghiệm của tập Val BDD100K** (theo định nghĩa đơn vị đã chọn), không phải trên mọi quần thể lái xe có thể có. Độ lệch của chính Evaluation pool so với quần thể lái xe thật không đo được từ dữ liệu này, và B-block không phát hiện được tương quan theo cùng tài xế vì các chuyến của cùng một người nằm rải rác ở nhiều super-block.

### 8.3. Experiment C: Đối Chứng Quét Cụm ($\alpha_C = 0.10$, Swept-Cluster Benchmark)
Exp C so sánh cách xử lý tương quan nội cụm trên **cùng dữ liệu**, quét $N_{\text{cluster}} \in \{20, 100, 200, 400, 800\}$ clip MOT với đúng $M = 200$ frame/clip (bước P1).

**Mức rủi ro $\alpha_C = 0.10$.** H1b là mệnh đề về cơ chế: bỏ qua tương quan nội cụm thì bảo chứng sụp đổ. Mệnh đề này không gắn với mức vận hành $\alpha = 0.02$ của H1a. Nếu giữ $\alpha = 0.02$, Arm 1 trả về $\emptyset$ ở hầu hết các mức $N_{\text{cluster}}$ khả dụng (bảng 5.4: không chứng nhận được ở 20 và 100; ở 200 cần cả 40.000 frame không có FP nào), và C5 chỉ còn cho thấy "quy trình từ chối gán nhãn thì không vi phạm". Đó không phải bằng chứng rằng hiệu chuẩn theo cụm khắc phục được vấn đề. Bộ MOT chỉ có 1.600 clip nên không thể tăng $N_{\text{cluster}}$ để bù. Với $\alpha_C = 0.10$ (bảng 5.5), Arm 1 có ngưỡng chứng nhận được không tầm thường ở mọi $N_{\text{cluster}} \ge 100$, nên C5 so sánh hai quy trình cùng thực sự gán nhãn. Cơ chế được kiểm chứng không phụ thuộc $\alpha$: ở mọi mức $\alpha$, ngưỡng chứng nhận theo $N_{\text{nominal}}$ tiến sát $\alpha$ (so sánh bảng 5.4 và 5.5). Exp C được chạy bổ sung ở $\alpha = 0.02$ với $\Lambda_{1D}$, chỉ để mô tả (D5).

**Đại lượng mục tiêu chung.** Cả ba arm cùng kiểm soát và cùng được đánh giá theo rủi ro trung bình theo frame (Mục 2.3):
$$\mathcal{R}_{\text{frame}}(\tau) = \mathbb{E}_c\left[Z_c(\tau)\right], \qquad Z_c(\tau) = \frac{1}{M}\sum_{t=1}^{M}\text{FDP}_{c,t}(\tau)$$
Vì mỗi clip có đúng $M$ frame, $\mathcal{R}_{\text{frame}}$ bằng kỳ vọng FDP của một frame ngẫu nhiên. Do đó Arm 2 (coi frame là đơn vị) và Arm 1 (coi clip là đơn vị với thống kê $Z_c$) ước lượng **cùng một** đại lượng, và chênh lệch giữa hai arm chỉ đến từ việc có tính đến tương quan nội cụm hay không. Rủi ro thật của mỗi $\hat\tau_b$ được đo bằng $\mathcal{R}^{C}_{\text{pool}}(\hat\tau_b)$, trung bình $Z_c(\hat\tau_b)$ trên toàn bộ clip MOT dùng trong Exp C (cùng lý do như Mục 8.2); trung bình trên các clip kiểm thử được báo cáo kèm như chỉ số mô tả. Mọi arm dùng đường đi $\Lambda_{1D}^{C}$.

**Phân hoạch.** Nhóm held-out của MOT là super-block (cùng định nghĩa với Val, bước P1). Mỗi phân hoạch hoán vị ngẫu nhiên các nhóm, cắt $K = \lfloor 1.200 / N_{\text{cluster}} \rfloor$ tập hiệu chuẩn rời nhau, mỗi tập gồm các nhóm liên tiếp cho đến khi đạt ít nhất $N_{\text{cluster}}$ clip (nhóm cuối giữ trọn; báo cáo $N_b$ thực tế). Tập kiểm thử (Unseen Test) của vòng $b$ là **mọi clip không thuộc** $\mathcal{D}_{\text{cal}}^{(b)}$, nên luôn có khoảng 800 clip trở lên. Biến cố vi phạm là $e_b = \mathbb{1}[\mathcal{R}^{C}_{\text{pool}}(\hat\tau_b) > \alpha_C]$, như Mục 8.2.

| $N_{\text{cluster}}$ | $K$ | $P$ | $B = P \cdot K$ | Vai trò |
|---|---|---|---|---|
| 20 | 60 | 50 | 3.000 | Mô tả: Arm 1 không chứng nhận được ($N < 29$) |
| 100 | 12 | 84 | 1.008 | Ứng viên C5 |
| 200 | 6 | 167 | 1.002 | Ứng viên C5 |
| 400 | 3 | 334 | 1.002 | Ứng viên C5 |
| 800 | 1 | 1.000 | 1.000 | Mô tả: một nửa bộ MOT nằm trong tập hiệu chuẩn |

**CI cho Violation Rate:** như 8.2, khi cố định bộ MOT các phân hoạch là i.i.d., còn các vòng trong cùng phân hoạch phụ thuộc nhau vì có tập hiệu chuẩn rời nhau trong cùng pool và tập kiểm thử chồng lấn gần như toàn bộ. Vì vậy mọi $N_{\text{cluster}}$ đều dùng **bootstrap theo phân hoạch** ($P \ge 50$ để bootstrap ổn định). Riêng $N_{\text{cluster}} = 800$ có $K = 1$ nên các vòng độc lập và Clopper-Pearson hợp lệ; mức này chỉ mang tính mô tả vì tập hiệu chuẩn chiếm một nửa bộ MOT. Mỗi vòng chỉ chạy lại LTT trên các dự đoán đã được tính sẵn một lần, nên $B \approx 1.000$ có chi phí tính toán nhỏ.

**Điều kiện kiểm định được (T2):** tính $\mathcal{R}^{C}_{\text{pool}}(\tau^{C,(k)})$, trung bình $Z_c$ trên toàn bộ clip MOT dùng trong Exp C, cho mọi điểm của $\Lambda_{1D}^{C}$; đặt $\mathcal{H}_0^{C,\text{pool}} = \{k : \mathcal{R}^{C}_{\text{pool}}(\tau^{C,(k)}) > \alpha_C\}$ và báo cáo, ở từng $N_{\text{cluster}}$, tỉ lệ vòng mà Arm 1 và Arm 2 kiểm định tới $\min \mathcal{H}_0^{C,\text{pool}}$ (D7).

1. **Arm 1 (Cluster-LTT):** Hiệu chuẩn trên $\{Z_c\}$ với $N = N_{\text{cluster}}$. Theo bảng 5.5:
   * $N_{\text{cluster}} = 20$: HB không chứng nhận được ($N \ge 29$ là tối thiểu) $\Rightarrow \hat\tau = \emptyset$.
   * $N_{\text{cluster}} = 100, 200, 400, 800$: chứng nhận được khi $\widehat{\mathcal{R}}_N$ lần lượt $\le 0.030,\, 0.055,\, 0.0675,\, 0.0775$. **Yield ở các mức này là đại lượng cần đo, không được khẳng định trước.**
   * Báo cáo kèm VR có điều kiện trên các vòng $\hat\tau \ne \emptyset$ và độ phủ tương đối của Arm 1 so với Arm 2.
2. **Arm 2 (Dense Frame-LTT):** Dùng cùng các clip nhưng coi $N_{\text{nominal}} = 200\,N_{\text{cluster}}$ giá trị $\text{FDP}_{c,t}$ là độc lập. Ngưỡng được chứng nhận tiến sát $\alpha_C$: từ 0.090 ($N_{\text{cluster}} = 20$) đến 0.098 ($N_{\text{cluster}} = 400$), bảng 5.5. Rủi ro thật của $\hat\tau$ trên các clip mới vì vậy dao động quanh mức sát trần. Điều này chỉ biến thành vi phạm được vì $\Lambda_{1D}^{C}$ có đoạn dày quanh $\alpha_C$ và đoạn kéo dài vượt qua $\alpha_C$ (Mục 6.2); nếu đường đi dừng dưới $\alpha_C$, Arm 2 không thể vi phạm dù bỏ qua tương quan. **Dự đoán (không ràng buộc):** tỷ lệ vi phạm vượt rõ $\delta = 0.05$ (mô phỏng E3: 14–35% tùy mức dị biệt giữa các clip, trong khi Arm 1 bằng 0), nhưng mức vượt trần về độ lớn có thể chỉ khiêm tốn. Tiêu chí C5 không phụ thuộc vào con số dự đoán này, nên không cần pilot trên dữ liệu MOT (một pilot như vậy sẽ chạm vào dữ liệu của chính Exp C).
3. **Arm 3 (Frame Context-Leakage, paired):** Với mỗi clip trong $\mathcal{D}_{\text{cal}}^{(b)}$, chia ngẫu nhiên 50/50 thành $100$ frame hiệu chuẩn (Cal) và $100$ frame kiểm thử rò rỉ (Leaked Test) (tổng $N_{\text{nominal}}/2$ frame cho Cal; ngưỡng chứng nhận từ 0.0855 đến 0.0978, bảng 5.5). Với **cùng một** ngưỡng $\hat\tau_{\text{leaked}}$ được chọn bằng LTT naive trên các frame Cal, đo đồng thời:
   * (i) $\text{FDP}_{\text{leaked}}$: trung bình $\text{FDP}_{c,t}$ trên các frame Leaked Test. Dự đoán thấp một cách biểu kiến ($\approx \widehat{\mathcal{R}}_{\text{cal}}$) do rò rỉ bối cảnh.
   * (ii) $\text{FDP}_{\text{unseen}}$: trung bình $Z_c$ trên Unseen Test của vòng đó.
   * Báo cáo chênh lệch từng cặp $\Delta_{\text{leak}} = \text{FDP}_{\text{unseen}} - \text{FDP}_{\text{leaked}}$ (kiểm định ghép cặp trên các vòng, CI bootstrap theo phân hoạch). Dự đoán $\Delta_{\text{leak}} > 0$ về trung bình (hiệu ứng nguyền rủa người thắng + khoảng cách tổng quát hóa) với độ lớn khiêm tốn. Arm 3 là bằng chứng bổ trợ cho cơ chế rò rỉ (Mục 4.2), không nằm trong C5.
* **Phân hệ Stress-test Riêng biệt (OOD Evaluation):** Đánh giá $\hat{\tau}$ từ Arm 1 tại $N_{\text{cluster}}$ lớn nhất còn có Yield đủ (ưu tiên 400, rồi 800) trên tập thời tiết cực đoan (tuyết dày / đêm mưa lớn) để đo mức suy thoái do Covariate Shift. Đây là giới hạn của giả định A3, không phải lỗi của khung LTT.

---

## IX. MA TRẬN TIÊU CHÍ BÁC BỎ / CHẤP NHẬN TIÊN NGHIỆM (DECISION MATRIX)

Ma trận quyết định tách độc lập ba mệnh đề: tính hợp lệ của bảo chứng (H1a), sự sụp đổ thực chứng của việc chia frame (H1b), và tính không tầm thường của quy trình trên detector cụ thể (H1c). **Quy tắc R-unit** (Mục 8.1) là bước thiết kế được thực thi trong bước P3, *trước* khi đăng ký và trước C2–C5, không phải tiêu chí chấp nhận/bác bỏ. Mọi tiêu chí kiểm định giả thuyết đều chia không gian kết quả thành ba vùng **Chấp nhận / Bác bỏ / Không kết luận**, theo cùng một nguyên tắc:
* **Chấp nhận** cần bằng chứng đủ chính xác rằng tính chất được khẳng định đúng.
* **Bác bỏ** cần bằng chứng dương tính rằng tính chất bị vi phạm; thiếu độ chính xác không phải là căn cứ bác bỏ.
* Nếu một kết quả rơi đồng thời vào điều kiện Chấp nhận và Bác bỏ, ghi nhận **Bác bỏ**.
* Nếu điều kiện kiểm định được (T1 cho C2, T2 cho C5) không thỏa, tiêu chí tương ứng được ghi là **Không kiểm định được**, kể cả khi điều kiện Chấp nhận thỏa, vì khi đó không kết quả nào có thể bác bỏ mệnh đề.

Mọi khoảng tin cậy là 95% hai phía, tính theo phương pháp ở Mục 8.2 (Exp B) và 8.3 (Exp C).

### Phần 1: Kiểm định Giả thuyết H1a (PAC Statistical Guarantee)
**Điều kiện kiểm định được (T1):** C2 chỉ được kết luận Chấp nhận hoặc Bác bỏ khi $\mathcal{H}_0^{\text{pool}} \ne \emptyset$ (Mục 8.2), tức trên chính Evaluation pool, đường đi $\Lambda_{1D}$ có ít nhất một điểm với rủi ro vượt $\alpha$. Nếu $\mathcal{H}_0^{\text{pool}} = \emptyset$, vi phạm là bất khả thi theo cấu trúc (bước 3 trong chứng minh Theorem 1): C2 được ghi là Không kiểm định được, và H1a được báo cáo là chưa được kiểm chứng thực nghiệm. Phần kéo dài của $\Lambda_{1D}$ tới $2\alpha$ (Mục 6.2) được thiết kế để T1 thỏa, trừ khi rủi ro trên Val thấp hơn nhiều so với trên Meta-Train.

| Mã số | Chỉ số kiểm chứng | CHẤP NHẬN H1a | BÁC BỎ H1a | Không kết luận |
|---|---|---|---|---|
| **C2** | Violation Rate của nhánh B-block, đo bằng rủi ro trên toàn pool (Exp B) | Cận trên CI của $\widehat{\text{VR}}$ $\le 0.065$ | Cận dưới CI của $\widehat{\text{VR}}$ $> 0.05$ | Các trường hợp còn lại |
| **C2b** | Design effect thực nghiệm $\hat D_{\text{eff}}^{\text{emp}}$ tại $\tau_{A1}$ (Exp B, phép kiểm A1) | Cận trên CI $\le 1.5$ | Cận dưới CI $> 1.3$ | Các trường hợp còn lại |

**Kết luận gộp cho H1a** (khi T1 thỏa):
* **CHẤP NHẬN:** C2 và C2b cùng Chấp nhận. Bảo chứng không bị vi phạm và tiền đề A1 của Theorem 1 được xác nhận ở cấp super-block.
* **BÁC BỎ:** C2 Bác bỏ. Bảo chứng thực sự bị vi phạm trên dữ liệu thật.
* **CHẤP NHẬN NHỜ BIÊN CHỊU ĐỰNG:** C2 Chấp nhận, C2b Bác bỏ, và cận trên CI của $\hat D_{\text{eff}}^{\text{emp}}$ không vượt $D_{\text{thr}} = \min(2, \hat D^*)$ (cùng ngưỡng an toàn với quy tắc R-unit, Mục 5.6 và 8.1). Bảo chứng đứng vững nhưng A1 bị vi phạm ở cấp super-block, nên kết quả là xấp xỉ, không còn là hệ quả của Theorem 1; điều này được nêu rõ trong paper cùng giá trị $\hat D_{\text{eff}}^{\text{emp}}$ và $\hat D^*$.
* **KHÔNG KẾT LUẬN:** các trường hợp còn lại.

*(Ghi chú: $e_b$ dùng rủi ro chính xác trên toàn pool nên không có sai số đo; CI chỉ phản ánh sai số Monte Carlo qua các phân hoạch. Mốc 0.065 bằng $\delta = 0.05$ cộng biên 0.015 cho sai số Monte Carlo; cùng mốc này được dùng cho điều kiện an toàn của Arm 1 trong C5. Vì HB bảo thủ, C2 gần như luôn chấp nhận khi design effect dưới $\hat D^*$ (Mục 5.6), nên C2 được hiểu là phép kiểm "bảo chứng không sụp đổ"; tiền đề A1 được kiểm riêng bằng C2b. Mốc của C2b: khi A1 đúng, tỉ số trong mô phỏng không vượt 1.18 ở ước lượng điểm, nên cận dưới CI vượt 1.3 là bằng chứng dương tính có tương quan; cận trên CI không quá 1.5 bảo đảm design effect nhỏ hơn nhiều so với biên chịu đựng điển hình. Năng lực của C2b được ước lượng trước ở bước P3.7. Phân vị 95% của $\mathcal{R}_{\text{pool}}(\hat\tau_b)$ là chỉ số mô tả D1, không phải tiêu chí.)*

*Nguyên tắc Chẩn đoán Khoa học khi Bác bỏ:* Do Theorem 1 là một định lý đã được chứng minh về mặt giải tích, nếu thực nghiệm bác bỏ H1a, kết luận khoa học **không phải là khung LTT bị sai**, mà là bằng chứng cho thấy ít nhất một tiền đề bị vi phạm. C2b và nhánh đối chứng B-rand giúp phân biệt nguyên nhân:
1. **C2b Bác bỏ (design effect thực nghiệm lớn) → Vi phạm Giả định A1:** các đơn vị cùng super-block tương quan với nhau (cùng khu vực, cùng ngày) mà quy tắc R-unit chưa gộp đủ.
2. **Cả B-block và B-rand cùng vi phạm → Vi phạm A2 hoặc A5, tương quan không thuộc cấu trúc không gian-thời gian (ví dụ cùng tài xế), hoặc lỗi cài đặt:** trọng số detector bị rò rỉ thông tin từ tập đánh giá, đường đi $\Lambda_{1D}$ bị xây dựng có dùng dữ liệu đánh giá, hoặc p-value/quy tắc ghép cặp được cài đặt sai.
3. **Vi phạm A3:** cal và test không cùng định nghĩa đơn vị/trọng số.

### Phần 2: Kiểm định Giả thuyết H1b (Frame-Level Structural Breakdown)
Gọi $E = \{N_{\text{cluster}} \in \{100, 200, 400\} : \text{Yield}_{\text{Arm1}}(N_{\text{cluster}}) \ge 50\%\}$ là tập các mức **đủ điều kiện**, tức các mức mà Arm 1 thực sự gán nhãn trong ít nhất một nửa số vòng. C5 chỉ được đánh giá trên $E$, để Arm 1 không "an toàn" chỉ nhờ từ chối.

**Điều kiện kiểm định được (T2):** C5 chỉ được kết luận khi $\mathcal{H}_0^{C,\text{pool}} \ne \emptyset$ (Mục 8.3), tức trên chính dữ liệu MOT, đường đi $\Lambda_{1D}^{C}$ có ít nhất một điểm với rủi ro vượt $\alpha_C$. Nếu rỗng, Arm 2 không thể vi phạm theo cấu trúc và C5 được ghi là Không kiểm định được.

| Mã số | Chỉ số kiểm chứng | CHẤP NHẬN | BÁC BỎ | Không kết luận |
|---|---|---|---|---|
| **C4** | Design effect cấp Frame $D_{\text{eff}}^{\text{frame}} = 1 + (M-1)\widehat{\text{ICC}}_{\text{frame}}$ tại $\tau_{\text{ref}}^{C}$ (Exp A, tầng 1) | Cận dưới CI của $D_{\text{eff}}^{\text{frame}} \ge \max(5,\, \hat D^*_C)$ | Cận trên CI của $D_{\text{eff}}^{\text{frame}} < 2$ | Các trường hợp còn lại |
| **C5** | Vi phạm của Arm 2 so với Arm 1 (Exp C, $\alpha_C = 0.10$, các mức thuộc $E$; vi phạm đo bằng rủi ro trên toàn bộ MOT) | $|E| \ge 2$, VÀ với **mọi** mức thuộc $E$: cận dưới CI của $\widehat{\text{VR}}_{\text{Arm2}} > \delta = 0.05$ và cận trên CI của $\widehat{\text{VR}}_{\text{Arm1}} \le 0.065$ | $|E| \ge 2$, VÀ với ít nhất 2 mức thuộc $E$: cận trên CI của $\widehat{\text{VR}}_{\text{Arm2}} \le 0.065$ | Các trường hợp còn lại, kể cả khi $|E| < 2$ |

**Kết luận gộp cho H1b:** CHẤP NHẬN khi cả C4 và C5 cùng chấp nhận (vừa có cơ chế tương quan, vừa có hậu quả vi phạm); BÁC BỎ khi C5 bác bỏ; KHÔNG KẾT LUẬN trong các trường hợp còn lại. C5 quyết định chiều bác bỏ vì nội dung cốt lõi của H1b là hậu quả vi phạm bảo chứng; tương quan cao mà không gây vi phạm không đủ để khẳng định H1b.

*(Ghi chú C4: chấp nhận khi design effect theo frame vượt biên chịu đựng $\hat D^*_C$ của Arm 2 (Mục 5.6, bước P3.7), tức cơ chế đủ mạnh để dự đoán Arm 2 vi phạm; mốc sàn 5 giữ yêu cầu có ý nghĩa khi $\hat D^*_C$ nhỏ. Trong mô phỏng, $D^*_C$ = 4.8–6.5. Mô phỏng E8: với ICC theo frame 0.04–0.21, tiêu chí cũ (ICC $\ge 0.30$) cho H1b Không kết luận ở cả ba kịch bản dù Arm 2 vi phạm 14–35%, còn tiêu chí mới chấp nhận ở cả ba; khi frame độc lập (E9), Arm 2 hợp lệ và tiêu chí mới không chấp nhận H1b.)*

*(Ghi chú C5: CI tính bằng bootstrap theo phân hoạch (Mục 8.3). Vi phạm đo bằng rủi ro chính xác trên toàn bộ MOT nên không thể quy cho sai số của tập kiểm thử; trong mô phỏng E3, đếm bằng cận dưới trên phần bù như bản 1.1 làm mất phần lớn vi phạm của Arm 2 và khiến C5 không chấp nhận được khi dị biệt giữa các clip thấp. Với $B \approx 1.000$ vòng mỗi mức, điều kiện an toàn của Arm 1 đạt được khi VR thật của Arm 1 ở mức vài phần trăm. Nếu $|E| < 2$, tức Arm 1 hiếm khi gán nhãn ngay cả ở $\alpha_C = 0.10$, kết quả được ghi là Không kết luận và báo cáo như một giới hạn của cỡ bộ MOT đối với detector đã chọn. Mức vi phạm kỳ vọng 30–50% của Arm 2 là dự đoán không ràng buộc; tiêu chí C5 chỉ yêu cầu vi phạm vượt rõ $\delta$. Arm 3 báo cáo $\Delta_{\text{leak}}$ như bằng chứng bổ trợ cho cơ chế rò rỉ, không nằm trong C5.)*

### Phần 3: Kiểm định Giả thuyết H1c (Non-Vacuity)
| Mã số | Chỉ số kiểm chứng | Điều kiện ĐẠT | Điều kiện KHÔNG ĐẠT | Diễn giải |
|---|---|---|---|---|
| **C3** | Tỷ lệ Không Rỗng (Yield) tại $N$ đã đăng ký (Exp B, B-block) | $\text{Yield} \ge 85\%$ | $\text{Yield} < 85\%$ | Không đạt **không** bác bỏ H1a: bảo chứng vẫn hợp lệ nhưng không mang thông tin thực tiễn cho cặp (detector $\mathcal{M}$, đường đi $\Lambda_{1D}$) đã chọn |
| **C3b** | Độ phủ tương đối so với baseline plug-in trên tập kiểm thử (Exp B, B-block, Mục 8.2) | $\text{Median}_b\, \rho_{\text{cov}}(\hat\tau_b) \ge \gamma_{\text{cov}} = 0.5$ | $\text{Median}_b\, \rho_{\text{cov}}(\hat\tau_b) < \gamma_{\text{cov}}$ | Như C3: không bác bỏ H1a; cho thấy cái giá của bảo chứng quá cao so với cách chọn ngưỡng thông thường, thường do detector yếu hoặc $\Lambda_{1D}$ dừng sớm |

*H1c chỉ được kết luận ĐẠT khi cả C3 và C3b cùng đạt. C3 và C3b là tiêu chí mô tả hiệu năng trên một tổng thể hữu hạn nên được giữ dạng hai vùng; ước lượng điểm kèm CI bootstrap theo phân hoạch vẫn được báo cáo. Nếu bước P3 chọn $\theta_{\text{IoU}} = 0.5$, điều này được nêu rõ cùng kết quả H1c, vì độ khó của H1c phụ thuộc trực tiếp vào $\theta_{\text{IoU}}$. Nếu bước P3 không tìm được baseline đạt Macro-FDP $\le 0.02$ trên Meta-Train, C3b được ghi là không xác định.*

### Phần 4: Chỉ số Mô tả (không dùng để quyết định)
| Mã số | Nội dung | Mục |
|---|---|---|
| **D1** | Trung vị và phân vị 95% của $\mathcal{R}_b$; $\Delta_{\text{gap}}$ | 8.2 |
| **D2** | $\widehat{\text{VR}}$ của B-rand và $\Delta\text{VR}$ (năng lực thấp; C2b thay thế trong việc phát hiện tương quan) | 8.2 |
| **D3** | Micro-FDP, tỉ lệ đơn vị rỗng, Spearman$(N^{\text{acc}}, \text{FDP})$; Macro-FDP của $\tau_{\text{base}}$ trên tập kiểm thử | 2.2, 8.2 |
| **D4** | Phân rã FP: ảo giác vs. lệch vị trí | 2.1, 8.2 |
| **D5** | $\Delta_{\text{leak}}$ của Arm 3; kết quả $N_{\text{cluster}} \in \{20, 800\}$; Exp C chạy lại ở $\alpha = 0.02$; Stress-test OOD | 8.3 |
| **D6** | Mọi ước lượng của Exp A tầng 2 (ICC, $D_{\text{eff}}$, $\tilde m$, Moran's $I$ liên super-block); tỉ lệ train bị loại (tách phần do ride proxy và phần do vùng chắn rò rỉ); phân bố số clip mỗi ride proxy; so sánh phân phối thuộc tính cảnh giữa Meta-Train và Val | 8.0, 8.1 |
| **D7** | Rủi ro dọc đường đi trên pool, vị trí null đúng đầu tiên và tỉ lệ vòng kiểm định tới nó (Exp B; Arm 1, Arm 2 của Exp C); $\hat D^*$, $\hat D^*_C$ và năng lực của C2b từ bước P3.7 | 8.0, 8.2, 8.3 |
| **D8** | Macro-FDP của đơn vị Val có và không có clip train lân cận còn giữ lại (độ nhạy với rò rỉ dư của ride proxy) | A2, 8.2 |

---

## X. DANH MỤC THAM KHẢO HÀN LÂM

1. **Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L.** (2025). Learn then test: Calibrating predictive algorithms to achieve risk control. *The Annals of Applied Statistics*, (in press; preprint arXiv:2110.01052).
2. **Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T.** (2024). Conformal risk control. *International Conference on Learning Representations (ICLR 2024)*.
3. **Bentkus, V.** (2004). On Hoeffding's inequalities. *The Annals of Probability*, 32(2), 1650-1673.
4. **Hoeffding, W.** (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13-30.
5. **Maurer, W., Hothorn, L. A., & Lehmacher, W.** (1995). Multiple comparisons in drug clinical trials and preclinical assays: a-priori ordered hypotheses. *Biometrie in der chemisch-pharmazeutischen Industrie*, 6, 3-18.
6. **Westfall, P. H., & Krishen, A.** (2001). Optimally weighted, fixed sequence and gatekeeping multiple testing procedures. *Journal of Statistical Planning and Inference*, 99(1), 25-40.
7. **Marcus, R., Peritz, E., & Gabriel, K. R.** (1976). On closed testing procedures with special reference to ordered analysis of variance. *Biometrika*, 63(3), 655-660.
8. **Vovk, V.** (2012). Conditional validity of inductive conformal predictors. *Asian Conference on Machine Learning (ACML 2012)*, PMLR 25:475-490.
9. **Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T.** (2020). BDD100K: A diverse driving dataset for heterogeneous multitask learning. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020)*, 2636-2645.
10. **Kish, L.** (1965). *Survey Sampling*. John Wiley & Sons.
11. **Hurlbert, S. H.** (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187-211.
12. **Waudby-Smith, I., & Ramdas, A.** (2024). Estimating means of bounded random variables by betting. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 86(1), 1-27.
13. **Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J.** (2024). DETRs beat YOLOs on real-time object detection. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*.
14. **Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J.** (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2), 816-845.
15. **Oliveira, R. I., Orenstein, P., Ramos, T., & Romano, J. V.** (2024). Split conformal prediction and non-exchangeable data. *Journal of Machine Learning Research*, 25.
16. **Cliff, A. D., & Ord, J. K.** (1981). *Spatial Processes: Models & Applications*. Pion, London.
