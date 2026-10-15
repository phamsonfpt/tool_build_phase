# GIẢI THÍCH TOÁN HỌC & KIẾN TRÚC TỪ CON SỐ 0
## Vì sao mỗi công thức của paper H1 + H2 + H3 được thiết kế như vậy, dựa trên paper nào

**Đi kèm:** [01_PAPER_BLUEPRINT.md](01_PAPER_BLUEPRINT.md). **Phiên bản:** 0.1, 15/10/2026.

**Viết cho ai:** người biết xác suất cơ bản (trung bình, phương sai, phân phối chuẩn) nhưng **chưa từng nghe** LTT, Hoeffding, Bentkus, Mahalanobis, Hungarian, EDL hay FCD. Mỗi khái niệm đi theo cùng một trình tự:

1. **Vấn đề:** cần giải quyết điều gì.
2. **Ý tưởng:** giải thích bằng lời.
3. **Công thức:** viết chính xác.
4. **Ví dụ số:** tính tay được. Mọi con số trong file này đã được kiểm lại bằng `numpy/scipy`.
5. **Vì sao thiết kế như vậy:** paper nào làm trước, và mình khác ở đâu.
6. **Cạm bẫy:** chỗ dễ sai, nhất là những chỗ tài liệu cũ đang sai.

---

## MỤC LỤC
- [Phần A. Bài toán: nhãn giả, FP, FDP, Coverage](#phần-a-bài-toán)
- [Phần B. Learn-then-Test (LTT) từ con số 0](#phần-b-learn-then-test-ltt-từ-con-số-0)
- [Phần C. Kiến trúc RT-DETR và gióng hàng Hungarian](#phần-c-kiến-trúc-rt-detr-và-gióng-hàng-hungarian)
- [Phần D. Năm tín hiệu bất định, từng cái một](#phần-d-năm-tín-hiệu-bất-định)
- [Phần E. So sánh các tín hiệu một cách thống kê](#phần-e-so-sánh-các-tín-hiệu-một-cách-thống-kê)
- [Phần F. Toán của vòng Active Learning (H3)](#phần-f-toán-của-vòng-active-learning-h3)
- [Phần G. Bảng tra: công thức, lý do, paper gốc](#phần-g-bảng-tra)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

# PHẦN A. BÀI TOÁN

## A1. Nhãn giả có chọn lọc là gì
Có một detector đã huấn luyện và một kho ảnh chưa có nhãn. Detector đưa ra các **box ứng viên** (vị trí, lớp, điểm tin cậy). Với mỗi box, ta quyết định:
* **chấp nhận**: box trở thành nhãn giả và được đưa vào tập huấn luyện, không cần người kiểm tra;
* **từ chối**: box không được dùng, hoặc được gửi cho người kiểm tra.

Quyết định dựa vào một **điểm bất định** $S(j)$: điểm càng lớn càng đáng nghi. Box được chấp nhận khi $S(j) \le \lambda$, với $\lambda$ là ngưỡng. **Toàn bộ H1 là bài toán chọn $\lambda$. Toàn bộ H2 là bài toán chọn $S$.**

## A2. TP, FP và FDP
Một box được chấp nhận là **True Positive (TP)** nếu nó khớp với một đối tượng thật (nhãn GT) cùng lớp và IoU $\ge \theta_{\text{IoU}}$. Ngược lại là **False Positive (FP)**, tức nhãn giả sai.

**IoU** (Intersection over Union) của hai hộp $A$, $B$ là $\frac{\text{diện tích}(A \cap B)}{\text{diện tích}(A \cup B)}$. Hai hộp trùng khít cho IoU bằng 1, hai hộp rời nhau cho IoU bằng 0.

Trong một đơn vị dữ liệu $i$ (ví dụ một chuyến đi), gọi $N^{\text{acc}}_i$ là số box được chấp nhận và $V_i$ là số FP trong đó. **Tỉ lệ phát hiện sai (False Discovery Proportion):**
$$\text{FDP}_i(\lambda) = \frac{V_i(\lambda)}{\max(N^{\text{acc}}_i(\lambda), 1)}$$

Phần $\max(\cdot, 1)$ đảm bảo không chia cho 0: nếu không chấp nhận box nào thì FDP bằng 0.

**Ví dụ.** Chuyến đi có 2 ảnh, detector cho 6 box ứng viên với điểm $S$ = 0.1, 0.2, 0.3, 0.5, 0.7, 0.9. Box có $S = 0.5$ và $S = 0.9$ là FP.
* $\lambda = 0.35$: chấp nhận 3 box (0.1, 0.2, 0.3), không có FP, FDP = 0/3 = 0.
* $\lambda = 0.6$: chấp nhận 4 box, có 1 FP, FDP = 1/4 = 0.25.
* $\lambda = 1.0$: chấp nhận 6 box, có 2 FP, FDP = 2/6 ≈ 0.33.

Ngưỡng càng lỏng thì nhận càng nhiều box nhưng FDP càng cao. Đây là **sự đánh đổi trung tâm** của paper.

## A3. Rủi ro quần thể và Coverage
* **Rủi ro** $\mathcal R(\lambda) = \mathbb E[\text{FDP}_i(\lambda)]$: FDP trung bình trên **mọi** chuyến đi có thể gặp, không chỉ những chuyến ta có trong tay. Ta không bao giờ biết chính xác $\mathcal R$; ta chỉ ước lượng được nó từ mẫu.
* **Coverage** $\operatorname{Cov}(\lambda) = \frac{\sum_i \text{TP}_i(\lambda)}{\sum_i M^{\text{gt}}_i}$: tỉ lệ đối tượng thật được nhãn giả đúng bao phủ, tức "thu hoạch" được bao nhiêu.

**Mục tiêu của paper:**
$$\max_{\lambda} \operatorname{Cov}(\lambda) \quad \text{với điều kiện} \quad \mathcal R(\lambda) \le \alpha = 0.02$$
Nghĩa là thu được nhiều nhãn giả đúng nhất có thể, trong khi trung bình không quá 2% nhãn giả là sai.

## A4. Vì sao 2% lại quan trọng: confirmation bias
Arazo et al. (2020) cho thấy khi mô hình học từ nhãn giả sai của chính nó, nó sẽ càng tin vào lỗi đó ở vòng sau. Lỗi tự củng cố, gọi là **confirmation bias**. Rizve et al. (2021, UPS) cho thấy chọn nhãn giả dựa trên độ bất định, thay vì chỉ dựa trên confidence, giảm mạnh lượng nhãn giả sai và cải thiện kết quả. Paper này đi xa hơn một bước: không chỉ "chọn cẩn thận" mà **có bảo chứng toán học** về tỉ lệ sai.

### A5. Macro-FDP và Micro-FDP
* **Macro-FDP** (mục tiêu chính): trung bình FDP của từng đơn vị.
* **Micro-FDP**: tổng FP chia tổng box được chấp nhận trên toàn bộ pool.

**Ví dụ chênh lệch.** Đơn vị A có 1 box được chấp nhận và nó đúng (FDP = 0). Đơn vị B có 99 box, 9 box sai (FDP ≈ 0.091). Macro-FDP = (0 + 0.091)/2 ≈ 0.045. Micro-FDP = 9/100 = 0.09.

Macro-FDP được chọn vì FDP của mỗi đơn vị nằm trong $[0, 1]$, nên các bất đẳng thức ở Phần B áp dụng trực tiếp. Micro-FDP được báo cáo kèm theo (tài liệu H1, Mục 2.2).

---

# PHẦN B. LEARN-THEN-TEST (LTT) TỪ CON SỐ 0

## B1. Vì sao không thể "chọn $\lambda$ sao cho FDP trên tập hiệu chuẩn $\le 2\%$"?
Cách tự nhiên nhất: lấy một tập có nhãn (tập hiệu chuẩn, $N$ đơn vị), tính FDP trung bình $\widehat{\mathcal R}_N(\lambda)$ cho từng $\lambda$, rồi chọn $\lambda$ lớn nhất thỏa $\widehat{\mathcal R}_N(\lambda) \le 0.02$. Cách này **không an toàn**, vì hai lý do.

**Lý do 1: nhiễu lấy mẫu.** $\widehat{\mathcal R}_N$ chỉ là ước lượng. Một ngưỡng có rủi ro thật 2.5% vẫn có thể cho $\widehat{\mathcal R}_N \le 2\%$ do may mắn.
* *Ví dụ số:* $N = 1.500$, rủi ro thật 2.5%, trường hợp xấu nhất mỗi đơn vị có FDP bằng 0 hoặc 1. Xác suất ta quan sát được $\widehat{\mathcal R}_N \le 2\%$ là $\mathbb P(\text{Bin}(1500, 0.025) \le 30) \approx$ **12%**.

**Lý do 2: lời nguyền người thắng (winner's curse).** Ta thử nhiều ngưỡng cùng lúc và chọn cái "đẹp nhất". Nếu có 10 ngưỡng không an toàn, mỗi cái có 12% cơ hội trông như an toàn, thì xác suất ít nhất một cái lọt qua là $1 - 0.88^{10} \approx$ **73%** (nếu chúng độc lập).

Vì vậy ta cần một quy trình mà **xác suất chọn phải ngưỡng nguy hiểm được chặn trên bởi một con số $\delta$ do mình đặt trước**. Đó chính là LTT.

## B2. Ý tưởng của LTT: biến "chọn ngưỡng" thành "kiểm định giả thuyết"
Angelopoulos, Bates, Candès, Jordan & Lei (2021) đưa ra ý tưởng: với **mỗi** ngưỡng $\lambda$, đặt giả thuyết không
$$H_0(\lambda): \mathcal R(\lambda) > \alpha \qquad (\text{"ngưỡng này nguy hiểm"})$$
Chỉ dùng $\lambda$ khi có đủ bằng chứng để **bác bỏ** $H_0(\lambda)$. Ở đây giả thuyết không được đặt là "nguy hiểm" một cách cố ý: gánh nặng chứng minh nằm ở phía an toàn, giống như một loại thuốc phải chứng minh là an toàn trước khi được dùng.

### B2.1. p-value là gì (nhắc lại)
**p-value** là một con số tính từ dữ liệu, có tính chất: **nếu $H_0$ đúng**, xác suất p-value $\le u$ không vượt quá $u$, với mọi $u$:
$$\mathbb P_{H_0}(p \le u) \le u \quad \forall u \in (0, 1)$$
Tính chất này gọi là **super-uniform**. Hệ quả: nếu ta chỉ bác bỏ $H_0$ khi $p \le \delta$, thì xác suất bác bỏ nhầm (tức tin rằng ngưỡng an toàn trong khi nó nguy hiểm) không quá $\delta$.

Câu hỏi còn lại: làm sao tính p-value cho $H_0: \mathcal R > \alpha$ khi chỉ biết $\widehat{\mathcal R}_N$? Cần một bất đẳng thức cho biết **trung bình mẫu khó lệch xa khỏi kỳ vọng đến mức nào**. Đó là các bất đẳng thức tập trung (concentration inequalities).

## B3. Bất đẳng thức tập trung: Hoeffding, dạng KL, Bentkus, HB

Bối cảnh: $Z_1, \dots, Z_N$ (FDP của $N$ đơn vị) là **độc lập, cùng phân phối (i.i.d.)**, mỗi cái nằm trong $[0, 1]$, có kỳ vọng $\mu = \mathcal R(\lambda)$. Trung bình mẫu là $\bar Z = \widehat{\mathcal R}_N$. Ta muốn chặn trên $\mathbb P(\bar Z \le s)$ khi $\mu > \alpha$: nếu ngưỡng thực sự nguy hiểm, khả năng ta nhìn thấy một FDP nhỏ cỡ $s$ là bao nhiêu?

### B3.1. Hoeffding (1963), dạng cơ bản
$$\mathbb P(\bar Z \le \mu - t) \le \exp(-2 N t^2)$$

*Ý tưởng chứng minh, không cần nhớ chi tiết:*
1. Bất đẳng thức Markov: với biến không âm $Y$, $\mathbb P(Y \ge a) \le \mathbb E[Y]/a$.
2. Áp dụng cho $Y = e^{-\theta N \bar Z}$ với $\theta > 0$ (thủ thuật Chernoff), rồi tách tích nhờ tính độc lập: $\mathbb E[e^{-\theta \sum Z_i}] = \prod \mathbb E[e^{-\theta Z_i}]$.
3. Bổ đề Hoeffding chặn từng $\mathbb E[e^{-\theta Z_i}]$ cho biến trong $[0, 1]$, rồi tối ưu theo $\theta$.

**Vì sao không đủ cho bài toán của mình:** với $N = 1.500$ và $\delta = 0.05$, đặt $\exp(-2Nt^2) = \delta$ cho $t = \sqrt{\ln(1/\delta)/(2N)} \approx$ **0.0316**. Muốn chứng nhận $\mathcal R \le 0.02$ thì cần $\widehat{\mathcal R}_N \le 0.02 - 0.0316 < 0$, tức **không bao giờ chứng nhận được**. Dạng này lỏng vì nó không tận dụng việc $\mu$ rất nhỏ, gần 0. Ở gần 0, phương sai $\mu(1-\mu)$ rất nhỏ, nhưng dạng cơ bản luôn giả định phương sai lớn nhất có thể.

### B3.2. Hoeffding dạng KL (chặt hơn nhiều)
Cũng trong bài báo gốc (Hoeffding, 1963, Theorem 1) có dạng chặt hơn:
$$\mathbb P(\bar Z \le s) \le \exp\big(-N \cdot \text{kl}(s \,\|\, \mu)\big), \qquad s < \mu$$
với **phân kỳ Kullback–Leibler giữa hai phân phối Bernoulli**:
$$\text{kl}(a \,\|\, b) = a \ln\frac{a}{b} + (1-a)\ln\frac{1-a}{1-b}$$

Hiểu nôm na: $\text{kl}(a\|b)$ đo mức "khó phân biệt" giữa đồng xu có xác suất ngửa $a$ và đồng xu có xác suất ngửa $b$. Hai số càng khác nhau, xác suất nhầm lẫn càng giảm theo hàm mũ.

Vì $\text{kl}(s\|\mu)$ tăng theo $\mu$ khi $\mu > s$, trường hợp xấu nhất dưới $H_0$ là $\mu$ sát $\alpha$. Từ đó có p-value:
$$p_{\text{Hoeff}}(\lambda) = \exp\Big(-N \cdot \text{kl}\big(\min(\widehat{\mathcal R}_N, \alpha) \,\|\, \alpha\big)\Big)$$

### B3.3. Bentkus (2004): so với phân phối nhị thức
Bentkus chứng minh rằng trong mọi biến nằm trong $[0, 1]$ có cùng kỳ vọng, biến **nhị phân** (chỉ nhận 0 hoặc 1) là trường hợp "phân tán nhất", và đuôi của tổng các biến bất kỳ bị chặn bởi đuôi nhị thức nhân một hằng số $e$:
$$\mathbb P(\bar Z \le s) \le e \cdot \mathbb P\big(\text{Bin}(N, \mu) \le \lceil N s \rceil\big)$$
p-value tương ứng:
$$p_{\text{Bentkus}}(\lambda) = \min\Big(1,\; e \cdot \sum_{k=0}^{\lceil N \widehat{\mathcal R}_N \rceil} \binom{N}{k} \alpha^k (1-\alpha)^{N-k}\Big)$$

### B3.4. Hoeffding–Bentkus (HB): lấy cái nhỏ hơn
Bates, Angelopoulos, Lei, Malik & Jordan (2021, RCPS) đề xuất:
$$p_{\text{HB}} = \min(p_{\text{Hoeff}},\, p_{\text{Bentkus}})$$

Lấy min của hai p-value bất kỳ thì thường **không** hợp lệ. Ở đây hợp lệ vì cả hai đều là **cận trên của cùng một xác suất** $\mathbb P_{H_0}(\bar Z \le s)$, nên cận nhỏ hơn vẫn là một cận trên đúng. LTT dùng đúng p-value này.

### B3.5. Ví dụ số: $N = 1.500$, $\alpha = 0.02$, $\delta = 0.05$
$k$ = tổng FDP trên tập hiệu chuẩn (với FDP nhị phân thì là số đơn vị có lỗi), $\widehat{\mathcal R}_N = k/N$.

| $k$ | $\widehat{\mathcal R}_N$ | $p_{\text{Hoeff}}$ | $p_{\text{Bentkus}}$ | $p_{\text{HB}}$ | Chứng nhận $\mathcal R \le 2\%$? |
|---|---|---|---|---|---|
| 10 | 0.67% | 0.0001 | 0.0001 | 0.0001 | Có |
| 15 | 1.00% | 0.0093 | 0.0049 | 0.0049 | Có |
| 17 | 1.13% | 0.0333 | 0.0186 | 0.0186 | Có |
| **18** | **1.20%** | 0.0576 | 0.0333 | **0.0333** | **Có (biên cuối)** |
| 19 | 1.27% | 0.0942 | 0.0568 | 0.0568 | Không |
| 25 | 1.67% | 0.6373 | 0.5594 | 0.5594 | Không |

**Đọc bảng:**
* Muốn chứng nhận rủi ro thật $\le 2\%$, FDP trên tập hiệu chuẩn phải $\le 1.2\%$. Khoảng cách 0.8% là "phí bảo hiểm" cho nhiễu lấy mẫu.
* Tại $k = 18$, riêng Hoeffding-KL thất bại (0.0576 > 0.05) nhưng Bentkus thành công (0.0333). Đây là lý do dùng HB thay vì một cận đơn lẻ.
* Mốc $N \ge 149$ trong tài liệu H1: khi $\widehat{\mathcal R}_N = 0$, $p_{\text{Hoeff}} = (1 - \alpha)^N = 0.98^N \le 0.05 \iff N \ge 149$. Nhánh Bentkus có thêm hằng số $e$ nên cần $N \ge 198$. HB lấy min nên mốc do Hoeffding-KL quyết định.

## B4. Thử nhiều ngưỡng: kiểm định tuần tự cố định (fixed-sequence)

### B4.1. Vấn đề đa kiểm định
Ta có $K = 33$ ngưỡng ứng viên. Nếu kiểm từng cái ở mức $\delta$ rồi chọn cái tốt nhất trong số được bác bỏ, ta gặp lại lời nguyền người thắng ở B1. Cách cổ điển là **Bonferroni**: kiểm mỗi ngưỡng ở mức $\delta / K = 0.05/33 \approx 0.0015$. Tại $N = 1.500$, mức này chỉ cho phép $k \le 13$ (so với $k \le 18$ khi không chia), tức mất khá nhiều công suất.

### B4.2. Fixed-sequence testing
Xếp các ngưỡng theo một **thứ tự cố định từ trước** $\lambda^{(1)}, \lambda^{(2)}, \dots, \lambda^{(K)}$, từ chặt nhất đến lỏng nhất, **không dùng dữ liệu hiệu chuẩn** để xếp. Sau đó:
* kiểm $\lambda^{(1)}$ ở mức $\delta$. Bác bỏ được thì sang $\lambda^{(2)}$, và cứ thế;
* **gặp ngưỡng đầu tiên không bác bỏ được thì dừng.** Chọn ngưỡng cuối cùng đã bác bỏ được.

Quy trình này do Maurer, Hothorn & Lehmacher (1995) đưa ra trong thử nghiệm lâm sàng, được Westfall & Krishen (2001) phân tích, và là trường hợp riêng của closed testing (Marcus, Peritz & Gabriel, 1976). Mỗi phép kiểm dùng **trọn** mức $\delta$ mà không cần chia.

### B4.3. Vì sao nó đúng (Theorem 1, bằng lời)
1. Gọi $j^*$ là **ngưỡng nguy hiểm đầu tiên** trên đường đi. Vì đường đi cố định và rủi ro thật cố định, $j^*$ là một chỉ số **không ngẫu nhiên**, dù ta không biết nó.
2. Muốn chọn **bất kỳ** ngưỡng nguy hiểm nào, thuật toán phải đi qua $j^*$, tức phải bác bỏ nhầm $H_0(\lambda^{(j^*)})$. Nếu không bác bỏ được $j^*$, thuật toán dừng tại $j^*$ và chỉ trả về các ngưỡng đứng trước nó, vốn đều an toàn.
3. Vậy $\mathbb P(\text{chọn ngưỡng nguy hiểm}) \le \mathbb P(\text{bác bỏ nhầm tại } j^*) \le \delta$.

Toàn bộ quy trình chỉ có **một** cơ hội để sai, tại $j^*$, nên không cần chia $\delta$.

**Bảo chứng thu được (PAC: Probably Approximately Correct):**
$$\mathbb P_{\mathcal D_{\text{cal}}}\big(\mathcal R(\hat\lambda) \le \alpha\big) \ge 1 - \delta$$
**Cách đọc đúng:** nếu 100 nhóm nghiên cứu độc lập, mỗi nhóm rút một tập hiệu chuẩn riêng và chạy quy trình, thì trung bình không quá 5 nhóm nhận được ngưỡng có rủi ro thật vượt 2%. Bảo chứng **không** nói "mọi ảnh có FDP ≤ 2%" và cũng không nói "tập hiệu chuẩn cụ thể của tôi chắc chắn tốt". Đây là bảo chứng biên theo phân phối của tập hiệu chuẩn (Vovk, 2012 bàn về sự khác biệt giữa bảo chứng biên và bảo chứng có điều kiện).

### B4.4. Ví dụ đồ chơi
Đường đi có 6 ngưỡng, $N = 1.500$. FDP tổng quan sát được dọc đường đi:

| Bước $j$ | $k$ quan sát | $p_{\text{HB}}$ | Quyết định |
|---|---|---|---|
| 1 | 3 | ≈ 0 | Bác bỏ, đi tiếp |
| 2 | 9 | ≈ 0 | Bác bỏ, đi tiếp |
| 3 | 14 | 0.0023 | Bác bỏ, đi tiếp |
| 4 | 18 | 0.0333 | Bác bỏ, đi tiếp |
| 5 | 21 | 0.1433 | **Không bác bỏ, DỪNG** |
| 6 | 27 | (không kiểm) | |

Kết quả $\hat\lambda = \lambda^{(4)}$. Bước 6 có thể an toàn hay không cũng không quan trọng: quy trình không bao giờ nhìn tới nó. Đó là cái giá của tính hợp lệ.

## B5. Đường đi được dựng ra sao: truy vết biên Pareto trên Meta-Train
* **Vì sao cần đường đi 1D:** khi có nhiều tham số (ví dụ $\lambda = (\lambda_{\text{cls}}, \lambda_{\text{loc}}, \lambda_{\text{fcd}})$), lưới 3D có hàng nghìn điểm và không có thứ tự tự nhiên. Fixed-sequence cần một thứ tự.
* **Cách dựng:** trên một tập riêng là **Meta-Train** (không chạm vào tập hiệu chuẩn hay tập kiểm thử), với mỗi mức rủi ro mục tiêu tăng dần $\alpha_1 < \alpha_2 < \dots$, chọn điểm lưới có Coverage cao nhất trong số các điểm có FDP trên Meta-Train $\le \alpha_k$. Dãy kết quả đi dọc biên Pareto (Coverage tối đa ở mỗi mức rủi ro).
* **Vì sao phải là tập riêng:** nếu dựng đường đi trên chính tập hiệu chuẩn, thứ tự phụ thuộc dữ liệu hiệu chuẩn, nên $j^*$ không còn cố định và bước 1 của B4.3 sụp đổ.
* **Vì sao đường đi được kéo dài vượt mức trần $\alpha$:** nếu mọi điểm trên đường đi đều an toàn thì không thể vi phạm, kể cả khi giả định độc lập sai. Khi đó thực nghiệm kiểm H1 "đạt" mà không kiểm tra được gì. Phần kéo dài đặt những ngưỡng nguy hiểm thật lên đường đi để một quy trình sai có cơ hội lộ ra (tài liệu H1, Mục 6.2; mô phỏng E2, E3).

## B6. Đơn vị thống kê: vì sao là chuyến đi, không phải frame

### B6.1. Giả định i.i.d. là trái tim của mọi thứ ở trên
Hoeffding và Bentkus đều yêu cầu $Z_1, \dots, Z_N$ **độc lập**. Frame thứ $t$ và $t+1$ của cùng một video gần như giống hệt nhau (cùng xe, cùng ánh sáng, cùng đường). Nếu detector sai ở frame $t$ thì rất có thể sai ở frame $t+1$. Coi 200 frame của một clip là 200 quan sát độc lập là **giả lặp** (pseudoreplication; Hurlbert, 1984).

### B6.2. Design effect (Kish, 1965)
Với $m$ quan sát trong một cụm và hệ số tương quan nội cụm $\rho$ (ICC):
$$D_{\text{eff}} = 1 + (m - 1)\rho, \qquad N_{\text{eff}} = \frac{N_{\text{nominal}}}{D_{\text{eff}}}$$

**Ví dụ (giả định minh họa từ tài liệu H1):** 20 clip × 200 frame = 4.000 frame, $\rho = 0.6$: $D_{\text{eff}} = 1 + 199 \times 0.6 = 120.4$, nên $N_{\text{eff}} = 4000/120.4 \approx 33$. Ta tưởng có 4.000 quan sát, thực ra chỉ có khoảng 33 quan sát độc lập.

**Kể cả khi $\rho$ nhỏ:** $\rho = 0.05$ đã cho $D_{\text{eff}} = 1 + 199 \times 0.05 \approx 11$. Mô phỏng của H1 cho thấy ICC của FDP theo frame chỉ khoảng 0.04–0.21 (vì nhiễu nhị thức lớn), nhưng $D_{\text{eff}}$ vẫn từ 9 đến 42. Vì vậy tiêu chí H1b đặt trên $D_{\text{eff}}$, không đặt trên ICC.

### B6.3. Hậu quả cụ thể
Nếu ta đưa $N = 4.000$ vào công thức HB, bảng $k_{\max}$ cho phép chứng nhận khi $\widehat{\mathcal R}_N \le 1.53\%$, sát mức 2%. Với $N = 80.000$ là $1.90\%$. Nhưng lượng thông tin thật chỉ tương đương vài chục đơn vị, nên ngưỡng được chọn dao động quanh mức trần và xác suất vi phạm có thể lên tới vài chục phần trăm thay vì $\le 5\%$. Đây là cơ chế H1b khẳng định.

### B6.4. Rò rỉ ngữ cảnh
Chia ngẫu nhiên theo frame còn đặt frame $t$ vào tập hiệu chuẩn và frame $t+1$ vào tập kiểm thử. Ngưỡng đã "thấy trước" bối cảnh của tập kiểm thử, nên FDP đo trên tập kiểm thử trông tốt hơn thực tế. Arm 3 của Exp C đo hiện tượng này bằng cách áp cùng một ngưỡng lên tập test rò rỉ và tập test chưa thấy.

### B6.5. Vị trí trong tài liệu
Barber et al. (2023) chứng minh độ phủ của conformal prediction suy giảm theo mức vi phạm tính trao đổi được (exchangeability). Oliveira et al. (2024) phân tích split conformal trên dữ liệu phụ thuộc. Paper này đi hướng khác: **không** đề xuất bất đẳng thức mới cho dữ liệu phụ thuộc, mà **chọn lại đơn vị** để khôi phục giả định độc lập, rồi kiểm chứng thực nghiệm lựa chọn đó.

## B7. LTT nằm ở đâu so với các khung khác

| Khung | Kiểm soát gì | Khác LTT ở đâu |
|---|---|---|
| Conformal prediction (Angelopoulos & Bates, 2023) | Xác suất tập dự đoán chứa nhãn đúng | Chỉ cho một loại rủi ro (miscoverage) |
| RCPS (Bates et al., 2021) | Rủi ro đơn điệu theo $\lambda$, có xác suất $1-\delta$ | Cần rủi ro đơn điệu theo $\lambda$. LTT bỏ được yêu cầu này |
| Conformal Risk Control (Angelopoulos et al., 2024) | $\mathbb E[\text{rủi ro}] \le \alpha$, không có $\delta$ | Bảo chứng theo kỳ vọng, không theo xác suất |
| Selective classification (Geifman & El-Yaniv, 2017) | Rủi ro trên tập được chấp nhận, dùng cận nhị thức | Ý tưởng gần nhất cho phân loại, nhưng dùng Bonferroni qua tìm kiếm nhị phân |
| Conformal selection (Jin & Candès, 2023) | FDR theo nghĩa Benjamini–Hochberg | FDR là kỳ vọng FDP trên một lần chọn. H1 kiểm soát kỳ vọng FDP theo đơn vị, có độ tin cậy $1-\delta$ |
| **LTT** (Angelopoulos et al., 2021) | Mọi rủi ro bị chặn, tham số bất kỳ, có xác suất $1-\delta$ | Chọn vì FDP không đơn điệu theo $\lambda$ và tham số có thể nhiều chiều |

**Tại sao FDP không đơn điệu:** nới ngưỡng có thể thêm 10 box đúng và 0 box sai, khiến FDP **giảm**. RCPS cần đơn điệu nên không dùng được. LTT không cần.

---

# PHẦN C. KIẾN TRÚC RT-DETR VÀ GIÓNG HÀNG HUNGARIAN

## C1. RT-DETR nói ngắn gọn
DETR (Carion et al., 2020) coi object detection là bài toán **dự đoán một tập hợp**. Mô hình có $K$ **object query**, là các vector học được. Mỗi query đi qua Transformer decoder và cuối cùng thành **một** dự đoán (box, lớp). RT-DETR (Zhao et al., 2024) là phiên bản thời gian thực với $K = 300$ query.

**Ba tensor ta lấy ra cho mỗi query $j$ của mô hình $m$:**
1. $\hat{\mathbf b}^{(m)}_j \in [0, 1]^4$: hộp $(x, y, w, h)$ chuẩn hóa theo kích thước ảnh.
2. $\mathbf l^{(m)}_j \in \mathbb R^C$: logit phân loại.
3. $\mathbf z^{(m)}_j \in \mathbb R^{256}$: **hidden state cuối của decoder**, vector mà cả đầu phân loại lẫn đầu hộp đều đọc. Đây là "không gian latent" của paper.

```
Ảnh ─► Backbone (ResNet-50) ─► Hybrid Encoder ─► Decoder (6 lớp, 300 query)
                                                          │
                                         z_j ∈ R^256 (latent, Phần D4–D5)
                                          ├──► Đầu phân loại: l_j = W z_j + b  (Phần D1–D3)
                                          └──► Đầu hộp (MLP): b_j ∈ [0,1]^4     (Phần D2)
```

**Lưu ý quan trọng:** RT-DETR huấn luyện đầu phân loại bằng **sigmoid** theo từng lớp (varifocal loss), không phải softmax. Điểm tin cậy là $\hat p_j = \max_c \sigma(l_{j,c})$ với $\sigma(x) = 1/(1 + e^{-x})$. Xem vấn đề P7 trong blueprint.

## C2. Vì sao phải gióng hàng giữa các mô hình
Ensemble có 3 mô hình, mỗi mô hình cho 300 query. **Thứ tự query không có nghĩa:** query số 17 của mô hình 1 có thể là chiếc xe đỏ, trong khi chiếc xe đỏ ở mô hình 2 lại là query số 203. Muốn đo "các mô hình có đồng ý về chiếc xe đỏ không", trước tiên phải biết query nào của mô hình 2 ứng với query nào của mô hình 1.

## C3. Bài toán gán (assignment) và thuật toán Hungarian
**Bài toán:** có ma trận chi phí $\mathbf C \in \mathbb R^{K \times K}$, với $C_{u,v}$ là "độ khác nhau" giữa query $u$ của anchor và query $v$ của member. Tìm một **song ánh** (mỗi $u$ ghép đúng một $v$, không $v$ nào bị dùng hai lần) có tổng chi phí nhỏ nhất:
$$\pi^* = \arg\min_{\pi} \sum_u C_{u, \pi(u)}$$
**Thuật toán Hungarian** (Kuhn, 1955; Munkres, 1957) giải chính xác bài toán này trong thời gian $O(K^3)$. Với $K = 300$, `scipy.optimize.linear_sum_assignment` chạy trong vài mili giây.

**Ví dụ vì sao không dùng cách tham lam.** Ma trận chi phí 2×2 với $C_{11} = 0.10$, $C_{12} = 0.20$, $C_{21} = 0.15$, $C_{22} = 0.90$.
* *Tham lam:* ghép cặp rẻ nhất trước, $(1,1)$ chi phí 0.10. Query 2 bị buộc nhận cột còn lại, $(2,2)$ chi phí 0.90. Tổng **1.00**.
* *Hungarian:* $(1,2) + (2,1) = 0.20 + 0.15 =$ **0.35**.

Cách tham lam ghép sai một box chỉ vì nó "tranh" trước. Với cảnh đông xe, sai kiểu này rất phổ biến.

**Vì sao chọn Hungarian:** chính DETR dùng Hungarian để ghép dự đoán với nhãn khi huấn luyện (Carion et al., 2020), nên dùng cùng cơ chế để ghép giữa các mô hình là tự nhiên. Miller et al. (2019) so sánh các chiến lược gộp dự đoán của nhiều mô hình trong detection và cho thấy chiến lược ghép ảnh hưởng đáng kể tới chất lượng bất định.

## C4. Ma trận chi phí
$$C_{u,v} = \lambda_{L_1}\,\|\hat{\mathbf b}_u - \hat{\mathbf b}_v\|_1 + \lambda_{\text{giou}}\,\big(1 - \text{GIoU}(\hat{\mathbf b}_u, \hat{\mathbf b}_v)\big) + \lambda_{\text{cls}}\, D_{\text{JS}}(\hat{\mathbf p}_u, \hat{\mathbf p}_v)$$
* **L1** đo lệch tọa độ.
* **GIoU** (Rezatofighi et al., 2019) là IoU mở rộng, vẫn có giá trị khi hai hộp không giao nhau (IoU khi đó luôn bằng 0, không phân biệt được hai hộp xa hay gần).
* Hệ số $5, 2, 2$ lấy theo DETR.
* **Ghép rỗng:** nếu cặp được ghép có IoU $< 0.2$ thì coi như member **không tìm thấy** đối tượng. Khuyến nghị lưu một biến riêng "số member không tìm thấy", thay vì gán box bằng 0 (vấn đề P13).

---

# PHẦN D. NĂM TÍN HIỆU BẤT ĐỊNH

Năm tín hiệu được xếp theo thiết kế giai thừa 2×2:

|  | **Không gian đầu ra** (box, xác suất) | **Không gian latent** ($\mathbf z$) |
|---|---|---|
| **Một mô hình** | D1. Min-confidence, D3. EDL vacuity | D4. Mahalanobis |
| **Ensemble 3 mô hình** | D2. Box variance + JS | D5. FCD (đề xuất) |

Thiết kế này cho phép tách hai câu hỏi: **"đo ở đâu"** (đầu ra hay latent) và **"bao nhiêu mô hình"** (1 hay 3). Nếu chỉ so FCD với min-confidence, reviewer sẽ hỏi lợi thế đến từ latent hay chỉ từ việc dùng gấp 3 compute.

---

## D1. Min-confidence (Maximum Class Probability)

**Công thức:** $S_{\text{conf}}(j) = 1 - \hat p_j$, với $\hat p_j = \max_c p_{j,c}$.

**Paper gốc:** Hendrycks & Gimpel (2017) cho thấy xác suất lớn nhất là một baseline mạnh và đơn giản cho phát hiện mẫu sai hoặc mẫu ngoài phân phối (OOD). Mọi paper về bất định đều phải có nó làm mốc.

**Ví dụ số (softmax, 3 lớp):** logit $(9.0, 2.0, 1.5)$ cho $p \approx (0.9985, 0.0009, 0.0006)$, nên $S_{\text{conf}} \approx 0.0015$, rất tự tin. Nhân logit với 0.3 thành $(2.7, 0.6, 0.45)$ thì $p \approx (0.81, 0.10, 0.09)$. **Độ tự tin phụ thuộc vào độ lớn của logit, không chỉ thứ tự của chúng.**

**Vì sao nó thất bại trước mẫu lạ (cơ sở lý thuyết thật, thay cho lập luận cũ):**
* Hein, Andriushchenko & Bitterwolf (2019) **chứng minh** rằng mạng ReLU là hàm tuyến tính từng mảnh. Khi đi đủ xa theo gần như mọi hướng ($\mathbf x \to t\mathbf x$, $t \to \infty$), logit tăng tuyến tính theo $t$ nên độ tự tin tiến tới 1. Tức là **càng xa dữ liệu huấn luyện, mô hình càng có thể tự tin**.
* Nguyen, Yosinski & Clune (2015) tạo ra ảnh nhiễu mà mạng phân loại với độ tự tin trên 99%.
* Guo et al. (2017) cho thấy mạng sâu hiện đại bị **quá tự tin** một cách hệ thống (miscalibration).

**Cách nói đúng trong paper:** đầu phân loại là một phép chiếu tuyến tính $\mathbf l = \mathbf W\mathbf z + \mathbf b$. Mọi thành phần của $\mathbf z$ nằm trong **không gian không** (null space) của $\mathbf W$, tức các hướng mà $\mathbf W\mathbf z_\perp = 0$, không ảnh hưởng gì tới logit. Với $\mathbf W \in \mathbb R^{8 \times 256}$, không gian này có ít nhất $256 - 8 = 248$ chiều. Một mẫu có thể rất lạ theo 248 chiều đó mà confidence không hề thay đổi. Đây là lập luận tuyến tính đúng, đã có trong tài liệu H2 (Bước 1). Nó cũng là động cơ trực tiếp để đo trong latent (D4).

**Cạm bẫy (P7):** RT-DETR dùng sigmoid. Lập luận "softmax ép tổng bằng 1 nên nhiễu nhỏ cũng thành tự tin" không áp dụng. Lập luận null space và Hein et al. vẫn đúng với sigmoid.

---

## D2. Bất đồng đầu ra của ensemble (box variance + JS)

**Paper gốc:** Lakshminarayanan, Pritzel & Blundell (2017), Deep Ensembles: huấn luyện $M$ mô hình với seed khác nhau, độ bất đồng giữa chúng là thước đo bất định đơn giản và mạnh. Ovadia et al. (2019) đánh giá trên nhiều loại dịch chuyển phân phối và thấy ensemble là phương pháp tốt nhất về độ tin cậy của bất định. Với detection, Miller et al. (2019) và tổng quan Feng et al. (2022) dùng phương sai của hộp và entropy của lớp.

### D2.1. Phương sai tọa độ
Sau khi gióng hàng, mỗi box anchor $u$ có $M$ hộp tương ứng. Trung bình $\bar{\mathbf b}_u = \frac1M \sum_m \hat{\mathbf b}^{(m)}_u$, và
$$\sigma^2_{\text{loc}}(u) = \frac{1}{M-1}\sum_{m=1}^M \|\hat{\mathbf b}^{(m)}_u - \bar{\mathbf b}_u\|^2$$
Chia cho $M - 1$ (không phải $M$) để có ước lượng không chệch của phương sai (hiệu chỉnh Bessel).

### D2.2. Entropy và Jensen–Shannon: đo bất đồng về lớp
* **Entropy Shannon** $H(\mathbf p) = -\sum_c p_c \ln p_c$ đo độ "không chắc" của **một** phân phối. Bằng 0 khi chắc chắn tuyệt đối, bằng $\ln C$ khi chia đều.
* **JS tổng quát** (Lin, 1991) giữa $M$ phân phối với trọng số đều:
$$D_{\text{JS}}(u) = \underbrace{H(\bar{\mathbf p}_u)}_{\text{bất định tổng}} - \underbrace{\frac1M\sum_m H(\hat{\mathbf p}^{(m)}_u)}_{\text{bất định trung bình của từng mô hình}}$$

**Ý nghĩa sâu:** đại lượng này chính là **thông tin tương hỗ** giữa nhãn và "mô hình nào được chọn". Trong học Bayes, nó là phần **bất định nhận thức** (epistemic: do mô hình thiếu kiến thức), tách khỏi phần **bất định ngẫu nhiên** (aleatoric: do bản thân dữ liệu mơ hồ). Houlsby et al. (2011, BALD) và Gal, Islam & Ghahramani (2017) dùng đúng đại lượng này để chọn mẫu trong AL. Depeweg et al. (2018) và Kendall & Gal (2017) phân tích sự phân rã này.

**Ví dụ số (3 lớp, 3 mô hình):**
* *Đồng ý:* cả 3 mô hình cho $(0.9, 0.05, 0.05)$. $H(\bar{\mathbf p}) = 0.394$, trung bình $H = 0.394$, $D_{\text{JS}} = 0$. Mỗi mô hình có một chút không chắc, nhưng **không bất đồng**.
* *Bất đồng:* $(0.9, 0.05, 0.05)$, $(0.1, 0.85, 0.05)$, $(0.3, 0.3, 0.4)$. $H(\bar{\mathbf p}) = 1.028$, trung bình $H = 0.667$, $D_{\text{JS}} = 0.360$ nat (tối đa là $\ln 3 \approx 1.10$).

### D2.3. Vì sao nó thất bại: thiên kiến chung
Ba mô hình cùng kiến trúc, cùng dữ liệu huấn luyện, nên có cùng **thiên kiến quy nạp**. Ví dụ CNN có xu hướng dựa vào kết cấu bề mặt thay vì hình dạng (Geirhos et al., 2019). Nếu một hàng rào sắt có kết cấu giống lưới tản nhiệt xe hơi, cả 3 mô hình có thể **cùng** nhận nhầm nó là xe, ở **cùng** vị trí. Khi đó $\sigma^2_{\text{loc}} \approx 0$, $D_{\text{JS}} \approx 0$, và tín hiệu không thấy gì. Ensemble chỉ bắt được lỗi mà các mô hình **không** cùng mắc.

### D2.4. Cạm bẫy: thang đo (P8) và cách sửa
* Tọa độ chuẩn hóa: lệch 0.02 cho $\sigma^2 \approx 4 \times 10^{-4}$, trong khi $D_{\text{JS}} \in [0, 2.08]$ với 8 lớp. Cộng thẳng $\sigma^2_{\text{loc}} + D_{\text{JS}}$ thì thành phần vị trí gần như biến mất.
* **Sửa 1, chuẩn hóa theo kích thước hộp:** đo lệch theo $\Delta x/w$, $\Delta y/h$, $\Delta \ln w$, $\Delta \ln h$, giống cách mã hóa hộp của Fast R-CNN (Girshick, 2015). Lệch 5 pixel với một chiếc xe ở xa (rộng 20 pixel) nghiêm trọng hơn nhiều so với một chiếc xe ở gần (rộng 400 pixel).
* **Sửa 2, đưa về cùng thang phân vị:** với mỗi thành phần $X$, tính hàm phân phối thực nghiệm $\hat F_X$ trên Meta-Train, rồi $S_{\text{out}} = \max(\hat F_{\text{loc}}(\sigma^2_{\text{loc}}), \hat F_{\text{cls}}(D_{\text{JS}}))$. Hai thành phần khi đó cùng nằm trong $[0, 1]$ và có cùng nghĩa ("lớn hơn bao nhiêu phần trăm box trên Meta-Train").
* **Khớp code và tài liệu:** code hiện chỉ tính $H(\bar{\mathbf p})$, không tính JS (P13). Chọn một và ghi vào paper.

---

## D3. Evidential Deep Learning (EDL vacuity)

**Paper gốc:** Sensoy, Kaplan & Kandemir (2018). Thay vì cho ra **một** phân phối xác suất, mô hình cho ra một **phân phối trên các phân phối**: phân phối Dirichlet. Nền tảng là Subjective Logic (Jøsang, 2016).

### D3.1. Dirichlet là gì (trực giác)
Tung một con xúc xắc 3 mặt chưa biết có đều hay không. Sau 3 lần tung, ta rất không chắc về xác suất từng mặt. Sau 3.000 lần tung, ta khá chắc. Dirichlet với tham số $\boldsymbol\alpha = (\alpha_1, \dots, \alpha_C)$ mô tả niềm tin đó: $\alpha_c - 1$ giống như "số lần đã thấy mặt $c$". Tổng $S = \sum_c \alpha_c$ càng lớn thì niềm tin càng chắc.

### D3.2. Công thức
* Bằng chứng không âm: $\mathbf e = g(\mathbf l) \ge 0$, với $g$ là ReLU, exp hoặc softplus.
* $\alpha_c = e_c + 1$, $S = \sum_c \alpha_c = \sum_c e_c + C$.
* Niềm tin vào lớp $c$: $b_c = e_c / S$. **Độ rỗng (vacuity):** $u = C / S$. Ta có $\sum_c b_c + u = 1$.
* $S_{\text{edl}}(j) = u_j = \frac{C}{\sum_c e_{j,c} + C} \in (0, 1]$.

**Ví dụ số ($C = 3$):**
| Bằng chứng $\mathbf e$ | $S$ | Vacuity $u$ | Niềm tin $\mathbf b$ |
|---|---|---|---|
| $(0, 0, 0)$ | 3 | **1.000** | $(0, 0, 0)$, hoàn toàn không biết |
| $(20, 1, 0)$ | 24 | 0.125 | $(0.83, 0.04, 0)$ |
| $(200, 0, 0)$ | 203 | 0.015 | $(0.985, 0, 0)$ |

### D3.3. Loss huấn luyện (vì sao có hai thành phần)
$$\mathcal L = \sum_c \Big[(y_c - \tfrac{\alpha_c}{S})^2 + \tfrac{\alpha_c(S - \alpha_c)}{S^2(S+1)}\Big] + \lambda_t\, \text{KL}\big(\text{Dir}(\tilde{\boldsymbol\alpha}) \,\|\, \text{Dir}(\mathbf 1)\big)$$
* Thành phần 1 là kỳ vọng của sai số bình phương khi $\mathbf p \sim \text{Dir}(\boldsymbol\alpha)$: (độ lệch của trung bình)² + phương sai. Nó thưởng cho dự đoán đúng **và** chắc.
* Thành phần 2 phạt bằng chứng dồn vào **lớp sai**: $\tilde{\boldsymbol\alpha}$ là $\boldsymbol\alpha$ sau khi bỏ bằng chứng của lớp đúng, và bị kéo về Dirichlet đều. $\lambda_t$ tăng dần theo epoch (annealing) để mô hình học phân loại trước.

### D3.4. Vì sao nó thất bại, và phê bình trong tài liệu
* Bằng chứng vẫn là hàm của logit $\mathbf l$, nên mẫu lạ kích hoạt mạnh một logit vẫn cho bằng chứng lớn và vacuity nhỏ. Cùng cơ chế với D1 (Hein et al., 2019).
* Bengs, Hüllermeier & Waegeman (2022) chỉ ra rằng học bất định nhận thức bằng cách tối thiểu hóa loss như EDL không cho đại lượng có nghĩa nhận thức rõ ràng. Ulmer, Hardmeier & Frellsen (2023) tổng quan các phương pháp evidential và hạn chế của chúng.
* EDL không có bất định cho tọa độ hộp, và cần **huấn luyện lại** detector, nên tập ứng viên khác (P9).

**Vì sao vẫn đưa vào:** EDL đại diện cho trường phái "một mô hình, một lần forward, có bất định nhận thức". Không có nó, reviewer sẽ hỏi "tại sao không dùng phương pháp một mô hình rẻ hơn?".

---

## D4. Khoảng cách Mahalanobis trong latent (một mô hình)

**Paper gốc:** Lee, Lee, Lee & Shin (2018) phát hiện mẫu OOD bằng cách khớp một phân phối Gauss cho đặc trưng của mỗi lớp, rồi đo khoảng cách Mahalanobis. Khoảng cách do Mahalanobis (1936) đề xuất. Với detection, Du et al. (2022, VOS) cũng làm việc trực tiếp trên đặc trưng của từng đối tượng để phát hiện OOD.

### D4.1. Từ Euclid đến Mahalanobis: ví dụ 2 chiều
Giả sử đặc trưng của lớp "xe" có tâm $\boldsymbol\mu = (0, 0)$ và ma trận hiệp phương sai
$$\boldsymbol\Sigma = \begin{pmatrix} 4 & 1.8 \\ 1.8 & 1 \end{pmatrix}$$
Chiều 1 trải rộng (độ lệch chuẩn 2), chiều 2 hẹp (độ lệch chuẩn 1), và hai chiều tương quan mạnh ($\rho = 1.8/(2 \cdot 1) = 0.9$). Đám mây điểm là một elip dài nghiêng theo đường chéo.

Xét hai điểm:
| Điểm | Euclid $\lVert\mathbf x - \boldsymbol\mu\rVert$ | Mahalanobis $\sqrt{(\mathbf x-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)}$ | Nhận xét |
|---|---|---|---|
| $\mathbf x_1 = (2, 1)$ | 2.24 | **1.03** | Nằm **dọc** trục dài của elip, hoàn toàn bình thường |
| $\mathbf x_2 = (1, -1)$ | 1.41 | **3.36** | Đi **ngược** chiều tương quan, rất bất thường |

Euclid nói $\mathbf x_2$ gần hơn. Mahalanobis nói $\mathbf x_2$ lạ hơn gấp ba. Mahalanobis đúng, vì nó đo khoảng cách **theo đơn vị độ lệch chuẩn theo từng hướng** của chính dữ liệu.

*(Kiểm tra: $\boldsymbol\Sigma^{-1} = \frac{1}{0.76}\begin{pmatrix}1 & -1.8\\-1.8 & 4\end{pmatrix}$. Với $\mathbf x_2$: $\frac{1}{0.76}(1 + 3.6 + 4) = 11.32$, căn bằng 3.36.)*

### D4.2. Vì sao lại có ý nghĩa xác suất
Nếu đặc trưng của lớp $c$ tuân theo phân phối Gauss $\mathcal N(\boldsymbol\mu_c, \boldsymbol\Sigma)$ thì
$$-\ln p(\mathbf z \mid c) = \tfrac12 D_c(\mathbf z)^2 + \text{hằng số}$$
**Khoảng cách Mahalanobis chính là log-mật độ** (đổi dấu) dưới mô hình Gauss. Khoảng cách lớn nghĩa là mật độ thấp, tức mẫu hiếm khi xuất hiện trong dữ liệu huấn luyện. Lee et al. (2018) gọi đây là bộ phân loại sinh (generative classifier) dựng trên đặc trưng của mạng. Mukhoti et al. (2023, DDU) dùng đúng ý tưởng "mật độ đặc trưng" này làm thước đo bất định nhận thức cho mô hình một lần forward.

### D4.3. Công thức trong paper
1. Trên tập huấn luyện, với mỗi box GT, lấy query của anchor khớp tốt nhất (IoU $\ge 0.5$, đúng lớp), giữ vector $\mathbf z \in \mathbb R^{256}$.
2. PCA xuống $\tilde{\mathbf z} \in \mathbb R^{64}$.
3. Với mỗi lớp $c$: tâm $\boldsymbol\mu_c$, hiệp phương sai $\boldsymbol\Sigma$ (xem D4.4 về dùng chung hay riêng).
4. $S_{\text{maha}}(j) = D(j) = \sqrt{(\tilde{\mathbf z}_j - \boldsymbol\mu_{\hat y_j})^\top \boldsymbol\Sigma^{-1} (\tilde{\mathbf z}_j - \boldsymbol\mu_{\hat y_j})}$, với $\hat y_j$ là lớp được dự đoán.

### D4.4. Các lựa chọn thiết kế và lý do
* **Vì sao PCA về 64 chiều:** ước lượng ma trận hiệp phương sai $d \times d$ cần ước lượng $d(d+1)/2$ tham số. Với $d = 256$ là 32.896 tham số; với $d = 64$ là 2.080. Spec M29 ước tính khoảng 7.000 vector cho 8 lớp, và lớp hiếm như `train` hay `rider` có thể chỉ vài chục mẫu. Ở 256 chiều, ma trận của các lớp đó **suy biến** (không nghịch đảo được).
* **Hiệp phương sai dùng chung (tied):** Lee et al. (2018) dùng **một** ma trận chung cho mọi lớp, chỉ tâm là riêng. Cách này ổn định hơn nhiều cho lớp hiếm. Tài liệu cũ dùng ma trận riêng cho từng lớp. **Khuyến nghị:** mặc định dùng chung, ma trận riêng là ablation.
* **Co rút (shrinkage) thay cho jitter:** thay vì cộng $\epsilon\mathbf I$ rất nhỏ ($10^{-5}$ hay $10^{-6}$), dùng ước lượng Ledoit & Wolf (2004). Nó tự chọn mức co rút tối ưu về phía ma trận đường chéo và luôn cho ma trận khả nghịch, điều kiện tốt.
* **Cạm bẫy của PCA:** Kamoi & Kobayashi (2020) cho thấy với phát hiện bất thường, các thành phần chính có **phương sai nhỏ** thường mang nhiều thông tin nhất. PCA giữ 64 thành phần phương sai **lớn** nhất có thể vứt bỏ đúng chỗ đó. **Ablation đăng ký trước:** $d_{\text{pca}} \in \{32, 64, 128\}$ và Ledoit–Wolf ở 256 chiều đầy đủ, chọn trên Meta-Train.
* **"Relative":** Ren et al. (2021) đề xuất *Relative Mahalanobis Distance*, trừ đi khoảng cách tới một Gauss nền chung. Tên "Relative Latent Discrepancy" trong tài liệu cũ dễ bị nhầm với phương pháp này. Nên đổi tên hoặc nói rõ là khác.

### D4.5. Hình học 64 chiều: khoảng cách "bình thường" là bao nhiêu
Đây là điểm mà tài liệu cũ sai, và cũng là điểm khó tin nhất với người mới.

Nếu $\tilde{\mathbf z} \sim \mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)$ trong $d$ chiều thì $D^2$ tuân theo **phân phối chi bình phương** $\chi^2_d$: tổng bình phương của $d$ biến chuẩn tắc độc lập.
* $\mathbb E[D^2] = d$. Với $d = 64$, $D^2 \approx 64$, tức $D \approx 8$.
* Chính xác hơn, $D$ tuân theo phân phối chi với 64 bậc tự do: **trung bình 7.97, độ lệch chuẩn 0.71**.
* Phân vị 95%: $D^2 = 83.7$, tức $D = 9.15$.
* $\mathbb P(D \le 2.1) \approx 4 \times 10^{-26}$.

**Hệ quả 1: mẫu bình thường không nằm gần tâm.** Trong không gian nhiều chiều, gần như toàn bộ xác suất của phân phối Gauss nằm trên một **vỏ cầu mỏng** bán kính khoảng $\sqrt d$, không phải gần tâm. Đây là hiện tượng tập trung độ đo. Một mẫu có $D = 2.1$ ở 64 chiều **không phải** "gần tâm nên an toàn". Nó gần tâm một cách bất thường. Ví dụ $D_1 = 2.1$, $D_2 = 6.8$, $D_3 = 5.2$ trong tài liệu H2 vì vậy không khớp hình học: cả ba số đều nhỏ hơn khoảng cách của một mẫu bình thường (khoảng 8).

**Hệ quả 2: ngưỡng phân vị có nghĩa cụ thể.** Ngưỡng phân vị 95% trong spec M29 tương ứng $D \approx 9.15$ nếu giả định Gauss đúng. Nó có nghĩa: **5% box bình thường sẽ bị gắn cờ nhầm**. Nó không có nghĩa "vượt ngưỡng thì chắc chắn là ảo giác" (P12). Giả định Gauss chỉ là xấp xỉ, nên trong paper ngưỡng luôn được chọn bằng LTT hoặc phân vị thực nghiệm, không lấy từ bảng $\chi^2$.

### D4.6. Vì sao nó vẫn có thể thất bại
Một mô hình đơn lẻ có thể ánh xạ một mẫu lạ vào **đúng** vùng vỏ cầu bình thường của một lớp. Khi đó $D$ trông bình thường. Ensemble (D5) có thêm cơ hội: mô hình khác có thể không bị lừa theo cùng cách.

---

## D5. FCD: tín hiệu latent của ensemble (phương pháp đề xuất)

### D5.1. Ý tưởng
Kết hợp hai điều đã biết:
1. Latent nhìn thấy những gì đầu ra không thấy (D1, D4: null space, mật độ đặc trưng).
2. Các mô hình khác seed không mắc cùng mọi lỗi (D2: Deep Ensembles).

Fort, Hu & Lakshminarayanan (2019) giải thích vì sao điều 2 đúng: huấn luyện từ các khởi tạo khác nhau đưa mô hình tới **các vùng khác nhau** (mode) của bề mặt loss, và các mô hình đó khác nhau về chức năng, không chỉ khác chút ít quanh một điểm. Lee et al. (2018) cũng đã gộp Mahalanobis từ **nhiều lớp** (layer) của cùng một mạng ("feature ensemble") và thấy tốt hơn dùng một lớp. FCD gộp Mahalanobis từ **nhiều mô hình**.

### D5.2. Kiến trúc: tại sao mỗi mô hình phải có thống kê riêng
Hai mô hình khác seed học được biểu diễn **giống nhau về cấu trúc nhưng khác hệ tọa độ**: gần như là phép xoay hoặc biến đổi tuyến tính của nhau (Li et al., 2016; Kornblith et al., 2019 đo bằng CKA). Vì vậy:
* **Không được** so sánh trực tiếp $\mathbf z^{(1)}$ với $\mathbf z^{(2)}$. Trục 17 của mô hình 1 không cùng nghĩa với trục 17 của mô hình 2. Đây là lỗi của `fcd_variance.py` hiện tại (P3).
* **Được** so sánh các **số vô hướng** $D_1, D_2, D_3$, vì mỗi $D_m$ đo trong hệ quy chiếu của chính mô hình $m$ và có cùng đơn vị (số độ lệch chuẩn).

Do đó mỗi mô hình $m$ có PCA $\mathcal P_m$, tâm $\boldsymbol\mu_{c,m}$ và hiệp phương sai $\boldsymbol\Sigma_m$ riêng, đều fit trên dữ liệu huấn luyện của chính nó.

**Quy trình cho một box ứng viên $j$ (query của anchor):**
1. Hungarian (Phần C) tìm query tương ứng $\pi_m(j)$ ở mỗi member.
2. Với mỗi $m$: $D_m(j) = \|\mathcal P_m(\mathbf z^{(m)}_{\pi_m(j)}) - \boldsymbol\mu_{\hat y_j, m}\|_{\boldsymbol\Sigma_m^{-1}}$, với $\hat y_j$ là lớp anchor dự đoán.
3. Gộp $(D_1, D_2, D_3)$ thành một điểm số (D5.3–D5.6).
4. Member không tìm thấy đối tượng: đặt $D_m = $ một giá trị lớn đăng ký trước (ví dụ phân vị 99.9% trên Meta-Train), hoặc đưa thẳng box sang vùng Red.

### D5.3. Bốn cách gộp và ví dụ số (thang đo đúng: mẫu bình thường có $D \approx 8 \pm 0.7$)

| Tình huống | $(D_1, D_2, D_3)$ | Trung bình $\bar D$ | Phương sai $s^2$ | $\bar D + s$ | $\max_m D_m$ |
|---|---|---|---|---|---|
| (a) Bình thường | (7.9, 8.4, 8.1) | 8.13 | 0.06 | 8.38 | 8.40 |
| (b) Một mô hình phát hiện lạ, một mô hình bị lừa | (8.2, 14.5, 12.9) | 11.87 | **10.72** | 15.14 | 14.50 |
| (c) Cả 3 cùng thấy lạ (đồng thuận) | (14.0, 14.6, 13.8) | 14.13 | **0.17** | 14.55 | 14.60 |
| (d) Chỉ một mô hình hơi lạ | (8.0, 8.2, 11.0) | 9.07 | 2.81 | 10.74 | 11.00 |
| (e) Cả 3 cùng bị lừa | (8.0, 8.1, 7.9) | 8.00 | 0.01 | 8.10 | 8.10 |

Phân vị 95% của mẫu bình thường (mô phỏng ở D5.4): $\bar D$: 8.64; $s^2$: 1.49; $\bar D + s$: 9.47; $\max$: 9.47.

**Đọc bảng:**
* **Phương sai thuần ($s^2$, định nghĩa FCD hiện tại) bỏ lọt tình huống (c).** $s^2 = 0.17 < 1.49$, nên mẫu được chấp nhận làm nhãn giả. Đây là mẫu lạ rõ ràng nhất: cả ba mô hình đều thấy nó ở rất xa tâm. Đây là **điểm mù nghiêm trọng** (P4).
* Trung bình bắt được (c) nhưng yếu với (d): $9.07 > 8.64$, bắt được nhưng sát.
* $\bar D + s$ và $\max$ bắt được cả (b), (c), (d).
* **Không** cách nào bắt được (e). Nếu cả ba mô hình đều bị lừa **trong cả latent**, mọi tín hiệu dựa trên các mô hình này đều mù. Đây là giới hạn cơ bản, phải viết rõ trong phần Limitations.

### D5.4. Mô phỏng công suất (power) đơn giản
**Mô hình đồ chơi:** mẫu bình thường có $D_m \sim \mathcal N(7.97, 0.71^2)$ độc lập giữa 3 mô hình. Ngưỡng của mỗi cách gộp đặt tại phân vị 95% của mẫu bình thường, nên mọi cách có cùng 5% báo động nhầm. Mẫu lạ có hai kiểu:
* **Lệch một phần:** một mô hình ngẫu nhiên có $D$ tăng thêm $\Delta$.
* **Lệch đồng thuận:** cả ba cùng tăng $\Delta$.

Tỉ lệ bắt được (200.000 lần lặp, kết quả làm tròn; tái tạo bằng `fcd_v3_research/scripts/02_statistical_validation.py`, mục `fcd_aggregators`, và có thể lệch ±0.01 do seed):

| Kiểu | $\Delta$ | Anchor ($D_1$) | $\bar D$ | $s^2$ | $\max_m D_m$ | $\bar D + s$ |
|---|---|---|---|---|---|---|
| Một phần | 1.5 | 0.26 | 0.34 | 0.32 | **0.52** | 0.51 |
| Một phần | 2.0 | 0.33 | 0.50 | 0.53 | **0.77** | 0.76 |
| Một phần | 3.0 | 0.37 | 0.79 | 0.89 | **0.98** | **0.98** |
| Một phần | 5.0 | 0.37 | 0.99 | 1.00 | 1.00 | 1.00 |
| Đồng thuận | 1.5 | 0.68 | **0.98** | **0.05** | 0.88 | 0.89 |
| Đồng thuận | 2.0 | 0.88 | **1.00** | **0.05** | 0.99 | 0.99 |
| Đồng thuận | 5.0 | 1.00 | 1.00 | **0.05** | 1.00 | 1.00 |

**Kết luận từ mô phỏng:**
1. **Phương sai thuần hoàn toàn mù với lệch đồng thuận:** tỉ lệ bắt được đúng bằng 5%, tức bằng tỉ lệ báo động nhầm, dù $\Delta$ lớn bao nhiêu.
2. **Mahalanobis một mô hình (anchor)** chỉ bắt tối đa khoảng 37% lệch một phần, vì 2/3 trường hợp mô hình nhận ra điều lạ không phải anchor. Đây là lợi ích thật của ensemble.
3. $\max_m D_m$ và $\bar D + s$ mạnh ở **cả hai** kiểu. Nên chọn một trong hai làm tín hiệu chính.

**Giới hạn của mô phỏng:** giả định các $D_m$ độc lập và có phân phối chuẩn. Thực tế các mô hình tương quan với nhau, nên lợi ích của ensemble sẽ nhỏ hơn. Vì vậy H2b là **giả thuyết thực nghiệm**, không phải định lý.

### D5.5. Phiên bản đúng của phần lý thuyết (thay cho "Theorem 2")
**Kỳ vọng của phương sai mẫu (công thức đúng).** Nếu $D_1, \dots, D_M$ độc lập với $\mathbb E[D_m] = \mu_m$ và $\operatorname{Var}(D_m) = \sigma^2_m$ thì
$$\mathbb E[s^2_D] = \frac1M\sum_m \sigma^2_m + \frac{1}{M-1}\sum_m (\mu_m - \bar\mu)^2$$
Tài liệu cũ chỉ có số hạng đầu. Số hạng thứ hai, **độ phân tán giữa các kỳ vọng**, mới là phần đo "các mô hình bất đồng". Nó bằng 0 khi mọi mô hình cùng lệch như nhau. Đó là cách viết bằng công thức của điểm mù ở D5.3(c).

> **Proposition (hai kiểu lệch, mô hình tất định).** Cho $M = 3$, mẫu bình thường có $D_m = \rho$ với mọi $m$.
> 1. **Lệch một phần** $(\rho, \rho, \rho + \Delta)$: $\bar D = \rho + \Delta/3$, $s^2_D = \Delta^2/3$.
> 2. **Lệch đồng thuận** $(\rho + \Delta, \rho + \Delta, \rho + \Delta)$: $\bar D = \rho + \Delta$, $s^2_D = 0$.
> 3. $\max_m D_m = \rho + \Delta$ trong cả hai trường hợp.
>
> *Chứng minh (trường hợp 1):* trung bình là $\rho + \Delta/3$. Độ lệch so với trung bình là $(-\Delta/3, -\Delta/3, 2\Delta/3)$. Tổng bình phương: $\Delta^2(1 + 1 + 4)/9 = 2\Delta^2/3$. Chia cho $M - 1 = 2$ được $\Delta^2/3$. Trường hợp 2 và 3 hiển nhiên. $\blacksquare$

**Ý nghĩa:** phương sai tăng theo **bình phương** độ lệch của một mô hình. Đây là phiên bản chính xác của câu "phương sai bùng nổ $O(\Delta^2)$" trong tài liệu cũ, nhưng chỉ đúng cho lệch một phần. Trung bình chỉ tăng $\Delta/3$ trong trường hợp này. Max luôn tăng đủ $\Delta$. Không có cách gộp nào mạnh nhất trong mọi trường hợp. Đó là lý do đề xuất $\bar D + \lambda s_D$ hoặc $\max$, và để dữ liệu (H2b, ablation) trả lời.

**Phân rã đẹp, dùng cho phần Discussion:**
$$\frac1M\sum_m D_m^2 = \underbrace{\bar D^2}_{\text{đồng thuận}} + \underbrace{\frac{M-1}{M}s_D^2}_{\text{bất đồng}}$$
Cấu trúc này giống hệt phân rã "bất định tổng = bất định trung bình + thông tin tương hỗ" ở D2.2. FCD đang làm cho latent đúng điều mà BALD làm cho đầu ra.

### D5.6. Đề xuất cuối cùng (bạn cần quyết định)
* **Tín hiệu chính:** $S_{\text{fcd}}(j) = \bar D(j) + \lambda\, s_D(j)$, với $\lambda \ge 0$ chọn trên Meta-Train (tối đa Coverage tại $\alpha$). Chọn trên Meta-Train là hợp lệ vì Meta-Train tách biệt khỏi tập hiệu chuẩn, nên không ảnh hưởng Theorem 1.
* **Phương án thay thế đơn giản hơn:** $S = \max_m D_m$, không có siêu tham số.
* **Ablation đăng ký trước:** $\bar D$ ($\lambda = 0$, "ensemble Mahalanobis"), $s^2_D$ (FCD cũ), $\max_m D_m$. So sánh này trả lời câu hỏi khoa học thật: **bất đồng giữa các mô hình trong latent có thêm thông tin ngoài khoảng cách trung bình không?**
* **Về tên gọi:** spec M29 dùng FCD = *Feature–Confidence Dissonance*, tức "confidence nói an toàn nhưng đặc trưng nói lạ". Đây là một khái niệm hay, mô tả đúng **cổng** của M29: ứng viên Green có confidence cao, rồi cổng latent loại những box có đặc trưng lạ. Gợi ý: dùng FCD cho **khái niệm cổng**, và đặt tên riêng cho **tín hiệu**, ví dụ *Ensemble Latent Discrepancy (ELD)*.

---

## D6. Từ box lên ảnh: vì sao dùng max (và cạm bẫy đa kiểm định)
Spec M29 định tuyến **cả ảnh**: $S_{\text{img}} = \max_j S(j)$ (box đáng nghi nhất) và $\text{conf}_{\text{img}} = \min_j \hat p_j$ (box kém tự tin nhất).
* **Lý do:** nguyên tắc mắt xích yếu nhất. Một ảnh chỉ an toàn để tự gán nhãn khi **mọi** box trong đó an toàn. Một box sai đủ để đưa một nhãn sai vào tập huấn luyện.
* **Cạm bẫy (P12):** nếu ngưỡng của mỗi box là phân vị 95% của box bình thường, ảnh có $n$ box bình thường độc lập bị gắn cờ với xác suất $1 - 0.95^n$. Với $n = 10$ là **40%**, với $n = 20$ là **64%**. Ảnh đông xe sẽ luôn bị đẩy sang Red dù hoàn toàn bình thường.
* **Sửa:** khi quyết định theo ảnh, hiệu chuẩn trực tiếp ngưỡng cho $\max_j S(j)$, nghĩa là đưa đại lượng cấp ảnh vào LTT. Hoặc giữ quyết định theo box (paper H2 làm theo box) và chỉ dùng cấp ảnh cho định tuyến của H3.

---

# PHẦN E. SO SÁNH CÁC TÍN HIỆU MỘT CÁCH THỐNG KÊ

## E1. Vì sao chỉ số chính là "Coverage tại rủi ro được chứng nhận", không phải AUROC
* **AUROC** đo khả năng xếp hạng TP trên FP **trên toàn bộ** các ngưỡng. Nhưng ta chỉ dùng **một** ngưỡng, ở vùng rủi ro rất thấp (2%). Hai tín hiệu có cùng AUROC vẫn có thể khác nhau rất xa ở vùng đó.
* **Coverage tại $\hat\lambda$ từ LTT** đo đúng thứ người dùng nhận được: "dưới bảo chứng 2%, tôi thu được bao nhiêu nhãn giả đúng?". Nó cũng gộp cả "phí bảo hiểm" của LTT (Phần B3.5), một tín hiệu có FDP dao động nhiều giữa các đơn vị sẽ bị phạt.
* AUROC và AURC (diện tích dưới đường risk–coverage) vẫn được báo cáo làm chỉ số phụ.

## E2. Wilcoxon signed-rank test (kiểm định dấu-hạng có cặp)
**Vấn đề:** so FCD với baseline trên cùng các đơn vị test. Hai số đo trên cùng đơn vị là **có cặp**, nên ta xét hiệu $d_i = \text{TP}_i(\text{FCD}) - \text{TP}_i(\text{baseline})$. Phân phối của $d_i$ thường lệch và có nhiều số 0, nên kiểm định t (cần gần chuẩn) không phù hợp. Wilcoxon (1945) chỉ dùng **thứ hạng**.

**Cách làm:**
1. Bỏ các $d_i = 0$.
2. Xếp hạng $|d_i|$ từ nhỏ đến lớn.
3. $W^+$ = tổng hạng của các $d_i > 0$.
4. So $W^+$ với phân phối của nó khi $H_0$ đúng (mỗi dấu $\pm$ có xác suất bằng nhau).

**Ví dụ tính tay:** 6 đơn vị, $d = (+5, +1, +3, -2, +4, +6)$. Hạng của $|d|$: 5→5, 1→1, 3→3, 2→2, 4→4, 6→6. $W^+ = 5 + 1 + 3 + 4 + 6 = 19$, $W^- = 2$. Dưới $H_0$, mỗi trong $2^6 = 64$ cách gán dấu có xác suất bằng nhau. $W^+ \ge 19$ tương đương $W^- \le 2$, xảy ra khi các hạng mang dấu âm thuộc $\{\}$, $\{1\}$ hoặc $\{2\}$: 3 cách. p-value một phía $= 3/64 \approx 0.047$.

Demšar (2006) khuyến nghị Wilcoxon làm phép so sánh mặc định giữa hai phương pháp học máy trên nhiều tập dữ liệu, vì nó không giả định phân phối chuẩn và ít bị ảnh hưởng bởi giá trị ngoại lai.

## E3. Hiệu chỉnh đa so sánh: Holm thay cho Bonferroni
FCD được so với 4 baseline, tức 4 phép kiểm. Nếu mỗi phép ở mức 0.01, xác suất có ít nhất một kết luận sai có thể lên tới 0.04.
* **Bonferroni:** mỗi phép ở mức $0.01/4 = 0.0025$.
* **Holm (1979):** xếp các p-value tăng dần $p_{(1)} \le \dots \le p_{(4)}$, so $p_{(k)}$ với $0.01/(4 - k + 1)$, dừng ở lần thất bại đầu tiên. Holm kiểm soát cùng mức lỗi như Bonferroni nhưng **luôn** bác bỏ ít nhất bằng, nên không có lý do gì để chọn Bonferroni. Holm cũng chính là một quy trình closed testing, cùng họ với fixed-sequence ở B4.

## E4. Ngưỡng là ngẫu nhiên: cần lặp lại phép chia
Ngưỡng $\hat\lambda$ của mỗi tín hiệu phụ thuộc vào tập hiệu chuẩn cụ thể. Wilcoxon trên một lần chia chỉ tính đến nhiễu của tập test. **Phân tích phụ:** lặp 100 lần chia cal/test theo super-block (giống Exp B), mỗi lần tính $\Delta\operatorname{Cov}$, rồi báo cáo phân phối và tỉ lệ số lần FCD thắng.

---

# PHẦN F. TOÁN CỦA VÒNG ACTIVE LEARNING (H3)

## F1. Vòng lặp
Ở mỗi vòng $t = 1, \dots, T$:
1. Mô hình $\theta_{t-1}$ dự đoán trên pool chưa có nhãn $\mathcal U_t$.
2. **Hiệu chuẩn trên tập mới** $\mathcal D^{(t)}_{\text{cal}}$ (P11) để có $\hat\lambda_t$.
3. Phân vùng: **Green** (mọi box qua LTT và cổng FCD) thành nhãn giả; **Red** (bị từ chối, hoặc ensemble bất đồng mạnh); **Grey** (còn lại, cất kho).
4. Chọn $B$ ảnh từ Red cho người gán nhãn, có lọc đa dạng (F3). Người sửa box mồi WBF (F4).
5. Huấn luyện $\theta_t$ trên $\mathcal L_{t-1} \cup \{\text{nhãn người mới}\} \cup \{\text{nhãn giả Green}\}$.
6. Đánh giá $\text{mAP}_{50:95}(\theta_t)$ trên tập test cố định.

Kết hợp "người gán phần khó, máy tự gán phần dễ" là ý tưởng của CEAL (Wang et al., 2017) cho phân loại. Elezi et al. (2022) làm tương tự cho object detection. Điểm mới của H3 là phần máy tự gán **có bảo chứng FDP** và phần chọn cho người dựa trên tín hiệu latent.

## F2. Vì sao mẫu bị từ chối lại có giá trị cho người gán: "hard negative"
Một box tự tin nhưng bị cổng latent từ chối thường là một **ảo giác tự tin**: mô hình chắc chắn có xe ở chỗ không có xe. Khi người gán nhãn xác nhận "không có xe ở đây", ảnh đó dạy mô hình đúng điều nó đang sai. Trong loss của detector, vùng đó trở thành **mẫu âm khó**. Các phương pháp chọn mẫu theo độ bất định trong AL cho detection (Haussmann et al., 2020, với dữ liệu lái xe; Choi et al., 2021; Yoo & Kweon, 2019) đều dựa trên nguyên tắc: nhãn có giá trị nhất là nhãn ở chỗ mô hình sai hoặc không chắc. Tổng quan cổ điển về AL: Settles (2009).

## F3. Lọc đa dạng bằng K-means
**Vấn đề:** 200 ảnh trong cùng một đường hầm tối đều bị gắn Red. Gán nhãn cả 200 lãng phí ngân sách, vì chúng dạy mô hình cùng một điều.

**Ý tưởng (Sener & Savarese, 2018, Core-set):** chọn tập con sao cho mọi ảnh trong pool đều **gần** một ảnh được chọn trong không gian đặc trưng. Bài toán gốc là k-center, khó giải chính xác. K-means là xấp xỉ thực tế:
$$\min_{\mu_1..\mu_B} \sum_{x \in \text{Red}} \min_b \|\phi(x) - \mu_b\|^2$$
với $\phi(x)$ là vector đặc trưng của cả ảnh. Từ mỗi cụm, chọn ảnh gần tâm cụm nhất (hoặc ảnh có $S_{\text{img}}$ cao nhất trong cụm).

## F4. Weighted Boxes Fusion (WBF) cho box mồi
Solovyev, Wang & Gabruseva (2021): gộp các hộp đã gióng hàng của $M$ mô hình bằng trung bình có trọng số theo độ tự tin:
$$\mathbf b_{\text{fused}} = \frac{\sum_m c_m\, \mathbf b_m}{\sum_m c_m}, \qquad c_{\text{fused}} = \frac{1}{M}\sum_m c_m$$
Khác với NMS (giữ một hộp, bỏ các hộp còn lại), WBF dùng thông tin của **mọi** mô hình, nên hộp gộp thường chính xác hơn từng hộp riêng. Mục đích ở đây là giảm thời gian người gán: sửa một box gần đúng nhanh hơn vẽ từ đầu.

## F5. mAP$_{50:95}$
* **Precision** = TP / (TP + FP), **Recall** = TP / số GT.
* **AP của một lớp** = diện tích dưới đường precision–recall khi quét ngưỡng confidence (COCO nội suy 101 điểm).
* **mAP$_{50:95}$** = trung bình AP qua các lớp và qua 10 ngưỡng IoU $0.50, 0.55, \dots, 0.95$. Chỉ số này thưởng cả phân loại đúng lẫn định vị chính xác.

## F6. Thiết kế thực nghiệm và kiểm định cho H3
**Các arm** (mọi arm dùng cùng ngân sách $B$ ảnh người mỗi vòng):
| Arm | Chọn cho người | Nhãn giả |
|---|---|---|
| Random | ngẫu nhiên | cùng chính sách với Proposed |
| Least-Conf | $\min_j \hat p_j$ thấp nhất | như trên |
| Ensemble-entropy | $S_{\text{out}}$ cao nhất | như trên |
| **Proposed** | Red (LTT + FCD) + K-means | như trên |
| *H3b:* Proposed-heuristic | như Proposed | ngưỡng cố định $\hat p \ge 0.9$ |

**Kiểm định:**
* Với mỗi seed $s$: $\Delta_s = \text{mAP}_T(\text{Proposed}) - \text{mAP}_T(\text{Random})$. Kiểm định t có cặp trên $\{\Delta_s\}$ với **ít nhất 5 seed**. Bouthillier et al. (2021) cho thấy nhiễu do seed, thứ tự dữ liệu và augmentation có thể lớn ngang chênh lệch giữa các phương pháp, nên 3 seed không đủ (P10).
* **"Vượt ở mọi vòng":** phép kiểm giao-hợp (Berger, 1982). Bác bỏ $H_0$ khi phép kiểm ở **từng** vòng đều có ý nghĩa ở mức 0.05. Không cần chia mức, vì kết luận chỉ đúng khi tất cả cùng đúng.
* **Chi phí công bằng:** nhãn của các tập hiệu chuẩn $\mathcal D^{(t)}_{\text{cal}}$ cũng tốn công người. Báo cáo kết quả theo **tổng** số nhãn người (gồm cả nhãn hiệu chuẩn), hoặc cho Random dùng thêm đúng số nhãn đó.

---

# PHẦN G. BẢNG TRA

| Thành phần | Công thức | Vì sao thiết kế như vậy | Paper gốc |
|---|---|---|---|
| Đơn vị thống kê = chuyến đi | $U_i$ i.i.d. | Frame tương quan cao, $D_{\text{eff}}$ lớn | Kish 1965; Hurlbert 1984; Barber et al. 2023 |
| Rủi ro = Macro-FDP | $\mathbb E[V_i / \max(N^{\text{acc}}_i, 1)]$ | Bị chặn trong $[0,1]$ nên dùng được các bất đẳng thức tập trung | Angelopoulos et al. 2021 |
| p-value HB | $\min(p_{\text{Hoeff}}, p_{\text{Bentkus}})$ | Chặt nhất trong các cận không cần biết phương sai, khi rủi ro gần 0 | Hoeffding 1963; Bentkus 2004; Bates et al. 2021 |
| Fixed-sequence | dừng ở lần thất bại đầu | Dùng trọn $\delta$ cho mỗi phép kiểm, hợp lệ khi thứ tự cố định | Maurer et al. 1995; Westfall & Krishen 2001 |
| Đường đi Pareto trên Meta-Train | $\arg\max \operatorname{Cov}$ s.t. FDP $\le \alpha_k$ | Thứ tự phải độc lập với tập hiệu chuẩn | Angelopoulos et al. 2021 |
| Hungarian | $\min_\pi \sum_u C_{u,\pi(u)}$ | Query không có thứ tự, cần ghép một-một tối ưu | Kuhn 1955; Carion et al. 2020 |
| Min-confidence | $1 - \hat p$ | Baseline bắt buộc | Hendrycks & Gimpel 2017 |
| Output disagreement | $\sigma^2_{\text{loc}}$, JS | Bất định nhận thức ở đầu ra | Lakshminarayanan et al. 2017; Lin 1991; Houlsby et al. 2011 |
| EDL vacuity | $C/S$ | Bất định nhận thức một mô hình | Sensoy et al. 2018 |
| Mahalanobis | $\sqrt{(\mathbf z - \boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf z - \boldsymbol\mu)}$ | Log-mật độ Gauss, thấy null space | Mahalanobis 1936; Lee et al. 2018 |
| PCA + shrinkage | $\mathcal P: \mathbb R^{256} \to \mathbb R^{64}$, Ledoit–Wolf | Ước lượng được hiệp phương sai với ít mẫu | Ledoit & Wolf 2004; Kamoi & Kobayashi 2020 |
| FCD | $\bar D + \lambda s_D$ (đề xuất) | Bắt cả lệch một phần lẫn lệch đồng thuận | Fort et al. 2019; Lee et al. 2018 |
| Max-pooling theo ảnh | $\max_j S(j)$ | Mắt xích yếu nhất | |
| K-means đa dạng | k-means trên $\phi(x)$ | Không lãng phí ngân sách cho ảnh trùng nhau | Sener & Savarese 2018 |
| WBF | trung bình có trọng số | Box mồi chính xác hơn NMS | Solovyev et al. 2021 |
| Wilcoxon + Holm | hạng của $\lvert d_i\rvert$ | Có cặp, không cần phân phối chuẩn | Wilcoxon 1945; Holm 1979; Demšar 2006 |
| ≥5 seed, IUT | t có cặp theo seed | Nhiễu seed lớn | Bouthillier et al. 2021; Berger 1982 |

---

# TÀI LIỆU THAM KHẢO

*Đã kiểm tra tên, tác giả, nơi đăng theo hiểu biết đến thời điểm viết. Các mục đánh dấu (†) cần kiểm lại số tập/trang hoặc phiên bản tạp chí trước khi nộp.*

**Kiểm soát rủi ro và suy diễn thống kê**
1. Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052. (†) Phiên bản tạp chí: *Annals of Applied Statistics*. Kiểm lại số tập.
2. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4). (arXiv:2107.07511)
3. Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal risk control. *ICLR 2024*.
4. Bates, S., Angelopoulos, A. N., Lei, L., Malik, J., & Jordan, M. I. (2021). Distribution-free, risk-controlling prediction sets. *Journal of the ACM*, 68(6).
5. Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *JASA*, 58(301), 13–30.
6. Bentkus, V. (2004). On Hoeffding's inequalities. *Annals of Probability*, 32(2), 1650–1673.
7. Maurer, W., Hothorn, L. A., & Lehmacher, W. (1995). Multiple comparisons in drug clinical trials and preclinical assays: a-priori ordered hypotheses. *Biometrie in der chemisch-pharmazeutischen Industrie*, 6, 3–18.
8. Westfall, P. H., & Krishen, A. (2001). Optimally weighted, fixed sequence and gatekeeper multiple testing procedures. *Journal of Statistical Planning and Inference*, 99(1), 25–40.
9. Marcus, R., Peritz, E., & Gabriel, K. R. (1976). On closed testing procedures with special reference to ordered analysis of variance. *Biometrika*, 63(3), 655–660.
10. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
11. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 2017*.
12. Jin, Y., & Candès, E. J. (2023). Selection by prediction with conformal p-values. *JMLR*, 24.
13. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289–300.
14. Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML 2012*, PMLR 25.
15. Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2).
16. Oliveira, R. I., Orenstein, P., Ramos, T., & Romano, J. V. (2024). Split conformal prediction and non-exchangeable data. *JMLR*, 25. (†)
17. Waudby-Smith, I., & Ramdas, A. (2024). Estimating means of bounded random variables by betting. *JRSS-B*, 86(1).
18. Kish, L. (1965). *Survey Sampling*. Wiley.
19. Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187–211.
20. Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics*, 24(4), 295–300.
21. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.
22. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30.
23. Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys 2021*.

**Conformal / kiểm soát rủi ro cho object detection**
24. Andéol, L., Fel, T., de Grancey, F., & Mossina, L. (2023). Confident object detection via conformal prediction and conformal risk control: an application to railway signaling. *COPA 2023*, PMLR 204. (†)
25. Timans, A., Straehle, C.-N., Sakmann, K., & Nalisnick, E. (2024). Adaptive bounding box uncertainties via two-step conformal prediction. *ECCV 2024*. (†)

**Detector và object detection**
26. Carion, N., et al. (2020). End-to-end object detection with transformers. *ECCV 2020*.
27. Zhao, Y., et al. (2024). DETRs beat YOLOs on real-time object detection. *CVPR 2024*.
28. Rezatofighi, H., et al. (2019). Generalized intersection over union. *CVPR 2019*.
29. Girshick, R. (2015). Fast R-CNN. *ICCV 2015*.
30. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2, 83–97.
31. Munkres, J. (1957). Algorithms for the assignment and transportation problems. *Journal of SIAM*, 5(1), 32–38.
32. Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion. *Image and Vision Computing*, 107.
33. Miller, D., Dayoub, F., Milford, M., & Sünderhauf, N. (2019). Evaluating merging strategies for sampling-based uncertainty techniques in object detection. *ICRA 2019*.
34. Feng, D., et al. (2022). A review and comparative study on probabilistic object detection in autonomous driving. *IEEE T-ITS*, 23(8).
35. Du, X., Wang, Z., Cai, M., & Li, Y. (2022). VOS: Learning what you don't know by virtual outlier synthesis. *ICLR 2022*.
36. Yu, F., et al. (2020). BDD100K: A diverse driving dataset for heterogeneous multitask learning. *CVPR 2020*.

**Tín hiệu bất định**
37. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR 2017*.
38. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
39. Hein, M., Andriushchenko, M., & Bitterwolf, J. (2019). Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. *CVPR 2019*.
40. Nguyen, A., Yosinski, J., & Clune, J. (2015). Deep neural networks are easily fooled. *CVPR 2015*.
41. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
42. Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS 2019*.
43. Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep ensembles: A loss landscape perspective. arXiv:1912.02757.
44. Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Trans. Information Theory*, 37(1), 145–151.
45. Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745.
46. Depeweg, S., et al. (2018). Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. *ICML 2018*.
47. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS 2017*.
48. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS 2018*.
49. Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer.
50. Bengs, V., Hüllermeier, E., & Waegeman, W. (2022). Pitfalls of epistemic uncertainty quantification through loss minimisation. *NeurIPS 2022*.
51. Ulmer, D., Hardmeier, C., & Frellsen, J. (2023). Prior and posterior networks: A survey on evidential deep learning methods for uncertainty estimation. *TMLR*. (†)
52. Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55.
53. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS 2018*.
54. Ren, J., et al. (2021). A simple fix to Mahalanobis distance for improving near-OOD detection. arXiv:2106.09022.
55. Kamoi, R., & Kobayashi, K. (2020). Why is the Mahalanobis distance effective for anomaly detection? arXiv:2003.00402.
56. Mukhoti, J., Kirsch, A., van Amersfoort, J., Torr, P. H. S., & Gal, Y. (2023). Deep deterministic uncertainty: A new simple baseline. *CVPR 2023*.
57. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
58. Li, Y., Yosinski, J., Clune, J., Lipson, H., & Hopcroft, J. (2016). Convergent learning: Do different neural networks learn the same representations? *ICLR 2016*.
59. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML 2019*.
60. Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR 2019*.

**Nhãn giả và Active Learning**
61. Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop on Challenges in Representation Learning*.
62. Arazo, E., Ortego, D., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. *IJCNN 2020*.
63. Rizve, M. N., Duarte, K., Rawat, Y. S., & Shah, M. (2021). In defense of pseudo-labeling: An uncertainty-aware pseudo-label selection framework for semi-supervised learning. *ICLR 2021*.
64. Sohn, K., et al. (2020). A simple semi-supervised learning framework for object detection (STAC). arXiv:2005.04757.
65. Liu, Y.-C., et al. (2021). Unbiased teacher for semi-supervised object detection. *ICLR 2021*.
66. Wang, K., Zhang, D., Li, Y., Zhang, R., & Lin, L. (2017). Cost-effective active learning for deep image classification. *IEEE TCSVT*, 27(12).
67. Settles, B. (2009). Active learning literature survey. University of Wisconsin–Madison, Computer Sciences Technical Report 1648.
68. Gal, Y., Islam, R., & Ghahramani, Z. (2017). Deep Bayesian active learning with image data. *ICML 2017*.
69. Sener, O., & Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. *ICLR 2018*.
70. Yoo, D., & Kweon, I. S. (2019). Learning loss for active learning. *CVPR 2019*.
71. Haussmann, E., et al. (2020). Scalable active learning for object detection. *IEEE Intelligent Vehicles Symposium (IV) 2020*.
72. Choi, J., Elezi, I., Lee, H.-J., Farabet, C., & Alvarez, J. M. (2021). Active learning for deep object detection via probabilistic modeling. *ICCV 2021*.
73. Elezi, I., Yu, Z., Anandkumar, A., Leal-Taixé, L., & Alvarez, J. M. (2022). Not all labels are equal: Rationalizing the labeling costs for training object detection. *CVPR 2022*.
