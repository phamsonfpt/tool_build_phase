# ĐỀ CƯƠNG NGHIÊN CỨU CHI TIẾT (FULL ACADEMIC RESEARCH PROPOSAL) - REVISED

**Tên đề tài:** Evaluating Epistemic Uncertainty Signals for Risk-Controlled Selective Pseudo-Labeling in Autonomous Driving
*(Đánh giá các Tín hiệu Bất định Nhận thức cho quy trình Gán nhãn giả có Kiểm soát Rủi ro trong Xe tự lái)*

**Mục tiêu xuất bản:** IEEE Transactions on Intelligent Vehicles (T-IV) / NeurIPS (Datasets & Benchmarks Track)

---

## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1. Bối cảnh và Nêu vấn đề (Context & Motivation)
Trong bối cảnh Xe tự hành (Autonomous Driving), việc gán nhãn dữ liệu thủ công là nút thắt cổ chai lớn nhất. Học tích cực (Active Learning - AL) thông qua nhãn giả (pseudo-labeling) là giải pháp hàng đầu. Tuy nhiên, các kiến trúc Deep Learning hiện đại thường mắc phải hiệu ứng **"Tự tin nhưng Sai" (Confidently Wrong / Hallucinations)** đối với dữ liệu ngoài phân phối (OOD) hoặc các mẫu khó trong phân phối (Hard-HID).
Khi các mẫu ảo giác này vượt qua ngưỡng tự tin truyền thống, chúng bị gán nhãn giả sai lệch và hòa trộn ngược vào tập huấn luyện, sinh ra **Thiên kiến Xác nhận (Confirmation Bias)**, khiến hệ thống tự củng cố những ảo giác của chính nó.

### 1.2. Đóng góp cốt lõi (Core Contributions)
Nghiên cứu đề xuất một Khung đánh giá Thực chứng (Empirical Evaluation Framework) có kiểm soát rủi ro thống kê nghiêm ngặt:
1.  **Ứng dụng Khung Learn-then-Test (LTT) vào Object Detection:** Chuyển đổi bài toán lọc nhãn giả thành bài toán kiểm định đa giả thuyết có giới hạn mẫu hữu hạn trên các đơn vị dữ liệu độc lập (Video Sequences), đảm bảo Kỳ vọng Tỷ lệ Khám phá Sai $\mathbb{E}[\text{FDP}] \le \alpha$.
2.  **Benchmark Định lượng Tín hiệu Bất định:** Đánh giá công bằng sức mạnh của 4 luồng tín hiệu bất định nhận thức (Confidence, Output Disagreement, EDL, và Latent Variance) dựa trên Năng suất Dữ liệu (Coverage) tại cùng một mức rủi ro cố định.
3.  **Chiến lược Khai thác Hard Negatives:** Thiết lập cơ chế định tuyến để chuyển các mẫu bị LTT từ chối sang hàng đợi đánh giá của con người, biến "ảo giác" thành các ràng buộc phủ định (negative constraints).

---

## 2. GIẢ THUYẾT NGHIÊN CỨU (HYPOTHESES)
Để tránh bẫy "overclaiming", nghiên cứu đăng ký trước (pre-register) 3 giả thuyết có thể bị bác bỏ:
*   **H1 (The Signal Hypothesis):** Tín hiệu phương sai không gian ẩn (Relative Latent Variance) phản ánh mức độ lệch so với phân phối tham chiếu tốt hơn, qua đó cải thiện Năng suất dữ liệu (Coverage) tại cùng một mức rủi ro $\alpha$ so với các tín hiệu không gian đầu ra (Output Disagreement / Min-Confidence).
*   **H2 (The Statistical Guarantee Hypothesis):** Khung kiểm soát rủi ro tuần tự (Sequential LTT) cung cấp các ngưỡng an toàn đạt đúng mục tiêu rủi ro (FDP $\le 2\%$) trên dữ liệu kiểm thử độc lập mà không vi phạm giả định I.I.D, thông qua thiết kế chia dữ liệu theo Video Sequence.
*   **H3 (The AL Utility Hypothesis):** Việc định tuyến (routing) các mẫu "Ảo giác tự tin" bị từ chối sang Human Review mang lại sự gia tăng hiệu suất (Downstream mAP Gain) cao hơn so với chiến lược Random Sampling trong AL.

---

## 3. PHƯƠNG PHÁP LUẬN (METHODOLOGY)

### 3.1. Căn chỉnh Truy vấn Bipartite (Bipartite Query Alignment)
Sử dụng **RT-DETR** làm kiến trúc lõi. Do các object queries mang tính bất biến hoán vị, việc so sánh trực tiếp index giữa các mô hình Ensemble là không hợp lệ. Khắc phục bằng:
1.  Chọn 1 mô hình có độ tự tin cao nhất làm **Mô hình Mỏ neo (Anchor Model)** $M_{anchor}$.
2.  Sử dụng tập dự đoán của $M_{anchor}$ làm **Pseudo-Ground Truth**. 
3.  Sử dụng Thuật toán Hungarian với chi phí (Focal Loss + $L_1$ + GIoU) để ghép cặp query từ các mô hình thành viên vào Pseudo-Ground Truth này. Các query không tìm thấy cặp được gán giá trị bất định cực đại.

### 3.2. Đánh giá Tín hiệu Bất định (Evaluated Uncertainty Signals)
*   **Baseline 1 (Min-Confidence):** Xác suất cực đại từ hàm Softmax.
*   **Baseline 2 (Output Disagreement):** Sự bất đồng của 5x Ensembles đo bằng Phương sai Hộp (Box Variance) và JS Divergence.
*   **Baseline 3 (Evidential Deep Learning - EDL):** Tham số Vacuity / Dissonance từ 1 mô hình huấn luyện với phân phối Dirichlet.
*   **Proposed (FCD Latent Variance):** Đo lường sự bất định qua không gian tiềm ẩn. 
    *   *Bước 1:* Đối với mỗi mô hình $k \in \{1..5\}$, tính khoảng cách Mahalanobis $D_k$ từ query $\mathbf{x}_k$ đến phân phối tham chiếu (Centroid $\boldsymbol{\mu}_k$) của mô hình đó: $D_k = \sqrt{(\mathbf{x}_k - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{x}_k - \boldsymbol{\mu}_k)}$.
    *   *Bước 2:* Tính phương sai của 5 khoảng cách vô hướng $\{D_1, \dots, D_5\}$ để đại diện cho mức độ dao động bất định nhận thức. (Áp dụng PCA nén xuống 64D và Jitter $\epsilon \mathbf{I}$ để chống suy biến ma trận $\Sigma_k$).

### 3.3. Khung LTT với Đơn vị Thống kê Độc lập (Statistical Unit & LTT)
Để tuân thủ tuyệt đối định lý tập trung (Concentration Inequalities) của LTT, rủi ro không được tính gộp trên toàn bộ boxes, mà phải tính trên **đơn vị độc lập là Video Sequence**.
*   Cho $N$ chuỗi video độc lập. Gọi $V_i$ là số False Positives, $R_i$ là tổng số boxes được chấp nhận trong chuỗi $i$.
*   Giá trị FDP của chuỗi $i$ là: $\text{FDP}_i(\tau) = \frac{V_i}{\max(R_i, 1)}$.
*   Rủi ro toàn cục là kỳ vọng: $\mathbb{E}[\text{FDP}(\tau)] = \frac{1}{N} \sum_{i=1}^N \text{FDP}_i(\tau)$.
*   Một Box là False Positive nếu: **Sai Lớp** HOẶC **IoU với Ground Truth $< 0.75$**. (Định nghĩa này được khóa cứng tiên nghiệm để chống P-hacking).

---

## 4. LỘ TRÌNH ĐÁNH GIÁ (EVALUATION PROTOCOL)

### 4.1. Dataset và Nguyên tắc Không Rò rỉ (Non-leakage Protocol)
*   **Dataset:** BDD100K. Chia `Meta-Train`, `Calibration`, và `Test` nghiêm ngặt theo cụm Video Sequence/Thời tiết để bảo vệ tính Exchangeability/I.I.D.
*   Mọi siêu tham số (ngưỡng IoU, PCA dims) được khóa cứng tiên nghiệm (A priori). Tuyệt đối không tinh chỉnh sau khi xem kết quả Calibration.

### 4.2. Study A & Study B
*   **Study A (Static Risk Gating):** Chạy 4 tín hiệu qua LTT. Metric chính: *Coverage @ Expected FDP $\le 2\%$*. Đánh giá thực nghiệm thông qua Bootstrap resampling để kiểm tra tỷ lệ vi phạm $\le \delta$.
*   **Study B (End-to-End AL):** Thực hiện 3 vòng lặp AL. Áp dụng *Adaptive LTT* (rút một tập Calibration hoàn toàn mới, chưa từng chạm tới ở mỗi vòng) để đối phó với hiện tượng trôi dạt phân phối (Distribution Shift) do mô hình được cập nhật trọng số. Metric chính: *Downstream mAP Gain*.
