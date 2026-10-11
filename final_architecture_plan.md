# 🔬 ĐỀ TÀI: M29 - CONFIDENCE-BASED HUMAN REVIEW TRONG ACTIVE LEARNING
**Hệ thống Lọc Điểm Mù (Blind-spot Mining) ứng dụng Deep Ensembles và RT-DETR**

---

## 📌 SƠ ĐỒ PIPELINE HOÀN CHỈNH (END-TO-END)

```mermaid
flowchart TD
    subgraph GĐ0 [Giai đoạn 0: Khởi tạo Baseline]
        A["BDD100K Dataset\n(100.000 ảnh)"] -->|Split 70/10/20| B["Tập Train (70k)\nTập Val (10k)\nTập Test (20k - Giấu nhãn)"]
        B -->|Train Base| C["Mô hình Gốc (Base RT-DETR)"]
    end

    subgraph GĐ1 [Giai đoạn 1: Deep Ensembles]
        B -->|Train 100% Data\nkhác Random Seed| D["5 Chuyên gia RT-DETR\n(Hội đồng giám khảo)"]
    end

    subgraph GĐ2 [Giai đoạn 2: Lõi Lọc Điểm Mù M29]
        E["20.000 ảnh Test (Dữ liệu thô)"] --> D
        D -->|Xuất 5 tập Bbox/ảnh| F["Hungarian Matching\n(Ghép cặp Bbox)"]
        F --> G["Tính Shannon Entropy &\nSpatial Variance"]
        G --> H["Phân luồng 3 Vùng (Elastic Grey Zone)"]
        
        H -->|Vùng Xám Động| I["Bỏ qua (Tránh nhiễu)"]
        H -->|Vùng Xanh (Min_Conf > 0.95)| FCD{"Cổng kiểm duyệt FCD v1.5\n(Đo khoảng cách Embedding)"}
        FCD -->|Bất thường (Hallucination)| K["Vùng Đỏ Thô (Top 8.000)"]
        FCD -->|Bình thường| J["Hard Pseudo-labels\n(Nhãn giả tự động)"]
        H -->|Vùng Đỏ Thô (Top 8.000)| K["Feature Caching & K-Means\n(Lọc Đa dạng - Diversity)"]
    end

    subgraph GĐ3 [Giai đoạn 3: CVAT Automation]
        K -->|Lấy Top 4.000 Đỏ Tinh| L["WBF (Weighted Boxes Fusion)\nTạo Box mồi (Pre-label)"]
        L --> M["Đẩy API lên CVAT Local"]
        M --> N["Human-in-the-Loop\n(Chỉ sửa rìa box bị lệch)"]
    end

    subgraph GĐ4 [Giai đoạn 4: Fine-Tuning & Đánh giá]
        N -->|Export 4.000 ảnh| O["Data Mixer"]
        J -->|N ảnh Pseudo-labels| O
        B -->|Stratified Sampling 12.000 ảnh| O
        O -->|16.000+ ảnh Train mới| P["Fine-tune Mô hình Gốc"]
        C --> P
        P --> Q["Thực nghiệm A/B trên Tập Ẩn\n(Đo lường mAP Gain)"]
    end
```

---

## GIAI ĐOẠN 0: Quản trị Dữ liệu & Khởi tạo (Data Preparation)
*   **Thiết kế:** Chia BDD100K (100.000 ảnh) thành 3 tập: Train (70k) - Val (10k) - Test (20k). Tập 20k Test bị giấu nhãn để giả lập mỏ dữ liệu thô.
> 💡 **Giải thích & Dẫn chứng:** Tại sao chia 70/10/20?
> *   **Nguyên lý Chống rò rỉ (Data Leakage):** Theo Goodfellow et al. (*Deep Learning, MIT Press, 2016*), mô hình không được phép kiểm tra trên dữ liệu đã học. Tập Val (10k) dùng để kiểm soát Overfitting (Early Stopping) trong lúc train. Tập Test (20k) phải bị cách ly hoàn toàn để dùng làm "Mỏ dữ liệu thô" cho M29 khám phá điểm mù. Nếu trộn chung, điểm Uncertainty sẽ bằng 0 (vì model đã thuộc lòng), phá hỏng toàn bộ lý thuyết Active Learning.

---

## GIAI ĐOẠN 1: Khởi tạo Hội đồng Chuyên gia (Deep Ensembles)
*   **Thiết kế:** Dùng kiến trúc RT-DETR. Huấn luyện 5 model RT-DETR trên cùng 70.000 ảnh nhưng khác Random Seed (khởi tạo trọng số ngẫu nhiên ban đầu).

> 💡 **Giải thích & Dẫn chứng:** 
> 1. **Tại sao chọn RT-DETR?** RT-DETR (*Zhao et al., CVPR 2024*) là kiến trúc Transformer loại bỏ hoàn toàn thuật toán NMS (Non-Maximum Suppression). NMS truyền thống (trong YOLO) dùng heuristic cứng nhắc để xóa bớt Bounding Box, làm méo mó phân phối xác suất thực (Raw logits). RT-DETR giữ nguyên phân phối này, giúp M29 tính toán Shannon Entropy chính xác tuyệt đối.
> 2. **Tại sao lại là 5 chuyên gia?** Theo *Lakshminarayanan et al. (NIPS 2017)*, 5 là "điểm ngọt" (sweet spot) của Deep Ensembles. Tăng từ 1 lên 5 giúp biểu diễn tốt sự đa dạng của Loss Landscape, nhưng từ 5 lên 10 không mang lại sự gia tăng mAP hay Calibration đáng kể, trong khi tốn gấp đôi VRAM tính toán.
> 3. **Tại sao Deep Ensembles (Đồng nhất) thay vì Heterogeneous (Nhiều model khác loại)?** Mục đích của M29 là đo **Epistemic Uncertainty** (Độ bất định do thiếu dữ liệu học). Theo *Fort et al. (2019)*, nếu dùng YOLO ghép với Faster-RCNN, sự cãi vã sinh ra là do khác biệt kiến trúc (Model bias/Aleatoric), làm nhiễu kết quả. Dùng 1 lõi RT-DETR duy nhất đảm bảo sự mâu thuẫn thuần túy đến từ việc bức ảnh đó chứa đặc trưng chưa từng xuất hiện.

---

## GIAI ĐOẠN 2: Lõi Lọc Điểm Mù (M29 - Uncertainty Engine)

### Bước 2.1: Căn chỉnh Bounding Box
*   **Thiết kế:** Đưa 1 ảnh qua 5 model. Dùng **Hungarian Algorithm** ghép cặp các hộp box.
> 💡 **Giải thích & Dẫn chứng:** Tại sao dùng Hungarian? (*Carion et al., ECCV 2020*)
> *   *Ví dụ trực quan (với 5 chuyên gia):* Giả sử trên cùng 1 ảnh, Model 1 vẽ 3 box, Model 2 vẽ 4 box, Model 3 vẽ 3 box, Model 4 vẽ 2 box, Model 5 vẽ 3 box. Không thể tính trung bình theo thứ tự (Index) vì số lượng box bị lệch. 
> *   *Cách giải quyết:* Thuật toán Hungarian (Bipartite Matching) sẽ lấy tập Box của 1 Model làm "Mỏ neo" (Anchor). Sau đó, nó tự động tìm phép ghép cặp tọa độ tối ưu nhất để gióng hàng 4 Model còn lại vào Anchor. Nó gom thành công 3 vật thể đồng thuận, và lập tức cô lập box thừa của Model 2 cũng như box bị miss của Model 4. Các box bị mâu thuẫn này sinh ra điểm Uncertainty cực lớn.

### Bước 2.2: Tính Độ Bất Định (Uncertainty)
*   **Thiết kế:** Dùng Shannon Entropy cho Class và Spatial Variance cho Tọa độ.
> 💡 **Giải thích & Dẫn chứng:** Tại sao kết hợp cả 2? (*Settles, Active Learning Literature Survey, 2009*)
> Shannon Entropy ($H = - \sum p \log p$) chỉ đo được mô hình lú lẫn về Môn loại (Classification - Vd: Không phân biệt được Xe tải hay Xe buýt). Spatial Variance đo sự lú lẫn về Tọa độ (Localization - Vd: Đoán đúng là xe tải, nhưng không xác định được đuôi xe ở đâu). Kết hợp cả 2 tạo ra điểm lú lẫn toàn diện cho bài toán Object Detection.

### Bước 2.3: Phân luồng 3 Vùng (Quyết định số phận 20.000 ảnh)
*   **Thiết kế:** Hệ thống phân luồng toàn bộ 20.000 ảnh thành 3 vùng:
    *   **Vùng Đỏ Thô (Top 8.000 ảnh lú lẫn nhất):** Dùng hàm Max-Pooling (`Image_Uncertainty = MAX(Object_Uncertainty)`). Trở thành Tập ứng viên đi CVAT.
    *   **12.000 ảnh dư thừa còn lại** được kiểm duyệt qua Ngưỡng Tuyệt đối với hàm Min-Pooling (`Image_Confidence = MIN(Object_Confidence)`):
        *   **Vùng Xám Động (Image_Confidence < 0.95):** Đóng băng, cất vào kho, không sử dụng ở vòng lặp này.
        *   **Vùng Xanh (Image_Confidence >= 0.95):** Ứng viên cho Hard Pseudo-labels. Bắt buộc đi qua cổng kiểm duyệt FCD ở Bước 2.4.
> 💡 **Giải thích & Dẫn chứng:** Logic hàm Gộp (Pooling) và Sự ra đời của các Vùng
> *   **Giới hạn truy vấn 20% (4.000 ảnh):** Theo *Sener & Savarese (ICLR 2018)*, học 20% dữ liệu khó nhất mang lại hiệu năng tương đương học 100%. Vượt quá 20%, gradient tiến về 0 (OHEM).
> *   **Tại sao cắt 8.000 (Oversampling ratio = 2) thay vì 4.000?** Việc chốt sổ đúng 4.000 ảnh (Ngân sách) chỉ diễn ra SAU KHI đã lọc đa dạng. Ở bước này, cắt 8.000 ảnh nhằm tạo ra một "Tập ứng viên" (Candidate Pool) đủ rộng, bảo đảm toàn bộ ứng viên đều là ảnh khó (Uncertainty cao), đồng thời tạo dư 4.000 slot khoảng trống cho thuật toán K-Means loại bỏ sự trùng lặp (Semantic Redundancy) ở bước tiếp theo.
> *   **Vùng Xanh & Cơ chế Min-Pooling:** Việc chuyển từ điểm Bbox (Object-level) sang điểm cả bức ảnh (Image-level) là bài toán Multiple Instance Learning (*Yuan et al., CVPR 2021*). Để 1 ảnh lọt vào Vùng Xanh, nó phải vượt qua hàm **Min-Pooling**. Nghĩa là: *Ngay cả vật thể mờ nhạt nhất trong ảnh cũng phải tự tin > 0.95*.
> *   **Vùng Xám (Tránh Noisy Labels):** Nếu gán nhãn giả cho 1 ảnh chỉ vì nó có 1 box < 0.95, ta sẽ bơm "Nhãn rác" vào tập train. Mô hình sẽ học cái sai của chính nó (Confirmation Bias - *Arazo et al., ICML 2019*).

### Bước 2.4: Cổng kiểm duyệt FCD v1.5 (Bảo vệ Vùng Xanh)
*   **Thiết kế:** Bất kỳ vật thể nào có Confidence > 0.95 muốn vào Vùng Xanh đều phải qua Cổng FCD (Feature-Confidence Dissonance):
    1. Trích xuất Content Embedding từ Base Model (1 model duy nhất để tránh chênh lệch tọa độ).
    2. Đo khoảng cách (Distance) từ Embedding đó đến Tâm (Centroid) của Class tương ứng.
    3. Nếu Khoảng cách vượt quá Ngưỡng Phân vị 95% (95th Empirical Quantile) đã tính trên tập Validation -> Coi là Ảo giác (Hallucination), tước quyền Vùng Xanh và ném sang Vùng Đỏ.
> 💡 **Giải thích & Dẫn chứng:** Tại sao cần thêm FCD? (*Lee et al., NeurIPS 2018*)
> Kể cả khi Min_Conf > 0.95, Deep Ensembles vẫn có thể mắc hội chứng **Confidently Wrong** (Cùng tự tin vào một vật thể sai, ví dụ nhận bóng cây là ô tô do bias chung). Bằng cách đối chiếu xác suất (Output Space) với Không gian đặc trưng (Feature Space), FCD v1.5 loại bỏ các điểm mù hệ thống này, đảm bảo Pseudo-label vô trùng tuyệt đối trước khi đem đi train lại.

### Bước 2.5: Đa dạng hóa (Diversity Sampling với K-Means)
*   **Thiết kế:** Lấy Feature Embeddings của 8.000 ảnh Vùng Đỏ Thô $\rightarrow$ Gom thành 4.000 cụm (K-Means) $\rightarrow$ Bốc 1 ảnh Uncertainty cao nhất mỗi cụm $\rightarrow$ Ra 4.000 ảnh Vùng Đỏ Tinh (Đi CVAT).
> 💡 **Giải thích & Dẫn chứng:** 
> *   **Tại sao phải Gom cụm khi BDD100K không có Trùng lặp Thời gian (Temporal)?** Theo bài báo gốc BDD100K (*Yu et al., CVPR 2020*), dữ liệu gặp vấn đề phân phối đuôi dài (Long-tail distribution): >70% ảnh là ban ngày/nắng. Dù không trùng lặp thời gian, chúng gây ra rủi ro **Trùng lặp Ngữ nghĩa (Semantic Redundancy)**. Nếu không dùng K-Means, Top 4.000 ảnh khó nhất có thể bị "nuốt chửng" bởi 4.000 ảnh cao tốc ban ngày. K-Means sẽ ép các ảnh giống hệt nhau về bối cảnh vào chung một vài cụm, giải phóng hàng ngàn slot ngân sách cho các Edge-case hiếm (Mưa, đêm).
> *   **Tại sao ép cứng K=4.000 cụm và chỉ bốc 1 ảnh/cụm?** (*Thuật toán Core-Set Selection, Sener & Savarese, ICLR 2018*). Về mặt toán học, K-Means phân bổ số cụm tỷ lệ thuận với mật độ dữ liệu (Density-aware). Nếu khu vực "Bão tuyết" có rất nhiều ảnh, K-Means tự động nhồi hàng ngàn "cụm vi mô" vào khu vực đó. Việc đặt $K = Ngân sách$ và bốc 1 ảnh/cụm giúp hệ thống tự động lấy NHIỀU ảnh hơn ở bối cảnh khó (vì có nhiều cụm vi mô), đồng thời đảm bảo khoảng cách hình học giữa 4.000 ảnh là xa nhất, triệt tiêu sự trùng lặp cục bộ.

---

## GIAI ĐOẠN 3: Tự động hóa Gán nhãn (CVAT Pre-labeling)
*   **Thiết kế:** Dùng **WBF (Weighted Boxes Fusion)** trộn 5 tập Bbox thành 1 Box mồi duy nhất, đẩy API lên CVAT.
> 💡 **Giải thích & Dẫn chứng:** Tại sao WBF? (*Solovyev et al., 2021*)
> Không dùng NMS vì NMS xóa bỏ box thô bạo. Không dùng Trung bình cộng vì box sẽ bị kéo lệch bởi model dự đoán kém. WBF lấy điểm Confidence làm "Lực hấp dẫn". Box của chuyên gia tự tin 95% sẽ kéo trọng tâm tọa độ về phía nó, bỏ qua chuyên gia 30%. Nhờ vậy, Pre-label đẩy lên CVAT cực kỳ bám sát vật thể, con người chỉ mất 1-2 giây tinh chỉnh thay vì 15 giây vẽ lại từ đầu (Annotation time saved).

---

## GIAI ĐOẠN 4: Fine-Tuning & Định lượng (M35 Integration)
*   **Thiết kế (Data Mixing):** Trộn 4.000 ảnh CVAT + 12.000 ảnh cũ (Stratified Sampling) + N ảnh Pseudo-labels (Vùng Xanh tự tin > 0.95) để train lại Mô hình Gốc.
> 💡 **Giải thích & Dẫn chứng:** Tại sao phải Stratified Sampling? (*McCloskey & Cohen, 1989 - Catastrophic Interference*)
> *   *Ví dụ trực quan:* Nếu chỉ lấy 4.000 ảnh khó (toàn ảnh sương mù ban đêm) dạy cho model, model sẽ bị "Tẩy não", trở nên xuất sắc ban đêm nhưng quên cách nhận diện ô tô giữa buổi trưa. 
> *   *Giải pháp:* Rút 12.000 ảnh cũ theo đúng tỷ lệ phân tầng (Stratified - giữ nguyên tỷ lệ xe tải/xe buýt hiếm) đóng vai trò làm **Replay Buffer (Bộ nhớ đệm)**, giúp model hấp thụ kiến thức Edge Case mới mà không đập bỏ kiến thức nền tảng cũ.

### Đo lường Thực nghiệm (A/B Testing - Đánh giá Đầu ra)
1.  **Error Recall:** Đo lường 4.000 ảnh do M29 bốc ra chứa bao nhiêu % lỗi thực sự của toàn hệ thống (So sánh với Ground Truth ẩn). Chứng minh M29 ưu việt hơn bốc Random.
2.  **Hard-set mAP Gain:** Cho Model trước và sau khi Fine-tune chạy trên tập Test Đêm/Mưa. Điểm mAP tăng vọt (Vd: +15%) chính là minh chứng định lượng cho thành công của toàn bộ quy trình.
