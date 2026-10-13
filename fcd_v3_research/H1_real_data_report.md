# Báo cáo thực nghiệm H1 trên dữ liệu thật BDD100K (bản 1.2 → 1.3)

Ngày chạy: 13–14/10/2026. Code: `fcd_v3_research/h1_real/` (Python, PyTorch, numpy, scipy, numba). Số liệu gốc: `h1_real/results/*.json`. Cách chạy lại ở Mục 7.

## 1. Mục đích và phạm vi

Báo cáo mô phỏng (`H1_simulation_report.md`) chỉ dùng dữ liệu tổng hợp. Báo cáo này chạy **đúng quy trình của tài liệu H1 bản 1.2** trên dữ liệu và detector thật:

* **Dữ liệu:** BDD100K Detection 100K (`ai_router/data/100k/{train,val,test}`: 70.000 / 10.000 / 20.000 keyframe kèm nhãn JSON từng ảnh) và 2.000 ảnh frame MOT (`ai_router/data/{train,val,test}`, mỗi clip MOT một frame số 100, không có nhãn).
* **Detector:** ba checkpoint `ai_router/experiments/expert_{1,2,3}.pth` (RT-DETR R50-vd của HuggingFace, fine-tune một lớp `car`). `expert_1` là detector $\mathcal{M}$ chính; `expert_2`, `expert_3` là hai lần lặp lại độc lập. Với mỗi anchor, hai expert còn lại tạo tín hiệu $\tau_{\text{loc}}$ (độ đồng thuận IoU) và $\tau_{\text{fcd}}$ (độ lệch cosine của đặc trưng decoder).
* **Thứ tự tiên nghiệm được giữ đúng:** P1 chỉ dùng metadata; P3 chỉ dùng Meta-Train; tham số được đóng băng vào `registration_<expert>.json` kèm SHA-256 **trước khi** đọc nhãn Val; Exp B và Exp C′ chỉ đọc file đăng ký.

**Không chạy được** (thiếu dữ liệu, không phải lỗi quy trình):

* Bộ dữ liệu không có file GPS/thời gian (`bdd100k_info`), nên không dựng được block, super-block, Moran's $I$ và C2b (phép kiểm A1 ở cấp super-block).
* Không có nhãn MOT 200 frame/clip, nên không chạy được Exp A tầng 1 và Exp C đúng như đăng ký (kiểm định H1b). Thay vào đó chạy **Exp C′**: cùng giao thức Exp C trên các ride của Val có ít nhất 2 keyframe ($M = 2$). Đây là đối chứng âm trên dữ liệu thật, không phải phép kiểm H1b.

## 2. Tóm tắt

### 2.1. Kết luận theo ma trận quyết định (Mục IX)

| Mệnh đề | expert_1 (chính) | expert_2 | expert_3 |
|---|---|---|---|
| T1 (có null trên pool) | Thỏa ($j^*$ tại $\alpha_k = 0.021$) | Thỏa ($\alpha_k = 0.029$) | Thỏa ($\alpha_k = 0.022$) |
| T1b (đối chứng dương bị bắt, bản 1.3) | Thỏa (plug-in VR = 0.475) | **Không thỏa** (plug-in VR = 0.001) | Thỏa (plug-in VR = 0.287) |
| C2: VR B-block [CI 95%] | 0.000 [0.000, 0.011] → Chấp nhận | 0.000 → Không kiểm định được (T1b) | 0.000 [0.000, 0.011] → Chấp nhận |
| C2b (A1 cấp super-block) | Không kiểm định được (thiếu GPS) | Không kiểm định được | Không kiểm định được |
| **H1a** | **Không kết luận** (C2 Chấp nhận, A1 cấp super-block chưa kiểm được) | Không kiểm định được | **Không kết luận** (như expert_1) |
| C3: Yield | 1.000 → Đạt | 1.000 → Đạt | 1.000 → Đạt |
| C3b: trung vị $\rho_{\text{cov}}$ | 1.058 → Đạt | 0.989 → Đạt | 1.096 → Đạt |
| **H1c** | **ĐẠT** | ĐẠT | ĐẠT |
| H1b | Không kiểm định được (thiếu nhãn MOT) | như trái | như trái |
| Exp C′ (đối chứng âm, $M = 2$) | C4′ Bác bỏ, C5′ Bác bỏ → không chấp nhận nhầm H1b | C4′, C5′ Bác bỏ | C4′ Bác bỏ, C5′ Không kiểm định được (T2b) |

Đọc đúng kết quả H1a: **bảo chứng không sụp đổ trên dữ liệu thật** (0 vi phạm trong 1.002 vòng ở cả ba detector, trong khi các quy trình không có bảo chứng trên cùng pool vi phạm tới 48%), **và** tiền đề A1 ở cấp chuyến đi được hỗ trợ gián tiếp; nhưng A1 ở cấp super-block (tương quan cùng khu vực, cùng ngày) chưa kiểm được vì thiếu metadata, nên theo đúng ma trận đã đăng ký, H1a chưa được Chấp nhận trọn vẹn.

### 2.2. Lỗi nghiêm trọng mới phát hiện nhờ dữ liệu thật (đã sửa ở bản 1.3)

| # | Lỗi | Bằng chứng | Sửa ở bản 1.3 |
|---|---|---|---|
| R1 | **Lưới $G$ không chạm tới $2\alpha$, $2\alpha_C$.** Phần kéo dài của Λ chỉ hoạt động khi $G$ chứa điểm có rủi ro vượt $2\alpha_C$; với detector thận trọng điều này không xảy ra | Lưới ban đầu ($\tau_{\text{cls}} \ge 0.05$): Macro-FDP lớn nhất trên Meta-Train chỉ 0.0214. $\Lambda_{1D}$ dừng ở 0.022, $\Lambda^C$ chỉ có 5 điểm và không điểm nào vượt $\alpha_C = 0.10$, nên T2 thất bại theo cấu trúc (`results/grid0_failed_reachability/`) | Thêm **quy tắc chạm** ở P3: hạ đầu dưới của $\tau_{\text{cls}}$ (trên Meta-Train) tới khi $\max_G \widehat{\text{FDP}}_{\text{meta}} \ge 2\alpha_C$. Sau khi hạ tới 0.004: max = 0.655 |
| R2 | **T1 không đủ để C2 có ý nghĩa.** Null đầu tiên có thể nằm xa $\alpha$, khi đó mọi quy trình, kể cả không có bảo chứng, đều "an toàn" | expert_2: null đầu tiên trên pool tại $\alpha_k = 0.029$ (rủi ro 0.0286); quy trình plug-in không bảo chứng chỉ có VR = 0.001; LTT phóng đại cỡ mẫu 256 lần VR = 0.001 | Thêm điều kiện **T1b** (và T2b cho C5): quy trình plug-in không bảo chứng phải bị bác bỏ (cận dưới CI của VR > δ) trên chính pool và đường đi; nếu không, tiêu chí ghi "Không kiểm định được" |
| R3 | **Ảnh và nhãn không khớp khi chạy local.** `ai_router/data/{train,val,test}` chứa frame số 100 của clip MOT (≈ giây 20), còn nhãn JSON 100K mô tả keyframe giây 10. Các expert **không** bị ảnh hưởng: chúng được huấn luyện trên Kaggle với keyframe 100K (69.863 ảnh khớp nhãn) | Sai khác điểm ảnh trung bình 33–59/255; box GT vẽ lên frame MOT lệch hoàn toàn khỏi xe. Khi `DATA_DIR` trỏ về `ai_router/data`, script huấn luyện, `01_baseline_fdp_eval.py` và `02_signal_superiority_benchmark.py` đều ghép `<name>-0000100.jpg` với nhãn 100K | Mọi tính toán H1 dùng `data/100k/<split>/<name>.jpg`. Bản 1.3 thêm bước P1 kiểm tra khớp ảnh–nhãn. Kết quả các script local đã chạy trên thư mục frame MOT cần chạy lại |
| R4 | **Chỉ số lớp sai.** Script huấn luyện dùng `COCO_CAR_ID = 3` (id COCO gốc) trong khi HF RT-DETR dùng id liên tục (car = 2, 3 = motorcycle) | Các expert phát hiện xe dưới nhãn 3 (`motorbike`); mô hình pretrained dưới nhãn 2. `02_signal_superiority_benchmark.py` dùng mô hình pretrained với id 3, tức đang chấm điểm lớp motorcycle | P1 thêm bước xác minh ánh xạ lớp trên Meta-Train; H1 dùng nhãn 3 cho expert và ghi rõ |
| R5 | **Giả định "BDD100K chia theo video, rò rỉ cấp chuyến đi lớn" là sai** | Tiền tố 8 ký tự hex của tên video hoạt động như ID chuyến đi: 100.000 video ↔ 60.138 tiền tố (1,66 video/tiền tố, khớp "100K clip từ hơn 50K chuyến"). Train và Val chung đúng **1** tiền tố (`fe189115`); nếu chia ngẫu nhiên theo video, 49,4% ride của Val sẽ có clip trong train | Ride = tiền tố tên video (đơn vị A1); quy tắc chuỗi GPS chỉ còn dùng cho block/super-block. A2 cấp chuyến đi: chỉ cần loại tiền tố chung. $\hat n = 5.930$ ride (không phải ≈ 7.000) |
| R6 | **Rò rỉ A2 cho Exp C qua bộ MOT.** Tên clip MOT train/val/test nằm trọn trong 100K train/val/test; các expert hiện có được huấn luyện trên 100K train, nên đã thấy keyframe của mọi clip MOT train | 1.400/1.400, 200/200, 399/400 tên trùng | Exp C chỉ dùng clip MOT có ride không thuộc tập huấn luyện detector. Với các expert hiện có chỉ còn 200 clip MOT val sạch, không đủ cho lưới $N_{\text{cluster}}$; cần huấn luyện lại detector trên $\text{Train}^{70\text{K}} \setminus \text{Rides}(\text{MOT})$ |
| R7 | **CI của VR suy biến khi không có vi phạm.** Bootstrap percentile theo phân hoạch cho [0, 0] | VR = 0 ở mọi detector | Cận trên CI = max(bootstrap, Clopper-Pearson với cỡ mẫu $P$): [0, 0.011] khi $P = 334$ |
| R8 | **Khẳng định của bản 1.2 rằng C2 không phát hiện được quy trình lỗi là phụ thuộc mô hình** | Mô phỏng E2: phóng đại N 32 lần chỉ VR = 0.033. Dữ liệu thật (expert_1): phóng đại 16 lần VR = 0.120 [0.087, 0.159], 64 lần VR = 0.259 → C2 Bác bỏ | Năng lực của C2 được đo trực tiếp bằng đối chứng dương D9 trên pool thật (T1b) thay vì suy từ mô phỏng |

### 2.3. Những gì được xác nhận trên dữ liệu thật

* Bảng 5.4 và 5.5 tái tạo chính xác bằng code độc lập; mốc $N \ge 149$ và $N \ge 29$ đúng.
* LTT với đơn vị ride: **0 vi phạm / 1.002 vòng** ở cả ba detector. Rủi ro pool tại $\hat\tau$: trung vị 0.010–0.012, phân vị 95% 0.013–0.015, khoảng bảo thủ $\Delta_{\text{gap}} \approx 0.008$–0.010.
* Biên chịu đựng $\hat D^* = 4.61$ (expert_1); $\text{Var}(Z)/[\alpha(1-\alpha)] = 0.39$, nằm trong khoảng mô phỏng (0.2–0.7). expert_2: $\hat D^* = 2.10$, tỉ số 0.86 (gần trường hợp nhị phân), đúng tình huống mà ngưỡng $\min(2, \hat D^*)$ được thiết kế để giữ an toàn.
* **Công cụ C2b (tỉ số phương sai) hoạt động đúng trên dữ liệu thật** khi kiểm ở cấp thấp hơn (ảnh trong ride): tỉ số = 1.03 [0.91, 1.17] (expert_1), 1.06 [0.94, 1.19], 1.02 [0.89, 1.16], khớp design effect Kish tính từ ICC ANOVA (1.006, 0.92, 1.00). Không có báo động giả: cận dưới CI không vượt 1.3.
* Các keyframe của cùng ride gần như độc lập về FDP: ICC ảnh-trong-ride 0.003 (Val, expert_1), $D_{\text{eff}} \le 1.08$ ở cận trên. Đây là lý do Exp C′ không thể tái hiện H1b, và cũng là bằng chứng ủng hộ chọn ride làm đơn vị.
* Quy tắc chọn $\theta_{\text{IoU}}$ chọn 0.75 ở cả ba detector (323 điểm của $G$ thỏa $r^*$ và $\rho_{\text{cov,meta}} \ge 0.6$ với expert_1).

## 3. P1: kiểm tra metadata (`p1_metadata.py`, `results/p1_report.json`)

| Kiểm tra | Kết quả |
|---|---|
| Số video / số tiền tố tên video | 100.000 / 60.138 (1,66 video mỗi tiền tố) |
| $\tilde m$ theo ride (train / val / test) | 2.87 / 2.85 / 2.78; ride lớn nhất 48 / 20 / 17 video; 66% ride chỉ có 1 video |
| Tiền tố chung giữa các split | train∩val = 1 (`fe189115`), train∩test = 0, val∩test = 0 |
| Phản thực tế: chia 70/10/20 ngẫu nhiên theo video | 49,4% ride của Val sẽ chung ride với train |
| Clip MOT ⊂ 100K cùng split | train 1.400/1.400, val 200/200, test 399/400 |
| Lớp box trong nhãn 100K | car, truck, bus, person, rider, bike, motor, train, traffic light, traffic sign |
| $\mathcal{C}_{\text{eval}}$ | {car}, vì các expert là detector một lớp |
| Evaluation pool (Val) | 5.930 ride, 9.995 ảnh (loại ride `fe189115`), $\tilde m = 2.85$, $m_{\max} = 20$ ảnh |
| Meta-Train | 3.000 ride ngẫu nhiên của split **test** 100K (5.027 ảnh): có nhãn, rời rạc theo ride với train và val, không được dùng để huấn luyện expert |
| Điều kiện khả thi | $5.930 \ge 3 \times 1.500 + 3$ (đơn vị = nhóm held-out = ride) |

**Kiểm tra rò rỉ trọng số (A2):** các expert được huấn luyện trên Kaggle với keyframe 100K train (69.863/70.000 ảnh ghép được nhãn, theo nhóm). Precision của expert_1 tại điểm 0.5 trên mỗi mẫu 150 ảnh ngẫu nhiên (IoU 0.5 / 0.75): hai mẫu ảnh train 0.993 / 0.988 và 0.990 / 0.971, Val 0.993 / 0.978, test 0.984 / 0.970. Precision trên ảnh đã huấn luyện không cao hơn trên Val và test, nên không có dấu hiệu ghi nhớ; Val và test chưa từng được dùng để cập nhật trọng số.

## 4. P3: pilot trên Meta-Train (`p3_pilot.py`, `results/p3_report_*.json`, `results/registration_*.json`)

Tín hiệu của box anchor $b$: $s_{\text{cls}}$ = điểm car của anchor; $s_{\text{loc}}$ = trung bình trên hai expert còn lại của IoU lớn nhất giữa $b$ và box của expert đó có điểm ≥ 0.1; $s_{\text{fcd}}$ = trung bình của $1 - \cos$ giữa đặc trưng decoder của $b$ và của box khớp (bằng 1 nếu không khớp). Box được chấp nhận khi $s_{\text{cls}} \ge \tau_{\text{cls}}$, $s_{\text{loc}} \ge \tau_{\text{loc}}$, $s_{\text{fcd}} \le \tau_{\text{fcd}}$. Lưới $G$: 35 mức $\tau_{\text{cls}}$ (0.004–0.99) × 6 mức $\tau_{\text{loc}}$ × 5 mức $\tau_{\text{fcd}}$ (phân vị 100/90/75/50/25% của $s_{\text{fcd}}$ trên Meta-Train) = 1.050 điểm.

| Chỉ số (θ = 0.75) | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| $\tau_{\text{base}}$ (chỉ lọc confidence) | 0.90 | 0.05 | 0.90 |
| Macro-FDP / độ phủ của $\tau_{\text{base}}$ trên Meta-Train | 0.0118 / 0.179 | 0.0142 / 0.186 | 0.0108 / 0.179 |
| $\theta_{\text{IoU}}$ chọn theo quy tắc $r^*$ | 0.75 | 0.75 | 0.75 |
| Max Macro-FDP trên $G$ (quy tắc chạm) | 0.655 | 0.696 | 0.731 |
| $K$ (lõi) / $K_C$ | 17 (10) / 12 | 16 (13) / 13 | 17 (10) / 11 |
| $\hat\sigma^2_\alpha$, tỉ số với $\alpha(1-\alpha)$ | 0.0077, 0.39 | 0.0169, 0.86 | 0.0077, 0.39 |
| $\hat D^*$ ($N = 1.500$) → $D_{\text{thr}}$ | 4.61 → 2 | 2.10 → 2 | 4.62 → 2 |
| $\hat D^*_C$ của arm frame ($M = 2$) | 2.27–2.37 | 1.92–2.00 | 1.99–2.07 |

Λ_1D của expert_1 (mức lưới : rủi ro Meta-Train / rủi ro pool Val): 0.002: 0.0013/0.0015 · 0.003: 0.0022/0.0021 · 0.005: 0.0048/0.0024 · 0.006: 0.0058/0.0050 · 0.007: 0.0063/0.0057 · 0.009: 0.0088/0.0079 · 0.010: 0.0096/0.0102 · 0.013: 0.0130/0.0133 · 0.014: 0.0131/0.0136 · 0.015: 0.0141/0.0149 · 0.016: 0.0158/0.0159 · 0.017: 0.0160/0.0167 · **0.021: 0.0206/0.0203** · 0.022: 0.0214/0.0219 · 0.024: 0.0238/0.0229 · 0.030: 0.0295/0.0292 · 0.040: 0.0397/0.0418.

Rủi ro trên Meta-Train dự báo rất sát rủi ro trên Val (sai lệch tối đa 0.0024), xác nhận Meta-Train lấy từ split test là gần cùng phân phối với Val.

**Lưới đầu tiên thất bại quy tắc chạm** (đã lưu nguyên trạng ở `results/grid0_failed_reachability/`): với $\tau_{\text{cls}} \ge 0.05$, max Macro-FDP = 0.0214, $\Lambda^C$ dừng ở 0.025. Nguyên nhân: điểm số của detector hai đỉnh, trung bình 3.0 truy vấn/ảnh có điểm ≥ 0.05 nhưng 12.2 truy vấn ≥ 0.005 và 92.6 truy vấn ≥ 0.001. Việc mở rộng lưới chỉ dùng Meta-Train, trước khi đọc nhãn Val.

## 5. Experiment B (`exp_b.py`, `results/exp_b_*.json`)

Đơn vị = ride, $N = 1.500$, $P = 334$ phân hoạch × $K = 3$ = 1.002 vòng. Không có super-block, nên nhóm held-out là chính ride và B-block trùng B-rand (trường hợp tài liệu đã dự liệu khi đơn vị là cấp thô nhất).

| Chỉ số | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| VR [CI] | 0.000 [0, 0.011] | 0.000 [0, 0.011] | 0.000 [0, 0.011] |
| Tỉ lệ vòng kiểm định tới $j^*_{\text{pool}}$ (D7) | 0.1% | 1.1% | 0.0% |
| Yield | 1.000 | 1.000 | 1.000 |
| Trung vị $\rho_{\text{cov}}$ | 1.058 | 0.989 | 1.096 |
| Độ phủ tuyệt đối tại $\hat\tau$ (trung vị) | 0.188 | 0.181 | 0.194 |
| D1: $\mathcal{R}_{\text{pool}}(\hat\tau)$ trung vị / p95 | 0.0102 / 0.0149 | 0.0110 / 0.0137 | 0.0119 / 0.0128 |
| D1: tỉ lệ vòng rủi ro tập kiểm thử > α | 0 | 0 | 0 |
| D3: Micro-FDP trung vị / p95 | 0.0128 / 0.0189 | 0.0125 / 0.0157 | 0.0148 / 0.0173 |
| D3: tỉ lệ ride rỗng; Spearman($N^{\text{acc}}$, FDP) | 0.17; 0.16 | 0.16; 0.13 | 0.17; 0.17 |
| D3: Macro-FDP của $\tau_{\text{base}}$ trên tập kiểm thử | 0.0100 | 0.0137 | 0.0096 |
| D4: tỉ lệ FP do ảo giác/sai lớp | 0.46 | 0.55 | 0.45 |

**D9, đối chứng dương trên cùng pool và cùng các tập hiệu chuẩn của nhánh chính (VR [CI] → vùng C2):**

| Quy trình | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| Plug-in, không bảo chứng (điểm cuối có $\widehat{\mathcal{R}}_{\text{cal}} \le \alpha$) | 0.475 [0.421, 0.531] Bác bỏ | 0.001 Chấp nhận | 0.287 [0.239, 0.339] Bác bỏ |
| LTT giả vờ cỡ mẫu ×4 | 0.012 Chấp nhận | 0.000 | 0.006 |
| ×16 | 0.120 [0.087, 0.159] Bác bỏ | 0.000 | 0.065 Không kết luận |
| ×64 | 0.259 Bác bỏ | 0.001 | 0.154 Bác bỏ |
| ×256 | 0.357 Bác bỏ | 0.001 | 0.213 Bác bỏ |

Micro-FDP cao hơn Macro-FDP khoảng 0.003 và Spearman dương nhỏ, đúng cơ chế đã cảnh báo ở Mục 2.2; Micro-FDP vẫn dưới α ở phân vị 95%.

**Phân nhóm (mô tả, không thuộc A3):** rủi ro trung vị tại $\hat\tau$ của expert_1 trên ride ban đêm 0.0096, mưa/tuyết 0.0134 (0,9% số vòng vượt α), cao tốc 0.0045. Nhóm mưa/tuyết có rủi ro cao nhất ở cả ba detector (0.013–0.016), là ứng viên cho stress-test OOD.

**Nhánh B-img (mô tả):** coi ảnh là đơn vị, bỏ qua ride: VR = 0.001 / 0.000 / 0.000. Do ICC ảnh-trong-ride gần 0, bỏ qua cấu trúc ride ở BDD100K keyframe không gây hại.

## 6. Experiment C′: đối chứng âm trên dữ liệu thật (`exp_c_prime.py`)

2.032 ride của Val có ít nhất 2 keyframe; mỗi ride lấy 2 keyframe đầu (theo tên). $\alpha_C = 0.10$, đường đi $\Lambda^C$, ba arm như Mục 8.3, $N_{\text{cluster}} \in \{100, 200, 400\}$ ($K$ = 12 / 6 / 3, $B \approx 1.000$).

| | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| ICC frame; $D_{\text{eff}}^{\text{frame}}$ [CI] | 0.021; 1.02 [0.97, 1.08] | 0.032; 1.03 [0.99, 1.09] | 0.020; 1.02 [0.97, 1.08] |
| C4′ | Bác bỏ (cận trên < 2) | Bác bỏ | Bác bỏ |
| VR Arm 1 / Arm 2 / Arm 3 (mọi $N_{\text{cluster}}$) | 0 / ≤ 0.001 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Yield Arm 1 | 1.00 ở cả ba mức | 1.00 | 1.00 |
| T2b: plug-in theo frame VR (100 / 200 / 400) | 0.27 / 0.16 / 0.08 | 0.36 / 0.19 / 0.10 | 0.07 / 0.02 / 0.00 |
| C5′ | Bác bỏ | Bác bỏ | Không kiểm định được (T2b) |
| $\Delta_{\text{leak}}$ trung bình | −0.0003 … +0.0001 | −0.0004 … 0.0000 | −0.0004 … +0.0002 |

Khi tương quan nội cụm không đáng kể ($D_{\text{eff}}^{\text{frame}} \approx 1.02 < \hat D^*_C \approx 2$), arm frame không vi phạm và các tiêu chí của bản 1.2 **không** chấp nhận nhầm H1b. Kết quả này khớp dự đoán của lý thuyết biên chịu đựng và bổ sung cho đối chứng âm E9 của mô phỏng. Nó không nói gì về H1b ở cấp frame MOT ($M = 200$), nơi $D_{\text{eff}}$ dự kiến hàng chục.

Hạn chế quan sát được: $\Lambda^C$ thưa quanh $\alpha_C$ (rủi ro pool nhảy từ 0.089 lên 0.111 giữa hai điểm liên tiếp) vì lưới $\tau_{\text{cls}}$ thô ở vùng 0.0125–0.015. Bản 1.3 thêm yêu cầu độ phân giải cho quy tắc chạm.

## 7. Cách chạy lại

```
cd fcd_v3_research/h1_real
python p1_metadata.py                        # P1, chỉ metadata
python infer_cache.py meta expert_1          # lặp cho meta|val × expert_1..3 (~3–5 phút mỗi lượt, RTX 3050)
python build_features.py meta expert_1       # lặp cho meta|val × expert_1..3; meta trước val
python p3_pilot.py expert_1                  # đóng băng registration_expert_1.json (SHA-256)
python exp_b.py expert_1                     # Experiment B + D9 + C2b-instrument
python exp_c_prime.py expert_1               # Experiment C′
python test_h1core.py                        # tự kiểm: ghép tham lam, Lemma A4, bảng HB, CI
```

Mọi bước có seed cố định. Bộ đếm lưới đã được kiểm chéo với tính trực tiếp từ cache dự đoán trên 300 ảnh × 4 ngưỡng: 0 sai khác.

## 8. Giới hạn

* Không có GPS/thời gian: C2b, block, super-block, Moran's $I$, quy tắc R-unit ở cấp trên ride và D8 không chạy được.
* Ride = tiền tố tên video là suy luận có bằng chứng mạnh (số lượng khớp công bố, split rời rạc theo tiền tố) nhưng chưa được BDD100K xác nhận chính thức.
* Detector một lớp và recall thấp (độ phủ ~0.19); H1c đạt theo tiêu chí tương đối, nhưng độ phủ tuyệt đối thấp phản ánh chất lượng detector (nguyên nhân chưa xác định), không phải khung LTT.
* Meta-Train lấy từ split test 100K thay vì từ train như tài liệu, vì không biết chắc tập huấn luyện của expert; điều kiện độc lập của A5 vẫn được giữ.
* Tôi (người chạy) đã xem precision của expert_1 trên 150 ảnh Val ở bước dò nhãn lớp trước khi đăng ký. Bước này không chọn tham số nào của LTT; nó chỉ xác định nhãn đầu ra của lớp car. Bản chạy chính thức nên làm bước này trên Meta-Train.
