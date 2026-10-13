# Báo cáo thực nghiệm H1 trên dữ liệu thật BDD100K (bản 1.2 → 1.3)

Lần chạy: 14/10/2026, chạy lại từ đầu bằng `h1_real/run_all.sh` (nhật ký có dấu thời gian: `h1_real/results/run_log.txt`). Code: `fcd_v3_research/h1_real/`. Số liệu gốc: `h1_real/results/*.json`. Cách chạy lại ở Mục 8.

## 1. Mục đích và phạm vi

Báo cáo mô phỏng (`H1_simulation_report.md`) chỉ dùng dữ liệu tổng hợp. Báo cáo này chạy **đúng quy trình của tài liệu H1** trên dữ liệu và detector thật, với phạm vi đầu vào cố định:

* **Dữ liệu:** chỉ `ai_router/data/100k/{train,val,test}` (70.000 / 10.000 / 20.000 keyframe kèm nhãn JSON từng ảnh). Không thư mục hay script nào khác của dự án được dùng.
* **Detector:** chỉ `ai_router/experiments/expert_{1,2,3}.pth` (RT-DETR R50-vd của HuggingFace, fine-tune một lớp `car` trên Kaggle với keyframe 100K train). `expert_1` là detector $\mathcal{M}$ chính; `expert_2`, `expert_3` là hai lần lặp lại độc lập. Với mỗi anchor, hai expert còn lại tạo tín hiệu $\tau_{\text{loc}}$ (độ đồng thuận IoU) và $\tau_{\text{fcd}}$ (độ lệch cosine của đặc trưng decoder).
* **Thứ tự tiên nghiệm:** P1 chỉ dùng metadata và lược đồ nhãn của split train; P3.0 và P3 chỉ dùng Meta-Train (3.000 ride của split test) và 100K train; tham số được đóng băng vào `registration_<expert>.json` kèm SHA-256 lúc 01:56:12; nhãn Val chỉ được đọc sau mốc này. Không bước nào trước đăng ký chạm vào nhãn Val.

**Không chạy được** (thiếu dữ liệu, không phải lỗi quy trình):

* Không có metadata GPS/thời gian (`bdd100k_info`), nên không dựng được block, super-block, Moran's $I$ và C2b (phép kiểm A1 ở cấp super-block).
* Không có bộ MOT (200 frame có nhãn mỗi clip), nên không chạy được Exp A tầng 1 và Exp C đúng như đăng ký (kiểm định H1b). Thay vào đó chạy **Exp C′**: cùng giao thức Exp C trên các ride của Val có ít nhất 2 keyframe ($M = 2$). Đây là đối chứng âm trên dữ liệu thật, không phải phép kiểm H1b.

## 2. Tóm tắt

### 2.1. Kết luận theo ma trận quyết định (Mục IX)

| Mệnh đề | expert_1 (chính) | expert_2 | expert_3 |
|---|---|---|---|
| T1: null đầu tiên trên pool (mức lưới : rủi ro) | 0.021 : 0.0203 | 0.025 : 0.0223 | 0.022 : 0.0212 |
| T1b: VR của plug-in không bảo chứng [CI] | 0.475 [0.421, 0.531] ✓ | 0.118 [0.084, 0.156] ✓ | 0.287 [0.239, 0.339] ✓ |
| C2: VR của LTT [CI 95%] | 0.000 [0.000, 0.011] → Chấp nhận | 0.000 [0.000, 0.011] → Chấp nhận | 0.000 [0.000, 0.011] → Chấp nhận |
| C2b (A1 cấp super-block) | Không kiểm định được (thiếu GPS) | Không kiểm định được | Không kiểm định được |
| **H1a** | **Không kết luận** | **Không kết luận** | **Không kết luận** |
| C3: Yield | 1.000 → Đạt | 1.000 → Đạt | 1.000 → Đạt |
| C3b: trung vị $\rho_{\text{cov}}$ | 1.057 → Đạt | 0.991 → Đạt | 1.096 → Đạt |
| **H1c** | **ĐẠT** | **ĐẠT** | **ĐẠT** |
| **H1b** | Không kiểm định được (thiếu bộ MOT) | như trái | như trái |
| Exp C′ (đối chứng âm, $M = 2$) | C4′ Bác bỏ, C5′ Bác bỏ (T2b thỏa) | như trái | như trái |

Đọc đúng kết quả H1a: **bảo chứng không sụp đổ trên dữ liệu thật** (0 vi phạm trong 1.002 vòng ở cả ba detector, trong khi quy trình không có bảo chứng trên cùng pool và cùng các tập hiệu chuẩn vi phạm 12–48%). Các keyframe cùng ride gần như độc lập, ủng hộ đơn vị ride. Nhưng A1 ở cấp super-block (tương quan cùng khu vực, cùng ngày) chưa kiểm được vì thiếu metadata, nên theo đúng ma trận đã đăng ký, H1a chưa được Chấp nhận trọn vẹn.

### 2.2. Lỗi phát hiện nhờ dữ liệu thật (đã sửa ở bản 1.3)

| # | Lỗi | Bằng chứng | Sửa ở bản 1.3 |
|---|---|---|---|
| R1 | **Lưới $G$ không chạm tới $2\alpha$, $2\alpha_C$.** Phần kéo dài của Λ chỉ hoạt động khi $G$ chứa điểm có rủi ro vượt $2\alpha_C$; với detector thận trọng điều này không xảy ra | Với $\tau_{\text{cls}} \ge 0.05$ (lưới bản 1.2), Macro-FDP lớn nhất trên Meta-Train chỉ là 0.0214 / 0.0142 / 0.0224, nên $\Lambda^C$ không thể vượt $\alpha_C = 0.10$ và T2 thất bại theo cấu trúc | **Quy tắc chạm** (P3): hạ đầu dưới của $\tau_{\text{cls}}$ trên Meta-Train cho tới khi $\max_G \widehat{\text{FDP}}_{\text{meta}} \ge 2\alpha_C$. Kết quả: dừng ở 0.0095 / 0.011 / 0.0105 (max 0.214 / 0.214 / 0.212) |
| R2 | **Lưới thô tạo khoảng nhảy rủi ro quanh α.** Điểm số của detector hai đỉnh, nên các bậc $\tau_{\text{cls}}$ thô (0.025 → 0.03) làm rủi ro nhảy cả chục bước lưới | Ablation D11 (lưới thô của bản nháp đầu, cùng dữ liệu): Λ_1D của expert_2 nhảy từ 0.0160 lên **0.0286** trên pool | **Quy tắc độ phân giải:** lưới $\tau_{\text{cls}}$ bước 0.0005 trong đoạn 0.004–0.05. Null đầu tiên của expert_2 trên pool còn 0.0223; quanh $\alpha_C$ hai điểm liên tiếp cách nhau ≤ 0.007 trên Meta-Train |
| R3 | **T1 không đủ để C2 có ý nghĩa.** Null đầu tiên nằm xa α thì mọi quy trình, kể cả không có bảo chứng, đều "an toàn" | Ablation D11, expert_2: quy trình plug-in không bảo chứng chỉ vi phạm 0.001 [0, 0.011]; nếu không có T1b, C2 sẽ "Chấp nhận" một phép thử không có năng lực | **Điều kiện T1b / T2b:** plug-in không bảo chứng phải bị bắt (cận dưới CI của VR > δ) trên chính pool và đường đi; T1b chặn kết luận Chấp nhận của C2, T2b chặn kết luận Bác bỏ của C5 |
| R4 | **Chỉ số lớp của detector không theo tên lớp trong cấu hình** | P3.0 trên Meta-Train: cả ba expert phát hiện xe ở chỉ số đầu ra 3 (tên trong cấu hình HuggingFace: `motorbike`), không chỉ số nào khác cho true positive; RT-DETR pretrained dùng chỉ số 2 cho car | Bước P3.0: xác định chỉ số lớp trên Meta-Train và đăng ký |
| R5 | **Giả định "BDD100K chia theo video, rò rỉ cấp chuyến đi lớn" là sai** | Tiền tố 8 ký tự hex của tên video hoạt động như ID chuyến đi: 100.000 video ↔ 60.138 tiền tố (1,66 video/tiền tố, khớp "100K clip từ hơn 50K chuyến"). Train và Val chung đúng **1** tiền tố (`fe189115`); nếu chia ngẫu nhiên theo video, 49,4% ride của Val sẽ có clip trong train | Ride = tiền tố tên video (đơn vị A1); quy tắc chuỗi GPS chỉ còn dùng cho block/super-block. A2 cấp chuyến đi: chỉ cần loại tiền tố chung. $\hat n = 5.930$ ride (không phải ≈ 7.000) |
| R6 | **CI của VR suy biến khi không có vi phạm.** Bootstrap percentile theo phân hoạch cho [0, 0] | VR = 0 ở mọi detector | Cận trên CI = max(bootstrap, Clopper-Pearson với cỡ mẫu $P$): [0, 0.011] khi $P = 334$ |
| R7 | **Khẳng định của bản 1.2 rằng C2 không phát hiện được quy trình lỗi là phụ thuộc mô hình** | Mô phỏng E2: phóng đại N 32 lần chỉ VR = 0.033. Dữ liệu thật (expert_1): LTT giả vờ cỡ mẫu ×16 có VR = 0.120 [0.087, 0.159], ×64 có VR = 0.259 → C2 Bác bỏ | Năng lực của C2 được đo trực tiếp bằng đối chứng dương D9 trên pool thật (T1b) thay vì suy từ mô phỏng |

### 2.3. Những gì được xác nhận trên dữ liệu thật

* Bảng 5.4 và 5.5 tái tạo chính xác bằng code độc lập; mốc $N \ge 149$ và $N \ge 29$ đúng.
* LTT với đơn vị ride: **0 vi phạm / 1.002 vòng** ở cả ba detector. Rủi ro pool tại $\hat\tau$: trung vị 0.010–0.012, phân vị 95% 0.014–0.015, khoảng bảo thủ $\Delta_{\text{gap}} \approx 0.008$–0.010.
* Biên chịu đựng $\hat D^* = 4.52$–4.62; $\text{Var}(Z)/[\alpha(1-\alpha)] = 0.39$–0.40, nằm trong khoảng mô phỏng (0.2–0.7).
* **Công cụ C2b (tỉ số phương sai) hoạt động đúng trên dữ liệu thật** khi kiểm ở cấp thấp hơn (ảnh trong ride, D10): tỉ số = 1.03 [0.91, 1.17], 1.05 [0.93, 1.18], 1.02 [0.89, 1.16], khớp design effect Kish tính từ ICC ANOVA (1.01, 0.94, 1.00). Không báo động giả: cận dưới CI không vượt 1.3.
* Các keyframe cùng ride gần như độc lập về FDP (ICC ảnh-trong-ride trên Val từ −0.03 đến 0.04). Đây là lý do Exp C′ không thể tái hiện H1b, và cũng là bằng chứng ủng hộ chọn ride làm đơn vị ở cấp trong-ride.
* Quy tắc chọn $\theta_{\text{IoU}}$ chọn 0.75 ở cả ba detector (666–1.320 điểm của $G$ thỏa $r^*$ và $\rho_{\text{cov,meta}} \ge 0.6$).
* Rủi ro trên Meta-Train dự báo sát rủi ro trên Val dọc toàn đường đi (sai lệch tối đa 0.0027, expert_1), xác nhận Meta-Train lấy từ split test gần cùng phân phối với Val.

## 3. P1: kiểm tra metadata (`p1_metadata.py`, `results/p1_report.json`)

| Kiểm tra | Kết quả |
|---|---|
| Số video / số tiền tố tên video | 100.000 / 60.138 (1,66 video mỗi tiền tố) |
| $\tilde m$ theo ride (train / val / test) | 2.87 / 2.85 / 2.78; ride lớn nhất 48 / 20 / 17 video; 66% ride chỉ có 1 video |
| Tiền tố chung giữa các split | train∩val = 1 (`fe189115`), train∩test = 0, val∩test = 0 |
| Phản thực tế: chia 70/10/20 ngẫu nhiên theo video | 49,4% ride của Val sẽ chung ride với train |
| Lớp box trong nhãn 100K (mẫu 2.000 ảnh train) | car, truck, bus, person, rider, bike, motor, train, traffic light, traffic sign |
| $\mathcal{C}_{\text{eval}}$ | {car}, vì các expert là detector một lớp |
| Evaluation pool (Val) | 5.930 ride, 9.995 ảnh (loại ride `fe189115`), $\tilde m = 2.85$, $m_{\max} = 20$ ảnh |
| Meta-Train | 3.000 ride ngẫu nhiên của split **test** 100K (5.027 ảnh): có nhãn, rời rạc theo ride với train và val, không được dùng để cập nhật trọng số expert |
| Điều kiện khả thi | $5.930 \ge 3 \times 1.500 + 3$ (đơn vị = nhóm held-out = ride) |

## 4. P3: pilot trên Meta-Train (`p3_class_map.py`, `p3_pilot.py`, `results/class_map.json`, `results/p3_report_*.json`, `results/registration_*.json`)

**P3.0, chỉ số lớp và kiểm tra ghi nhớ.** Trên 300 ảnh Meta-Train, chỉ số đầu ra cho nhiều true positive nhất với xe là **3** ở cả ba expert (không chỉ số nào khác có true positive). Precision tại điểm 0.5 (IoU 0.5 / 0.75), 150 ảnh Meta-Train so với 150 ảnh 100K train (tập huấn luyện của expert): expert_1 0.998 / 0.981 so với 0.993 / 0.986; expert_2 0.994 / 0.984 so với 0.997 / 0.991; expert_3 0.998 / 0.982 so với 0.998 / 0.993. Chênh lệch trên ảnh đã huấn luyện tối đa 1.1 điểm % ở IoU 0.75, không có dấu hiệu ghi nhớ đáng kể.

**Tín hiệu và lưới.** Tín hiệu của box anchor $b$: $s_{\text{cls}}$ = điểm car của anchor; $s_{\text{loc}}$ = trung bình trên hai expert còn lại của IoU lớn nhất giữa $b$ và box của expert đó có điểm ≥ 0.1; $s_{\text{fcd}}$ = trung bình của $1 - \cos$ giữa đặc trưng decoder của $b$ và của box khớp (bằng 1 nếu không khớp). Box được chấp nhận khi $s_{\text{cls}} \ge \tau_{\text{cls}}$, $s_{\text{loc}} \ge \tau_{\text{loc}}$, $s_{\text{fcd}} \le \tau_{\text{fcd}}$. Thang ứng viên của $\tau_{\text{cls}}$: 0.004–0.05 bước 0.0005, rồi 0.10–0.95 bước 0.05, 0.97, 0.99; 6 mức $\tau_{\text{loc}}$; 5 mức $\tau_{\text{fcd}}$ (phân vị 100/90/75/50/25% của $s_{\text{fcd}}$ trên Meta-Train). Quy tắc chạm chọn đầu dưới của $\tau_{\text{cls}}$, từ đó ra $G$.

| Chỉ số (θ = 0.75) | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| $\tau_{\text{base}}$ (chỉ lọc confidence) | 0.90 | 0.05 | 0.90 |
| Macro-FDP / độ phủ của $\tau_{\text{base}}$ trên Meta-Train | 0.0118 / 0.179 | 0.0142 / 0.186 | 0.0108 / 0.179 |
| $\theta_{\text{IoU}}$ chọn theo quy tắc $r^*$ | 0.75 | 0.75 | 0.75 |
| Quy tắc chạm: max Macro-FDP khi $\tau_{\text{cls}} \ge 0.05$ → đầu dưới chọn (max) | 0.0214 → 0.0095 (0.214) | 0.0142 → 0.011 (0.214) | 0.0224 → 0.0105 (0.212) |
| $\lvert G \rvert$ | 3.060 | 2.970 | 3.000 |
| $K$ (lõi) / $K_C$ | 26 (10) / 30 | 18 (13) / 25 | 23 (11) / 27 |
| Độ phân giải quanh α: điểm cuối ≤ α → điểm đầu > α (Meta-Train) | 0.0160 → 0.0206 | 0.0169 → 0.0202 | 0.0158 → 0.0208 |
| Độ phân giải quanh $\alpha_C$ | 0.0957 → 0.1003 | 0.0976 → 0.1043 | 0.0961 → 0.1014 |
| $\hat\sigma^2_\alpha$, tỉ số với $\alpha(1-\alpha)$ | 0.0077, 0.39 | 0.0078, 0.40 | 0.0077, 0.39 |
| $\hat D^*$ ($N = 1.500$) → $D_{\text{thr}}$ | 4.61 → 2 | 4.52 → 2 | 4.62 → 2 |
| $\hat D^*_C$ của arm frame ($M = 2$) | 2.56–2.67 | 2.25–2.34 | 2.44–2.54 |

Λ_1D của expert_1 (mức lưới : rủi ro Meta-Train / rủi ro pool Val): 0.002: 0.0013/0.0015 · 0.003: 0.0022/0.0021 · 0.005: 0.0048/0.0024 · 0.006: 0.0058/0.0050 · 0.007: 0.0063/0.0057 · 0.009: 0.0088/0.0079 · 0.010: 0.0096/0.0102 · 0.013: 0.0130/0.0134 · 0.014: 0.0135/0.0146 · 0.015: 0.0150/0.0152 · 0.016: 0.0158/0.0159 · 0.017: 0.0160/0.0167 · **0.021: 0.0206/0.0203** · 0.022: 0.0217/0.0220 · 0.023: 0.0221/0.0223 · 0.024: 0.0237/0.0231 · 0.025: 0.0245/0.0241 · 0.026: 0.0257/0.0248 · 0.027: 0.0270/0.0260 · 0.028: 0.0275/0.0276 · 0.029: 0.0284/0.0280 · 0.030: 0.0297/0.0304 · 0.0325: 0.0322/0.0349 · 0.035: 0.0346/0.0372 · 0.0375: 0.0358/0.0385 · 0.040: 0.0397/0.0418.

## 5. Experiment B (`exp_b.py`, `results/exp_b_*.json`)

Đơn vị = ride, $N = 1.500$, $P = 334$ phân hoạch × $K = 3$ = 1.002 vòng. Không có super-block, nên nhóm held-out là chính ride và B-block trùng B-rand (trường hợp tài liệu đã dự liệu khi đơn vị là cấp thô nhất).

| Chỉ số | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| VR [CI] | 0.000 [0, 0.011] | 0.000 [0, 0.011] | 0.000 [0, 0.011] |
| Tỉ lệ vòng kiểm định tới $j^*_{\text{pool}}$ (D7) | 0.1% | 0.0% | 0.0% |
| Yield | 1.000 | 1.000 | 1.000 |
| Trung vị $\rho_{\text{cov}}$ | 1.057 | 0.991 | 1.096 |
| Độ phủ tuyệt đối tại $\hat\tau$ (trung vị) | 0.188 | 0.181 | 0.194 |
| D1: $\mathcal{R}_{\text{pool}}(\hat\tau)$ trung vị / p95 | 0.0102 / 0.0146 | 0.0110 / 0.0145 | 0.0119 / 0.0143 |
| D1: tỉ lệ vòng rủi ro tập kiểm thử > α | 0 | 0 | 0 |
| D3: Micro-FDP trung vị / p95 | 0.0128 / 0.0188 | 0.0127 / 0.0192 | 0.0148 / 0.0196 |
| D3: tỉ lệ ride rỗng; Spearman($N^{\text{acc}}$, FDP) | 0.17; 0.16 | 0.16; 0.14 | 0.17; 0.17 |
| D3: Macro-FDP của $\tau_{\text{base}}$ trên tập kiểm thử | 0.0100 | 0.0137 | 0.0096 |
| D4: tỉ lệ FP do ảo giác/sai lớp | 0.46 | 0.54 | 0.45 |

**D9, đối chứng dương trên cùng pool và cùng các tập hiệu chuẩn của nhánh chính (VR [CI] → vùng C2):**

| Quy trình | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| Plug-in, không bảo chứng (điểm cuối có $\widehat{\mathcal{R}}_{\text{cal}} \le \alpha$) | 0.475 [0.421, 0.531] Bác bỏ | 0.118 [0.084, 0.156] Bác bỏ | 0.287 [0.239, 0.339] Bác bỏ |
| LTT giả vờ cỡ mẫu ×4 | 0.012 Chấp nhận | 0.002 Chấp nhận | 0.006 Chấp nhận |
| ×16 | 0.120 [0.087, 0.159] Bác bỏ | 0.010 Chấp nhận | 0.065 Không kết luận |
| ×64 | 0.259 Bác bỏ | 0.044 Không kết luận | 0.154 Bác bỏ |
| ×256 | 0.357 Bác bỏ | 0.070 Không kết luận | 0.213 Bác bỏ |

**D11, ablation lưới thô (`ablation_coarse_grid.py`):** dựng lại Λ_1D bằng lưới $\tau_{\text{cls}}$ thô của bản nháp đầu, cùng quy tắc chạm, cùng thuật toán, cùng pool và cùng phân hoạch:

| | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| Null đầu tiên trên pool (mức lưới : rủi ro) | 0.021 : 0.0203 | **0.029 : 0.0286** | 0.022 : 0.0212 |
| VR plug-in [CI] → T1b | 0.475 → thỏa | **0.001 [0, 0.011] → không thỏa** | 0.287 → thỏa |
| VR LTT | 0 | 0 | 0 |

Ablation cho thấy cả hai cơ chế sửa ở R2–R3 đều cần: không có quy tắc độ phân giải, đường đi của expert_2 có khoảng nhảy làm C2 mất năng lực; không có T1b, khoảng nhảy đó dẫn tới một kết luận "Chấp nhận" vô nghĩa.

Micro-FDP cao hơn Macro-FDP khoảng 0.002–0.003 và Spearman dương nhỏ, đúng cơ chế đã cảnh báo ở Mục 2.2; Micro-FDP vẫn dưới α ở phân vị 95%.

**Phân nhóm (mô tả, không thuộc A3):** rủi ro trung vị tại $\hat\tau$ của expert_1 trên ride ban đêm 0.0096, mưa/tuyết 0.0134 (0,9% số vòng vượt α), cao tốc 0.0045. Nhóm mưa/tuyết có rủi ro cao nhất ở cả ba detector (0.013–0.016), là ứng viên cho stress-test OOD.

**Nhánh B-img (mô tả):** coi ảnh là đơn vị, bỏ qua ride: VR = 0.001 / 0.000 / 0.000. Do ICC ảnh-trong-ride gần 0, bỏ qua cấu trúc ride ở cấp keyframe không gây hại.

## 6. Experiment C′: đối chứng âm trên dữ liệu thật (`exp_c_prime.py`)

2.032 ride của Val có ít nhất 2 keyframe; mỗi ride lấy 2 keyframe đầu (theo tên). $\alpha_C = 0.10$, đường đi $\Lambda^C$, ba arm như Mục 8.3, $N_{\text{cluster}} \in \{100, 200, 400\}$ ($K$ = 12 / 6 / 3, $B \approx 1.000$).

| | expert_1 | expert_2 | expert_3 |
|---|---|---|---|
| ICC frame; $D_{\text{eff}}^{\text{frame}}$ [CI] | 0.011; 1.01 [0.97, 1.07] | 0.042; 1.04 [0.99, 1.11] | −0.004; 1.00 [0.96, 1.04] |
| C4′ | Bác bỏ (cận trên < 2) | Bác bỏ | Bác bỏ |
| VR Arm 1 / Arm 2 / Arm 3 (mọi $N_{\text{cluster}}$) | 0 / ≤ 0.002 / ≤ 0.002 | 0 / 0.001 / ≤ 0.001 | 0 / ≤ 0.003 / ≤ 0.002 |
| Yield Arm 1 | 1.00 ở cả ba mức | 1.00 | 1.00 |
| T2b: plug-in theo frame VR (100 / 200 / 400) | 0.41 / 0.33 / 0.26 | 0.44 / 0.39 / 0.31 | 0.44 / 0.36 / 0.29 |
| C5′ | Bác bỏ | Bác bỏ | Bác bỏ |
| $\Delta_{\text{leak}}$ trung bình | −0.0004 … +0.0002 | −0.0002 … 0.0000 | −0.0004 … +0.0002 |

Khi tương quan nội cụm không đáng kể ($D_{\text{eff}}^{\text{frame}} \approx 1 < \hat D^*_C \approx 2.3$–2.7), arm frame không vi phạm dù quy trình plug-in theo frame trên cùng dữ liệu vi phạm 26–44%, và các tiêu chí **không** chấp nhận nhầm H1b. Kết quả này khớp dự đoán của lý thuyết biên chịu đựng và bổ sung cho đối chứng âm E9 của mô phỏng. Nó không nói gì về H1b ở cấp frame MOT ($M = 200$), nơi $D_{\text{eff}}$ dự kiến hàng chục.

## 7. Thay đổi so với lần chạy trước

Lần chạy trước bị loại bỏ toàn bộ và làm lại từ đầu với phạm vi đầu vào cố định. Khác biệt:

* Chỉ số lớp được xác định ở P3.0 trên Meta-Train (lần trước dò trên 150 ảnh Val trước khi đăng ký).
* Quy tắc chạm và quy tắc độ phân giải được cài vào code P3 (lần trước lưới được mở rộng bằng tay và còn thô). Hệ quả: expert_2 có Λ_1D dày quanh α, T1b thỏa và C2 Chấp nhận (lần trước: Không kiểm định được); expert_3 thỏa T2b ở Exp C′ (lần trước: không thỏa). Hiện tượng của lưới thô được giữ lại dưới dạng ablation D11 tái lập được.
* P1 và các bước trước đăng ký chỉ đọc `ai_router/data/100k` và metadata; lược đồ lớp lấy từ nhãn split train.

## 8. Cách chạy lại

```
cd fcd_v3_research/h1_real
bash run_all.sh                              # toàn bộ pipeline theo đúng thứ tự, ghi nhật ký vào stdout
```

`run_all.sh` chạy lần lượt: P1 (`p1_metadata.py`) → P3.0 (`p3_class_map.py`) → suy luận (`infer_cache.py`, bỏ qua nếu đã có cache) → bộ đếm Meta-Train (`build_features.py meta`) → P3 và đăng ký (`p3_pilot.py`) → bộ đếm Val → Exp B → Exp C′ → ablation D11 → tự kiểm (`test_h1core.py`). Thời gian khoảng 30 phút trên RTX 3050 nếu chạy cả suy luận. Mọi bước có seed cố định. Bộ đếm lưới đã được kiểm chéo với tính trực tiếp từ cache dự đoán (0 sai khác).

## 9. Giới hạn

* Không có GPS/thời gian: C2b, block, super-block, Moran's $I$, quy tắc R-unit ở cấp trên ride và D8 không chạy được.
* Ride = tiền tố tên video là suy luận có bằng chứng mạnh (số lượng khớp công bố, split rời rạc theo tiền tố) nhưng chưa được BDD100K xác nhận chính thức.
* Detector một lớp và recall thấp (độ phủ ~0.19 ở IoU 0.75); H1c đạt theo tiêu chí tương đối, nhưng độ phủ tuyệt đối thấp phản ánh chất lượng detector (nguyên nhân chưa xác định), không phải khung LTT.
* Meta-Train lấy từ split test 100K thay vì từ train như tài liệu, vì các expert đã học trên toàn bộ 100K train; điều kiện độc lập của A5 vẫn được giữ.
