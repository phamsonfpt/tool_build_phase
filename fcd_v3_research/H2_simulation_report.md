# Báo cáo mô phỏng thiết kế cho Giả thuyết H2 (bản 1.3 → 1.4)

Ngày chạy: 14/10/2026. Code nằm trong `h2_simulation/` (Python, numpy, scipy). Mỗi script có seed cố định và in ra đúng các số trong báo cáo này.

## 1. Mục đích và phạm vi

Báo cáo này **không** đo kết quả thật của H2. Chưa có đặc trưng và dự đoán của 3 detector trên Meta-Train, và dữ liệu cục bộ chỉ có 200 ảnh Val, vốn thuộc Evaluation pool nên không được dùng cho pilot. Mô phỏng chạy đúng quy trình của tài liệu H2 (đường đi Pareto trên Meta-Path, LTT với p-value HB, các vòng B-block, bootstrap theo super-block) trên dữ liệu tổng hợp, để trả lời hai câu hỏi:

* **S1:** trong các phương án tín hiệu (bản 1.2, các ô của thiết kế 2×2, phương án lai), phương án nào giữ được nhiều Coverage nhất dưới LTT, và kết luận đó có bền khi thay đổi các giả định không?
* **S2:** quy tắc quyết định của bản 1.2 (Wilcoxon theo đơn vị trên một lần chia) và của bản 1.3 (trung bình qua các vòng B-block, bootstrap theo super-block, quy tắc ba vùng) có tỷ lệ kết luận sai là bao nhiêu?

## 2. Mô hình sinh dữ liệu

Mỗi pool gồm các đơn vị (ride), gom vào super-block với kích thước trung bình 2.5 đơn vị. Độ khó của đơn vị gồm hiệu ứng riêng ($\sigma_u = 0.5$) và hiệu ứng super-block ($\sigma_b = 0.3$). Số GT mỗi đơn vị là $1 + \text{Poisson}(7)$; mỗi GT được phát hiện với xác suất 0.92. Ứng viên có ba loại:

| Loại | Số lượng | Logit tin cậy (anchor) | Khoảng cách tiềm ẩn chuẩn hóa $\zeta_m = \mu + b\,\xi_m$ |
|---|---|---|---|
| TP | phát hiện đúng | $3.5 - 0.4u + \varepsilon$ | $\mu \sim N(0, 0.6^2)$, $b = 0.8$ |
| FP lệch vị trí | tỷ lệ $p_{\text{loc}}$ của các phát hiện (tăng theo độ khó) | $1.0 - 0.4u + \varepsilon$ | $\mu \sim N(\delta_{\text{loc}}, 0.6^2)$, $b = 0.8$ |
| FP ảo giác | $\text{Poisson}(r_{\text{hal}}\, e^{u})$ | 30% tự tin: $2.5 + \varepsilon$; 70%: $-0.5 + \varepsilon$ | $\mu \sim N(2.5, 1)$, $b = b_h$ |

* Điểm tin cậy của từng thành viên: $c_m = c + 0.6\,\eta_m$. Khoảng cách tiềm ẩn: $\zeta_m = \mu + b\,\xi_m$. Cả $\eta$ và $\xi$ có tương quan $\rho$ giữa các thành viên (thiên kiến chung).
* $r_{\text{cl}}$: tương quan âm giữa nhiễu tin cậy và nhiễu tiềm ẩn trong cùng loại, tức mức độ thông tin tiềm ẩn trùng với thông tin tin cậy.
* **Ba chế độ FP:** "hal" ($p_{\text{loc}} = 0.02$, $r_{\text{hal}} = 0.6$; ảo giác chiếm 82% số FP), "mixed" (0.06, 0.4; 50%), "loc" (0.14, 0.15; 14%).
* **Đại lượng quan sát được tương ứng với $\rho$:** tương quan giữa $\zeta_m$ và $\zeta_{m'}$ trên TP bằng $(0.36 + 0.64\rho)/1.0$, tức $\rho = 0.3$, $0.7$, $0.95$ tương ứng $\bar r_\zeta \approx 0.55$, $0.81$, $0.97$.

Mỗi lần lặp sinh một Meta-Path (3.500 đơn vị) để chọn siêu tham số và dựng đường đi, và một Evaluation pool (7.000 đơn vị) độc lập. Siêu tham số chọn theo quy tắc P-H2.3 (Coverage lớn nhất tại ngưỡng plug-in có Macro-FDP $\le 0.010$ trên Meta-Path, tối đa 5 giá trị mỗi thành phần). Lưới rủi ro 33 điểm của H1, $N = 1.500$, $\alpha = 0.02$, $\delta = 0.05$.

## 3. S1: so sánh các phương án tín hiệu (`s1_signal_design.py`, `summarize_s1.py`)

18 kịch bản × 4 lần lặp × 120 vòng B-block. Đơn vị: Coverage trung bình trên tập kiểm thử (trần Coverage 0.79–0.90 tùy chế độ FP).

| Kịch bản | B1 tin cậy | B2 ens. tin cậy | B4 Maha | v1.2 phương sai | A1 ens. Maha | A2 $\bar\zeta + \lambda s$ | B8 lai đơn | **Đề xuất lai ens.** | Đề xuất − B8 | B8 − B1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ρ=0.3, hal | 0.384 | 0.431 | 0.041 | 0.034 | 0.087 | 0.129 | 0.735 | **0.819** | +8.4 | +35.1 |
| ρ=0.3, mixed | 0.456 | 0.503 | 0.029 | 0.020 | 0.039 | 0.041 | 0.633 | **0.708** | +7.5 | +17.7 |
| ρ=0.3, loc | 0.459 | 0.516 | 0.016 | 0.008 | 0.018 | 0.018 | 0.504 | **0.567** | +6.3 | +4.5 |
| ρ=0.7, hal | 0.390 | 0.400 | 0.044 | 0.037 | 0.065 | 0.105 | 0.733 | **0.793** | +6.0 | +34.3 |
| ρ=0.7, mixed | 0.479 | 0.499 | 0.031 | 0.016 | 0.035 | 0.041 | 0.617 | **0.673** | +5.6 | +13.8 |
| ρ=0.7, loc | 0.460 | 0.486 | 0.015 | 0.008 | 0.017 | 0.016 | 0.504 | **0.532** | +2.8 | +4.4 |
| ρ=0.95, hal | 0.374 | 0.382 | 0.039 | 0.039 | 0.041 | 0.056 | 0.728 | **0.752** | +2.4 | +35.4 |
| ρ=0.95, mixed | 0.439 | 0.446 | 0.027 | 0.019 | 0.029 | 0.031 | 0.618 | **0.629** | +1.1 | +17.9 |
| ρ=0.95, loc | 0.452 | 0.452 | 0.014 | 0.008 | 0.014 | 0.015 | 0.488 | **0.491** | +0.3 | +3.6 |
| ρ=0.7, hal, δ_loc=1 | 0.389 | 0.397 | 0.075 | 0.035 | 0.127 | 0.314 | 0.772 | **0.838** | +6.6 | +38.3 |
| ρ=0.7, mixed, δ_loc=1 | 0.434 | 0.464 | 0.106 | 0.020 | 0.138 | 0.170 | 0.706 | **0.743** | +3.7 | +27.2 |
| ρ=0.7, loc, δ_loc=1 | 0.451 | 0.478 | 0.082 | 0.007 | 0.095 | 0.104 | 0.588 | **0.617** | +2.9 | +13.7 |
| ρ=0.7, hal, $b_h = b$ | 0.356 | 0.380 | 0.147 | 0.012 | 0.153 | 0.136 | 0.759 | **0.764** | +0.5 | +40.3 |
| ρ=0.7, mixed, $b_h = b$ | 0.478 | 0.497 | 0.041 | 0.010 | 0.043 | 0.042 | 0.654 | **0.668** | +1.4 | +17.6 |
| ρ=0.7, loc, $b_h = b$ | 0.461 | 0.477 | 0.017 | 0.008 | 0.018 | 0.018 | 0.510 | **0.530** | +2.0 | +4.9 |
| ρ=0.7, hal, $r_{\text{cl}}$=0.6 | 0.388 | 0.414 | 0.038 | 0.037 | 0.056 | 0.100 | 0.690 | **0.761** | +7.1 | +30.2 |
| ρ=0.7, mixed, $r_{\text{cl}}$=0.6 | 0.454 | 0.479 | 0.028 | 0.018 | 0.033 | 0.036 | 0.576 | **0.612** | +3.6 | +12.2 |
| ρ=0.7, loc, $r_{\text{cl}}$=0.6 | 0.462 | 0.480 | 0.016 | 0.008 | 0.016 | 0.018 | 0.465 | **0.487** | +2.2 | +0.3 |

Mặc định: $b_h = 1.6$, $\delta_{\text{loc}} = 0.25$, $r_{\text{cl}} = 0$. Hai cột cuối tính bằng pp. Toàn bộ output ở `h2_simulation/s1_output.txt`. Yield bằng 1.00 và VR không vượt 0.004 ở mọi tín hiệu và mọi kịch bản.

**Kết luận của S1 và quyết định thiết kế cho bản 1.4:**

1. **Tín hiệu chỉ dùng không gian tiềm ẩn không cạnh tranh được với điểm tin cậy** ở cả 18 kịch bản, kể cả khi ảo giác chiếm 82% số FP (Coverage 0.7–31% so với 36–48% của B1). Nguyên nhân: tại Macro-FDP khoảng 1%, một tín hiệu không thấy lỗi lệch vị trí phải dùng ngưỡng rất chặt. Kết quả này khớp với Jaeger et al. (2023). → **Bản 1.4 bỏ H2a dạng "tiềm ẩn thắng đầu ra"** và thay bằng "tiềm ẩn bổ sung cho đầu ra".
2. **Tín hiệu phương sai của bản 1.2 kém nhất ở mọi kịch bản** (0.7–3.9%). → Loại khỏi vai trò tín hiệu đề xuất; giữ làm ablation A3.
3. **Tín hiệu lai ensemble (đề xuất của bản 1.4) có Coverage cao nhất ở cả 18/18 kịch bản.** Lợi ích của thông tin tiềm ẩn (B8 − B1) là +0.3 đến +40 pp. Lợi ích này lớn khi ảo giác chiếm đa số, và gần bằng 0 khi lỗi lệch vị trí chiếm đa số đồng thời thông tin tiềm ẩn trùng với thông tin tin cậy.
4. **Hiệu ứng ensemble trên tín hiệu lai (đề xuất − B8) giảm theo $\rho$:** +6.3 đến +8.4 pp ở $\rho = 0.3$; +2.8 đến +6.0 pp ở $\rho = 0.7$; +0.3 đến +2.4 pp ở $\rho = 0.95$. Hiệu ứng chỉ vượt $\gamma = 5$ pp khi $\rho \le 0.7$ và ảo giác chiếm từ một nửa số FP trở lên. Khi các thành viên không bất đồng hơn trên ảo giác ($b_h = b$), hiệu ứng chỉ còn +0.5 đến +2.0 pp. → Đăng ký trước các dự đoán này trong P-H2.2 và khuyến nghị huấn luyện lại detector với seed thật sự khác nhau.
5. **$\lambda^*$ thay đổi theo kịch bản** (từ 0 đến 2), nên H2b′ là câu hỏi thực nghiệm. Giữ $\lambda$ trong lưới thay vì cố định.

## 4. S2: quy tắc quyết định (`s2_decision_rule.py`)

Cảnh ρ=0.7 mixed và các biến thể, 40 cặp pool độc lập cho mỗi trường hợp. Quy tắc bản 1.2: một lần chia cal/test, Wilcoxon một phía trên ΔTP theo đơn vị kiểm thử, chấp nhận khi $p < 0.0025$ và hiệu Coverage $\ge \gamma$. Quy tắc bản 1.3/1.4: ước lượng điểm trên 120 vòng B-block; bootstrap theo super-block với 600 lần lặp, mỗi lần 20 vòng; mức một phía $\alpha_{\text{each}} = 0.00625$; quy tắc ba vùng với $\gamma = 5$ pp.

| Trường hợp | Δ thật (pp) | SD điểm 1.3 / SD một lần chia 1.2 / SD bootstrap (pp) | 1.2: P(p<0.0025) | 1.2: P(chấp nhận) | 1.3: P(Đạt) / P(KKL) / P(Không đạt) | 1.3: P(LCB>0) |
|---|---|---|---|---|---|---|
| Đối chứng âm: B8 anchor 1 vs B8 anchor 2 | −0.17 (thật = 0) | 1.07 / 2.55 / 1.30 | **0.33** | 0.03 | 0.00 / 0.03 / 0.97 | **0.00** |
| Đề xuất vs B8, ρ=0.95 mixed | +1.51 | 0.80 / 1.89 / 0.73 | 0.75 | 0.07 | 0.00 / 0.05 / 0.95 | 0.57 |
| Đề xuất vs B8, ρ=0.7 loc | +2.58 | 0.74 / 2.29 / 0.87 | 0.85 | 0.07 | 0.00 / 0.38 / 0.62 | 0.78 |
| Đề xuất vs B8, ρ=0.7 mixed | +4.47 | 1.31 / 2.58 / 1.09 | 0.97 | 0.33 | 0.20 / 0.80 / 0.00 | 0.97 |
| Đề xuất vs B8, ρ=0.3 hal | +8.82 | 1.17 / 3.10 / 1.29 | 1.00 | 0.95 | 1.00 / 0.00 / 0.00 | 1.00 |
| B8 vs B1, ρ=0.7 mixed | +17.01 | 2.26 / 5.82 / 2.59 | 1.00 | 0.97 | 1.00 / 0.00 / 0.00 | 1.00 |

*KKL: Không kết luận. Δ thật xấp xỉ bằng trung bình ước lượng điểm qua 40 pool. Output đầy đủ ở `h2_simulation/s2_output.txt`.*

**Kết luận của S2:**

1. **Wilcoxon theo đơn vị (bản 1.2) không kiểm soát được sai lầm loại I.** Khi Δ thật bằng 0, $p < 0.0025$ xảy ra ở 33% số lần, gấp khoảng 130 lần mức danh nghĩa. Nguyên nhân: Wilcoxon coi hai ngưỡng $\hat\tau$ của một lần hiệu chuẩn là cố định, nên nó kiểm tra "với **hai ngưỡng này**, tín hiệu nào cho nhiều TP hơn" chứ không kiểm tra hiệu kỳ vọng qua các lần hiệu chuẩn. Với khoảng 5.500 đơn vị kiểm thử, mọi khác biệt ngẫu nhiên do hiệu chuẩn đều trở thành "có ý nghĩa". Bản 1.2 không chấp nhận sai thường xuyên hơn chỉ nhờ điều kiện $\Delta \ge \gamma$.
2. **Một lần chia cho ước lượng nhiễu hơn 2–2.6 lần** so với trung bình qua 120 vòng (cột SD). Lấy trung bình qua các vòng B-block là cải thiện đáng kể.
3. **Quy tắc bản 1.3/1.4 kiểm soát được sai lầm loại I:** khi Δ = 0, P(LCB > 0) = 0. Khi Δ thật nhỏ hơn γ một cách rõ rệt (1.5–2.6 pp), quy tắc không bao giờ cho "Đạt", và cho "Không đạt" ở 62–95% số lần.
4. **SD bootstrap gần với SD thật** của ước lượng điểm qua các pool độc lập. Lệch không quá khoảng 20%, phần lớn theo hướng bảo thủ; một trường hợp (ρ=0.7 mixed) thấp hơn 17%.
5. **Giới hạn của quy tắc ba vùng:** khi Δ thật **ngay dưới** γ (4.5 pp so với 5 pp), "Đạt" xảy ra ở 20% số lần, vì "Đạt" chỉ yêu cầu ước lượng điểm $\ge \gamma$ cộng với $\text{LCB} > 0$. Nói cách khác, quy tắc kiểm soát sai lầm cho giả thuyết không $\Delta \le 0$, không phải cho $\Delta \le \gamma$. Phương án chặt hơn là yêu cầu $\text{LCB} > \gamma$; với SE khoảng 1.1–1.3 pp, phương án này cần Δ thật khoảng 8 pp để có năng lực 50%. Bản 1.4 giữ quy tắc hiện tại và ghi rõ giới hạn này (H2 Mục IX). Nếu nhóm muốn "Đạt" có nghĩa là "chắc chắn vượt γ", hãy đổi sang $\text{LCB} > \gamma$ trước khi đăng ký.
6. **Hệ quả cho dự đoán H2b:** với các mức Δ của S1 ở ρ ≥ 0.7 (0.3–6 pp), H2b gần như không bao giờ "Đạt" trừ trường hợp ρ=0.7 với ảo giác chiếm đa số.

## 5. Giới hạn

* Mô hình sinh dữ liệu là giả định của chúng tôi. Điều được kiểm chứng là **chiều** và **cơ chế** (ví dụ: tín hiệu không thấy lỗi lệch vị trí thì thua ở FDP thấp; hiệu ứng ensemble giảm theo $\rho$), không phải các con số tuyệt đối.
* Mô phỏng không có lỗi trùng lặp, không có bất đồng về hộp giữa các thành viên (B2 trong mô phỏng chỉ dùng điểm tin cậy trung bình), không có B3, B5, B6, B7.
* Đặc trưng tiềm ẩn được mô phỏng trực tiếp ở dạng khoảng cách đã chuẩn hóa $\zeta$; mô phỏng không kiểm tra việc ước lượng hiệp phương sai, Ledoit-Wolf hay PCA.
* Trong mô phỏng, tham số được chuẩn hóa robust trên TP của Meta-Path, thay cho Meta-Fit.

## 6. Cách chạy lại

```
pip install numpy scipy
cd h2_simulation
python s1_signal_design.py > s1_output.txt   # khoảng 1.5 phút
python summarize_s1.py
python s2_decision_rule.py > s2_output.txt   # khoảng 20 phút
```
Đặt `PYTHONIOENCODING=utf-8` nếu console không in được tiếng Việt.
