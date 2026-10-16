# Báo cáo mô phỏng edge case cho Giả thuyết H1 (bản 1.1 → 1.2)

Ngày chạy: 13/10/2026. Code đi kèm trong `h1_simulation_code.zip` (Python, numpy, scipy); mỗi script in ra đúng các số trong báo cáo này.

## 1. Mục đích và phạm vi

Chưa có dữ liệu BDD100K đã xử lý và detector đã huấn luyện, nên báo cáo này **không** đo kết quả thật của H1. Nó chạy đúng các quy trình trong tài liệu giả thuyết (p-value Hoeffding-Bentkus, Fixed-Sequence LTT, Exp B, Exp C, quy tắc R-unit, ride proxy) trên dữ liệu tổng hợp, dưới các kịch bản biên, để trả lời: **thiết kế có hoạt động như khẳng định không, và có chỗ nào khiến thực nghiệm thật cho kết luận sai không?**

Dữ liệu tổng hợp giữ các đặc điểm quan trọng của FDP thật: phần lớn đơn vị có FDP = 0 tại ngưỡng chặt, số box thay đổi theo ngưỡng, độ khó khác nhau giữa các chuyến, tương quan theo super-block (điều chỉnh được), và tự tương quan theo thời gian giữa các frame. Rủi ro dọc đường đi được hiệu chỉnh để khớp đúng lưới của bản 1.1 (sai lệch dưới $10^{-5}$).

## 2. Tóm tắt kết quả

### Bốn lỗi nghiêm trọng mới (bản 1.1), đã sửa ở bản 1.2

| # | Lỗi | Bằng chứng | Sửa ở bản 1.2 |
|---|---|---|---|
| 1 | **C2 không kiểm được A1.** HB bảo thủ tới mức VR vẫn dưới δ dù A1 bị vi phạm nặng | C2 chấp nhận H1a ở mọi kịch bản tới $D_{\text{eff}} \approx 3$; chỉ rơi vào Không kết luận ở $D_{\text{eff}} \approx 4.8$ (E4). Quy trình lỗi phóng đại cỡ mẫu 32 lần chỉ đạt VR = 0.033 (E2), nên bước MDV của bản 1.1 cho kết quả không xác định | Thêm C2b: design effect thực nghiệm (tỉ số phương sai B-block/B-rand); thêm Mục 5.6 về biên chịu đựng $D^*$; thay bước MDV |
| 2 | **Tiêu chí $e^{\text{cons}}$ trên phần bù làm mất tín hiệu.** C5 không chấp nhận được H1b khi Arm 2 vi phạm rõ | Arm 2 vi phạm thật 14.2% nhưng $\widehat{\text{VR}}^{\text{cons}}$ chỉ 1.4% (E3, N = 400) | Vi phạm đo bằng rủi ro chính xác trên toàn pool (Exp B và C) |
| 3 | **C4 đặt trên ICC ≥ 0.30, sai đại lượng.** ICC của FDP theo frame nhỏ do nhiễu nhị thức dù design effect lớn | ICC 0.04–0.21 trong khi $D_{\text{eff}}$ = 9–42; theo tiêu chí cũ H1b Không kết luận ở cả ba kịch bản dù Arm 2 vi phạm 14–35% (E8) | C4 đặt trên $D_{\text{eff}}^{\text{frame}}$ (chấp nhận khi cận dưới CI ≥ $\max(5, \hat D^*_C)$) |
| 4 | **Quy tắc R-unit gộp đơn vị thô sai rất thường xuyên** khi A1 đúng | Ngưỡng 1.2: từ bỏ đơn vị ride trong 27–85% số lần với $\tilde m_{\text{Val}} \ge 3$ và Meta-Train 1.000–2.000 chuyến (E5, E10) | Ngưỡng $\min(2, \hat D^*)$ (còn 0–28%), Meta-Train 10% super-block |

Ngoài ra có một **giới hạn được định lượng**: nếu các clip của cùng một chuyến cách nhau hơn 10 phút, khoảng 20–25% clip train dính rò rỉ vẫn lọt qua quy tắc ride proxy (E6). Bản 1.2 nêu rõ giới hạn này và thêm phân tích độ nhạy D8.

### Những gì được xác nhận đúng

* p-value HB hợp lệ ở mọi kịch bản thử, kể cả trường hợp xấu nhất $Z$ nhị phân và lấy mẫu không hoàn lại từ pool hữu hạn (E1).
* Mốc $N \ge 149$ đúng; bảng 5.4 và 5.5 được tái tạo chính xác.
* **Việc kéo dài đường đi ở bản 1.1 là bắt buộc.** Với đường đi cũ, VR luôn bằng 0 bất kể A1 (E2), và trong Exp C, Arm 2 có VR = 0 ở mọi kịch bản, rơi đúng vào điều kiện **Bác bỏ** của C5. Thiết kế cũ sẽ bác bỏ nhầm H1b (E3).
* Phần kéo dài không làm thay đổi $\hat\tau$ ở bất kỳ vòng nào dừng trong phần lõi, và Yield giữ nguyên (E2, kiểm trên 24.000 vòng).
* Gộp đơn vị lên super-block khôi phục tính hợp lệ khi A1 sai (E2).
* Arm 1 (hiệu chuẩn theo clip) luôn hợp lệ; khi frame độc lập, Arm 2 cũng hợp lệ, nên tiêu chí mới không chấp nhận H1b một cách sai (E9, đối chứng âm).

## 3. Kết quả chi tiết

### E1. Tính hợp lệ của p-value HB (`e1_hb_validity.py`, `e1d_bernoulli_pool.py`)

Xác suất bác bỏ dưới $H_0$ khi rủi ro thật $R = \alpha(1 + \varepsilon)$, 40.000 lần lặp (giới hạn hợp lệ 0.05):

| Phân phối của $Z$ | $N = 149$, $\varepsilon = 0.001$ | $N = 400$ | $N = 1.500$ |
|---|---|---|---|
| Nhị phân (xấu nhất) | 0.0486 | 0.0134 | 0.0127 |
| FDP 15 box/đơn vị | 0.0000 | 0.0000 | 0.0000 |
| FDP số box thay đổi | 0.0000 | 0.0000 | 0.0000 |
| Lệch, 90% bằng 0 | 0.0000 | 0.0000 | 0.0000 |

* Làm tròn $\lceil Ns \rceil$ và $\lfloor Ns \rfloor$ cho kết quả gần như trùng nhau (chênh lệch dưới 0.002), đều hợp lệ.
* Lấy mẫu không hoàn lại 1.500 từ pool 7.000 (pool là quần thể), $Z$ nhị phân, mean pool 0.02014: xác suất bác bỏ 0.0052, thấp hơn i.i.d. (0.0112). Đây là cơ sở để dùng rủi ro trên toàn pool làm tiêu chí vi phạm.
* **Hệ quả quan trọng:** với $Z$ dạng FDP thật, HB gần như không bao giờ bác bỏ khi rủi ro chỉ vừa vượt α. Độ bảo thủ này tạo biên chịu đựng tương quan (E7) và là nguyên nhân của lỗi số 1.

### E2. LTT đầu-cuối, có và không có phần kéo dài (`e2_ltt_path.py`)

$N = 1.500$, lấy mẫu theo trọn super-block, 3.000 vòng mỗi kịch bản. VR = P(rủi ro thật của $\hat\tau$ > α).

| $\tilde m$ | $\sigma_b$ | $D_{\text{eff}}$ | VR (kéo dài) | VR (chỉ lõi) | Yield | Vòng vượt lõi |
|---|---|---|---|---|---|---|
| 3 | 0 | 0.99 | 0.0000 | 0.0000 | 1.000 | 0.3% |
| 3 | 2 | 1.58 | 0.0000 | 0.0000 | 1.000 | 8.0% |
| 3 | 3 | 2.09 | 0.0007 | 0.0000 | 1.000 | 14.6% |
| 8 | 1 | 1.41 | 0.0000 | 0.0000 | 1.000 | 2.9% |
| 8 | 2 | 3.06 | 0.0017 | 0.0000 | 1.000 | 16.2% |
| 8 | 3 | 4.72 | 0.0290 | 0.0000 | 1.000 | 27.1% |

* Đường đi chỉ có lõi cho VR = 0 ở mọi kịch bản: không thí nghiệm nào có thể phát hiện vi phạm.
* $\hat\tau$ giống hệt nhau giữa hai đường đi ở mọi vòng dừng trong lõi (khẳng định ở Mục 6.2 đúng).
* Quy trình lỗi phóng đại cỡ mẫu $c$ lần (A1 đúng): VR thật = 0, 0, 0, 0.0015, 0.0090, 0.0330 với $c$ = 1, 2, 4, 8, 16, 32.
* Gộp đơn vị lên super-block (600 nhóm) khi A1 sai nặng: VR = 0.

### E3 và E8. Experiment C (`e3_expC.py`, `e8_decisions.py`)

1.600 clip × 200 frame, 1.000 vòng mỗi mức. VR đo bằng rủi ro trên toàn bộ MOT ($\text{VR}_{\text{pool}}$), trên phần bù ($\text{VR}$), và cận dưới trên phần bù ($\text{VR}_{\text{cons}}$, tiêu chí bản 1.1).

| $\sigma_u$ | ICC frame | $D_{\text{eff}}$ | N | Arm 1 VR | Arm 2 $\text{VR}_{\text{pool}}$ | Arm 2 VR | Arm 2 $\text{VR}_{\text{cons}}$ |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.038 | 8.6 | 100 / 200 / 400 | 0 / 0 / 0 | 0.187 / 0.163 / 0.142 | 0.187 / 0.163 / 0.142 | 0.075 / 0.041 / 0.014 |
| 0.7 | 0.085 | 17.9 | 100 / 200 / 400 | 0 / 0 / 0 | 0.269 / 0.258 / 0.239 | 0.269 / 0.258 / 0.239 | 0.096 / 0.135 / 0.069 |
| 1.2 | 0.206 | 42.0 | 100 / 200 / 400 | 0 / 0 / 0 | 0.274 / 0.227 / 0.171 | 0.349 / 0.343 / 0.327 | 0.207 / 0.150 / 0.082 |

* **Đường đi cũ (dừng ở 0.095):** mọi arm có VR = 0 ở mọi kịch bản. C5 cũ sẽ **Bác bỏ H1b** dù cơ chế có thật.
* Arm 1 Yield = 0 ở $N = 20$ (đúng như bảng 5.5) và = 1 ở $N \ge 100$.
* VR trên phần bù bị thổi phồng khi dị biệt lớn (0.327 so với 0.171 ở $\sigma_u = 1.2$, $N = 400$), do hiệu ứng phần bù.
* $\Delta_{\text{leak}}$ của Arm 3 nhỏ (tối đa +0.0045), đúng với dự đoán "độ lớn khiêm tốn".

Quyết định H1b theo hai bộ tiêu chí (CI Clopper-Pearson, vì mỗi vòng E3 là một phân hoạch độc lập):

| $\sigma_u$ | Bản 1.1 (C4: ICC ≥ 0.30; C5: $\text{VR}_{\text{cons}}$) | Bản 1.2 (C4: $D_{\text{eff}} \ge \max(5, D^*_C)$; C5: $\text{VR}_{\text{pool}}$) |
|---|---|---|
| 0.3 | C4 Bác bỏ, C5 Không kết luận → **Không kết luận** | C4, C5 Chấp nhận → **Chấp nhận** |
| 0.7 | C4 Bác bỏ, C5 Chấp nhận → **Không kết luận** | **Chấp nhận** |
| 1.2 | C4 Không kết luận, C5 Chấp nhận → **Không kết luận** | **Chấp nhận** |

### E9. Đối chứng âm cho Exp C (`e9_negative_control.py`)

Frame độc lập (không hiệu ứng clip, không tự tương quan): ICC = −0.0001, $D_{\text{eff}}$ = 0.99; Arm 1, Arm 2, Arm 3 đều có $\text{VR}_{\text{pool}}$ = 0 ở $N$ = 100 và 400. Tiêu chí bản 1.2 không chấp nhận H1b; C5 rơi vào vùng Bác bỏ, đúng như mong muốn.

### E4. Experiment B và hai phép kiểm A1 (`e4_expB.py`)

Pool 7.000 chuyến, 334 phân hoạch × 3 = 1.002 vòng mỗi nhánh.

| $\tilde m$ | $\sigma_b$ | $D_{\text{eff}}$ | HB $\text{VR}_{\text{pool}}$ | C2 (bản 1.1) | C2 (pool) | Tỉ số phương sai [CI] | CLT-LTT B-block / B-rand |
|---|---|---|---|---|---|---|---|
| 3 | 0 | 0.99 | 0.000 | Chấp nhận | Chấp nhận | 0.85 [0.75, 0.96] | 0.032 / 0.029 |
| 3 | 2 | 1.57 | 0.000 | Chấp nhận | Chấp nhận | 1.63 [1.44, 1.84] | 0.061 / 0.026 |
| 3 | 3 | 2.04 | 0.000 | Chấp nhận | Chấp nhận | 1.37 [1.21, 1.56] | 0.095 / 0.030 |
| 8 | 0 | 1.02 | 0.000 | Chấp nhận | Chấp nhận | 1.09 [0.96, 1.26] | 0.031 / 0.021 |
| 8 | 1 | 1.47 | 0.000 | Chấp nhận | Chấp nhận | 1.59 [1.39, 1.82] | 0.033 / 0.001 |
| 8 | 2 | 3.03 | 0.003 | Chấp nhận | Chấp nhận | 4.10 [3.66, 4.64] | 0.226 / 0.015 |
| 8 | 3 | 4.82 | 0.031 | Không kết luận | Chấp nhận | 3.54 [3.12, 3.97] | 0.215 / 0.038 |

* C2 (cả hai phiên bản) chấp nhận H1a dù A1 sai nặng: C2 chỉ kiểm "bảo chứng không sụp đổ".
* Tỉ số phương sai phát hiện rõ vi phạm A1, nên được chọn làm C2b. LTT dùng cận trên CLT (không bảo thủ) cũng phát hiện được nhưng kém nhạy hơn, nên không được dùng.

### E5 và E10. Quy tắc R-unit và báo động giả của C2b (`e5_runit_varratio.py`, `e10_thresholds.py`)

Tỉ lệ quy tắc **không** giữ được đơn vị ride khi A1 hoàn toàn đúng, ngưỡng 1.2 → 2.0 (60 lần rút Meta-Train mỗi ô):

| $\tilde m_{\text{Meta}}$ | Cỡ Meta-Train | $\tilde m_{\text{Val}} = 2$ | $\tilde m_{\text{Val}} = 3$ | $\tilde m_{\text{Val}} = 5$ |
|---|---|---|---|---|
| 3 | 1.000 | 0.48 → 0.00 | 0.73 → 0.03 | 0.80 → 0.28 |
| 3 | 2.000 | 0.25 → 0.00 | 0.70 → 0.03 | 0.85 → 0.15 |
| 3 | 4.000 | 0.08 → 0.00 | 0.55 → 0.00 | 0.73 → 0.03 |
| 8 | 1.000 | 0.10 → 0.00 | 0.38 → 0.00 | 0.65 → 0.05 |
| 8 | 2.000 | 0.00 → 0.00 | 0.27 → 0.00 | 0.63 → 0.00 |
| 8 | 4.000 | 0.02 → 0.00 | 0.15 → 0.00 | 0.48 → 0.02 |

Tỉ số phương sai B-block/B-rand khi A1 đúng (qua nhiều pool) và khi A1 sai:

| Kịch bản | Tại $\alpha_k = 0.008$ | Tại $\alpha_k = 0.020$ |
|---|---|---|
| A1 đúng, $\tilde m = 3$ (20 pool tại 0.008, 8 pool tại 0.020) | 0.92–1.18 | 0.89–1.10 |
| A1 đúng, $\tilde m = 8$ (20 pool tại 0.008, 8 pool tại 0.020) | 0.87–1.15 | 0.89–1.12 |
| A1 sai, $\tilde m = 3$, $\sigma_b = 2$ | 1.41–1.65 | 2.06–2.13 |
| A1 sai, $\tilde m = 8$, $\sigma_b = 1$ | 1.25–1.83 | 1.90–2.71 |
| A1 sai, $\tilde m = 8$, $\sigma_b = 2$ | 2.46–2.75 | 3.48–4.53 |

Tại $\alpha_k = 0.020$ phép kiểm nhạy hơn, nên C2b được đặt tại điểm này. Mốc quyết định (bác bỏ A1 khi cận dưới CI > 1.3) không bị vượt ở bất kỳ pool nào khi A1 đúng.

### E7. Biên chịu đựng tương quan $D^*$ (`e7_tolerance.py`)

| $\tilde m$ | $\sigma_b$ | $\text{Var}(Z)/[\alpha(1-\alpha)]$ | $D_{\text{eff}}$ | $D^*$ | VR |
|---|---|---|---|---|---|
| 8 | 3 | 0.41 | 5.95 | 4.38 | 0.025 |
| 8 | 4 | 0.52 | 6.60 | 3.50 | 0.056 |
| 16 | 4 | 0.54 | 13.10 | 3.33 | 0.161 |
| 16 | 6 | 0.68 | 14.00 | 2.67 | 0.071 (Yield 0.07) |

Ở cấp chuyến đi, VR chỉ vượt δ khi $D_{\text{eff}}$ khoảng 1.9 lần $D^*$. Dòng cuối có Yield rất thấp vì dị biệt quá lớn khiến quy trình hầu như luôn trả về rỗng.

Nhưng trong Exp C, biên chịu đựng của Arm 2 tính trực tiếp từ mô phỏng là $D^*_C$ = 6.47, 5.81, 4.75 (với $\sigma_u$ = 0.3, 0.7, 1.2; không đổi giữa $N_{\text{cluster}}$ = 100 và 400), và Arm 2 đã vi phạm 14–19% ở $\sigma_u = 0.3$, khi $D_{\text{eff}}$ = 8.6 chỉ bằng khoảng 1.3 lần $D^*_C$. Vậy $D^*$ đúng chiều nhưng **không bảo thủ đồng đều**: nó là thước đo cỡ độ lớn, và bản 1.2 chỉ dùng nó kèm biên an toàn (ngưỡng $\min(2, \hat D^*)$).

### E6. Quy tắc ride proxy (`e6_ride_proxy.py`)

Thành phố tổng hợp 40 × 40 km, 60 ngày, mỗi chuyến 1–3 clip; clip gán ngẫu nhiên vào Val với xác suất 1/8.

| Lấy clip trong chuyến | Chuyến/ngày | Thành phần liên thông: lớn nhất / bắt rò rỉ / loại train | Chuỗi + vùng chắn: lớn nhất / bắt rò rỉ / loại train |
|---|---|---|---|
| Ngẫu nhiên | 20 | 7 / 0.78 / 0.136 | 6 / 0.76 / 0.129 |
| Ngẫu nhiên | 200 | 33 / 0.84 / 0.344 | 11 / 0.80 / 0.259 |
| Ngẫu nhiên | 1.000 | 3.749 / 0.98 / 0.939 | 34 / 0.94 / 0.756 |
| Liên tiếp | 20 | 7 / 1.00 / 0.151 | 5 / 1.00 / 0.151 |
| Liên tiếp | 200 | 26 / 1.00 / 0.279 | 12 / 1.00 / 0.248 |
| Liên tiếp | 1.000 | 675 / 1.00 / 0.875 | 38 / 1.00 / 0.688 |

* Thành phần liên thông bùng nổ ở mật độ cao (một "chuyến" 3.749 clip, loại 94% tập train); quy tắc chuỗi giới hạn được kích thước.
* Khi clip lấy ở vị trí ngẫu nhiên trong chuyến (khoảng cách có thể vượt 10 phút), các quy tắc chỉ bắt được 70–84% rò rỉ ở mật độ 20–200 chuyến/ngày (chuỗi kèm vùng chắn: 76–80%).
* Phần train bị loại ở mật độ 20–200 chuyến/ngày là 13–26%, cao hơn ước tính 10–15% trước đây.

## 4. Giới hạn của mô phỏng

* Mô hình sinh dữ liệu là giả định của chúng tôi. Các con số tuyệt đối (VR, $D^*$, tỉ lệ gộp thô sai) phụ thuộc mô hình; điều được kiểm chứng là **chiều** và **cơ chế** của các hiện tượng, cùng việc các tiêu chí có phản ứng đúng hay không.
* Mật độ chuyến đi và cách BDD100K cắt clip trong một chuyến chưa biết, nên E6 quét nhiều mức.
* Trong E3, mỗi vòng là một phân hoạch độc lập (không cắt $K$ tập mỗi phân hoạch như tài liệu), nên dùng Clopper-Pearson thay cho bootstrap theo phân hoạch; điều này không ảnh hưởng ước lượng VR.
* Bootstrap theo cụm được thay bằng xấp xỉ chuẩn theo cụm để giảm chi phí.

## 5. Cách chạy lại

```
pip install numpy scipy
python3 h1core.py                 # tự kiểm: tái tạo bảng 5.4, 5.5
python3 e1_hb_validity.py         # ~1.5 phút
python3 e1d_bernoulli_pool.py
python3 e2_ltt_path.py            # ~5 phút
python3 e3_expC.py                # ~3.5 phút
python3 e4_expB.py                # ~3.5 phút
python3 e5_runit_varratio.py      # ~2 phút
python3 e6_ride_proxy.py          # ~0.5 phút
python3 e7_tolerance.py           # ~2 phút
python3 e8_decisions.py
python3 e9_negative_control.py    # ~1 phút
python3 e10_thresholds.py         # ~3 phút
```

Thời gian đo trên máy 2 nhân. Mỗi script có seed cố định nên kết quả lặp lại được.
