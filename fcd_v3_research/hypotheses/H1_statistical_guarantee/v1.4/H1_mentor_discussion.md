# H1 — Kịch bản thảo luận với mentor

*Tài liệu đi kèm: `H1_statistical_guarantee_research.md` (thiết kế đầy đủ, công thức, chứng minh) và `H1_real_data_report.md` (số liệu). Bản này chỉ giữ mạch lập luận. SV = sinh viên, M = mentor.*

---

## Phần 1. Vấn đề

**SV:** Nhóm em đang làm pipeline gán nhãn giả cho phát hiện xe trên BDD100K. Detector tự gán nhãn cho ảnh mới, box nào đủ chắc thì đưa thẳng vào tập huấn luyện vòng sau. Vấn đề là box sai cũng lọt vào, mô hình học lại chính lỗi của nó, và lỗi tích lũy qua các vòng.

**M:** Thì đặt ngưỡng confidence cao, ví dụ 0,95?

**SV:** Đó là cách mọi người hay làm, nhưng con số 0,95 không đi kèm lời hứa nào. Không ai trả lời được "với ngưỡng này, bao nhiêu phần trăm box tự gán là sai". Trên dữ liệu của em, detector hầu như không cho điểm nào giữa 0,05 và 0,5, nên đổi ngưỡng từ 0,5 lên 0,95 gần như không thay đổi gì, và ta cũng không biết nó an toàn đến đâu.

**M:** Vậy em muốn gì?

**SV:** Em muốn chọn ngưỡng kèm một **lời hứa kiểm chứng được**: *"trong số box được tự gán nhãn, tỉ lệ sai trung bình không quá 2%, và lời hứa này đúng với độ tin cậy 95%"*. Box qua ngưỡng vào vùng xanh (tự gán), phần còn lại vào vùng xám hoặc đỏ (người duyệt hoặc loại).

---

## Phần 2. Giải pháp: Learn-then-Test

**M:** Có phương pháp nào làm được vậy chưa?

**SV:** Có, Learn-then-Test (LTT) của Angelopoulos và cộng sự. Ý tưởng là biến việc chọn ngưỡng thành một chuỗi kiểm định thống kê. Em xếp các ngưỡng từ chặt đến lỏng thành một đường đi, dựng trước trên một tập dữ liệu riêng. Rồi trên tập hiệu chuẩn, em đi dọc đường đi: ở mỗi ngưỡng, em hỏi "dữ liệu có đủ bằng chứng rằng tỉ lệ sai ở đây dưới 2% không?". Đủ thì đi tiếp, không đủ thì dừng ngay. Ngưỡng cuối cùng qua được là ngưỡng được chọn.

**M:** Vì sao phải dừng ngay mà không thử tiếp?

**SV:** Vì chính việc dừng ngay làm cho cả quá trình chỉ "đặt cược" một lần. Muốn chọn nhầm một ngưỡng không an toàn, thuật toán phải vượt qua ngưỡng không an toàn *đầu tiên* trên đường đi, và xác suất đó tối đa 5%. Nhờ vậy không cần hiệu chỉnh đa kiểm định. Đây là định lý đã được chứng minh trong bài LTT; em chỉ trích dẫn.

**M:** Ba ngưỡng của em là gì?

**SV:** Một box được chấp nhận khi qua cả ba điều kiện: điểm tin cậy đủ cao; ba detector trong ensemble đồng ý về vị trí box (IoU giữa chúng đủ lớn); và biểu diễn bên trong của chúng đủ giống nhau. Ba ngưỡng này được gộp thành một đường đi một chiều theo biên Pareto giữa độ phủ và tỉ lệ sai, dựng trên tập Meta-Train.

---

## Phần 3. Vấn đề sâu hơn: dữ liệu video không độc lập

**M:** LTT đã có rồi. Áp vào BDD100K thì em đóng góp gì?

**SV:** Lời hứa của LTT chỉ đúng khi dữ liệu hiệu chuẩn là các mẫu **độc lập**. Dữ liệu lái xe không như vậy, ở ba tầng:
1. **Các frame trong một video** rất giống nhau.
2. **Các clip trong cùng một chuyến đi** cùng tuyến đường, cùng ánh sáng.
3. **Các chuyến đi cùng khu vực, cùng ngày** cùng thời tiết, cùng kiểu đường.

Nếu coi dữ liệu tương quan là độc lập, ta tưởng mình có nhiều bằng chứng hơn thực tế, và lời hứa 2% có thể sai mà không ai biết. Câu hỏi nghiên cứu của H1 là: **áp LTT vào video lái xe thì lời hứa còn đứng không, và phải làm gì để nó đứng?**

**M:** Cụ thể em tách thành những giả thuyết nào?

**SV:** Ba giả thuyết:
- **H1a:** nếu chọn đúng đơn vị thống kê, lời hứa đứng vững, kể cả khi dùng ở khu vực-ngày chưa có trong tập hiệu chuẩn.
- **H1b:** nếu chia dữ liệu theo frame, coi mỗi frame là một mẫu, lời hứa sụp đổ. Đây là lỗi hay gặp khi làm với video, và là **đóng góp chính** của em.
- **H1c:** lời hứa không rỗng: quy trình thực sự chọn được ngưỡng và giữ được một lượng nhãn giả đáng kể.

**M:** Đơn vị thống kê của em là gì?

**SV:** Là **chuyến đi**. Ban đầu em suy ra từ tên video, vì 8 ký tự đầu của tên lặp lại giữa các clip. Sau đó em tải file metadata của BDD100K, trong đó có trường `rideID`. Kết quả: trên 99.999 video, 8 ký tự đầu của tên khớp đúng với `rideID`, tổng cộng 60.137 chuyến đi. Em còn phát hiện BDD100K chia train/val **theo chuyến đi**: hai tập chỉ chung đúng 1 chuyến, trong khi nếu chia ngẫu nhiên theo video thì khoảng 49% chuyến đi của Val sẽ có clip nằm trong train.

**M:** Còn khu vực-ngày?

**SV:** Từ GPS và giờ ghi hình, em gom các chuyến đi vào ô 5 km × 5 km × một ngày theo giờ địa phương, gọi là super-block. Dữ liệu nằm ở New York (79%), vùng vịnh San Francisco (12%) và Tel Aviv (9%), toàn bộ ghi năm 2017.

---

## Phần 4. Kiểm thử: làm sao biết lời hứa đứng vững?

**M:** Em kiểm chứng lời hứa thế nào? Không thể chỉ chạy một lần rồi bảo "rủi ro dưới 2%".

**SV:** Đúng ạ. Lời hứa nói về xác suất trên *nhiều lần* hiệu chuẩn, nên em lặp lại thí nghiệm khoảng 1.000 lần: mỗi lần rút một tập hiệu chuẩn 1.500 chuyến đi từ pool đánh giá, chạy LTT, rồi xem rủi ro thật của ngưỡng được chọn trên toàn pool có vượt 2% không. Tỉ lệ số lần vượt phải dưới 5%.

**M:** Nghe hợp lý. Có điểm yếu nào không?

**SV:** Có một điểm em mới nhận ra, và nó làm thay đổi cách em diễn giải kết quả cũ. Nếu rút tập hiệu chuẩn **ngẫu nhiên theo từng chuyến đi** từ một pool cố định, thì việc không vi phạm là **định lý**: nó đúng kể cả khi các chuyến đi tương quan với nhau. Phép thử đó chỉ kiểm tra được rằng code chạy đúng.

**M:** Vậy phép thử thật sự là gì?

**SV:** Là rút tập hiệu chuẩn theo **trọn khu vực-ngày**. Ngưỡng được hiệu chuẩn ở một số khu vực-ngày rồi dùng ở những nơi, những ngày khác, giống hệt lúc triển khai thật. Nếu các chuyến đi cùng khu vực-ngày tương quan mà ta lại coi chúng độc lập, phép thử này sẽ lộ ra. Em kèm thêm một phép đo trực tiếp: so độ dao động của kết quả hiệu chuẩn khi rút theo khu vực-ngày với khi rút theo chuyến đi. Nếu tỉ số xấp xỉ 1 thì không có tương quan đáng kể; nếu lớn hơn rõ rệt thì có.

**M:** Khi em chạy thử, có gì bất ngờ không?

**SV:** Có ba bất ngờ, và đều dẫn tới sửa thiết kế.

1. **Lưới ngưỡng ban đầu không đủ rộng.** Detector của nhóm rất thận trọng: mỗi ảnh chỉ có 2–3 box với điểm từ 0,05 trở lên, trong khi có tới 11–13 box với điểm từ 0,005 trở lên. Với ngưỡng thấp nhất là 0,05, tỉ lệ sai cao nhất trên cả lưới chỉ 1,4–2,2%. Nghĩa là trên đường đi gần như *không có ngưỡng nào không an toàn* ở mức 2%, và hoàn toàn không có ở mức 10% dùng cho H1b, nên phép thử không thể thất bại dù quy trình có sai. Em thêm quy tắc mở rộng lưới xuống cho tới khi chạm vùng rủi ro cao.

2. **Lưới thô tạo khoảng nhảy.** Với một detector, rủi ro trên đường đi nhảy thẳng từ 1,6% lên 2,9% quanh mốc 2%. Khi đó, ngay cả một quy trình *không có bảo chứng* cũng chỉ vi phạm 0,1%, tức thí nghiệm "chấp nhận" mà không kiểm tra được gì. Em làm mịn lưới (bước 0,0005) và thêm một điều kiện bắt buộc: **đối chứng dương**.

3. **Đối chứng dương.** Trên cùng dữ liệu, em chạy song song một quy trình "ăn gian": chọn ngưỡng lỏng nhất mà tỉ lệ sai đo được trên tập hiệu chuẩn dưới 2%, không có biên an toàn. Quy trình này vi phạm 12–48% số lần, tùy detector. Chỉ khi phép thử bắt được quy trình ăn gian thì việc LTT không vi phạm mới có ý nghĩa.

**M:** Còn đối chứng âm?

**SV:** Có ạ. Em chạy lại toàn bộ giao thức của H1b trên các cặp ảnh cùng chuyến đi. Hóa ra hai ảnh cùng chuyến gần như độc lập về tỉ lệ sai (tương quan nội cụm gần 0), vì các clip cùng chuyến cách nhau trung vị khoảng 21 phút. Kết quả: quy trình coi ảnh là độc lập *không* vi phạm, dù quy trình ăn gian trên cùng dữ liệu vi phạm 26–44%, và tiêu chí của em không chấp nhận nhầm H1b. Tức là khi không có tương quan, phép thử không báo động giả.

---

## Phần 5. Kết quả đến hiện tại

**M:** Tóm lại em đã chứng minh được gì?

**SV:** Em chia làm ba loại.

**Chứng minh toán học (trích dẫn hoặc tự chứng minh):** LTT hợp lệ khi dữ liệu độc lập; kéo dài đường đi qua vùng rủi ro cao không làm mất tính hợp lệ; các quy tắc thiết kế chọn trên dữ liệu riêng không làm mất tính hợp lệ; nhãn bị bỏ sót chỉ làm hệ thống bảo thủ hơn; lấy mẫu từ pool hữu hạn vẫn giữ tính hợp lệ. Riêng nhánh Bentkus của mệnh đề cuối em mới có phác thảo chứng minh và cần kiểm lại khi viết paper.

**Bằng chứng thực nghiệm (thăm dò, trên Val):**
- Đơn vị chuyến đi đã được xác nhận bằng ID chính thức.
- Ảnh cùng chuyến gần như độc lập.
- Công cụ đo tương quan không báo động giả.
- Phép thử có năng lực: bắt được quy trình ăn gian.
- H1c đạt: quy trình luôn chọn được ngưỡng và giữ được khoảng 99–110% độ phủ của cách chọn thông thường. Nhưng em sẽ chạy lại với một baseline công bằng hơn, vì baseline cũ chỉ lọc theo confidence trên lưới thô.

**Chưa chứng minh:**
- **H1a ở mức xác nhận:** phép thử theo khu vực-ngày chưa chạy.
- **H1b:** cần bộ MOT 2020, vì phải có nhiều frame có nhãn trong cùng một video.

**M:** Sao không kết luận H1a luôn từ kết quả trên Val?

**SV:** Vì em đã nhìn kết quả trên Val rồi mới đặt ra một số quy tắc, như làm mịn lưới và đối chứng dương. Nếu dùng lại Val để kết luận thì đó là "con đường rẽ nhánh": quy tắc được may đo theo dữ liệu. Em giữ riêng một pool 9.033 chuyến đi của split test **chưa từng đọc nhãn**. Em sẽ chốt mọi quy tắc, lưu kèm mã băm, rồi chạy đúng một lần trên pool đó.

---

## Phần 6. Câu hỏi mentor có thể hỏi

**M:** Detector huấn luyện trên train, mà 91,7% chuyến đi trong pool đánh giá có chung khu vực-ngày với tập train. Như vậy có rò rỉ không?

**SV:** Ở cấp chuyến đi thì không, vì không chuyến nào trùng. Ở cấp khu vực-ngày thì có chung bối cảnh. Điều này không làm sai lời hứa trên pool, nhưng có thể làm pool "dễ" hơn so với một khu vực hoàn toàn mới. Em sẽ đo riêng rủi ro trên khoảng 700 chuyến đi thuộc khu vực-ngày hoàn toàn mới.

**M:** Tỉ lệ sai 2% đo theo cách nào?

**SV:** Theo từng chuyến đi: lấy tỉ lệ box sai trong mỗi chuyến rồi lấy trung bình giữa các chuyến (Macro-FDP). Em báo cáo kèm tỉ lệ sai trên toàn bộ box gộp lại (Micro-FDP). Trên Val, Micro-FDP cao hơn khoảng 0,2–0,3 điểm % nhưng vẫn dưới 2%. Lời hứa chính thức chỉ áp dụng cho Macro-FDP.

**M:** Box "sai" được định nghĩa thế nào?

**SV:** Box không ghép được với nhãn chuẩn cùng lớp ở IoU ≥ 0,75 (chuẩn COCO chặt). Quy tắc chọn 0,75 hay 0,5 được cố định trước và áp dụng trên Meta-Train; cả ba detector đều giữ được 0,75.

**M:** Nhãn BDD100K có lỗi thì sao?

**SV:** Nhãn bị bỏ sót chỉ làm box đúng bị đếm là sai, nên hệ thống càng bảo thủ; em đã chứng minh điều này cho đúng quy tắc ghép của mình. Nhãn thừa hoặc sai lớp thì theo chiều ngược lại; em giả định loại lỗi này hiếm và đề xuất kiểm thủ công 300 box.

**M:** Độ phủ thực tế bao nhiêu?

**SV:** Khoảng 19% số xe ở IoU 0,75, khá thấp vì detector có recall thấp. Lời hứa vẫn đúng, nhưng lượng nhãn giả thu được ít. Điều này ảnh hưởng tới H2 và H3 hơn là H1.

**M:** Tương quan cùng tài xế?

**SV:** Không đo được, vì `rideID` là chuyến đi chứ không phải tài xế. Em ghi đây là giới hạn.

---

## Phần 7. Kế hoạch và các quyết định cần mentor góp ý

**Kế hoạch:**
1. Chạy chẩn đoán tương quan theo khu vực-ngày trên Val, để quyết định đơn vị là chuyến đi hay khu vực-ngày.
2. Chốt và lưu toàn bộ quy tắc kèm mã băm.
3. Chạy kiểm định xác nhận **một lần** trên 9.033 chuyến đi chưa đụng tới.
4. Tải MOT 2020 và huấn luyện lại detector loại các chuyến đi của MOT, rồi chạy H1b.

**Em cần thầy/cô góp ý:**
1. Mức rủi ro 2% và độ tin cậy 95% có phù hợp mục tiêu gán nhãn giả không?
2. Chỉ làm lớp car (vì detector một lớp) có đủ cho paper không, hay cần detector đa lớp?
3. Huấn luyện lại detector cho H1b có đáng chi phí, hay chấp nhận H1b chỉ chạy trên các clip MOT sạch?
4. H1 nên là một paper riêng (trọng tâm H1b và phương pháp kiểm chứng), hay là chương nền tảng cho H2/H3?

---

## Phụ lục: các con số nên nhớ

| Nội dung | Số |
|---|---|
| Video / chuyến đi (`rideID`) | 100.000 / 60.137 |
| Train và Val chung | 1 chuyến đi (49% nếu chia theo video) |
| Khoảng cách giữa các clip cùng chuyến | trung vị 21,6 phút |
| Meta-Train / Val / Pool xác nhận | 3.000 / 5.930 / 9.033 chuyến đi |
| LTT trên Val (kiểm tra cài đặt) | 0 vi phạm / 1.002 lần |
| Quy trình ăn gian trên cùng dữ liệu | vi phạm 12–48% |
| Tương quan ảnh cùng chuyến | gần 0 |
| Tỉ lệ sai cao nhất khi ngưỡng ≥ 0,05 | ~2% (lưới quá hẹp) |
| Độ phủ tuyệt đối tại ngưỡng LTT | ~19% số xe (IoU 0,75) |
| Pool xác nhận chung khu vực-ngày với train | 91,7% |
