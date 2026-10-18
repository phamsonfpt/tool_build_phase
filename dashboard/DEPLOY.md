# Đưa dashboard lên production: Render + GPU Kaggle + máy của bạn

```text
Trình duyệt (ai có link cũng dùng được) ──► Render: trang web, API, hàng đợi job, kết quả   [không GPU, không torch]
                                   ▲  │ có job chờ mà chưa có máy GPU → gọi Kaggle API bật kernel
                                   │  ▼
               worker.py ◄─────────┴── Kaggle kernel (GPU, tự tắt sau 10 phút rảnh)
               (cùng code)            hoặc máy bạn (RTX 3050 Ti, nhận job ngay khi đang bật)
```

- Worker **tự gọi ra** Render để nhận job (`/api/worker/claim`), tải ảnh, chạy đúng code `ai_router` B0→B7, giải thích từng ảnh, rồi gửi kết quả về. Không cần mở cổng hay tunnel nào.
- Máy nào đang kết nối thì nhận job trước. Kaggle chỉ được bật khi có job chờ mà **chưa có máy nào** kết nối.
- Render gói miễn phí ngủ sau 15 phút không có truy cập, và **xoá ổ đĩa** mỗi lần ngủ hoặc khởi động lại. Job và ảnh đã gửi lên sẽ mất; 2 ví dụ và thống kê train thì đi kèm code nên không mất.

## Không bao giờ vào git

- Ảnh BDD100K và `train_index.npz` (chứa nhãn GT của 70.000 ảnh). License BDD100K (© The Regents of the University of California) cho phép dùng, sao chép và phân phối vì mục đích giáo dục, nghiên cứu, phi lợi nhuận, **với điều kiện** kèm thông báo bản quyền và nội dung license trong mọi bản sao; dùng thương mại phải xin phép UC Berkeley. Nhóm chọn cách an toàn: link công khai không hiện ảnh hay nhãn BDD cho tới khi ban tổ chức xác nhận mục đích phi lợi nhuận.
- Token: `WORKER_TOKEN`, `KAGGLE_API_TOKEN` chỉ đặt trên Render.

## Trang công khai

- Không có mật khẩu: ai có link cũng xem và gửi ảnh được. Muốn khoá lại thì đặt `DASH_PASSWORD` trên Render.
- **Không hiển thị ảnh BDD100K**: server công khai từ chối ảnh và nhãn BDD mà worker định nạp lên. Trang Dữ liệu train vẫn có đủ thống kê trên 70.000 ảnh, chỉ không có lưới ảnh. Hai ví dụ vẫn có đủ số liệu, biểu đồ và từng bước tính, chỗ ảnh hiện khung trống kèm box đúng vị trí. Ảnh người dùng tự gửi hiển thị bình thường.
- Worker không bao giờ gửi ground truth lên, kể cả khi ảnh người dùng gửi trùng với một ảnh BDD có nhãn trên máy worker.
- Giới hạn để không ai dùng hết quota GPU: tối đa `MAX_JOB_IMAGES` = 300 ảnh mỗi lần gửi, `MAX_QUEUED_JOBS` = 5 job chờ cùng lúc, mỗi ảnh tối đa 15 MB (`MAX_IMAGE_MB`), tối đa 20 ảnh chờ máy GPU tính từng bước (`MAX_EXPLAIN_QUEUE`). Ảnh mà máy GPU báo không tính được (ví dụ ảnh của lô ví dụ khi chỉ có Kaggle) sẽ không được gửi lại trong 10 phút, và Kaggle chỉ được bật cho việc Kaggle làm được.
- Bản riêng tư cho nhóm (có `DASH_PASSWORD`) có thể đặt thêm `ALLOW_SEED=1` và chạy worker với `--seed` để hiện ảnh BDD.
- Checkpoint expert (3 × 164 MB) chỉ nằm trong dataset **riêng tư** trên Kaggle.

`dashboard/bundled/` (đi kèm code) chỉ có số liệu tổng hợp và kết quả của 2 ví dụ, đã bỏ đường dẫn ảnh và ground truth.

## Bước 1. Kaggle (một lần)

1. Tạo tài khoản, **xác minh số điện thoại** (Settings → Phone verification): không có bước này thì không dùng được GPU và Internet trong kernel.
2. Settings → API → **Generate New Token**. Lưu token ở máy: `~/.kaggle/access_token` (hoặc biến môi trường `KAGGLE_API_TOKEN`).
3. Máy hiện có 2 bản `kagglesdk` cài chồng nhau nên lệnh `kaggle` báo lỗi `get_access_token_from_env`. Cài lại sạch (trên môi trường mới thì `kaggle==2.2.4` + `kagglesdk==0.1.37` chạy bình thường, đã thử):

   ```bash
   pip install --force-reinstall --no-deps "kaggle==2.2.4" "kagglesdk==0.1.37"
   ```

4. Đóng gói và tải lên dataset riêng tư (khoảng 500 MB):

   Gói cần `ai_router/calibrated_params/` (`centroids.pt`, `thresholds.yaml`). Thư mục này không có trong git: tạo bằng `python ai_router/scripts/setup_centroids.py` rồi `setup_thresholds.py`, hoặc đặt `M29_AI_ROUTER` trỏ tới một bản `ai_router` đã có sẵn.

   ```bash
   python dashboard/kaggle/make_bundle.py --user <tên-kaggle>
   kaggle datasets create -p "%USERPROFILE%/m29_kaggle_bundle" --dir-mode zip
   ```

   Sau này sửa code `ai_router`/`dashboard` hoặc đổi checkpoint thì chạy lại `make_bundle.py` rồi `kaggle datasets version -p ... --dir-mode zip -m "mô tả"`.

## Bước 2. Đẩy code lên GitHub

Render lấy code từ GitHub: `dashboard/` (trừ `dashboard/data/`, đã có trong `.gitignore`) và `render.yaml` ở gốc repo phải nằm trên nhánh bạn muốn deploy.

- Repo của tổ chức (ví dụ P-022, riêng tư): Render chỉ đọc được khi **chủ tổ chức** cài ứng dụng Render GitHub cho tổ chức đó và cấp quyền repo. Không xin được thì deploy từ một bản fork hoặc repo cá nhân có cùng code.
- Repo cá nhân: Render → Account → GitHub → cấp quyền repo đó.

## Bước 3. Render

1. Dashboard Render → **New → Blueprint** → chọn repo và nhánh có `render.yaml`. Render đọc file đó và tạo service `m29-dashboard` (gói Free).
2. Điền các biến Render hỏi:
   - `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`.
   - `WORKER_TOKEN` được Render tự sinh. Mở Environment để chép nó cho worker ở máy bạn.
3. Deploy. Địa chỉ có dạng `https://m29-dashboard-xxxx.onrender.com`. Kaggle kernel gọi về địa chỉ này qua biến `RENDER_EXTERNAL_URL` do Render tự cấp.
4. Kiểm tra: `https://<địa chỉ>/healthz` trả `{"ok": true, "mode": "web"}`.

## Bước 4. Worker ở máy bạn (tuỳ chọn, nhưng nên có)

```bash
python dashboard/worker.py --server https://<địa chỉ>.onrender.com --token <WORKER_TOKEN> --name may-son
```

- Trang "1 ảnh" của lô ví dụ: 30 ảnh đầu hàng đợi review đã tính sẵn và đi kèm code. Ảnh khác của lô ví dụ chỉ worker ở máy này tính được (Kaggle không có ảnh gốc BDD), trang sẽ báo rõ nếu chỉ có Kaggle.
- Để worker chạy, máy nhận job ngay. Tắt worker thì job chờ Kaggle.

## Kaggle chạy thế nào

- Khi có job chờ và không có máy nào kết nối, Render đẩy một phiên bản mới của kernel riêng tư `<user>/m29-gpu-worker` (bật GPU và Internet, gắn dataset). Kaggle cần 3–5 phút để khởi động máy, cài `transformers==4.57.3`, nạp 3 expert.
- Kernel chạy `worker.py --idle-exit 600`: nhận hết job, rảnh 10 phút thì tự thoát để không phí quota (khoảng 30 giờ GPU/tuần).
- Trang **Test ảnh** có ô "GPU Kaggle" với nút **Bật GPU Kaggle ngay** và lỗi gần nhất nếu có.
- Token worker nằm trong mã nguồn kernel riêng tư của bạn. Đừng chuyển kernel đó sang công khai; nếu lỡ, đổi `WORKER_TOKEN` trên Render.

## Đã kiểm và chưa kiểm

- Đã chạy thử ở máy: server chế độ web (không torch) + worker kết nối qua HTTP, đủ vòng gửi ảnh → GPU → kết quả → trang batch / 1 ảnh / tracing. Bài `dashboard/tests/test_web_mode.py` kiểm cả mật khẩu tuỳ chọn, chặn ảnh BDD, giới hạn upload, chặn đường dẫn lạ.
- **Chưa** chạy trên Render và Kaggle thật (cần tài khoản của bạn). Hai điểm dễ lỗi nhất: phiên bản Python/thư viện trên Kaggle, và đường dẫn dataset sau khi Kaggle giải nén. Lần đầu nên mở trang kernel trên Kaggle để xem log.
