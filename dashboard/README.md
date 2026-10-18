# M29 Dashboard: xem hệ thống `ai_router` vận hành

Trang web để nhìn thấy dữ liệu train và từng bước một ảnh hoặc một lô ảnh đi qua `ai_router` đang ship. Mọi phép tính trên trang đều do chính code gốc chạy trên GPU. Dashboard chỉ móc vào để ghi lại, **không sửa gì** trong `ai_router`.

## Chạy ở máy

```bash
python dashboard/train_stats.py      # một lần: thống kê 70,000 file nhãn train -> dashboard/data/train_stats.json
python dashboard/server.py           # http://127.0.0.1:8770
```

- Lần chạy đầu, server tự tạo 2 ví dụ: 1 ảnh test và một lô 1,000 ảnh test (`flow_compare/batch_input`). Trên RTX 3050 Ti, lô này mất khoảng 3 phút.
- Đường dẫn mặc định được đặt trong `settings.py`. Có thể đổi bằng biến môi trường `M29_AI_ROUTER`, `M29_DATA_ROOT`, `M29_DASH_STORE`.

## Production (Render + Kaggle)

`M29_MODE=web` chạy server không cần GPU. Job chờ trong hàng đợi để `worker.py` (trên Kaggle hoặc máy có GPU) nhận và chạy. Hướng dẫn từng bước: [`DEPLOY.md`](DEPLOY.md).

## Các trang

| Trang | Nội dung |
|---|---|
| ① Dữ liệu train | % số box theo lớp; mean/variance của rộng, cao, diện tích, tỉ lệ, tâm box theo lớp; số object/ảnh; thời tiết, bối cảnh; centroid FCD của từng lớp trong không gian ẩn; ảnh train có box GT |
| ② Test ảnh | 3 detector, các ngưỡng đang dùng, 2 ví dụ, nút "Đẩy ảnh của bạn", danh sách các lần chạy |
| 1 ảnh (`#/single/<job>/<ảnh>`) | B0 → B7 từng bước, gồm ma trận Hungarian, logit/softmax/entropy, variance toạ độ, WBF, MAX/MIN, vector 256 chiều, z 64 chiều, D_M tới centroid mọi lớp, Σ⁻¹, luật xếp vùng, K-Means, JSON đầu ra, so với GT. Tab "Kiểm chứng" so mọi số tính tay với code gốc |
| 1 batch (`#/batch/<job>`) | Số ảnh mỗi vùng trước và sau K-Means; mean/variance theo lớp; histogram so với τ; không gian ẩn của object kèm centroid; các cụm K-Means; lưới ảnh theo vùng |
| Job (`#/job/<id>`) | **Tracing:** các giai đoạn của job, thời gian từng bước B0–B7 lấy từ tracer của ai_router. **Monitoring:** tốc độ, VRAM, RAM theo batch; tỉ lệ lớp so với GT train; kiểm định KS theo ảnh so với lô tham chiếu; cảnh báo |

## Cấu trúc

```text
dashboard/
├── server.py        FastAPI: API + trang tĩnh; chế độ local (có GPU) hoặc web (Render)
├── jobs.py          hàng đợi job; worker GPU trong tiến trình (local) hoặc qua HTTP (web); tracing vào job.json
├── worker.py        worker GPU từ xa cho chế độ web: nhận job, chạy engine, gửi kết quả, nạp ảnh BDD lên server
├── kaggle_launch.py tự bật kernel Kaggle khi có job chờ mà chưa có worker
├── kaggle/          make_bundle.py (dataset riêng tư cho Kaggle), kernel_run.py (mã kernel)
├── export_bundled.py ghi bundled/: số liệu tổng hợp + kết quả 2 ví dụ, không ảnh, không GT
├── bundled/         đi kèm code cho server web
├── engine.py        chạy M29Pipeline gốc (kèm tracer), giữ lại vector object/cảnh để vẽ, giải thích 1 ảnh
├── monitor.py       monitoring hệ thống và dữ liệu
├── train_stats.py   thống kê GT của tập train
├── settings.py      đường dẫn
├── static/          index.html, app.js, style.css (Plotly qua CDN)
├── tests/           smoke_engine.py (cần CUDA)
└── data/            đầu ra tự sinh, không commit
```

## Lưu ý

- **Ảnh BDD100K trên bản công khai:** license BDD100K cho phép phân phối vì mục đích giáo dục, nghiên cứu, phi lợi nhuận nếu kèm thông báo bản quyền và license. Nhóm vẫn để mặc định là link công khai chỉ hiện thống kê và ảnh người dùng tự gửi; bản có ảnh BDD dành cho giám khảo (có mật khẩu, xem `DEPLOY.md`).
- Vùng Xanh luôn rỗng vì `τ_conf = 1.01 > 1`: LTT của bản ship không chứng nhận được. Trang hiển thị đúng như vậy, không làm đẹp.
- Render không có GPU: ở chế độ web, GPU là worker trên Kaggle hoặc máy của bạn (xem `DEPLOY.md`).
