/* M29 dashboard — single-page app. Pages: #/train, #/test, #/job/<id>, #/batch/<id>, #/single/<id>/<image> */
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (x, n = 4) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : (typeof x === "number" ? x.toFixed(n) : esc(x));
const fi = (x) => (x === null || x === undefined) ? "–" : Number(x).toLocaleString("vi-VN");
const pct = (x, n = 1) => (x === null || x === undefined) ? "–" : (100 * x).toFixed(n) + " %";
const sum = (a) => a.reduce((s, v) => s + v, 0);
const main = () => $("#main");
const CLS_COLORS = ["#e6194b", "#f58231", "#4363d8", "#911eb4", "#46f0f0", "#808000", "#f032e6", "#3cb44b", "#ffe119", "#00a3a3"];
const ZONE_VI = { RED: "Đỏ", GREY: "Xám", GREEN: "Xanh" };
const zone = (z) => `<span class="zone ${esc(z)}">${ZONE_VI[z] || esc(z)}</span>`;
let NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"];
const ccol = (name) => CLS_COLORS[Math.max(0, NAMES.indexOf(name)) % CLS_COLORS.length];
const timers = [];
const clearTimers = () => { while (timers.length) clearInterval(timers.pop()); };
const cache = {};

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) { /* not JSON */ }
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.json();
}

const PCFG = { displayModeBar: false, responsive: true };
function plot(el, data, layout = {}) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el || !window.Plotly) return;
  Plotly.newPlot(el, data, Object.assign({
    margin: { l: 50, r: 16, t: 28, b: 40 }, font: { family: "Segoe UI, sans-serif", size: 12 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
  }, layout), PCFG);
}
const vline = (x, label, color = "#dc2626") => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color, width: 2, dash: "dash" }, label: { text: label, textposition: "end", font: { color, size: 11 } } });

function matrix(rows, { rowLabels = [], colLabels = [], digits = 4, hl = () => false, heat = false } = {}) {
  let lo = Infinity, hi = -Infinity;
  if (heat) rows.flat().forEach((v) => { if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
  const bg = (v) => {
    if (!heat || v === null || hi === lo) return "";
    const t = (v - lo) / (hi - lo);
    return `background:rgba(37,99,235,${(0.08 + 0.55 * t).toFixed(2)})`;
  };
  let s = '<table class="mat">';
  if (colLabels.length) s += "<tr><th></th>" + colLabels.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr>";
  rows.forEach((r, i) => {
    s += `<tr><th>${esc(rowLabels[i] ?? i)}</th>` + r.map((v, j) => `<td class="${hl(i, j) ? "hl" : ""}" style="${bg(v)}">${f(v, digits)}</td>`).join("") + "</tr>";
  });
  return s + "</table>";
}

function imageWithBoxes(src, W, H, boxes) {
  // boxes: [{b:[x1,y1,x2,y2], color, label, id, sel, dash}]
  const r = boxes.map((o) => {
    const [x1, y1, x2, y2] = o.b;
    return `<rect ${o.id !== undefined ? `class="click${o.sel ? " sel" : ""}" data-id="${o.id}"` : ""} x="${x1}" y="${y1}" width="${Math.max(x2 - x1, 1)}" height="${Math.max(y2 - y1, 1)}" style="stroke:${o.color}${o.dash ? ";stroke-dasharray:8 6" : ""}"></rect>` +
      (o.label ? `<text x="${x1 + 3}" y="${Math.max(y1 - 6, 20)}" fill="${o.sel ? "#facc15" : o.color}">${esc(o.label)}</text>` : "");
  }).join("");
  return `<div class="imgbox" style="aspect-ratio:${W} / ${H}"><img src="${src}" alt="" onerror="this.parentElement.classList.add('noimg')"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${r}</svg></div>`;
}

function modal(html) { $("#modal-box").innerHTML = html; $("#modal").classList.add("on"); }
function closeModal() { $("#modal").classList.remove("on"); }
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal" || e.target.dataset.close !== undefined) closeModal(); });

// ------------------------------------------------------------------ sidebar status
async function pollSystem() {
  try {
    const s = await api("/api/system");
    cache.system = s.engine_loaded ? s : cache.system;
    if (s.classes) NAMES = s.classes;
    document.body.classList.toggle("nobdd", s.bdd_images === false);
    const busy = s.running.length > 0;
    const dev = (x) => String(x || "GPU").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "");
    if (s.mode === "web") {
      const w = s.workers || [];
      const kg = s.kaggle || {};
      const booting = kg.last_push && s.now - kg.last_push < 15 * 60 && !w.length;
      $("#gpu-dot").className = "dot " + (w.length ? (busy ? "busy" : "ok") : booting ? "busy" : "err");
      $("#gpu-text").textContent = w.length ? w.map((x) => `${x.name} (${dev(x.device)})`).join(", ") + (busy ? " · đang chạy job" : " · sẵn sàng")
        : booting ? "đang bật GPU Kaggle (3–5 phút)…" : "chưa có máy GPU nào kết nối";
    } else {
      $("#gpu-dot").className = "dot " + (s.engine_loaded ? (busy ? "busy" : "ok") : "busy");
      $("#gpu-text").textContent = s.engine_loaded ? (dev(s.device) + (busy ? " · đang chạy job" : " · sẵn sàng"))
        : s.engine_loading ? "đang nạp 3 expert lên GPU…" : "chưa nạp model";
    }
    $("#queue-text").textContent = s.queue ? `${s.queue} job đang chờ` : "";
  } catch (e) {
    $("#gpu-dot").className = "dot err"; $("#gpu-text").textContent = "mất kết nối server";
  }
}
pollSystem(); setInterval(pollSystem, 3000);

// ------------------------------------------------------------------ router
async function route() {
  clearTimers(); closeModal();
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const page = parts[0] || "intro";
  $$("#side a.nav").forEach((a) => a.classList.toggle("on", a.dataset.nav === (["intro", "train"].includes(page) ? page : "test")));
  window.scrollTo(0, 0);
  try {
    if (page === "intro") await pageIntro();
    else if (page === "train") await pageTrain();
    else if (page === "test") await pageTest();
    else if (page === "job") await pageJob(parts[1]);
    else if (page === "batch") await pageBatch(parts[1]);
    else if (page === "single") await pageSingle(parts[1], parts.slice(2).join("/"));
    else main().innerHTML = `<p>Không có trang này. <a href="#/train">Về trang chủ</a></p>`;
  } catch (e) {
    main().innerHTML = `<div class="alert err">Lỗi: ${esc(e.message)}</div>`;
    console.error(e);
  }
}
window.addEventListener("hashchange", route);
route();

// ================================================================== GIỚI THIỆU (trang đầu)
async function pageIntro() {
  main().innerHTML = `
    <div class="kick" style="color:#b45309;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px">AI20K Build Phase · Cohort 4A · Đội 022 / 113 · Bài M29</div>
    <h1 style="font-size:30px;margin-top:6px">M29 AI-Router: Confidence-Based Human Review</h1>
    <p class="lead" style="font-size:16px;max-width:1000px">
      <b>Tóm tắt.</b> Gán nhãn lại toàn bộ ảnh thô rất tốn công (khoảng 35 giây cho một box), trong khi phần lớn ảnh mô hình đã đoán đúng.
      M29 AI-Router cho <b>3 detector RT-DETR</b> (train trên BDD100K) cùng chấm từng ảnh, đo mức <b>bất đồng</b> giữa chúng và mức <b>bất thường</b>
      của từng object so với dữ liệu train, rồi chỉ đưa cho người review những prediction đáng ngờ. Phần còn lại được gán nhãn tự động, kèm cam kết
      thống kê về tỉ lệ nhãn sai. Trang này cho xem toàn bộ cách hệ thống tính toán, từ thống kê dữ liệu tới từng phép tính trên một ảnh, và cho bạn tự gửi ảnh để chạy thử.</p>

    <div class="grid g3 section">
      <div class="card"><h3>1 · Bài toán</h3><p style="font-size:14px;line-height:1.5">Đề M29: <i>chỉ bắt reviewer xem prediction có confidence thấp hoặc bất thường</i>. Đầu ra là chính sách review, hàng đợi review và benchmark. Hai thước đo: <b>Error Recall</b> (bao nhiêu lỗi lọt vào hàng đợi) và <b>Workload Reduction</b> (bớt được bao nhiêu công).</p></div>
      <div class="card"><h3>2 · Cách hệ thống làm</h3><p style="font-size:14px;line-height:1.5">3 expert chạy cùng ảnh → ghép object bằng <b>Hungarian</b>, gộp box bằng <b>WBF</b> → đo <b>entropy</b> và variance toạ độ → khoảng cách <b>Mahalanobis</b> tới centroid từng lớp (FCD) → ngưỡng chứng nhận bằng <b>Learn-Then-Test</b> → <b>K-Means</b> để hàng đợi không lặp cảnh → ảnh vào vùng Đỏ (cần xem), Xám (chờ) hoặc Xanh (tự gán).</p></div>
      <div class="card"><h3>3 · Kết quả (2.000 ảnh test)</h3><p style="font-size:14px;line-height:1.5">Xem 20 % số ảnh, tỉ lệ lỗi bắt được (lỗi = box sai + object bị sót): ngẫu nhiên 20,0 %, bản ship 25,6 %, <b>bản v2 35,6 %</b>, mức trần oracle 38,3 %. Bản v2 còn cho <b>9.220 nhãn tự động</b> dùng được ngay, tỉ lệ sai 1,53 % (cam kết ≤ 2 %).</p></div>
    </div>

    <h2 class="section">Dùng thử trong 3 bước</h2>
    <div class="grid g3">
      <a class="card big" href="#/train" style="color:inherit;text-decoration:none"><div class="ico">📊</div><div class="t">Dữ liệu train</div><div style="font-size:14px">70.000 ảnh, 1,29 triệu box: tỉ lệ lớp, mean/variance, không gian ẩn 3D của từng lớp.</div></a>
      <a class="card big" href="#/single/example-single/cd92dc38-3ee11e29.jpg" style="color:inherit;text-decoration:none"><div class="ico">🖼</div><div class="t">Một ảnh, từng bước</div><div style="font-size:14px">B0 → B7 với ma trận và số thật. Bấm một object để xem vector đặc trưng và khoảng cách tới centroid. 76/76 phép tính khớp code gốc.</div></a>
      <a class="card big" href="#/batch/example-batch" style="color:inherit;text-decoration:none"><div class="ico">🗂</div><div class="t">Cả lô 1.000 ảnh</div><div style="font-size:14px">Số ảnh mỗi vùng, phân bố điểm so với ngưỡng, không gian ẩn, cụm K-Means, tracing và monitoring.</div></a>
    </div>
    <p class="hint" style="margin-top:10px">Muốn chạy với ảnh của bạn: vào <a href="#/test">Test ảnh</a> → <b>Đẩy ảnh của bạn</b>. Ảnh được xử lý trên GPU của nhóm; nếu ô "Máy xử lý ảnh" ở góc trái báo chưa có máy GPU, ảnh sẽ chờ trong hàng đợi.</p>

    <div class="grid g2 section">
      <div class="card"><h3>Nói thẳng: những gì chưa đạt</h3><ul style="font-size:14px;line-height:1.55;padding-left:18px">
        <li>Mục tiêu Error Recall 65 % chưa đạt ở ngân sách 20 %: kể cả mức trần oracle chỉ 38 %.</li>
        <li>Detector còn bỏ sót nhiều object; vùng Xanh của bản ship chưa mở được.</li>
        <li>Bản công khai <b>không hiển thị ảnh BDD100K</b> (license chỉ cho phân phối phi lợi nhuận kèm thông báo bản quyền); box, số liệu và các bước tính vẫn hiển thị đúng.</li>
      </ul></div>
      <div class="card"><h3>Hướng phát triển</h3><ul style="font-size:14px;line-height:1.55;padding-left:18px">
        <li>Đo thời gian review thật trên CVAT để chọn ngân sách và cách hiện box mồi.</li>
        <li>Train lại expert với ảnh do hệ thống chọn, đo mAP sau mỗi vòng.</li>
        <li>Nối giao diện với chiến lược v2 (review theo prediction).</li>
      </ul>
      <p class="hint">Công nghệ: PyTorch, HuggingFace RT-DETR, scikit-learn, FastAPI, Plotly · Hạ tầng: Render + GPU Kaggle / máy cá nhân.</p></div>
    </div>`;
}

// ================================================================== ① DỮ LIỆU TRAIN
const ATTR_VI = {
  weather: { _: "Thời tiết", clear: "quang đãng", overcast: "u ám", "partly cloudy": "có mây", rainy: "mưa", snowy: "tuyết", foggy: "sương mù", undefined: "không ghi" },
  scene: { _: "Bối cảnh", "city street": "đường phố", highway: "cao tốc", residential: "khu dân cư", "parking lot": "bãi đỗ xe", tunnel: "đường hầm", "gas stations": "cây xăng", undefined: "không ghi" },
  timeofday: { _: "Thời điểm", daytime: "ban ngày", night: "ban đêm", "dawn/dusk": "bình minh / hoàng hôn", undefined: "không ghi" },
};
const attrLabel = (k, v) => `${v} (${ATTR_VI[k]?.[v] || v})`;
const GEOM = {
  w: { t: "Rộng", u: "px", d: "chiều rộng box = x2 − x1" },
  h: { t: "Cao", u: "px", d: "chiều cao box = y2 − y1" },
  area: { t: "Diện tích", u: "px²", d: "rộng × cao" },
  aspect: { t: "Tỉ lệ rộng/cao", u: "", d: "rộng ÷ cao; > 1 là box nằm ngang (car, bus), < 1 là box đứng (pedestrian, traffic light)" },
  cx: { t: "Tâm x", u: "px", d: "toạ độ ngang của tâm box; ảnh rộng 1280 px, 640 là giữa ảnh" },
  cy: { t: "Tâm y", u: "px", d: "toạ độ dọc của tâm box; ảnh cao 720 px, số nhỏ là gần mép trên (đèn, biển báo)" },
};
const unit2 = (u) => u ? u.replace("px²", "px⁴").replace(/^px$/, "px²") : "";
const fnum = (x, n) => x === null || x === undefined ? "–" : Number(x).toLocaleString("vi-VN", { maximumFractionDigits: n, minimumFractionDigits: n });

async function pageTrain() {
  main().innerHTML = `<div class="loading">Đang tải thống kê dữ liệu train…</div>`;
  const t = cache.train || (cache.train = await api("/api/train/stats"));
  NAMES = t.class_names;
  const cls = t.classes, total = t.boxes, opi = t.objects_per_image, L = t.latent;
  const sb = t.size_buckets;
  main().innerHTML = `
    <h1>Dữ liệu train (ground truth)</h1>
    <p class="lead">BDD100K train là tập dữ liệu 3 expert đã được train. Mọi số dưới đây được đếm trên <b>toàn bộ ${fi(t.images)} file nhãn</b>, không lấy mẫu. Cả ${fi(t.images_with_boxes)} ảnh đều có ít nhất 1 box GT.
      Chỉ tính 10 lớp mà detector biết. Nhãn khác (làn đường, vùng lái được) bị bỏ, giống <code>gt_map</code> của ai_router.</p>
    <div class="grid g4">
      <div class="card kpi"><div class="v">${fi(t.images)}</div><div class="l">ảnh train</div></div>
      <div class="card kpi"><div class="v">${fi(total)}</div><div class="l">box GT thuộc 10 lớp</div></div>
      <div class="card kpi"><div class="v">${fnum(opi.mean, 1)}</div><div class="l">box GT trung bình trên 1 ảnh (trung vị ${opi.median}, nhiều nhất ${opi.max})</div></div>
      <div class="card kpi"><div class="v">${pct(cls.find((c) => c.name === "car").share)}</div><div class="l">số box là <b>car</b>, lớp chiếm đa số</div></div>
    </div>

    <div class="card section"><h2>1. Tỉ lệ số box theo lớp</h2>
      <div class="grid" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:center">
        <div id="p-share" style="height:380px"></div>
        <div style="overflow-x:auto"><table class="t"><tr><th>Lớp</th><th>% tổng số box</th><th>Số box / tổng</th><th>Số ảnh có lớp này</th></tr>
          ${cls.map((c) => `<tr><td><i style="display:inline-block;width:11px;height:11px;background:${ccol(c.name)};border-radius:2px"></i> ${esc(c.name)}</td><td><b>${pct(c.share, 2)}</b></td><td>${fi(c.boxes)} / ${fi(total)}</td><td>${fi(c.images)} / ${fi(t.images)}</td></tr>`).join("")}
          <tr><td><b>Tổng</b></td><td>100 %</td><td>${fi(total)}</td><td></td></tr></table></div>
      </div>
      <div class="hint">Lớp hiếm có rất ít mẫu: train chỉ có ${fi(cls.find((c) => c.name === "train").boxes)} box, motorcycle ${fi(cls.find((c) => c.name === "motorcycle").boxes)} box. Detector ít được học các lớp này, nên dễ sai ở chúng.</div></div>

    <div class="card section"><h2>2. Mean và variance theo lớp</h2>
      <div class="why"><b>Mean</b> = giá trị trung bình trên mọi box của lớp. <b>Variance</b> = trung bình của (giá trị − mean)²: các box của lớp khác nhau nhiều hay ít. Variance có đơn vị bình phương (px → px²), nên khó so trực tiếp với mean. <b>Std</b> = √variance thì cùng đơn vị với mean: phần lớn box nằm trong khoảng mean ± 2·std.</div>
      <div class="row"><span>Đại lượng:</span><select id="geom">${Object.entries(GEOM).map(([k, v]) => `<option value="${k}">${v.t}${v.u ? " (" + v.u + ")" : ""}</option>`).join("")}</select><span class="hint" id="geom-def"></span></div>
      <div class="grid g2"><div id="p-mean" style="height:320px"></div><div id="p-var" style="height:320px"></div></div></div>

    <div class="card section"><h2>3. Bảng mean / variance theo lớp</h2>
      <div class="hint" style="margin-bottom:8px">Mỗi đại lượng có 2 cột: <b>mean</b> (đơn vị như tiêu đề) và <b>variance</b> của chính đại lượng đó (đơn vị bình phương). Ví dụ "Rộng · variance" là variance của chiều rộng box, đơn vị px².</div>
      <div class="scroll"><table class="t"><tr><th rowspan="2">Lớp</th><th rowspan="2">Số box</th>${Object.values(GEOM).map((g) => `<th colspan="2" style="text-align:center;border-left:2px solid var(--line)">${g.t}</th>`).join("")}</tr>
        <tr>${Object.values(GEOM).map((g) => `<th style="border-left:2px solid var(--line)">mean${g.u ? " (" + g.u + ")" : ""}</th><th>variance${unit2(g.u) ? " (" + unit2(g.u) + ")" : ""}</th>`).join("")}</tr>
        ${cls.map((c) => `<tr><td>${esc(c.name)}</td><td>${fi(c.boxes)}</td>${Object.keys(GEOM).map((k) => `<td style="border-left:2px solid var(--line)">${fnum(c[k].mean, k === "aspect" ? 2 : 1)}</td><td>${fnum(c[k].var, k === "aspect" ? 3 : 0)}</td>`).join("")}</tr>`).join("")}
      </table></div>
      <div class="hint">${Object.values(GEOM).map((g) => `<b>${g.t}</b>: ${g.d}`).join(" · ")}</div></div>

    <div class="card section"><h2>4. Cỡ box theo lớp: nhỏ, vừa, lớn</h2>
      <div class="why">Theo chuẩn COCO, box được chia theo diện tích: <b>nhỏ (small)</b> &lt; 32 × 32 = ${fi(sb.small_lt)} px²; <b>vừa (medium)</b> từ ${fi(sb.small_lt)} đến dưới 96 × 96 = ${fi(sb.large_ge)} px²; <b>lớn (large)</b> ≥ ${fi(sb.large_ge)} px². Ảnh BDD100K là 1280 × 720 = 921.600 px², nên box nhỏ chiếm dưới 0,11 % diện tích ảnh. Detector thường sót box nhỏ nhiều nhất.</div>
      <div id="p-size" style="height:380px"></div>
      <h3 style="margin-top:12px">Phân bố chi tiết diện tích box</h3>
      <div class="row"><span>Lớp:</span><select id="area-cls" multiple size="4" style="min-width:200px">${NAMES.map((n, i) => `<option ${[0, 2, 8, 9].includes(i) ? "selected" : ""}>${esc(n)}</option>`).join("")}</select><span class="hint">Giữ Ctrl để chọn nhiều lớp. Mỗi đường là % số box của lớp đó rơi vào từng khoảng diện tích. Hai vạch dọc là ranh giới nhỏ/vừa và vừa/lớn.</span></div>
      <div id="p-area" style="height:320px"></div></div>

    <div class="card section"><h2>5. Số object trên mỗi ảnh</h2>
      <div class="why">Đếm trong mỗi ảnh có bao nhiêu box GT thuộc 10 lớp. Cột ở vị trí 17 cao bao nhiêu nghĩa là có bấy nhiêu ảnh chứa đúng 17 box. Trung bình ${fnum(opi.mean, 1)} box/ảnh, std ${fnum(Math.sqrt(opi.var), 1)}, trung vị ${opi.median}, nhiều nhất ${opi.max}. Đại lượng này quan trọng với router: ai_router lấy MAX/MIN trên các object, nên ảnh càng nhiều object càng dễ bị một object xấu kéo vào vùng Đỏ.</div>
      <div id="p-opi" style="height:300px"></div></div>

    <div class="grid g3 section">
      ${Object.keys(t.attributes).map((k) => `<div class="card"><h3>${ATTR_VI[k]?._ || k}</h3><div id="p-at-${k}" style="height:${60 + 30 * Object.keys(t.attributes[k]).length}px"></div>
        <table class="t" style="table-layout:fixed"><colgroup><col><col style="width:92px"><col style="width:58px"></colgroup>
          ${Object.entries(t.attributes[k]).map(([v, n]) => `<tr><td style="white-space:normal">${esc(attrLabel(k, v))}</td><td>${fi(n)} ảnh</td><td>${pct(n / t.images)}</td></tr>`).join("")}</table></div>`).join("")}
    </div>

    <div class="card section"><h2>6. Không gian ẩn của train (3D, kéo để xoay)</h2>
      <div class="why">Mỗi box GT của train được ghép với một query của Anchor. Vector ẩn 256 chiều của query đó được PCA đưa xuống ${L.pca_dim} chiều: đây là không gian FCD dùng để tính khoảng cách Mahalanobis.
        <b>Centroid</b> ◆ của một lớp là mean của mọi vector thuộc lớp đó. ${L.points ? `Mỗi chấm là 1 vector train thật (tối đa ${L.points.per_class_cap} chấm mỗi lớp, trong ${fi(L.points.n_total)} vector).` : ""}
        Để vẽ được, 64 chiều được chiếu tiếp xuống 3 trục chính (PC1–PC3, giữ ${L.points ? pct(sum(L.points.explained)) : "?"} phương sai). Khoảng cách thật luôn tính ở 64 chiều, nên hai điểm trông gần nhau trong hình 3D chưa chắc đã gần nhau thật.</div>
      <div class="row"><span>Tô màu theo:</span><select id="lat-color"><option value="cls">lớp</option><option value="d">D_M tới centroid lớp của chính nó</option></select>
        <label><input type="checkbox" id="lat-pts" checked> hiện chấm train</label><span class="hint">Bấm vào một ◆ centroid để xem chi tiết lớp đó.</span></div>
      <div class="grid" style="grid-template-columns:minmax(0,3fr) minmax(0,1fr)"><div id="p-lat" style="height:560px"></div><div id="lat-info" class="why" style="align-self:start">Bấm vào một centroid.</div></div></div>

    <div class="card section"><h2>7. Mean / variance trong không gian ẩn</h2>
      <table class="t"><tr><th>Lớp</th><th>Số vector</th><th>‖μ‖</th><th>trace(Σ)</th><th>λ lớn nhất</th><th>λ nhỏ nhất</th><th>Centroid gần nhất</th><th>Covariance</th></tr>
        ${L.classes.map((c, i) => { const dr = L.centroid_dist[i]; let j = -1; dr.forEach((v, k) => { if (k !== i && (j < 0 || v < dr[j])) j = k; });
          return `<tr><td>${esc(c.name)}</td><td>${fi(c.n_vectors)}</td><td>${fnum(c.mu_norm, 2)}</td><td>${fnum(c.trace_cov, 2)}</td><td>${fnum(c.max_eig, 2)}</td><td>${fnum(c.min_eig, 4)}</td><td>${esc(L.classes[j].name)} (${fnum(dr[j], 2)})</td><td>${c.pooled_cov ? '<span class="tag warn">dùng Σ chung</span>' : '<span class="tag ok">Σ riêng</span>'}</td></tr>`; }).join("")}</table>
      <div class="why" style="margin-top:10px">
        <b>Số vector</b>: số box GT của lớp được ghép với một query của Anchor trên ${fi(L.images)} ảnh train, dùng để tính centroid μ và covariance Σ của lớp.<br>
        <b>‖μ‖</b> (mean): độ dài vector centroid trong ${L.pca_dim} chiều, tức khoảng cách từ centroid lớp tới tâm chung của mọi vector. Lớp càng "khác số đông" thì càng xa.<br>
        <b>trace(Σ)</b> (variance): tổng variance theo cả ${L.pca_dim} chiều, tức các vector của lớp tản rộng tới đâu quanh centroid.<br>
        <b>λ lớn nhất / nhỏ nhất</b>: variance theo hướng lớp tản rộng nhất và hẹp nhất. Mahalanobis chia độ lệch theo từng hướng cho variance của hướng đó, nên lệch dù nhỏ theo hướng có λ nhỏ cũng làm D_M tăng mạnh.<br>
        <b>Centroid gần nhất</b>: lớp có centroid gần nhất (khoảng cách Euclid ở ${L.pca_dim} chiều). Hai lớp gần nhau thì dễ bị nhầm với nhau.<br>
        <b>Covariance</b>: lớp có dưới 100 vector không đủ để ước lượng ma trận ${L.pca_dim}×${L.pca_dim}, nên dùng covariance gộp của mọi lớp. Lớp <b>train</b> chỉ có ${L.classes.find((c) => c.name === "train")?.n_vectors} vector, nên centroid của nó không đáng tin.</div></div>

    <div class="card section"><h2>8. Ảnh train và box ground truth</h2>
      <div class="row" style="margin-bottom:8px">
        ${Object.keys(ATTR_VI).map((k) => `<span>${ATTR_VI[k]._}:</span><select data-f="${k}"><option value="">tất cả</option>${Object.entries(t.attributes[k]).map(([v, n]) => `<option value="${esc(v)}">${esc(attrLabel(k, v))} · ${fi(n)}</option>`).join("")}</select>`).join("")}
        <span>Có lớp:</span><select data-f="cls"><option value="">tất cả</option>${cls.map((c) => `<option value="${esc(c.name)}">${esc(c.name)} · ${fi(c.images)} ảnh</option>`).join("")}</select>
        <span>Xếp:</span><select data-f="sort"><option value="random">ngẫu nhiên</option><option value="most">nhiều object nhất</option><option value="fewest">ít object nhất</option><option value="name">theo tên</option></select></div>
      <div class="legend" style="margin:8px 0">${NAMES.map((n) => `<span><i style="background:${ccol(n)}"></i>${esc(n)}</span>`).join("")}</div>
      <div class="kpi"><span class="v" id="gal-total">…</span> <span class="l" id="gal-of"></span></div>
      <div id="gal" class="thumbs" style="margin-top:8px"></div><div class="row" style="margin-top:10px"><button class="btn" id="gal-more">Xem thêm</button><span class="hint" id="gal-count"></span></div></div>`;

  // 1. donut
  plot("p-share", [{ type: "pie", hole: 0.45, labels: cls.map((c) => c.name), values: cls.map((c) => c.boxes), marker: { colors: cls.map((c) => ccol(c.name)) }, sort: false, direction: "clockwise",
    textinfo: "percent", textposition: "inside", insidetextorientation: "horizontal",
    hovertemplate: "%{label}<br>%{value:,} / " + fi(total) + " box<br>%{percent}<extra></extra>" }],
  { height: 380, margin: { l: 10, r: 10, t: 10, b: 10 }, annotations: [{ text: `${fi(total)}<br>box`, showarrow: false, font: { size: 15 } }] });

  // 2. mean and variance
  const drawGeom = () => {
    const k = $("#geom").value, g = GEOM[k];
    $("#geom-def").textContent = g.d;
    const dig = k === "aspect" ? 2 : 1;
    plot("p-mean", [{ type: "bar", x: cls.map((c) => c.name), y: cls.map((c) => c[k].mean), marker: { color: cls.map((c) => ccol(c.name)) },
      text: cls.map((c) => fnum(c[k].mean, dig)), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}<br>mean %{y:.2f}<extra></extra>" }],
    { title: { text: `Mean của ${g.t.toLowerCase()}${g.u ? " (" + g.u + ")" : ""}`, font: { size: 13 } }, height: 320, margin: { l: 60, r: 10, t: 40, b: 80 } });
    plot("p-var", [{ type: "bar", x: cls.map((c) => c.name), y: cls.map((c) => c[k].var), marker: { color: cls.map((c) => ccol(c.name)), opacity: 0.75 },
      text: cls.map((c) => fnum(c[k].var, k === "aspect" ? 3 : 0)), textposition: "outside", cliponaxis: false,
      customdata: cls.map((c) => c[k].std), hovertemplate: "%{x}<br>variance %{y:.2f}<br>std = √variance = %{customdata:.2f}<extra></extra>" }],
    { title: { text: `Variance của ${g.t.toLowerCase()}${unit2(g.u) ? " (" + unit2(g.u) + ")" : ""}`, font: { size: 13 } }, height: 320, margin: { l: 70, r: 10, t: 40, b: 80 } });
  };
  $("#geom").onchange = drawGeom; drawGeom();

  // 4. size buckets + detail
  const sz = (b) => cls.map((c) => c.size[b] / Math.max(c.boxes, 1) * 100);
  plot("p-size", [["small", "nhỏ (< 32²)", "#f59e0b"], ["medium", "vừa (32²–96²)", "#3b82f6"], ["large", "lớn (≥ 96²)", "#16a34a"]].map(([b, lab, col]) => ({
    type: "bar", orientation: "h", name: lab, y: cls.map((c) => c.name), x: sz(b), marker: { color: col },
    text: cls.map((c) => `${(c.size[b] / Math.max(c.boxes, 1) * 100).toFixed(1)} % (${fi(c.size[b])})`), textposition: "inside", insidetextanchor: "middle",
    hovertemplate: "%{y} · " + lab + "<br>%{x:.1f} %<extra></extra>" })),
  { barmode: "stack", height: 380, xaxis: { title: "% số box của lớp", range: [0, 100] }, yaxis: { autorange: "reversed" }, showlegend: true, legend: { orientation: "h", y: -0.18 }, margin: { l: 100, r: 10, t: 10, b: 60 } });
  const drawArea = () => {
    const sel = $$("#area-cls option").filter((o) => o.selected).map((o) => o.value);
    const xs = [...Array(24).keys()].map((i) => Math.pow(10, (i + 0.5) * 0.25));
    plot("p-area", sel.map((n) => { const c = cls.find((x) => x.name === n); const tot = sum(c.log_area_hist) || 1;
      return { type: "scatter", mode: "lines+markers", name: n, x: xs, y: c.log_area_hist.map((v) => 100 * v / tot), line: { color: ccol(n), width: 2 }, marker: { size: 4 },
        hovertemplate: `${n}<br>diện tích ≈ %{x:,.0f} px²<br>%{y:.1f} % box<extra></extra>` }; }),
    { xaxis: { type: "log", title: "diện tích box (px², trục log)" }, yaxis: { title: "% số box của lớp" }, showlegend: true, legend: { orientation: "h" }, height: 320,
      shapes: [sb.small_lt, sb.large_ge].map((x) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#64748b", dash: "dash" } })),
      annotations: [{ x: Math.log10(sb.small_lt), y: 1, yref: "paper", text: `nhỏ | vừa<br>${fi(sb.small_lt)} px²`, showarrow: false, yanchor: "bottom", font: { size: 11 } },
        { x: Math.log10(sb.large_ge), y: 1, yref: "paper", text: `vừa | lớn<br>${fi(sb.large_ge)} px²`, showarrow: false, yanchor: "bottom", font: { size: 11 } }],
      margin: { l: 50, r: 10, t: 40, b: 50 } });
  };
  $("#area-cls").onchange = drawArea; drawArea();

  // 5. objects per image + attributes
  const oh = opi.hist;
  plot("p-opi", [{ type: "bar", x: oh.map((_, i) => i), y: oh, marker: { color: "#2563eb" }, hovertemplate: "%{y:,} ảnh có đúng %{x} box GT<extra></extra>" }],
    { xaxis: { title: "số box GT trong 1 ảnh (cột 80 gộp mọi ảnh có từ 80 box trở lên)" }, yaxis: { title: "số ảnh" }, height: 300,
      shapes: [{ type: "line", x0: opi.mean, x1: opi.mean, yref: "paper", y0: 0, y1: 1, line: { color: "#dc2626", dash: "dash" } }],
      annotations: [{ x: opi.mean, y: 1, yref: "paper", text: `mean = ${fnum(opi.mean, 1)}`, showarrow: false, xanchor: "left", font: { color: "#dc2626" } }] });
  // horizontal bars: long labels sit on the left axis instead of being cut under the bars
  Object.entries(t.attributes).forEach(([k, v]) => {
    const n = Object.values(v), max = Math.max(...n);
    plot(`p-at-${k}`, [{ type: "bar", orientation: "h", y: Object.keys(v).map((x) => ATTR_VI[k]?.[x] || x), x: n, marker: { color: "#64748b" },
      text: n.map((x) => fi(x)), textposition: "outside", textfont: { size: 11 }, cliponaxis: false, hovertemplate: "%{y}: %{x:,} ảnh<extra></extra>" }],
    { height: 60 + 30 * n.length, margin: { l: 10, r: 10, t: 10, b: 30 }, bargap: 0.3,
      yaxis: { autorange: "reversed", automargin: true }, xaxis: { range: [0, max * 1.3], tickformat: "~s", title: { text: "số ảnh", standoff: 4 } } });
  });

  // 6. latent 3D
  const drawLat = () => {
    const mode = $("#lat-color").value, show = $("#lat-pts").checked, P = L.points;
    const traces = [];
    if (P && show) {
      if (mode === "cls") NAMES.forEach((n, c) => {
        const ix = P.cls.map((v, i) => v === c ? i : -1).filter((i) => i >= 0);
        if (ix.length) traces.push({ type: "scatter3d", mode: "markers", name: n, x: ix.map((i) => P.xyz[i][0]), y: ix.map((i) => P.xyz[i][1]), z: ix.map((i) => P.xyz[i][2]),
          marker: { size: 2.2, color: ccol(n), opacity: 0.55 }, customdata: ix.map((i) => P.d[i]), hovertemplate: `${n}<br>D_M = %{customdata:.2f}<extra></extra>` });
      });
      else traces.push({ type: "scatter3d", mode: "markers", name: "vector train", x: P.xyz.map((p) => p[0]), y: P.xyz.map((p) => p[1]), z: P.xyz.map((p) => p[2]),
        marker: { size: 2.2, color: P.d, colorscale: "Viridis", cmin: 0, cmax: 15, opacity: 0.7, colorbar: { title: "D_M" } },
        customdata: P.cls.map((c, i) => [NAMES[c], P.d[i]]), hovertemplate: "%{customdata[0]}<br>D_M = %{customdata[1]:.2f}<extra></extra>" });
    }
    traces.push({ type: "scatter3d", mode: "markers+text", name: "centroid", x: L.classes.map((c) => c.xyz[0]), y: L.classes.map((c) => c.xyz[1]), z: L.classes.map((c) => c.xyz[2]),
      text: L.classes.map((c) => c.name), textposition: "top center", textfont: { size: 12, color: "#111" },
      marker: { size: 9, symbol: "diamond", color: L.classes.map((c) => ccol(c.name)), line: { color: "#111", width: 2 } },
      customdata: L.classes.map((c, i) => i), hovertemplate: "centroid %{text}<extra></extra>" });
    plot("p-lat", traces, { height: 560, margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: mode === "cls", legend: { orientation: "h" },
      scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } } });
    $("#p-lat").on("plotly_click", (e) => {
      const p = e.points.find((q) => q.data.name === "centroid"); if (!p) return;
      const i = p.customdata, c = L.classes[i], dr = L.centroid_dist[i];
      const near = dr.map((v, k) => [v, k]).filter(([, k]) => k !== i).sort((a, b) => a[0] - b[0]).slice(0, 3);
      $("#lat-info").innerHTML = `<b style="color:${ccol(c.name)}">◆ ${esc(c.name)}</b><br>${fi(c.n_vectors)} vector train<br>‖μ‖ = ${fnum(c.mu_norm, 2)}<br>trace(Σ) = ${fnum(c.trace_cov, 2)}<br>
        ${c.pooled_cov ? "dùng covariance chung" : "covariance riêng"}<br><br><b>Centroid gần nhất (64 chiều):</b><br>${near.map(([v, k]) => `${esc(L.classes[k].name)}: ${fnum(v, 2)}`).join("<br>")}`;
    });
  };
  $("#lat-color").onchange = drawLat; $("#lat-pts").onchange = drawLat; drawLat();

  // 8. gallery over all train images
  let shown = 0, items = [], gen = 0;
  const q = () => { const p = new URLSearchParams(); $$("[data-f]").forEach((s) => { if (s.value) p.set(s.dataset.f, s.value); }); return p; };
  const more = async () => {
    const my = gen;                                   // a newer filter change makes this response stale
    const p = q(); p.set("offset", shown); p.set("limit", 24);
    $("#gal-more").disabled = true;
    let r;
    try { r = await api("/api/train/gallery?" + p.toString()); } catch (e) {
      $("#gal-total").textContent = "–"; $("#gal-of").textContent = "";
      $("#gal").innerHTML = `<div class="alert info">${esc(e.message)}</div>`; return;
    }
    if (my !== gen) return;
    items = items.concat(r.items); shown += r.items.length;
    $("#gal-total").textContent = fi(r.total);
    $("#gal-of").textContent = r.subset ? `ảnh có sẵn trên server (trong ${fi(r.total_all)} ảnh train khớp bộ lọc)` : `/ ${fi(t.images)} ảnh khớp bộ lọc`;
    $("#gal").insertAdjacentHTML("beforeend", r.items.map((g) => `<div class="thumb" data-name="${esc(g.name)}">${imageWithBoxes(`/api/train/image/${encodeURIComponent(g.name)}?w=360`, 1280, 720,
      g.boxes.map((b) => ({ b: b.slice(0, 4), color: ccol(NAMES[b[4]]) })))}<div class="cap"><span>${esc(ATTR_VI.weather[g.weather] || g.weather)} · ${esc(ATTR_VI.scene[g.scene] || g.scene)} · ${esc(ATTR_VI.timeofday[g.timeofday] || g.timeofday)}</span><span>${g.boxes.length} box</span></div></div>`).join(""));
    $("#gal-count").textContent = `đang hiện ${shown} / ${fi(r.total)} ảnh`;
    $("#gal-more").disabled = shown >= r.total;
  };
  const reset = () => { gen++; $("#gal").innerHTML = ""; shown = 0; items = []; more(); };
  $$("[data-f]").forEach((s) => s.onchange = reset);
  $("#gal-more").onclick = more; reset();
  $("#gal").onclick = (e) => {
    const el = e.target.closest(".thumb"); if (!el) return;
    const g = items.find((x) => x.name === el.dataset.name);
    const counts = {}; g.boxes.forEach((b) => { counts[NAMES[b[4]]] = (counts[NAMES[b[4]]] || 0) + 1; });
    modal(`<div class="row"><h2 style="margin:0">${esc(g.name)}</h2><span class="tag">${esc(attrLabel("weather", g.weather))}</span><span class="tag">${esc(attrLabel("scene", g.scene))}</span><span class="tag">${esc(attrLabel("timeofday", g.timeofday))}</span><span class="spacer"></span><button class="btn" data-close>Đóng</button></div>
      <div style="margin:10px 0">${imageWithBoxes(`/api/train/image/${encodeURIComponent(g.name)}`, 1280, 720, g.boxes.map((b) => ({ b: b.slice(0, 4), color: ccol(NAMES[b[4]]), label: NAMES[b[4]] })))}</div>
      <div class="legend">${Object.entries(counts).map(([n, c]) => `<span><i style="background:${ccol(n)}"></i>${esc(n)}: ${c}</span>`).join("")}</div>`);
  };
}

// ================================================================== ② TEST ẢNH
const jobStatus = (j) => !j ? '<span class="tag bad">chưa có</span>' : j.status === "done" ? '<span class="tag ok">sẵn sàng</span>' : j.status === "error" ? '<span class="tag bad">lỗi</span>'
  : `<span class="tag warn">${j.status === "running" ? "đang chạy " + (j.progress ? j.progress.done + "/" + j.progress.total : "") : "đang chờ GPU"}</span>`;

function detectorHtml(sys) {
  if (!sys.members) return `<div class="card"><span class="dot busy"></span> ${sys.mode === "web" ? "Chưa có thông tin detector: cần một máy GPU kết nối" : "Đang nạp 3 expert lên GPU (khoảng 40 giây sau khi mở server)…"}</div>`;
  const th = sys.thresholds;
  return `<div class="grid g4">
    ${sys.members.map((m, i) => `<div class="card"><div class="row"><b>${esc(m.name)}</b>${m.role ? '<span class="tag ok">Anchor</span>' : '<span class="tag">member</span>'}</div>
      <div class="hint">${esc(m.arch)} · ${esc(m.checkpoint)} · ${m.mb} MB</div><div class="hint">${i === 0 ? "Định ra object (query có score ≥ " + sys.detection.score_threshold + ") và cung cấp vector ẩn cho FCD và K-Means" : "Được ghép 1-1 với object của Anchor bằng Hungarian"}</div></div>`).join("")}
    <div class="card"><b>Ngưỡng đang dùng</b><div class="mono" style="margin-top:6px">τ_unc = ${f(th.tau_unc)}<br>τ_conf = ${f(th.tau_conf)}<br>τ_fcd = ${f(th.tau_fcd)}<br>ngân sách K = ${sys.diversity.budget_ratio * 100} % số ảnh</div>
      <div class="hint">${th.tau_conf > 1 ? "τ_conf > 1: LTT không chứng nhận được vùng Xanh, nên vùng Xanh luôn rỗng." : ""}</div></div>
  </div><div class="hint">GPU: ${esc(sys.device)} · 10 lớp: ${sys.classes.map(esc).join(", ")}</div>`;
}

function gpuPanelHtml(sys) {
  if (sys.mode !== "web") return "";
  const w = sys.workers || [], kg = sys.kaggle || {};
  const ago = (t) => t ? `${Math.max(0, Math.round((sys.now - t) / 60))} phút trước` : "chưa";
  return `<div class="grid g2">
    <div class="card"><b>Máy GPU đang kết nối</b>
      ${w.length ? `<table class="t" style="margin-top:6px"><tr><th>Tên</th><th>GPU</th><th>Đang làm</th></tr>${w.map((x) => `<tr><td>${esc(x.name)}</td><td>${esc(x.device)}</td><td style="white-space:normal">${esc(x.job || "rảnh")}</td></tr>`).join("")}</table>`
        : `<div class="hint">Chưa có máy nào. Job gửi lên sẽ nằm trong hàng đợi cho đến khi có máy GPU nhận.</div>`}
      <div class="hint">Máy của bạn: chạy <code>python dashboard/worker.py --server &lt;địa chỉ trang&gt; --token &lt;WORKER_TOKEN&gt;</code>. Máy nào nhận job trước thì chạy job đó.</div></div>
    <div class="card"><b>GPU Kaggle</b>
      ${kg.enabled ? `<div class="hint">Kernel <code>${esc(kg.kernel)}</code>, dataset <code>${esc(kg.dataset)}</code>. Tự bật khi có job chờ mà chưa có máy nào kết nối; tự tắt sau 10 phút không có việc. Lần bật gần nhất: ${ago(kg.last_push)}.</div>
        ${kg.last_error ? `<div class="alert err">${esc(kg.last_error)}</div>` : ""}
        <button class="btn small" id="kaggle-start" style="margin-top:8px">Bật GPU Kaggle ngay</button>`
        : `<div class="hint">Chưa cấu hình. Thiếu: ${esc((kg.missing || []).join(", "))}.</div>`}</div>
  </div>`;
}

function jobsHtml(jobs) {
  const link = (h, t) => `<a class="btn small" href="${h}">${t}</a>`;
  return `<div style="overflow-x:auto"><table class="t"><tr><th>Mô tả</th><th>Số ảnh</th><th>Trạng thái</th><th>Vùng</th><th>Lúc</th><th>Xem</th></tr>
    ${jobs.map((j) => { const id = encodeURIComponent(j.id); return `<tr>
      <td style="white-space:normal;min-width:180px">${esc(j.title || (j.kind === "single" ? "1 ảnh" : "batch"))}<div class="hint mono" style="margin:0">${esc(j.id)}</div></td>
      <td>${j.n_images}</td><td>${jobStatus(j)}</td>
      <td style="white-space:normal">${j.zones ? Object.entries(j.zones).map(([z, n]) => `<span style="white-space:nowrap">${zone(z)} ${n}</span>`).join(" ") : ""}</td>
      <td style="white-space:normal">${new Date(j.created * 1000).toLocaleString("vi-VN")}</td>
      <td style="white-space:normal;text-align:left"><div class="row" style="gap:4px">${link(`#/job/${id}`, "Tracing")}${j.status === "done" ? link(`#/batch/${id}`, "1 batch") + (j.n_images === 1 ? link(`#/single/${id}/${encodeURIComponent(j.image_names[0])}`, "Từng bước") : "") : ""}</div></td></tr>`; }).join("")}
  </table></div>`;
}

async function pageTest() {
  let [sys, ex, jobs] = await Promise.all([api("/api/system"), api("/api/examples"), api("/api/jobs")]);
  if (sys.classes) NAMES = sys.classes;
  main().innerHTML = `
    <div class="row"><div><h1>Test ảnh</h1><p class="lead" style="margin:0">Ảnh đi qua đúng code <code>ai_router</code> đang ship: 3 expert, ghép Hungarian, WBF, entropy và variance, FCD, LTT, K-Means. Hệ thống không sửa gì trong code gốc.</p></div>
      <span class="spacer"></span><button class="btn primary" id="upload">⬆ Đẩy ảnh của bạn</button></div>
    ${sys.mode === "web" ? `<h2 class="section">Máy xử lý ảnh</h2><div id="gpu-panel">${gpuPanelHtml(sys)}</div>` : ""}
    <h2 class="section">Detector</h2><div id="det">${detectorHtml(sys)}</div>
    <h2 class="section">Xem hệ thống vận hành</h2>
    <div class="grid g2">
      <div class="card big" id="go-single" role="button" tabindex="0"><div class="ico">🖼</div><div class="t">1 ảnh</div>
        <div>Đi từng bước một ảnh qua hệ thống, cho đến khi ảnh được xếp vào vùng Xanh, Xám hoặc Đỏ. Mỗi bước hiện công thức, ma trận và số thật. Bấm vào một object để xem vector đặc trưng và khoảng cách Mahalanobis tới centroid từng lớp.</div>
        <div class="hint">Dành cho người muốn đào sâu · ví dụ: ${esc(ex["example-single"]?.image_names?.[0] || "")} <span id="st-single">${jobStatus(ex["example-single"])}</span></div></div>
      <div class="card big" id="go-batch" role="button" tabindex="0"><div class="ico">🗂</div><div class="t">1 batch</div>
        <div>Cả lô qua pipeline: số ảnh mỗi vùng, mean và variance theo lớp, phân bố điểm so với ngưỡng, không gian ẩn với centroid, các cụm K-Means trên vùng Đỏ.</div>
        <div class="hint">Bức tranh lớn · ví dụ: ${ex["example-batch"] ? ex["example-batch"].n_images + " ảnh test BDD100K" : ""} <span id="st-batch">${jobStatus(ex["example-batch"])}</span></div></div>
    </div>
    <h2 class="section">Các lần chạy</h2><div class="card" id="jobs">${jobsHtml(jobs)}</div>`;
  const goSingle = () => { const j = ex["example-single"]; location.hash = j?.status === "done" ? `#/single/example-single/${encodeURIComponent(j.image_names[0])}` : "#/job/example-single"; };
  const goBatch = () => { location.hash = ex["example-batch"]?.status === "done" ? "#/batch/example-batch" : "#/job/example-batch"; };
  $("#go-single").onclick = goSingle; $("#go-batch").onclick = goBatch;
  $("#go-single").onkeydown = (e) => { if (e.key === "Enter") goSingle(); };
  $("#go-batch").onkeydown = (e) => { if (e.key === "Enter") goBatch(); };
  $("#upload").onclick = () => uploadDialog(sys, ex["example-batch"]);
  if ($("#gpu-panel")) $("#gpu-panel").onclick = async (e) => {
    if (e.target.id !== "kaggle-start") return;
    e.target.disabled = true; e.target.textContent = "Đang gửi lệnh bật…";
    try { await api("/api/kaggle/start", { method: "POST" }); e.target.textContent = "Đã gửi, GPU sẽ sẵn sàng sau 3–5 phút"; }
    catch (err) { e.target.textContent = "Lỗi: " + err.message; }
  };

  // while something is loading or running, refresh only the status parts (the cards stay in place)
  const pending = () => sys.mode === "web" || !sys.engine_loaded || [ex["example-single"], ex["example-batch"], ...jobs].some((j) => j && (j.status === "queued" || j.status === "running"));
  if (!pending()) return;
  let busy = false;
  const tm = setInterval(async () => {
    if (busy) return;
    busy = true;
    try {
      const wasLoaded = sys.engine_loaded;
      [sys, ex, jobs] = await Promise.all([api("/api/system"), api("/api/examples"), api("/api/jobs")]);
      if (!$("#jobs")) { clearInterval(tm); return; }
      if (sys.engine_loaded !== wasLoaded || !$("#det .grid")) $("#det").innerHTML = detectorHtml(sys);
      if ($("#gpu-panel") && !$("#gpu-panel").contains(document.activeElement)) $("#gpu-panel").innerHTML = gpuPanelHtml(sys);
      $("#st-single").innerHTML = jobStatus(ex["example-single"]); $("#st-batch").innerHTML = jobStatus(ex["example-batch"]);
      $("#jobs").innerHTML = jobsHtml(jobs);
      if (!pending()) clearInterval(tm);
    } catch (e) { console.warn("refresh", e); } finally { busy = false; }
  }, 3000);
  timers.push(tm);
}

function uploadDialog(sys, ref) {
  const ratio = sys?.diversity?.budget_ratio ?? 0.2;
  const redShare = ref?.zones_b5 ? ref.zones_b5.RED / ref.n_images : null;
  const lim = sys?.limits;
  modal(`<div class="row"><h2 style="margin:0">Đẩy ảnh của bạn vào hệ thống</h2><span class="spacer"></span><button class="btn" data-close>Đóng</button></div>
    <div class="tabs" style="margin-top:10px"><button class="on" data-mode="single">1 ảnh</button><button data-mode="batch">1 batch</button></div>
    <div id="up-body"></div>`);
  let mode = "single", files = [];
  const body = () => {
    const n = files.length, K = Math.ceil(ratio * n);
    $("#up-body").innerHTML = `
      <div class="drop" id="drop">${mode === "single" ? "Kéo 1 ảnh vào đây, hoặc" : "Kéo nhiều ảnh vào đây, hoặc"}<br><br>
        <label class="btn"><input type="file" id="fi" accept="image/*" ${mode === "batch" ? "multiple" : ""} hidden>Chọn ${mode === "single" ? "ảnh" : "các ảnh"}</label>
        ${mode === "batch" ? '<label class="btn"><input type="file" id="fd" webkitdirectory hidden>Chọn cả thư mục</label>' : ""}</div>
      <div style="margin-top:10px">${n ? `<b>${n}</b> ảnh · ${(sum(files.map((x) => x.size)) / 1048576).toFixed(1)} MB` : "Chưa chọn ảnh"}</div>
      ${lim ? `<div class="hint">Bản công khai: tối đa ${lim.images} ảnh mỗi lần gửi, mỗi ảnh tối đa ${lim.image_mb} MB, cả server nhận tối đa ${lim.queued_jobs} job chờ cùng lúc (để không ai dùng hết quota GPU).</div>` : ""}
      ${lim && n > lim.images ? `<div class="alert warn">Quá ${lim.images} ảnh: hãy chia thành nhiều lần gửi.</div>` : ""}
      ${mode === "batch" ? `<div class="why">Ngân sách review K = ⌈${ratio} × N⌉${n ? ` = <b>${K}</b>` : ""}. K-Means chỉ chạy khi số ảnh rơi vào vùng Đỏ <b>lớn hơn K</b>. Có nhiều ảnh hơn K là chưa đủ, phải có hơn K ảnh Đỏ.
        ${redShare !== null ? `Trên lô tham chiếu (${ref.n_images} ảnh test BDD100K), ${pct(redShare)} số ảnh rơi vào vùng Đỏ.${n ? ` Nếu lô của bạn giống vậy: khoảng <b>${Math.round(redShare * n)}</b> ảnh Đỏ so với K = ${K}, nên K-Means ${Math.round(redShare * n) > K ? "<b>sẽ chạy</b>" : "<b>có thể không chạy</b>"}.` : ""}` : ""}</div>` : `<div class="why">Ảnh được chạy qua đủ 7 bước. Trang "từng bước" sẽ mở khi xong. Ảnh lẻ có ngân sách K = 1, nên K-Means không chạy.</div>`}
      <div class="row" style="margin-top:10px"><input type="text" id="title" placeholder="Mô tả (không bắt buộc)" style="flex:1"><button class="btn primary" id="send" ${n && !(lim && n > lim.images) ? "" : "disabled"}>Gửi vào hệ thống</button></div>
      <div id="up-prog" style="margin-top:10px"></div>`;
    const take = (list) => { files = [...list].filter((x) => /\.(jpe?g|png|bmp|webp)$/i.test(x.name)); if (mode === "single") files = files.slice(0, 1); body(); };
    $("#fi").onchange = (e) => take(e.target.files);
    if ($("#fd")) $("#fd").onchange = (e) => take(e.target.files);
    const d = $("#drop");
    d.ondragover = (e) => { e.preventDefault(); d.classList.add("on"); };
    d.ondragleave = () => d.classList.remove("on");
    d.ondrop = (e) => { e.preventDefault(); d.classList.remove("on"); take(e.dataTransfer.files); };
    $("#send").onclick = () => send();
  };
  const send = () => {
    const fd = new FormData();
    files.forEach((x) => fd.append("files", x, x.name));
    fd.append("title", $("#title").value || "");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("#up-prog").innerHTML = `<div class="progress"><div style="width:${(100 * e.loaded / e.total).toFixed(1)}%"></div></div><div class="hint">Đang tải lên ${(e.loaded / 1048576).toFixed(1)} / ${(e.total / 1048576).toFixed(1)} MB</div>`; };
    xhr.onload = () => {
      if (xhr.status >= 300) {
        let msg = xhr.responseText; try { msg = JSON.parse(msg).detail || msg; } catch (_) { /* plain text */ }
        $("#up-prog").innerHTML = `<div class="alert err">${esc(msg)}</div>`; $("#send").disabled = false; return;
      }
      const j = JSON.parse(xhr.responseText); closeModal(); location.hash = `#/job/${encodeURIComponent(j.id)}`;
    };
    xhr.onerror = () => { $("#up-prog").innerHTML = `<div class="alert err">Không gửi được: mất kết nối server</div>`; };
    $("#send").disabled = true; xhr.send(fd);
  };
  $$("#modal .tabs button").forEach((b) => b.onclick = () => { mode = b.dataset.mode; files = []; $$("#modal .tabs button").forEach((x) => x.classList.toggle("on", x === b)); body(); });
  body();
}

// ================================================================== JOB: tracing + monitoring
const STAGE_VI = { upload: "Tải ảnh lên", wait_gpu: "Chờ máy GPU", load_models: "Nạp 3 expert", pipeline: "Pipeline B0→B7", explain: "Giải thích từng bước", monitoring: "Tính monitoring",
  gpu_worker: "Máy GPU: tải ảnh, B0→B7, giải thích" };
const SPAN_VI = { "B0.load_wait": "B0 đọc + resize ảnh", "B1.to_device": "B1 đưa lên GPU", "B1.to_host": "B1 lấy kết quả về CPU", "B2-B4.match_score_aggregate": "B2–B4 Hungarian, điểm, gộp", "B5.route_fcd": "B5 xếp vùng + FCD", "B6.diversity_kmeans": "B6 K-Means", "B7.write_queues": "B7 ghi queue" };
const spanName = (n) => SPAN_VI[n] || (n.startsWith("B1.forward.") ? "B1 forward " + n.slice(11) : n);

async function pageJob(id) {
  const render = async () => {
    const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
    const t0 = j.created, tEnd = j.finished || Date.now() / 1000, span = Math.max(tEnd - t0, 0.001);
    const done = j.status === "done";
    const head = `<div class="crumb"><a href="#/test">Test ảnh</a> › job</div>
      <div class="row"><h1>${esc(j.title || j.id)}</h1>${done ? '<span class="tag ok">xong</span>' : j.status === "error" ? '<span class="tag bad">lỗi</span>' : `<span class="tag warn">${j.status === "running" ? "đang chạy" : "đang chờ GPU"}</span>`}
        <span class="spacer"></span>${done ? `<a class="btn primary" href="#/batch/${encodeURIComponent(id)}">🗂 Xem cả batch</a>${j.n_images === 1 ? ` <a class="btn primary" href="#/single/${encodeURIComponent(id)}/${encodeURIComponent(j.image_names[0])}">🖼 Xem từng bước</a>` : ""}` : ""}</div>
      <p class="lead mono">${esc(j.id)} · ${j.n_images} ảnh · tạo lúc ${new Date(j.created * 1000).toLocaleString("vi-VN")}</p>`;
    const prog = j.progress || { done: 0, total: j.n_images };
    const stages = j.stages.map((s) => {
      const e = s.end || Date.now() / 1000;
      return `<div class="lane"><span>${esc(STAGE_VI[s.name] || s.name)}</span><div class="track"><div class="bar" style="left:${(100 * (s.start - t0) / span).toFixed(2)}%;width:${Math.max(0.4, 100 * (e - s.start) / span).toFixed(2)}%;background:${s.end ? "#2563eb" : "#f59e0b"}"></div></div><span class="mono">${(e - s.start).toFixed(1)} s${s.end ? "" : " …"}</span></div>`;
    }).join("");
    let html = head + `<div class="card"><h2>Tracing: job đang ở đâu</h2>
      <div class="row"><div style="flex:1"><div class="progress"><div style="width:${(100 * prog.done / Math.max(prog.total, 1)).toFixed(1)}%"></div></div></div><span class="mono">${prog.done} / ${prog.total} ảnh</span></div>
      <div class="gantt" style="margin-top:12px">${stages}</div>
      <div class="hint">Tổng: ${span.toFixed(1)} s. Vàng = đang chạy. Bước "Chờ GPU" dài nghĩa là còn job khác đang chạy trước.</div>
      ${j.status === "error" ? `<div class="alert err">${esc(j.error)}</div><pre class="formula">${esc(j.traceback || "")}</pre>` : ""}</div>`;
    if (j.monitoring_ready) html += `<div id="mon"></div>`;
    main().innerHTML = html;
    if (j.monitoring_ready) await renderMonitoring(id);
    return j.status;
  };
  const st = await render();
  if (st === "queued" || st === "running") {
    let busy = false;
    const tm = setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        const s = await render();
        if (s !== "queued" && s !== "running") clearInterval(tm);
      } catch (e) {
        clearInterval(tm);
        main().insertAdjacentHTML("afterbegin", `<div class="alert err">Mất theo dõi job: ${esc(e.message)}</div>`);
      } finally { busy = false; }
    }, 1000);
    timers.push(tm);
  }
}

async function renderMonitoring(id) {
  const [m, s] = await Promise.all([api(`/api/jobs/${encodeURIComponent(id)}/monitoring`), api(`/api/jobs/${encodeURIComponent(id)}/summary`)]);
  const sys = m.system, d = m.data;
  $("#mon").innerHTML = `
    <div class="card section"><h2>Tracing: thời gian từng bước của pipeline</h2>
      <div class="hint">Lấy từ tracer có sẵn của ai_router (<code>trace/spans.jsonl</code>). Mỗi thanh là một lần chạy bước đó trên một batch 8 ảnh.</div>
      <div id="p-spans" class="plot" style="min-height:320px"></div>
      <table class="t"><tr><th>Bước</th><th>Số lần</th><th>Tổng (ms)</th><th>Trung bình (ms)</th><th>% thời gian</th></tr>
        ${(() => { const tot = sum(sys.spans.map((x) => x.total_ms)) || 1; return sys.spans.map((x) => `<tr><td>${esc(spanName(x.name))}</td><td>${x.n}</td><td>${fi(Math.round(x.total_ms))}</td><td>${f(x.mean_ms, 1)}</td><td>${pct(x.total_ms / tot)}</td></tr>`).join(""); })()}</table></div>
    <div class="grid g2 section">
      <div class="card"><h2>Monitoring hệ thống: tốc độ từng batch</h2><div id="p-ips" class="plot"></div><div class="hint">Trung bình ${f(s.images_per_s, 1)} ảnh/s · đỉnh VRAM ${f(s.peak_vram_mb, 0)} MB · tổng ${f(s.elapsed_s, 1)} s</div></div>
      <div class="card"><h2>Monitoring hệ thống: bộ nhớ</h2><div id="p-mem" class="plot"></div></div>
    </div>
    <div class="card section"><h2>Monitoring dữ liệu: dữ liệu bạn gửi có giống dữ liệu đã thấy không?</h2>
      ${m.alerts.length ? m.alerts.map((a) => `<div class="alert ${a.level}">${esc(a.text)}</div>`).join("") : '<div class="alert info">Không có cảnh báo.</div>'}
      <div class="grid g2" style="margin-top:10px">
        <div><h3>Tỉ lệ lớp: phát hiện trong lô vs GT train (TVD = ${f(d.class_share_tvd, 3)})</h3><div id="p-cls" class="plot"></div></div>
        <div><h3>So với lô tham chiếu (1,000 ảnh test BDD100K): kiểm định Kolmogorov–Smirnov</h3>
          ${d.vs_reference ? `<table class="t" style="table-layout:fixed"><colgroup><col><col style="width:62px"><col style="width:72px"><col style="width:46px"><col style="width:56px"><col style="width:52px"></colgroup>
            <tr><th style="white-space:normal">Đại lượng (mean)</th><th style="white-space:normal">Lô này</th><th style="white-space:normal">Tham chiếu</th><th>KS</th><th>p</th><th></th></tr>
            ${d.vs_reference.tests.map((x) => `<tr><td style="white-space:normal">${esc(x.label)}</td><td>${f(x.mean, 3)}</td><td>${f(x.mean_ref, 3)}</td><td>${f(x.ks, 2)}</td><td>${x.p >= 0.001 ? x.p.toFixed(3) : x.p.toExponential(0)}</td><td>${x.p < 0.01 ? '<span class="tag warn">lệch</span>' : '<span class="tag ok">giống</span>'}</td></tr>`).join("")}</table>
            <div class="hint">KS = khoảng cách lớn nhất giữa hai đường phân phối tích luỹ (0 là giống hệt, 1 là khác hẳn). Mỗi ảnh tính 1 giá trị. p &lt; 0.01 thì báo "lệch". Lô càng ít ảnh thì phép kiểm càng khó thấy lệch (lô này ${s.n_images} ảnh).</div>
            <div class="hint">Tỉ lệ ảnh Đỏ trước K-Means: lô này ${pct(d.red_share)}, tham chiếu ${pct(d.vs_reference.red_share_ref)}.</div>` : '<div class="hint">Đây là lô tham chiếu, hoặc lô tham chiếu chưa chạy xong.</div>'}
          <h3 style="margin-top:12px">Object / ảnh</h3><div class="hint">Lô này: <b>${f(d.objects_per_image.batch_mean, 2)}</b> object được phát hiện mỗi ảnh (query của Anchor có score ≥ 0.30, gồm cả box sai) · GT train: <b>${f(d.objects_per_image.train_gt_mean, 2)}</b> box/ảnh.
            ${d.objects_per_image.batch_mean > d.objects_per_image.train_gt_mean ? "Detector báo nhiều object hơn số box GT trung bình: có thể ảnh của bạn đông object hơn train, hoặc có nhiều box thừa (FP)." : "Detector báo ít object hơn số box GT trung bình: thường là do bỏ sót object nhỏ hoặc bị che."}</div>
          <div class="hint">Cỡ ảnh: ${Object.entries(d.image_sizes).map(([k, v]) => `${esc(k)} (${v})`).join(", ")} · ảnh không có object: ${d.empty_images}</div></div>
      </div></div>`;
  const tl = sys.timeline;
  const rows = [...new Set(tl.map((x) => spanName(x.name)))];
  plot("p-spans", [{ type: "bar", orientation: "h", y: tl.map((x) => spanName(x.name)), base: tl.map((x) => x.start_s * 1000), x: tl.map((x) => x.dur_ms),
    marker: { color: tl.map((x) => x.name.startsWith("B1") ? "#2563eb" : x.name.startsWith("B0") ? "#94a3b8" : "#16a34a") },
    customdata: tl.map((x) => [x.batch, x.dur_ms]), hovertemplate: "%{y}<br>batch %{customdata[0]} · %{customdata[1]:.1f} ms<extra></extra>" }],
  { xaxis: { title: "thời điểm từ lúc bắt đầu chạy (ms)" }, yaxis: { categoryorder: "array", categoryarray: rows.reverse() }, margin: { l: 190, r: 10, t: 10, b: 40 }, height: 60 + 26 * rows.length, bargap: 0.3 });
  const mt = sys.metrics;
  plot("p-ips", [{ type: "scatter", mode: "lines+markers", x: mt.map((x) => x.batch), y: mt.map((x) => x.images_per_s), line: { color: "#2563eb" } }], { xaxis: { title: "batch" }, yaxis: { title: "ảnh / giây" }, height: 260 });
  plot("p-mem", [{ type: "scatter", mode: "lines", name: "VRAM đỉnh (MB)", x: mt.map((x) => x.batch), y: mt.map((x) => x.cuda_peak_mb), line: { color: "#dc2626" } },
    { type: "scatter", mode: "lines", name: "RAM tiến trình (MB)", x: mt.map((x) => x.batch), y: mt.map((x) => x.rss_mb), line: { color: "#64748b" }, yaxis: "y2" }],
  { xaxis: { title: "batch" }, yaxis: { title: "VRAM (MB)" }, yaxis2: { title: "RAM (MB)", overlaying: "y", side: "right" }, showlegend: true, legend: { orientation: "h" }, height: 260, margin: { l: 50, r: 60, t: 10, b: 40 } });
  if (d.class_share) plot("p-cls", [
    { type: "bar", name: "phát hiện trong lô", x: d.class_share.map((x) => x.name), y: d.class_share.map((x) => 100 * x.detected), marker: { color: "#2563eb" } },
    { type: "bar", name: "GT train", x: d.class_share.map((x) => x.name), y: d.class_share.map((x) => 100 * x.train_gt), marker: { color: "#94a3b8" } }],
  { barmode: "group", yaxis: { title: "% số box" }, showlegend: true, legend: { orientation: "h" }, height: 280, margin: { l: 50, r: 10, t: 10, b: 80 } });
}

// ================================================================== 1 BATCH
async function pageBatch(id) {
  main().innerHTML = `<div class="loading">Đang tải kết quả batch…</div>`;
  let j = await api(`/api/jobs/${encodeURIComponent(id)}`);
  if (j.status !== "done") { location.hash = `#/job/${encodeURIComponent(id)}`; return; }
  const s = await api(`/api/jobs/${encodeURIComponent(id)}/summary`);
  const th = s.thresholds, km = s.kmeans, imgs = s.images;
  const metrics = { unc: ["unc_total = entropy + 10 × variance toạ độ", "unc_mean", "unc_var"], conf: ["mean_conf (confidence trung bình 3 expert)", "conf_mean", "conf_var"], fcd: ["D_M (khoảng cách Mahalanobis tới centroid lớp)", "fcd_mean", "fcd_var"] };
  main().innerHTML = `
    <div class="crumb"><a href="#/test">Test ảnh</a> › 1 batch</div>
    <div class="row"><h1>${esc(j.title || id)}</h1><span class="spacer"></span><a class="btn" href="#/job/${encodeURIComponent(id)}">⏱ Tracing và monitoring</a></div>
    <p class="lead">${s.n_images} ảnh đi qua B0→B7 trong ${f(s.elapsed_s, 1)} s (${f(s.images_per_s, 1)} ảnh/s). Ngân sách review K = ⌈0.2 × ${s.n_images}⌉ = <b>${s.budget}</b>.</p>
    <div class="grid g4">
      <div class="card kpi"><div class="v" style="color:var(--red)">${s.zones_b5.RED} → ${s.zones.RED}</div><div class="l">Đỏ: sau bước xếp vùng (B5) → sau K-Means (B6)</div></div>
      <div class="card kpi"><div class="v" style="color:var(--grey)">${s.zones_b5.GREY} → ${s.zones.GREY}</div><div class="l">Xám (ảnh Đỏ bị K-Means loại thì chuyển sang Xám)</div></div>
      <div class="card kpi"><div class="v" style="color:var(--green)">${s.zones.GREEN}</div><div class="l">Xanh${th.tau_conf > 1 ? " (τ_conf = " + th.tau_conf + " > 1: không thể mở)" : ""}</div></div>
      <div class="card kpi"><div class="v">${fi(s.latent ? s.latent.n_total : 0)}</div><div class="l">object (query của Anchor có score ≥ 0.30)</div></div>
    </div>
    <div class="grid g2 section">
      <div class="card"><h2>Mean và variance theo lớp</h2><div class="row"><select id="bm">${Object.entries(metrics).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join("")}</select></div>
        <div id="p-bm" class="plot"></div><div class="hint">Cột = mean, thanh lỗi = ± 1 std. Số trên cột là số object của lớp trong lô.</div></div>
      <div class="card"><h2>Ảnh được xếp vùng thế nào (B4 → B5)</h2><div class="row"><select id="bh"><option value="unc">img_unc = MAX unc_total của ảnh, so với τ_unc</option><option value="conf">img_conf = MIN mean_conf của ảnh, so với τ_conf</option><option value="fcd">D_M của từng object, so với τ_fcd</option></select></div>
        <div id="p-bh" class="plot"></div></div>
    </div>
    <div class="grid g2 section">
      <div class="card"><h2>Không gian ẩn của object + centroid từng lớp</h2><div id="p-lat" class="plot" style="min-height:420px"></div>
        <div class="hint">Vector 256 chiều của Anchor → PCA 64 chiều (giống FCD) → chiếu 2 chiều (PC1 ${s.latent ? pct(s.latent.explained[0]) : ""}, PC2 ${s.latent ? pct(s.latent.explained[1]) : ""} phương sai). ★ = centroid lớp, fit trên train. Điểm viền đỏ = D_M > τ_fcd. Bấm một điểm để mở ảnh chứa object đó.</div></div>
      <div class="card"><h2>K-Means trên các ảnh vùng Đỏ</h2>${km.ran ? `<div id="p-km" class="plot" style="min-height:420px"></div>
        <div class="hint">${km.red_before} ảnh Đỏ > K = ${km.budget}, nên K-Means chia thành ${km.budget} cụm trên vector cảnh 256 chiều (đã chuẩn hoá L2). Mỗi cụm giữ 1 ảnh: ảnh ưu tiên cao nhất (P1 trước P2) và gần centroid nhất (◆ viền đen). ✕ = centroid. Inertia = ${f(km.inertia, 2)}.
        ${km.replicates_pipeline ? '<span class="tag ok">Chạy lại khớp đúng ảnh mà pipeline đã chọn</span>' : '<span class="tag bad">Khác với pipeline</span>'}</div>
        <div id="p-kmsz" class="plot" style="min-height:180px"></div>` : `<div class="why">K-Means không chạy: số ảnh Đỏ (${km.red_before}) ≤ K (${km.budget}), nên giữ hết. Hãy gửi lô lớn hơn để có hơn K ảnh Đỏ.</div>`}</div>
    </div>
    <div class="card section"><div class="row"><h2 style="margin:0">Ảnh trong batch</h2><span class="spacer"></span>
      <div class="tabs" id="zt" style="margin:0;border:0">${["RED", "GREY", "DEDUP", "GREEN", "ALL"].map((z, i) => `<button data-z="${z}" class="${i === 0 ? "on" : ""}">${{ RED: "Đỏ (vào queue)", GREY: "Xám", DEDUP: "Bị K-Means loại", GREEN: "Xanh", ALL: "Tất cả" }[z]}</button>`).join("")}</div>
      <select id="sort"><option value="unc">xếp theo img_unc ↓</option><option value="conf">xếp theo img_conf ↑</option><option value="fcd">xếp theo D_M lớn nhất ↓</option><option value="name">theo tên</option></select></div>
      <div id="grid" class="thumbs" style="margin-top:10px"></div><div class="row" style="margin-top:10px"><button class="btn" id="more">Xem thêm</button><span class="hint" id="cnt"></span></div></div>`;

  const drawBM = () => {
    const [label, mk, vk] = metrics[$("#bm").value];
    const pc = s.per_class.filter((c) => c.n > 0 && c[mk] !== null);
    plot("p-bm", [{ type: "bar", x: pc.map((c) => c.name), y: pc.map((c) => c[mk]), marker: { color: pc.map((c) => ccol(c.name)) }, text: pc.map((c) => c.n), textposition: "outside",
      error_y: { type: "data", array: pc.map((c) => Math.sqrt(c[vk])), visible: true, color: "#475569" }, hovertemplate: "%{x}<br>mean %{y:.4f}<extra></extra>" }],
    { yaxis: { title: label.split(" (")[0] }, height: 320, margin: { l: 60, r: 10, t: 20, b: 80 },
      shapes: $("#bm").value === "fcd" ? [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: th.tau_fcd, y1: th.tau_fcd, line: { color: "#dc2626", dash: "dash" }, label: { text: "τ_fcd", textposition: "end" } }] : [] });
  };
  const drawBH = () => {
    const k = $("#bh").value;
    const byZ = (z) => imgs.filter((x) => x.zone_b5 === z && x.n > 0);
    let data, shapes, title;
    if (k === "fcd") {
      data = [{ type: "histogram", x: s.objects.d.filter((v) => v !== null), nbinsx: 60, marker: { color: "#64748b" } }];
      shapes = [vline(th.tau_fcd, "τ_fcd = " + th.tau_fcd.toFixed(2))]; title = "D_M của object";
    } else {
      const key = k === "unc" ? "img_unc" : "img_conf";
      data = ["RED", "GREY", "GREEN"].map((z) => ({ type: "histogram", name: ZONE_VI[z], x: byZ(z).map((x) => x[key]), nbinsx: 40, marker: { color: { RED: "#dc2626", GREY: "#9ca3af", GREEN: "#16a34a" }[z] }, opacity: 0.8 }));
      shapes = [vline(k === "unc" ? th.tau_unc : Math.min(th.tau_conf, 1.0), k === "unc" ? "τ_unc = " + th.tau_unc.toFixed(3) : "τ_conf = " + th.tau_conf)]; title = key;
    }
    plot("p-bh", data, { barmode: "stack", xaxis: { title }, yaxis: { title: "số lượng" }, shapes, showlegend: k !== "fcd", legend: { orientation: "h" }, height: 320 });
  };
  $("#bm").onchange = drawBM; drawBM(); $("#bh").onchange = drawBH; drawBH();

  const openImg = (i) => { location.hash = `#/single/${encodeURIComponent(id)}/${encodeURIComponent(imgs[i].name)}`; };
  if (s.latent) {
    const L = s.latent;
    const traces = NAMES.map((n, c) => {
      const ix = L.cls.map((v, i) => v === c ? i : -1).filter((i) => i >= 0);
      return { type: "scattergl", mode: "markers", name: n, x: ix.map((i) => L.x[i]), y: ix.map((i) => L.y[i]), customdata: ix.map((i) => [L.img[i], L.d[i], imgs[L.img[i]].name]),
        marker: { size: 5, color: ccol(n), opacity: 0.6, line: { color: ix.map((i) => L.d[i] !== null && L.d[i] > th.tau_fcd ? "#dc2626" : "rgba(0,0,0,0)"), width: 1.2 } },
        hovertemplate: `${n}<br>D_M = %{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>` };
    }).filter((t) => t.x.length);
    traces.push({ type: "scatter", mode: "markers+text", name: "centroid", x: L.centroids.map((c) => c.xy[0]), y: L.centroids.map((c) => c.xy[1]), text: L.centroids.map((c) => c.name), textposition: "top center",
      marker: { symbol: "star", size: 16, color: L.centroids.map((c) => ccol(c.name)), line: { color: "#111", width: 1.5 } }, hovertemplate: "centroid %{text}<extra></extra>" });
    plot("p-lat", traces, { showlegend: true, legend: { orientation: "h" }, xaxis: { title: "PC1", zeroline: false }, yaxis: { title: "PC2", zeroline: false }, height: 420, margin: { l: 50, r: 10, t: 10, b: 40 } });
    $("#p-lat").on("plotly_click", (e) => { const p = e.points[0]; if (p.customdata && p.customdata.length === 3) openImg(p.customdata[0]); });
  }
  if (km.ran) {
    const kept = new Set(km.kept);
    plot("p-km", [
      { type: "scattergl", mode: "markers", x: km.x, y: km.y, customdata: km.img.map((i, j) => [i, imgs[i].name, km.label[j], km.sizes[km.label[j]]]),
        marker: { size: km.img.map((i) => kept.has(i) ? 11 : 6), symbol: km.img.map((i) => kept.has(i) ? "diamond" : "circle"), color: km.label, colorscale: "Turbo", line: { color: km.img.map((i) => kept.has(i) ? "#000" : "rgba(0,0,0,0)"), width: 1.5 }, opacity: 0.85 },
        hovertemplate: "%{customdata[1]}<br>cụm %{customdata[2]} (%{customdata[3]} ảnh)<extra></extra>" },
      { type: "scatter", mode: "markers", x: km.cx, y: km.cy, marker: { symbol: "x", size: 7, color: "#111" }, hoverinfo: "skip" }],
    { xaxis: { title: "PC1 (vector cảnh)", zeroline: false }, yaxis: { title: "PC2", zeroline: false }, height: 420, margin: { l: 50, r: 10, t: 10, b: 40 } });
    $("#p-km").on("plotly_click", (e) => { const p = e.points[0]; if (p.customdata) openImg(p.customdata[0]); });
    const h = km.cluster_size_hist;
    plot("p-kmsz", [{ type: "bar", x: h.map((_, i) => i), y: h, marker: { color: "#7c3aed" } }], { xaxis: { title: "số ảnh trong một cụm" }, yaxis: { title: "số cụm" }, height: 180, margin: { l: 50, r: 10, t: 10, b: 40 } });
  }

  // image grid
  let zsel = "RED", shown = 0;
  const filt = () => {
    let L = imgs.map((x, i) => ({ ...x, i }));
    if (zsel === "DEDUP") L = L.filter((x) => x.dedup); else if (zsel !== "ALL") L = L.filter((x) => x.zone === zsel);
    const k = $("#sort").value;
    const key = { unc: (x) => -(x.img_unc ?? -1), conf: (x) => (x.img_conf ?? 9), fcd: (x) => -(x.fcd_max_all ?? -1), name: (x) => x.name };
    L.sort((a, b) => { const A = key[k](a), B = key[k](b); return A < B ? -1 : A > B ? 1 : 0; });
    return L;
  };
  const more = () => {
    const L = filt(), next = L.slice(shown, shown + 30); shown += next.length;
    $("#grid").insertAdjacentHTML("beforeend", next.map((x) => `<div class="thumb ${x.zone}" data-i="${x.i}"><div class="imgbox" style="aspect-ratio:${x.size ? x.size[0] + " / " + x.size[1] : "16 / 9"}"><img src="/api/jobs/${encodeURIComponent(id)}/image/${encodeURIComponent(x.name)}?w=320" loading="lazy" onerror="this.parentElement.classList.add('noimg')"></div>
      <div class="cap"><span>${zone(x.zone)} ${x.priority !== "P3" ? `<span class="tag">${x.priority}</span>` : ""}${x.dedup ? '<span class="tag warn">K-Means loại</span>' : ""}</span><span class="mono">unc ${f(x.img_unc, 2)} · ${x.n} obj</span></div></div>`).join(""));
    $("#cnt").textContent = `${shown} / ${L.length} ảnh`; $("#more").disabled = shown >= L.length;
  };
  const reset = () => { $("#grid").innerHTML = ""; shown = 0; more(); };
  $$("#zt button").forEach((b) => b.onclick = () => { zsel = b.dataset.z; $$("#zt button").forEach((x) => x.classList.toggle("on", x === b)); reset(); });
  $("#sort").onchange = reset; $("#more").onclick = more; reset();
  $("#grid").onclick = (e) => { const el = e.target.closest(".thumb"); if (el) openImg(+el.dataset.i); };
}

// ================================================================== 1 ẢNH: từng bước
const STEPS = [["B0", "Ảnh vào"], ["B1", "3 expert"], ["B2", "Hungarian"], ["B3", "Entropy, variance, WBF"], ["B4", "Gộp MAX/MIN"], ["B5", "FCD + xếp vùng"], ["B6", "K-Means"], ["B7", "Đầu ra"], ["CHK", "Kiểm chứng"]];

async function pageSingle(id, name) {
  main().innerHTML = `<div class="loading">Đang tính lại từng bước cho ${esc(name)} trên GPU (vài giây cho lần đầu)…</div>`;
  const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
  if (j.status !== "done") { location.hash = `#/job/${encodeURIComponent(id)}`; return; }
  let d = await api(`/api/jobs/${encodeURIComponent(id)}/explain/${encodeURIComponent(name)}`);
  for (let tries = 0; d.pending; tries++) {
    main().innerHTML = `<div class="loading">Ảnh ${esc(name)} chưa được tính từng bước. Đã gửi yêu cầu cho máy GPU${d.workers ? "" : d.kaggle ? " (Kaggle đang được bật, mất 3–5 phút)" : ". Hiện chưa có máy GPU nào kết nối"}… (${tries * 3} s)</div>`;
    await new Promise((r) => setTimeout(r, 3000));
    if (!location.hash.includes(encodeURIComponent(name))) return;
    d = await api(`/api/jobs/${encodeURIComponent(id)}/explain/${encodeURIComponent(name)}`);
  }
  NAMES = d.names;
  const S = { sel: d.objects.length ? (d.B4.k_unc ?? 0) : null, step: "B0", layer: "obj", member: 0, gt: false };
  const src = `/api/jobs/${encodeURIComponent(id)}/image/${encodeURIComponent(name)}`;
  const W = d.B0.W, H = d.B0.H, th = d.thresholds;
  main().innerHTML = `
    <div class="crumb"><a href="#/test">Test ảnh</a> › ${j.n_images > 1 ? `<a href="#/batch/${encodeURIComponent(id)}">batch</a> › ` : ""}1 ảnh</div>
    <div class="row"><h1>${esc(name)}</h1>${zone(d.B5.zone)} <span class="tag">${esc(d.B5.priority)}</span>${d.B5.reasons.map((r) => `<span class="tag">${esc(r)}</span>`).join("")}
      <span class="spacer"></span><span class="tag ${d.checks.every((c) => c.ok) ? "ok" : "bad"}">${d.checks.filter((c) => c.ok).length}/${d.checks.length} phép tính tay khớp code gốc</span></div>
    ${d.gt ? `<div class="why row" style="gap:12px"><label style="white-space:nowrap;font-weight:600"><input type="checkbox" id="showgt"> Hiện ground truth</label>
      <span>Ảnh này có sẵn file nhãn trong bộ dữ liệu (thư mục <code>${esc(d.gt.source)}</code> của dự án có nhãn cho mọi ảnh). <b>Hệ thống không đọc nhãn này ở bất kỳ bước nào.</b> Mặc định ẩn; bật lên chỉ để chấm xem hệ thống đúng hay sai. Ảnh bạn tự gửi lên không có nhãn nên không có ô này.</span></div>` : ""}
    <div class="grid" style="grid-template-columns:minmax(0,5fr) minmax(0,6fr);align-items:start;margin-top:10px">
      <div style="position:sticky;top:10px">
        <div class="card tight"><div class="row" style="margin-bottom:6px"><span>Lớp phủ:</span><select id="layer">
          </select></div>
          <div id="img"></div><div class="hint">Ảnh gốc ${W}×${H}. Bấm vào một box để chọn object. Các bước B2, B3 và B5 sẽ hiện số của object đó.</div></div>
        <div class="card tight" style="margin-top:10px"><div class="scroll" style="max-height:300px"><table class="t" id="objtab"></table></div></div>
      </div>
      <div><div class="card"><div class="steps" id="steps">${STEPS.map(([k, v]) => `<button data-s="${k}"><span class="n">${k.startsWith("B") ? k : ""}</span>${v}</button>`).join("")}</div>
        <div id="step"></div>
        <div class="row" style="margin-top:14px"><button class="btn" id="prev">← Bước trước</button><span class="spacer"></span><button class="btn primary" id="next">Bước sau →</button></div></div>
        <div class="card" id="gtpanel" style="margin-top:10px;display:none"></div></div>
    </div>`;
  const layerOptions = () => {
    $("#layer").innerHTML = `<option value="obj">Object sau WBF (bấm để chọn)</option>${d.members.map((m, i) => `<option value="m${i}">Box thô của ${esc(m)}</option>`).join("")}`
      + (S.gt ? '<option value="gt">Ground truth</option><option value="objgt">Object + GT (đúng xanh, sai đỏ)</option>' : "");
    if (!S.gt && (S.layer === "gt" || S.layer === "objgt")) S.layer = "obj";
    $("#layer").value = S.layer;
  };

  const objColor = (o) => d.gt && o.tp !== undefined && S.layer === "objgt" ? (o.tp ? "#22c55e" : "#ef4444") : ccol(o.cls);
  const drawImg = () => {
    let boxes = [];
    if (S.layer === "obj" || S.layer === "objgt") boxes = d.objects.map((o) => ({ b: o.box, color: objColor(o), label: `#${o.k} ${o.cls}`, id: o.k, sel: o.k === S.sel }));
    if (S.layer.startsWith("m")) { const m = d.B1.models[+S.layer.slice(1)]; boxes = m.dets.map((x) => ({ b: x.box, color: ccol(x.cls), label: `${x.cls} ${x.score.toFixed(2)}` })); }
    if (S.layer === "gt" || S.layer === "objgt") boxes = boxes.concat(d.gt.boxes.map((b, i) => ({ b, color: "#ffffff", dash: true, label: S.layer === "gt" ? d.gt.cls[i] : "" })));
    $("#img").innerHTML = imageWithBoxes(src, W, H, boxes);
    $$("#img rect.click").forEach((r) => r.onclick = () => select(+r.dataset.id));
  };
  const drawTab = () => {
    $("#objtab").innerHTML = `<tr><th>#</th><th>Lớp</th><th>mean_conf</th><th>unc_total</th><th>D_M</th><th></th></tr>` + d.objects.map((o) => `<tr class="click ${o.k === S.sel ? "sel" : ""}" data-k="${o.k}">
      <td>${o.k}</td><td><i style="display:inline-block;width:9px;height:9px;background:${ccol(o.cls)}"></i> ${esc(o.cls)}</td><td>${f(o.mean_conf, 3)}</td><td>${f(o.unc_total, 3)}</td><td>${f(o.dm, 2)}</td>
      <td>${o.fcd_flag ? '<span class="tag bad">FCD</span>' : ""}${o.unmatched ? '<span class="tag warn">lệch</span>' : ""}${S.gt && o.tp === true ? '<span class="tag ok">TP</span>' : S.gt && o.tp === false ? '<span class="tag bad">FP</span>' : ""}</td></tr>`).join("");
    $$("#objtab tr.click").forEach((r) => r.onclick = () => select(+r.dataset.k));
  };
  const select = (k) => { S.sel = k; drawImg(); drawTab(); if (["B2", "B3", "B5"].includes(S.step)) drawStep(); else { S.step = "B3"; drawStep(); } };
  const order = STEPS.map(([k]) => k);
  const drawStep = () => {
    $$("#steps button").forEach((b) => b.classList.toggle("on", b.dataset.s === S.step));
    const fn = STEP_RENDER[S.step];
    $("#step").innerHTML = fn ? fn.html(d, S) : "";
    if (fn && fn.after) fn.after(d, S, { select, redraw: drawStep });
    $("#prev").disabled = order.indexOf(S.step) === 0; $("#next").disabled = order.indexOf(S.step) === order.length - 1;
  };
  $$("#steps button").forEach((b) => b.onclick = () => { S.step = b.dataset.s; drawStep(); });
  $("#prev").onclick = () => { S.step = order[Math.max(0, order.indexOf(S.step) - 1)]; drawStep(); };
  $("#next").onclick = () => { S.step = order[Math.min(order.length - 1, order.indexOf(S.step) + 1)]; drawStep(); };
  $("#layer").onchange = (e) => { S.layer = e.target.value; drawImg(); };
  if ($("#showgt")) $("#showgt").onchange = (e) => {
    S.gt = e.target.checked;
    if (S.gt && S.layer === "obj") S.layer = "objgt";
    layerOptions(); drawImg(); drawTab();
    $("#gtpanel").style.display = S.gt ? "" : "none";
    $("#gtpanel").innerHTML = S.gt ? GT_PANEL(d) : "";
  };
  layerOptions(); drawImg(); drawTab(); drawStep();
}

const noObj = '<div class="why">Ảnh không có object nào có score ≥ ngưỡng, nên bước này không có gì để tính.</div>';
const STEP_RENDER = {
  B0: { html: (d) => `<h2>B0 · Đọc ảnh và đưa về 640×640</h2>
    <div class="formula">ảnh gốc        ${d.B0.W} × ${d.B0.H} px, ${fi(d.B0.bytes)} byte
resize bilinear → ${d.B0.in[1]} × ${d.B0.in[0]}   (RT-DETR processor: không giữ tỉ lệ, không normalize)
hệ số           sx = ${d.B0.W}/${d.B0.in[1]} = <span class="r">${f(d.B0.sx, 4)}</span>    sy = ${d.B0.H}/${d.B0.in[0]} = <span class="r">${f(d.B0.sy, 4)}</span>
pixel ∈ [0,255] → chia 255 → tensor [1, 3, 640, 640]
mean RGB        ${d.B0.mean_rgb.map((v) => f(v, 4)).join("  ")}
std  RGB        ${d.B0.std_rgb.map((v) => f(v, 4)).join("  ")}</div>
    <h3 style="margin-top:12px">8 pixel ở hàng y = 320, cột x = 316…323 (giá trị uint8 sau resize)</h3>
    ${matrix(d.B0.pixel_sample, { rowLabels: [316, 317, 318, 319, 320, 321, 322, 323].map((x) => "x=" + x), colLabels: ["R", "G", "B"], digits: 0, heat: true })}
    <div class="why">Box do model trả về được chuẩn hoá về [0, 1], rồi nhân với W, H của ảnh gốc để vẽ. Vì vậy resize không giữ tỉ lệ vẫn vẽ đúng chỗ.</div>
    <div class="hint">Thời gian: ${JSON.stringify(d.timings_ms)} ms</div>` },

  B1: { html: (d) => `<h2>B1 · Ba expert chạy cùng một tensor</h2>
    <div class="formula">mỗi expert: RT-DETR r50vd → ${d.B1.logit_shape[2]} query, mỗi query có 1 box (cx, cy, w, h) và ${d.B1.logit_shape[3]} logit lớp
score của query = max_c sigmoid(logit_c)        giữ query có score ≥ <span class="r">${d.detection.score_threshold}</span>
Anchor (${esc(d.members[0])}) còn trả vector ẩn ${d.B1.hidden_shape.slice(1).join(" × ")} cho mỗi query và vector cảnh ${d.B1.scene_dim} chiều (‖·‖ = ${f(d.B1.scene_norm, 2)})</div>
    <div class="grid g3" style="margin-top:10px">${d.B1.models.map((m, i) => `<div class="card tight"><b>${esc(m.name)}</b>${i === 0 ? ' <span class="tag ok">Anchor</span>' : ""}<div class="kpi"><span class="v">${m.kept}</span> <span class="l">/ ${m.queries} query ≥ ${d.detection.score_threshold}</span></div><div id="h${i}" style="height:150px"></div></div>`).join("")}</div>
    <h3 style="margin-top:10px">Các query được giữ</h3><div class="grid g3">${d.B1.models.map((m) => `<div class="scroll" style="max-height:240px"><table class="t"><tr><th>query</th><th>lớp</th><th>score</th></tr>${m.dets.map((x) => `<tr><td>${x.q}</td><td>${esc(x.cls)}</td><td>${f(x.score, 3)}</td></tr>`).join("")}</table></div>`).join("")}</div>
    <div class="why">Chọn lớp phủ "Box thô của expert_k" bên trái để xem box của từng model. Box của 3 model thường chồng lên nhau ở cùng một object: B2 sẽ ghép chúng lại.</div>`,
  after: (d) => d.B1.models.forEach((m, i) => plot(`h${i}`, [{ type: "bar", x: ["<.05", ".05–.1", ".1–.2", ".2–.3", ".3–.5", ".5–.7", ".7–.9", "≥.9"], y: m.hist, marker: { color: ["#cbd5e1", "#cbd5e1", "#cbd5e1", "#cbd5e1", "#2563eb", "#2563eb", "#2563eb", "#2563eb"] } }],
    { height: 150, margin: { l: 40, r: 5, t: 5, b: 40 }, yaxis: { type: "log" }, xaxis: { tickfont: { size: 9 } } })) },

  B2: { html: (d, S) => {
    if (!d.B2.K) return `<h2>B2 · Hungarian</h2>` + noObj;
    const hg = d.B2.hungarian[S.member], k = S.sel, w = d.B2.weights;
    const ci = hg.cols.indexOf(hg.chosen[k]);
    return `<h2>B2 · Ghép object của Anchor với từng member (Hungarian)</h2>
    <div class="formula">Anchor có <span class="r">K = ${d.B2.K}</span> object (query score ≥ ${d.detection.score_threshold}, tối đa ${d.detection.max_objects})
chi phí ghép object i của Anchor với query j của member:
  C[i,j] = <span class="k">${w.l1}</span>·L1(box_i, box_j) − <span class="k">${w.giou}</span>·GIoU(box_i, box_j) − <span class="k">${w.cls}</span>·sigmoid(logit_j[lớp của i])
Hungarian (linear_sum_assignment) chọn K cặp 1-1 sao cho tổng C nhỏ nhất
IoU(cặp) < ${d.B2.match_iou} → object "lệch" (UNMATCHED)</div>
    <div class="row" style="margin:10px 0"><span>Member:</span>${d.B2.hungarian.map((h, i) => `<button class="btn small ${i === S.member ? "primary" : ""}" data-m="${i}">${esc(h.member)}</button>`).join("")}
      <span class="hint">Ma trận C: K hàng × ${hg.cols.length} cột (các query được chọn + 3 query rẻ nhất của mỗi hàng; member có 300 query)</span></div>
    <div id="hm" style="height:${Math.min(520, 120 + 22 * d.B2.K)}px"></div>
    <h3>Object #${k} (${esc(d.B2.cls[k])}, score Anchor ${f(d.B2.score[k], 3)}): chi phí với từng query</h3>
    <div class="scroll" style="max-height:200px">${matrix([hg.l1[k], hg.giou[k], hg.pc[k], hg.cost[k]], { rowLabels: ["L1", "GIoU", "p(lớp)", "C"], colLabels: hg.cols.map((c, j) => "q" + c + " " + hg.member_cls[j]), hl: (i, j) => j === ci, digits: 3 })}</div>
    <div class="formula">query được chọn: <span class="r">q${hg.chosen[k]}</span> (${esc(hg.member_cls[ci] || "")})
C = ${w.l1}×${f(hg.l1[k][ci], 4)} − ${w.giou}×${f(hg.giou[k][ci], 4)} − ${w.cls}×${f(hg.pc[k][ci], 4)} = <span class="r">${f(hg.cost[k][ci], 4)}</span>
IoU(box Anchor, box member) = <span class="r">${f(hg.iou[k], 4)}</span> ${hg.iou[k] < d.B2.match_iou ? "< " + d.B2.match_iou + " → LỆCH" : "≥ " + d.B2.match_iou + " → khớp"}</div>`; },
  after: (d, S, ctx) => {
    if (!d.B2.K) return;
    $$("#step [data-m]").forEach((b) => b.onclick = () => { S.member = +b.dataset.m; ctx.redraw(); });
    const hg = d.B2.hungarian[S.member];
    const ylab = d.B2.cls.map((c, i) => `#${i} ${c}`), xlab = hg.cols.map((c) => "q" + c);
    plot("hm", [{ type: "heatmap", z: hg.cost, x: xlab, y: ylab, colorscale: "RdBu", reversescale: false, hovertemplate: "%{y} ↔ %{x}<br>C = %{z:.3f}<extra></extra>" },
      { type: "scatter", mode: "markers", x: hg.chosen.map((c) => "q" + c), y: ylab, marker: { symbol: "square-open", size: 14, color: "#facc15", line: { width: 3 } }, hoverinfo: "skip" }],
    { margin: { l: 120, r: 10, t: 10, b: 60 }, yaxis: { autorange: "reversed" }, xaxis: { tickfont: { size: 9 } } });
    $("#hm").on("plotly_click", (e) => { const y = e.points[0].y; const k = ylab.indexOf(y); if (k >= 0) ctx.select(k); });
  } },

  B3: { html: (d, S) => {
    if (S.sel === null) return `<h2>B3</h2>` + noObj;
    const o = d.objects[S.sel], M = d.members.length, C = d.names.length;
    const mrow = d.members.map((m, i) => `${m} (q${o.member_q[i]})`);
    return `<h2>B3 · Object #${o.k}: entropy, variance toạ độ, WBF</h2>
    <h3>1. Logit 3 model × ${C} lớp → softmax từng hàng</h3>${matrix(o.L, { rowLabels: mrow, colLabels: d.names, digits: 3, heat: true })}
    <div style="margin-top:6px">${matrix([...o.P, o.pbar, o.terms], { rowLabels: [...d.members.map((m) => "P " + m), "p̄ = mean", "−p̄·ln p̄"], colLabels: d.names, digits: 4, heat: false, hl: (i, j) => i === M && j === o.c })}</div>
    <div class="formula">lớp dự đoán c = argmax p̄ = <span class="r">${esc(o.cls)}</span>
unc_cls = Σ −p̄ ln p̄ = ${o.terms.map((v) => f(v, 4)).join(" + ")}
        = <span class="r">${f(o.unc_cls, 4)}</span>      (tối đa ln ${C} = ${Math.log(C).toFixed(4)} khi 3 model phân vân đều)</div>
    <h3 style="margin-top:10px">2. Box chuẩn hoá (cx, cy, w, h) của 3 model → variance theo từng toạ độ</h3>
    ${matrix([...o.B, o.var], { rowLabels: [...d.members, "variance"], colLabels: ["cx", "cy", "w", "h"], digits: 6 })}
    <div class="formula">unc_loc   = mean(variance) = (${o.var.map((v) => f(v, 7)).join(" + ")}) / 4 = <span class="r">${f(o.unc_loc, 7)}</span>
unc_total = unc_cls + ${d.detection.unc_loc_weight} × unc_loc = ${f(o.unc_cls, 4)} + ${d.detection.unc_loc_weight} × ${f(o.unc_loc, 7)} = <span class="r">${f(o.unc_total, 4)}</span></div>
    <h3 style="margin-top:10px">3. Confidence của lớp c ở từng model và WBF</h3>
    <table class="t"><tr><th>model</th><th>lớp riêng</th><th>sigmoid(logit_c)</th><th>IoU với Anchor</th><th>vào cụm WBF (IoU ≥ ${d.detection.wbf_iou})</th></tr>
      ${d.members.map((m, i) => `<tr><td>${esc(m)}</td><td>${esc(o.model_cls[i])}</td><td>${f(o.confs[i], 4)}</td><td>${i === 0 ? "—" : f(o.ious[i - 1], 3)}</td><td>${o.in_cluster[i] ? "✓" : "✗"}</td></tr>`).join("")}</table>
    <div class="formula">mean_conf = (${o.confs.map((v) => f(v, 4)).join(" + ")}) / ${M} = <span class="r">${f(o.mean_conf, 4)}</span>
box WBF   = Σ wᵢ·boxᵢ / Σ wᵢ trên ${o.T} box trong cụm (w = confidence) → xyxy chuẩn hoá ${o.fused.map((v) => f(v, 4)).join(", ")}
wbf_conf  = mean(w) × min(T, M)/M = ${f(o.wts.reduce((a, b) => a + b, 0) / o.wts.length, 4)} × ${Math.min(o.T, M)}/${M} = <span class="r">${f(o.wbf_conf, 4)}</span>
box (px)  = ${o.box.map((v) => f(v, 1)).join(", ")}${o.unmatched ? "\n⚠ có member IoU < " + d.detection.match_iou + " → object LỆCH" : ""}</div>`; } },

  B4: { html: (d) => {
    if (!d.objects.length) return `<h2>B4</h2>` + noObj;
    return `<h2>B4 · Gộp object thành 2 số của ảnh</h2>
    <div class="formula">img_unc  = MAX unc_total của các object = <span class="r">${f(d.B4.img_unc, 4)}</span>  (object #${d.B4.k_unc})
img_conf = MIN mean_conf của các object = <span class="r">${f(d.B4.img_conf, 4)}</span>  (object #${d.B4.k_conf})</div>
    <div id="pool" style="height:${100 + 24 * d.objects.length}px"></div>
    <div class="why">MAX và MIN nghĩa là chỉ cần một object xấu là cả ảnh bị coi là xấu. Ảnh càng nhiều object thì càng dễ bị một object kéo vào vùng Đỏ hoặc Xám.</div>`; },
  after: (d, S, ctx) => {
    if (!d.objects.length) return;
    const y = d.objects.map((o) => `#${o.k} ${o.cls}`);
    plot("pool", [{ type: "bar", orientation: "h", name: "unc_total", y, x: d.objects.map((o) => o.unc_total), marker: { color: d.objects.map((o) => o.k === d.B4.k_unc ? "#dc2626" : "#93c5fd") }, xaxis: "x" },
      { type: "bar", orientation: "h", name: "mean_conf", y, x: d.objects.map((o) => o.mean_conf), marker: { color: d.objects.map((o) => o.k === d.B4.k_conf ? "#d97706" : "#cbd5e1") }, xaxis: "x2" }],
    { grid: { rows: 1, columns: 2 }, xaxis: { domain: [0, 0.47], title: "unc_total (MAX → img_unc)" }, xaxis2: { domain: [0.53, 1], title: "mean_conf (MIN → img_conf)", range: [0, 1] },
      yaxis: { autorange: "reversed" }, margin: { l: 110, r: 10, t: 10, b: 40 },
      shapes: [{ type: "line", xref: "x", yref: "paper", x0: d.thresholds.tau_unc, x1: d.thresholds.tau_unc, y0: 0, y1: 1, line: { color: "#dc2626", dash: "dash" } }] });
    $("#pool").on("plotly_click", (e) => { const k = +e.points[0].y.split(" ")[0].slice(1); ctx.select(k); });
  } },

  B5: { html: (d, S) => {
    const th = d.thresholds;
    let s = `<h2>B5 · FCD (khoảng cách tới centroid lớp) và xếp vùng</h2>`;
    if (S.sel !== null) {
      const o = d.objects[S.sel];
      s += `<div class="formula">vector ẩn Anchor e ∈ ℝ^${d.B5.gate.emb_dim} → z = (e − mean_PCA) · Wᵀ ∈ ℝ^${d.B5.gate.dim}     (PCA fit trên train)
D_M(z, lớp k) = √( (z − μ_k)ᵀ Σ_k⁻¹ (z − μ_k) )     μ_k, Σ_k: mean và covariance của lớp k trên train
object #${o.k} dự đoán "${esc(o.cls)}" → so với centroid của chính lớp đó${o.dm === null ? "\n⚠ lớp này không có centroid → coi là bất thường" : `
D² = Σᵢ (z−μ)ᵢ · (Σ⁻¹(z−μ))ᵢ = <span class="r">${f(o.d2, 3)}</span>   →   D_M = <span class="r">${f(o.dm, 4)}</span>   ${o.dm > th.tau_fcd ? "&gt;" : "≤"} τ_fcd = ${f(th.tau_fcd, 4)}  →  ${o.fcd_flag ? "BẤT THƯỜNG (nghi ảo giác)" : "trong phân phối"}`}</div>
      <div class="grid g2" style="margin-top:10px">
        <div><h3>Vector đặc trưng e (256 chiều, xếp 16×16)</h3><div id="emb" style="height:260px"></div></div>
        <div><h3>Khoảng cách tới centroid của mọi lớp</h3><div id="dist" style="height:260px"></div></div>
      </div>
      ${o.dm !== null ? `<h3>z và μ_${esc(o.cls)} (64 chiều): đóng góp của từng chiều vào D²</h3><div id="zmu" style="height:260px"></div>
      <div class="row"><button class="btn small" id="showinv">Mở ma trận Σ⁻¹ của lớp ${esc(o.cls)} (64×64)</button><span class="hint">tải từ centroids.pt</span></div><div id="inv"></div>` : ""}`;
    }
    s += `<h3 style="margin-top:14px">Luật xếp vùng (zone_router.py)</h3><table class="t"><tr><th>Kiểm tra</th><th>Giá trị</th><th>Kết quả</th><th>Nếu đúng</th></tr>
      ${d.B5.steps.map((x) => `<tr><td class="mono">${esc(x.test)}</td><td class="mono">${esc(x.value)}</td><td>${x.result ? '<span class="tag ok">đúng</span>' : '<span class="tag">sai</span>'}</td><td>${esc(x.then)}${x.else && !x.result ? " · sai → " + esc(x.else) : ""}</td></tr>`).join("")}</table>
      <div class="formula">luật áp dụng: ${esc(d.B5.rule)}
→ vùng <span class="r">${ZONE_VI[d.B5.zone]}</span>, ưu tiên ${esc(d.B5.priority)}, lý do: ${d.B5.reasons.map(esc).join(", ") || "—"}</div>`;
    return s;
  },
  after: (d, S) => {
    if (S.sel === null) return;
    const o = d.objects[S.sel], th = d.thresholds;
    const z16 = []; for (let r = 0; r < 16; r++) z16.push(o.emb.slice(16 * r, 16 * r + 16));
    plot("emb", [{ type: "heatmap", z: z16, colorscale: "RdBu", zmid: 0, hovertemplate: "e[%{customdata}] = %{z:.3f}<extra></extra>", customdata: z16.map((row, r) => row.map((_, c) => 16 * r + c)) }],
      { margin: { l: 30, r: 10, t: 5, b: 25 }, yaxis: { autorange: "reversed" } });
    const cd = o.centroid_dist;
    plot("dist", [{ type: "bar", x: cd.map((c) => c.cls), y: cd.map((c) => c.D), marker: { color: cd.map((c) => c.cls === o.cls ? "#2563eb" : "#cbd5e1") }, customdata: cd.map((c) => c.euclid),
      hovertemplate: "%{x}<br>Mahalanobis %{y:.2f}<br>Euclid %{customdata:.2f}<extra></extra>" }],
    { margin: { l: 40, r: 10, t: 5, b: 70 }, yaxis: { title: "D_M" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: th.tau_fcd, y1: th.tau_fcd, line: { color: "#dc2626", dash: "dash" }, label: { text: "τ_fcd", textposition: "end" } }] });
    if (o.dm !== null) {
      const ix = o.z.map((_, i) => i);
      plot("zmu", [{ type: "scatter", mode: "lines", name: "z (object)", x: ix, y: o.z, line: { color: "#2563eb" } },
        { type: "scatter", mode: "lines", name: "μ (centroid)", x: ix, y: o.mu, line: { color: "#111", dash: "dot" } },
        { type: "bar", name: "(z−μ)ᵢ·(Σ⁻¹(z−μ))ᵢ", x: ix, y: o.contrib, marker: { color: "#f59e0b" }, yaxis: "y2", opacity: 0.6 }],
      { showlegend: true, legend: { orientation: "h" }, margin: { l: 40, r: 50, t: 5, b: 30 }, yaxis: { title: "giá trị" }, yaxis2: { title: "đóng góp D²", overlaying: "y", side: "right" } });
      $("#showinv").onclick = async () => {
        $("#inv").innerHTML = '<div class="loading">Đang tải…</div>';
        const c = await api(`/api/centroid/${encodeURIComponent(o.cls)}`);
        $("#inv").innerHTML = `<div id="invhm" style="height:460px"></div><div class="hint">Σ⁻¹ của lớp ${esc(o.cls)}. Đường chéo lớn nghĩa là chiều đó ít biến động trên train: lệch một chút ở chiều đó cũng làm D² tăng mạnh. Variance trên đường chéo Σ: min ${f(Math.min(...c.cov_diag), 4)}, max ${f(Math.max(...c.cov_diag), 4)}.</div>`;
        plot("invhm", [{ type: "heatmap", z: c.inv_cov, colorscale: "RdBu", zmid: 0, hovertemplate: "Σ⁻¹[%{y},%{x}] = %{z:.3f}<extra></extra>" }], { margin: { l: 40, r: 10, t: 5, b: 30 }, yaxis: { autorange: "reversed" } });
      };
    }
  } },

  B6: { html: (d) => {
    const b = d.B6;
    if (!b.in_batch) return `<h2>B6 · K-Means đa dạng hoá</h2><div class="formula">K = ⌈0.2 × 1⌉ = ${b.budget}</div><div class="why">${esc(b.note)}</div>`;
    return `<h2>B6 · K-Means đa dạng hoá (trong batch ${b.n_images} ảnh)</h2>
    <div class="formula">K = ⌈0.2 × ${b.n_images}⌉ = <span class="r">${b.budget}</span>      ảnh Đỏ trước K-Means: ${b.red_before}
${b.ran ? `${b.red_before} &gt; ${b.budget} → K-Means chia ${b.red_before} vector cảnh (256 chiều, chuẩn hoá L2) thành ${b.budget} cụm` : `${b.red_before} ≤ ${b.budget} → không chạy K-Means, giữ hết`}
ảnh này: vùng sau B5 = ${ZONE_VI[b.zone_b5]}${b.cluster !== null && b.cluster !== undefined ? `, thuộc cụm #${b.cluster} (${b.cluster_size} ảnh), khoảng cách tới centroid = ${f(b.dist_to_centroid, 4)}
→ ${b.kept ? "ĐƯỢC GIỮ: ảnh đại diện của cụm (ưu tiên cao nhất, gần centroid nhất)" : "BỊ LOẠI (DIVERSITY_DEDUP): cụm đã có ảnh đại diện khác → chuyển sang Xám"}` : ""}
vùng cuối cùng = <span class="r">${ZONE_VI[b.zone_final]}</span></div>
    <div class="why">Xem toàn bộ các cụm ở trang <a href="${location.hash.replace("#/single/", "#/batch/").split("/").slice(0, 3).join("/")}">1 batch</a>.</div>`; } },

  B7: { html: (d) => `<h2>B7 · Ghi ra file queue</h2>
    <div class="formula">vùng ${ZONE_VI[d.B7.zone]} → ${esc(d.B7.file)}</div>
    <pre class="formula" style="max-height:420px">${esc(JSON.stringify(d.B7.labels, null, 2))}</pre>
    <div class="why">Ảnh Đỏ vào <code>review_queue.json</code>, kèm box mồi (pre_labels) và cờ <code>is_fcd_anomaly</code>. Ảnh Xám vào <code>grey_pool.json</code>, chờ vòng sau. Ảnh Xanh vào <code>green_labels.json</code> làm nhãn giả.</div>` },

};
const GT_PANEL = (d) => `<h2>Chấm bằng ground truth (hệ thống không dùng)</h2>
    <div class="grid g3"><div class="card kpi"><div class="v" style="color:var(--green)">${d.gt.tp ?? "–"}</div><div class="l">TP: object đúng</div></div>
      <div class="card kpi"><div class="v" style="color:var(--red)">${d.gt.fp ?? "–"}</div><div class="l">FP: object sai (sai lớp, lệch box, hoặc không có thật)</div></div>
      <div class="card kpi"><div class="v" style="color:var(--amber)">${d.gt.fn ?? "–"}</div><div class="l">FN: object thật bị sót</div></div></div>
    <div class="why">${d.gt.n} box GT, lấy từ thư mục ${esc(d.gt.source)}. Ghép tham lam theo confidence, cùng lớp, IoU ≥ 0.5. Lớp phủ "Object + GT" tô object đúng màu xanh, object sai màu đỏ, GT nét đứt trắng.</div>`;
STEP_RENDER.CHK = {
  html: (d) => `<h2>Kiểm chứng: mọi số tính tay so với code gốc</h2>
    <p class="hint">Mỗi số trên trang được tính lại bằng numpy, rồi so với kết quả của chính các module trong ai_router (sai số ≤ 1e-6, D_M ≤ 1e-2).</p>
    <div class="scroll"><table class="t"><tr><th>Phép kiểm</th><th>Kết quả</th></tr>${d.checks.map((c) => `<tr><td>${esc(c.name)}</td><td>${c.ok ? '<span class="tag ok">khớp</span>' : '<span class="tag bad">lệch</span>'}</td></tr>`).join("")}</table></div>`,
};
