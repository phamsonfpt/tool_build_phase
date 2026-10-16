"""Lõi mô phỏng cho H2: sinh dữ liệu tổng hợp, các tín hiệu bất định, LTT với p-value HB.

Mô hình sinh dữ liệu là giả định (xem H2_simulation_report.md, Mục 2). Mục đích là kiểm tra
chiều và cơ chế của các phương án thiết kế, không phải dự đoán con số thật trên BDD100K.
"""
import numpy as np
from scipy.stats import binom

ALPHA = 0.02
DELTA = 0.05
N_CAL = 1500
M = 3  # số thành viên ensemble (3 detector có sẵn)


def risk_grid():
    """Lưới rủi ro của H1 tại N = 1.500: lõi 0.002-0.015, kéo dài tới 0.040 (K = 33)."""
    core = np.arange(14) * 0.001 + 0.002
    ext1 = np.arange(15) * 0.001 + 0.016
    ext2 = np.array([0.0325, 0.035, 0.0375, 0.040])
    g = np.round(np.concatenate([core, ext1, ext2]), 4)
    assert len(g) == 33
    return g


def hb_pvalues(r_hat, n, alpha=ALPHA):
    """p-value Hoeffding-Bentkus như H1 Mục V. r_hat: (R, K); n: (R,)."""
    n = np.asarray(n, dtype=float)[:, None]
    p = np.minimum(r_hat, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * np.log(p / alpha), 0.0) + (1 - p) * np.log((1 - p) / (1 - alpha))
    p_hoeff = np.exp(-n * kl)
    k = np.ceil(n * r_hat - 1e-12)
    p_bent = np.minimum(1.0, np.e * binom.cdf(k, n, alpha))
    return np.minimum(p_hoeff, p_bent)


def ltt_index(r_hat, n, delta=DELTA):
    """Fixed-sequence: trả về chỉ số điểm cuối cùng được chứng nhận, -1 nếu rỗng."""
    pv = hb_pvalues(r_hat, n)
    fail = pv > delta
    first_fail = np.where(fail.any(1), fail.argmax(1), fail.shape[1])
    return first_fail - 1


# ----------------------------------------------------------------------------- sinh dữ liệu
DEFAULT = dict(
    sigma_u=0.5, sigma_b=0.3, mean_gt=7.0, p_det=0.92,
    p_loc=0.06, r_hal=0.4, pi_conf=0.3,
    rho=0.7,            # tương quan giữa thành viên (thiên kiến chung), dùng cho cả latent và output
    s_mu=0.6,           # dị biệt chung của TP trong không gian tiềm ẩn
    b=0.8,              # dao động riêng của mô hình (TP và LOC)
    b_h=1.6,            # dao động riêng của mô hình trên ảo giác
    delta_loc=0.25,     # độ lệch tiềm ẩn của lỗi lệch vị trí (gần ID)
    delta_h=2.5, s_h=1.0,  # độ lệch tiềm ẩn của ảo giác
    sigma_c=0.6,        # dao động riêng của điểm tin cậy giữa thành viên
    c_tp=3.5, c_loc=1.0, c_hal_hi=2.5, c_hal_lo=-0.5,  # logit tin cậy trung bình theo loại
    r_cl=0.0,           # tương quan (âm) giữa tin cậy và khoảng cách tiềm ẩn trong cùng loại
)


def _superblocks(rng, n_units):
    sizes = rng.geometric(0.4, n_units)  # trung bình 2.5 đơn vị mỗi super-block
    sizes = sizes[: np.searchsorted(np.cumsum(sizes), n_units) + 1]
    sb = np.repeat(np.arange(len(sizes)), sizes)[:n_units]
    return sb


def _equicorr(rng, n, m, rho):
    z0 = rng.standard_normal((n, 1))
    return np.sqrt(rho) * z0 + np.sqrt(1 - rho) * rng.standard_normal((n, m))


def make_pool(rng, n_units, sc):
    """Trả về dict: đơn vị, super-block, số GT, và các ứng viên (loại, điểm tin cậy, khoảng cách)."""
    sb = _superblocks(rng, n_units)
    b_g = rng.normal(0, sc["sigma_b"], sb.max() + 1)
    u = rng.normal(0, sc["sigma_u"], n_units) + b_g[sb]
    G = 1 + rng.poisson(sc["mean_gt"], n_units)
    det = rng.binomial(G, sc["p_det"])
    lp = np.log(sc["p_loc"] / (1 - sc["p_loc"])) + 0.5 * u
    n_loc = rng.binomial(det, 1 / (1 + np.exp(-lp)))
    n_tp = det - n_loc
    n_hal = rng.poisson(sc["r_hal"] * np.exp(u))
    unit = np.concatenate([np.repeat(np.arange(n_units), n_tp),
                           np.repeat(np.arange(n_units), n_loc),
                           np.repeat(np.arange(n_units), n_hal)])
    typ = np.concatenate([np.zeros(n_tp.sum(), int), np.ones(n_loc.sum(), int), np.full(n_hal.sum(), 2)])
    n = len(unit)
    uu = u[unit]
    # Nhiễu dùng chung e làm điểm tin cậy và khoảng cách tiềm ẩn tương quan âm trong cùng loại
    # (r_cl = 0: hai tín hiệu độc lập khi biết loại; r_cl lớn: tiềm ẩn trùng thông tin với tin cậy).
    r = sc["r_cl"]
    e = rng.standard_normal(n)
    nc = np.sqrt(1 - r * r) * rng.standard_normal(n) + r * e
    nz = np.sqrt(1 - r * r) * rng.standard_normal(n) - r * e
    # điểm tin cậy dùng chung (logit)
    c = np.empty(n)
    i0, i1, i2 = typ == 0, typ == 1, typ == 2
    c[i0] = sc["c_tp"] + nc[i0] - 0.4 * uu[i0]
    c[i1] = sc["c_loc"] + nc[i1] - 0.4 * uu[i1]
    conf = rng.random(i2.sum()) < sc["pi_conf"]
    c[i2] = np.where(conf, sc["c_hal_hi"], sc["c_hal_lo"]) + nc[i2]
    cm = c[:, None] + sc["sigma_c"] * _equicorr(rng, n, M, sc["rho"])
    # khoảng cách tiềm ẩn đã chuẩn hóa zeta_m = mu + b * xi_m
    mu = np.empty(n); bb = np.empty(n)
    mu[i0] = sc["s_mu"] * nz[i0]; bb[i0] = sc["b"]
    mu[i1] = sc["delta_loc"] + sc["s_mu"] * nz[i1]; bb[i1] = sc["b"]
    mu[i2] = sc["delta_h"] + sc["s_h"] * nz[i2]; bb[i2] = sc["b_h"]
    zeta = mu[:, None] + bb[:, None] * _equicorr(rng, n, M, sc["rho"])
    return dict(n_units=n_units, sb=sb, G=G, unit=unit, typ=typ, cm=cm, zeta=zeta)


# ----------------------------------------------------------------------------- tín hiệu
def _rz(x, ref):
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (x - med) / mad


LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
KAPPAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def base_signals(p):
    """Các tín hiệu không có siêu tham số. Quy ước: càng lớn càng bất định."""
    z, cm = p["zeta"], p["cm"]
    zbar = z.mean(1)
    sd = z.std(1, ddof=1)
    return {
        "B1_conf": -cm[:, 0],
        "B2_ens_conf": -cm.mean(1),
        "B4_maha": z[:, 0],
        "B4_maha_m2": z[:, 1],  # mô hình thứ hai làm anchor: dùng cho đối chứng âm (Δ thật = 0)
        "P0_var_v12": z.var(1, ddof=1),
        "A1_ens_maha": zbar,
        "A3_max": z.max(1),
        "_zbar": zbar, "_sd": sd,
    }


def tuned_signal(name, p, params, ref_p=None):
    """Tín hiệu có siêu tham số. ref_p: pool dùng để chuẩn hóa (Meta-Fit; ở đây dùng TP của Meta-Path)."""
    s = base_signals(p)
    if name == "P2_mean_plus_sd":
        return s["_zbar"] + params["lam"] * s["_sd"]
    r = base_signals(ref_p if ref_p is not None else p)
    tp = (ref_p if ref_p is not None else p)["typ"] == 0
    if name == "H_ens_hybrid":
        lat = s["_zbar"] + params["lam"] * s["_sd"]
        lat_ref = r["_zbar"] + params["lam"] * r["_sd"]
        return _rz(s["B2_ens_conf"], r["B2_ens_conf"][tp]) + params["kappa"] * _rz(lat, lat_ref[tp])
    if name == "H_single_hybrid":
        return _rz(s["B1_conf"], r["B1_conf"][tp]) + params["kappa"] * _rz(s["B4_maha"], r["B4_maha"][tp])
    if name == "H_single_hybrid_m2":  # cùng công thức, anchor là mô hình thứ hai (đối chứng âm)
        c2, c2r = -p["cm"][:, 1], -(ref_p if ref_p is not None else p)["cm"][:, 1]
        return _rz(c2, c2r[tp]) + params["kappa"] * _rz(s["B4_maha_m2"], r["B4_maha_m2"][tp])
    raise KeyError(name)


# ----------------------------------------------------------------------------- bảng theo đơn vị
def unit_tables(p, U, thr):
    """Trả về TP, V (FP), N_acc theo đơn vị × điểm lưới, với chấp nhận khi U <= thr[g]."""
    g = np.searchsorted(thr, U, side="left")  # chấp nhận tại mọi điểm lưới >= g
    G_ = len(thr)
    keep = g < G_
    idx = p["unit"][keep] * G_ + g[keep]
    tp = np.bincount(idx[p["typ"][keep] == 0], minlength=p["n_units"] * G_).reshape(p["n_units"], G_)
    fp = np.bincount(idx[p["typ"][keep] != 0], minlength=p["n_units"] * G_).reshape(p["n_units"], G_)
    tp = tp.cumsum(1); fp = fp.cumsum(1)
    return tp, fp


def fdp(tp, fp):
    n = tp + fp
    return np.where(n > 0, fp / np.maximum(n, 1), 0.0)


def build_path(meta, U_meta, n_grid=300):
    """Đường đi Pareto 1D trên Meta-Path (H1 Mục 6.2): mỗi α_k chọn ngưỡng có Coverage lớn nhất
    trong các ngưỡng có Macro-FDP trên Meta-Path <= α_k."""
    thr = np.quantile(U_meta, np.linspace(0.002, 1.0, n_grid))
    tp, fp = unit_tables(meta, U_meta, thr)
    f = fdp(tp, fp).mean(0)
    cov = tp.sum(0) / meta["G"].sum()
    path = []
    for a in risk_grid():
        ok = np.where(f <= a)[0]
        path.append(thr[ok[np.argmax(cov[ok])]] if len(ok) else thr[0])
    return np.array(path)


def plugin_cov(meta, U_meta, r_star=0.010, n_grid=300):
    """Coverage trên Meta-Path tại ngưỡng plug-in lớn nhất có Macro-FDP <= r*: dùng để chọn siêu tham số."""
    thr = np.quantile(U_meta, np.linspace(0.002, 1.0, n_grid))
    tp, fp = unit_tables(meta, U_meta, thr)
    f = fdp(tp, fp).mean(0)
    cov = tp.sum(0) / meta["G"].sum()
    ok = f <= r_star
    return cov[ok].max() if ok.any() else 0.0


def eval_tables(p, U, path):
    tp, fp = unit_tables(p, U, path)
    return fdp(tp, fp), tp


# ----------------------------------------------------------------------------- các vòng B-block
def bblock_rounds(rng, sb, n_part, weights=None, n_cal=N_CAL):
    """Ma trận trọng số tập hiệu chuẩn (R × đơn vị): mỗi phân hoạch hoán vị super-block rồi cắt 3 tập."""
    n_units = len(sb)
    w = np.ones(n_units) if weights is None else weights
    sb_w = np.bincount(sb, weights=w)
    rows = []
    for _ in range(n_part):
        perm = rng.permutation(len(sb_w))
        cum = np.cumsum(sb_w[perm])
        lo, base = 0, 0.0
        for k in range(3):
            hi = np.searchsorted(cum, base + n_cal, side="left")  # super-block cuối được giữ trọn
            mask = np.zeros(len(sb_w)); mask[perm[lo:hi + 1]] = 1
            rows.append(mask[sb] * w)
            lo, base = hi + 1, cum[hi]
    return np.array(rows)


def run_rounds(Wcal, w_pool, F, TP, G):
    """Với mỗi vòng: LTT trên tập hiệu chuẩn, Coverage trên tập kiểm thử, rủi ro trên toàn pool."""
    n = Wcal.sum(1)
    r_hat = (Wcal @ F) / n[:, None]
    k = ltt_index(r_hat, n)
    Wtest = w_pool[None, :] - Wcal
    cov_all = (Wtest @ TP) / (Wtest @ G)[:, None]
    cov = np.where(k >= 0, cov_all[np.arange(len(k)), np.maximum(k, 0)], 0.0)
    r_pool = (w_pool @ F) / w_pool.sum()
    viol = np.where(k >= 0, r_pool[np.maximum(k, 0)] > ALPHA, False)
    return cov, k, viol
