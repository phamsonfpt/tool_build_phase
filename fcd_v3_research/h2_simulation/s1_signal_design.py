"""S1: so sánh các phương án tín hiệu (bản 1.2, các ô 2x2, phương án lai) dưới LTT.

Mỗi kịch bản: R lần lặp độc lập, mỗi lần sinh Meta-Path (3.500 đơn vị) và Evaluation pool
(7.000 đơn vị), chọn siêu tham số trên Meta-Path, dựng đường đi, rồi chạy 40 phân hoạch × 3
vòng B-block. Báo cáo Coverage trung bình trên tập kiểm thử, Yield và tỉ lệ vi phạm.
"""
import sys
import numpy as np
import h2core as hc

SIGNALS = ["B1_conf", "B2_ens_conf", "B4_maha", "P0_var_v12", "A1_ens_maha", "A3_max",
           "P2_mean_plus_sd", "H_single_hybrid", "H_ens_hybrid"]


def choose_params(name, meta):
    if name == "P2_mean_plus_sd":
        grid = [dict(lam=l) for l in hc.LAMBDAS]
    elif name == "H_ens_hybrid":
        # chọn λ theo P2 trước, rồi κ (ngân sách 5 + 5 giá trị)
        lam = choose_params("P2_mean_plus_sd", meta)["lam"]
        grid = [dict(lam=lam, kappa=k) for k in hc.KAPPAS]
    elif name == "H_single_hybrid":
        grid = [dict(kappa=k) for k in hc.KAPPAS]
    else:
        return {}
    scores = [hc.plugin_cov(meta, hc.tuned_signal(name, meta, g, meta)) for g in grid]
    return grid[int(np.argmax(scores))]


def signal(name, p, params, meta):
    if params:
        return hc.tuned_signal(name, p, params, meta)
    return hc.base_signals(p)[name]


def run_scenario(sc, seed, n_rep=4, n_part=40, signals=SIGNALS):
    rng = np.random.default_rng(seed)
    out = {s: dict(cov=[], yld=[], vr=[], par=[]) for s in signals}
    ceil = []
    for _ in range(n_rep):
        meta = hc.make_pool(rng, 3500, sc)
        pool = hc.make_pool(rng, 7000, sc)
        Wcal = hc.bblock_rounds(rng, pool["sb"], n_part)
        w = np.ones(pool["n_units"])
        ceil.append((pool["typ"] == 0).sum() / pool["G"].sum())
        for s in signals:
            par = choose_params(s, meta)
            path = hc.build_path(meta, signal(s, meta, par, meta))
            F, TP = hc.eval_tables(pool, signal(s, pool, par, meta), path)
            cov, k, viol = hc.run_rounds(Wcal, w, F, TP, pool["G"].astype(float))
            out[s]["cov"].append(cov.mean()); out[s]["yld"].append((k >= 0).mean())
            out[s]["vr"].append(viol.mean()); out[s]["par"].append(par)
    return out, float(np.mean(ceil))


REGIMES = {"hal": dict(p_loc=0.02, r_hal=0.6), "mixed": dict(p_loc=0.06, r_hal=0.4),
           "loc": dict(p_loc=0.14, r_hal=0.15)}
SCENARIOS = []
for rho in (0.3, 0.7, 0.95):
    for reg, o in REGIMES.items():
        SCENARIOS.append((f"rho={rho} FP={reg}", dict(hc.DEFAULT, rho=rho, **o)))
for reg, o in REGIMES.items():  # độ nhạy: lỗi lệch vị trí lộ rõ hơn trong không gian tiềm ẩn
    SCENARIOS.append((f"rho=0.7 FP={reg} delta_loc=1.0", dict(hc.DEFAULT, rho=0.7, delta_loc=1.0, **o)))
for reg, o in REGIMES.items():  # độ nhạy: ảo giác không làm thành viên bất đồng hơn
    SCENARIOS.append((f"rho=0.7 FP={reg} b_h=0.8", dict(hc.DEFAULT, rho=0.7, b_h=0.8, **o)))
for reg, o in REGIMES.items():  # độ nhạy: tiềm ẩn trùng một phần thông tin với tin cậy
    SCENARIOS.append((f"rho=0.7 FP={reg} r_cl=0.6", dict(hc.DEFAULT, rho=0.7, r_cl=0.6, **o)))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    scen = SCENARIOS[:1] if quick else SCENARIOS
    for i, (lab, sc) in enumerate(scen):
        out, ceil = run_scenario(sc, seed=1000 + i, n_rep=1 if quick else 4)
        # tỷ trọng ảo giác trong FP (toàn bộ ứng viên)
        p = hc.make_pool(np.random.default_rng(1), 7000, sc)
        hal_share = (p["typ"] == 2).sum() / (p["typ"] != 0).sum()
        print(f"\n== {lab} | trần Cov = {ceil:.3f} | tỷ trọng ảo giác trong FP = {hal_share:.2f}")
        best = max(out, key=lambda s: np.mean(out[s]["cov"]))
        for s in out:
            c = np.array(out[s]["cov"])
            par = out[s]["par"][0]
            print(f"  {s:18s} Cov={c.mean():.3f} (±{c.std(ddof=1) if len(c) > 1 else 0:.3f})  "
                  f"Yield={np.mean(out[s]['yld']):.2f}  VR={np.mean(out[s]['vr']):.3f}  "
                  f"{'<- tốt nhất' if s == best else ''} {par if par else ''}")
