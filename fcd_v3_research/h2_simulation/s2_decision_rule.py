"""S2: đặc tính vận hành của quy tắc quyết định bản 1.2 (Wilcoxon theo đơn vị trên một lần chia)
và bản 1.3 (trung bình qua các vòng B-block + bootstrap theo super-block, quy tắc ba vùng).

Mỗi cặp (A, B) được chạy trên REP cặp pool độc lập (Meta-Path 3.500 + Evaluation 7.000 đơn vị).
Δ thật xấp xỉ bằng trung bình các ước lượng điểm qua REP lần (mỗi lần 120 vòng).
"""
import sys
import numpy as np
from scipy.stats import wilcoxon
import h2core as hc
import s1_signal_design as s1

GAMMA = 0.05
A_EACH = 0.025 / 4
REP = 40
R_BOOT = 600
ROUNDS_BOOT = 20


def tables(name, par, meta, pool):
    path = hc.build_path(meta, s1.signal(name, meta, par, meta))
    return hc.eval_tables(pool, s1.signal(name, pool, par, meta), path)


def one_rep(rng, sc, a, b):
    meta = hc.make_pool(rng, 3500, sc)
    pool = hc.make_pool(rng, 7000, sc)
    pa = s1.choose_params(a if a != "H_single_hybrid_m2" else "H_single_hybrid", meta)
    pb = s1.choose_params(b if b != "H_single_hybrid_m2" else "H_single_hybrid", meta)
    Fa, Ta = tables(a, pa, meta, pool)
    Fb, Tb = tables(b, pb, meta, pool)
    G = pool["G"].astype(float)
    w = np.ones(pool["n_units"])
    # --- ước lượng điểm của bản 1.3: 40 phân hoạch × 3 vòng
    W = hc.bblock_rounds(rng, pool["sb"], 40)
    ca, ka, _ = hc.run_rounds(W, w, Fa, Ta, G)
    cb, kb, _ = hc.run_rounds(W, w, Fb, Tb, G)
    point = (ca - cb).mean()
    # --- quy tắc bản 1.2: một lần chia (vòng đầu tiên), Wilcoxon trên ΔTP theo đơn vị kiểm thử
    test = W[0] == 0
    d = (Ta[test, max(ka[0], 0)] * (ka[0] >= 0) - Tb[test, max(kb[0], 0)] * (kb[0] >= 0)).astype(float)
    p_w = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    v12_accept = (p_w < 0.0025) and (ca[0] - cb[0] >= GAMMA)
    # --- bootstrap theo super-block của bản 1.3
    n_sb = pool["sb"].max() + 1
    boots = np.empty(R_BOOT)
    for r in range(R_BOOT):
        mult = np.bincount(rng.integers(0, n_sb, n_sb), minlength=n_sb).astype(float)
        wb = mult[pool["sb"]]
        Wb = hc.bblock_rounds(rng, pool["sb"], ROUNDS_BOOT // 3 + 1, weights=wb)[:ROUNDS_BOOT]
        cab, _, _ = hc.run_rounds(Wb, wb, Fa, Ta, G)
        cbb, _, _ = hc.run_rounds(Wb, wb, Fb, Tb, G)
        boots[r] = (cab - cbb).mean()
    lcb, ucb = np.quantile(boots, A_EACH), np.quantile(boots, 1 - A_EACH)
    v13 = "Đạt" if (point >= GAMMA and lcb > 0) else ("Không đạt" if ucb < GAMMA else "Không kết luận")
    return dict(point=point, single=ca[0] - cb[0], p_w=p_w, v12=v12_accept, lcb=lcb, ucb=ucb, v13=v13, bsd=boots.std(ddof=1))


CASES = [
    ("Đối chứng âm: Hs(anchor 1) vs Hs(anchor 2), rho=0.7 mixed", dict(hc.DEFAULT), "H_single_hybrid", "H_single_hybrid_m2"),
    ("He vs Hs, rho=0.95 mixed", dict(hc.DEFAULT, rho=0.95), "H_ens_hybrid", "H_single_hybrid"),
    ("He vs Hs, rho=0.7 loc", dict(hc.DEFAULT, p_loc=0.14, r_hal=0.15), "H_ens_hybrid", "H_single_hybrid"),
    ("He vs Hs, rho=0.7 mixed", dict(hc.DEFAULT), "H_ens_hybrid", "H_single_hybrid"),
    ("He vs Hs, rho=0.3 hal", dict(hc.DEFAULT, rho=0.3, p_loc=0.02, r_hal=0.6), "H_ens_hybrid", "H_single_hybrid"),
    ("Hs vs B1, rho=0.7 mixed", dict(hc.DEFAULT), "H_single_hybrid", "B1_conf"),
]

if __name__ == "__main__":
    reps = 6 if "--quick" in sys.argv else REP
    for i, (lab, sc, a, b) in enumerate(CASES):
        rng = np.random.default_rng(2000 + i)
        res = [one_rep(rng, sc, a, b) for _ in range(reps)]
        pt = np.array([r["point"] for r in res]); sg = np.array([r["single"] for r in res])
        print(f"\n== {lab}")
        print(f"  Δ thật ≈ {100*pt.mean():+.2f} pp | SD ước lượng điểm 1.3 = {100*pt.std(ddof=1):.2f} pp | "
              f"SD hiệu một lần chia (1.2) = {100*sg.std(ddof=1):.2f} pp | "
              f"SD bootstrap trung bình = {100*np.mean([r['bsd'] for r in res]):.2f} pp")
        print(f"  Bản 1.2: P(p_Wilcoxon < 0.0025) = {np.mean([r['p_w'] < 0.0025 for r in res]):.2f} | "
              f"P(chấp nhận) = {np.mean([r['v12'] for r in res]):.2f}")
        for z in ("Đạt", "Không kết luận", "Không đạt"):
            print(f"  Bản 1.3: P({z}) = {np.mean([r['v13'] == z for r in res]):.2f}", end="")
        print(f" | P(LCB > 0) = {np.mean([r['lcb'] > 0 for r in res]):.2f}")
