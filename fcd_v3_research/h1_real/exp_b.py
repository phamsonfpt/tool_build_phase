"""Experiment B on BDD100K Val (alpha = 0.02), using only what registration_<anchor>.json froze.

Main arm  : units = rides (video-name prefix), N = 1500, P = 334 partitions x K = 3 disjoint cal sets.
            Without GPS/time no super-block exists, so the held-out group is the ride itself and B-block
            coincides with B-rand (the doc's own fallback when the unit is the coarsest group).
Arm B-img : images treated as units, ignoring rides (descriptive).
C2b-instr : variance-ratio instrument of C2b validated one level down (images in rides) - descriptive.
usage: python exp_b.py <anchor_expert>
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from h1common import (ALPHA, CACHE, DATA, RESULTS, cp_ci, fixed_sequence, icc_anova, partition_bootstrap, vr_ci,
                      write_json)
from h1units import aggregate, fdp, grid_params, load_counts, ride_index


def three_zone(lo, hi, acc_hi, rej_lo):
    if lo > rej_lo:
        return "Bac bo"
    return "Chap nhan" if hi <= acc_hi else "Khong ket luan"


def run_ltt_rounds(Z, R_pool, n_cal, P, K, rng, extra=None, unit_cut=None):
    """Z: units x path. Returns dict of per-round arrays shaped (P, K)."""
    n = Z.shape[0]
    jstar = int(np.nonzero(R_pool > ALPHA)[0][0]) if (R_pool > ALPHA).any() else None
    out = {k: np.zeros((P, K)) for k in ("idx", "e", "reach", "Rb", "Rpool")}
    out["cal_sets"] = []
    for p in range(P):
        perm = rng.permutation(n)
        for k in range(K):
            cal = perm[k * n_cal:(k + 1) * n_cal] if unit_cut is None else unit_cut(perm, k)
            mask = np.zeros(n, bool); mask[cal] = True
            idx, stop = fixed_sequence(Z[mask].mean(0), mask.sum(), ALPHA)
            out["idx"][p, k] = idx
            out["e"][p, k] = (idx >= 0) and R_pool[idx] > ALPHA
            out["reach"][p, k] = jstar is not None and stop >= jstar
            out["Rpool"][p, k] = R_pool[idx] if idx >= 0 else 0.0
            out["Rb"][p, k] = Z[~mask, idx].mean() if idx >= 0 else 0.0
            out["cal_sets"].append(mask)
    out["jstar"] = jstar
    return out


def main(anchor):
    reg = json.load(open(os.path.join(CACHE, f"registration_{anchor}.json")))
    c = load_counts("val", anchor)
    gidx, rides = ride_index("val", c["names"])
    params = grid_params(c["tau_fcd"])
    ti, path, base = reg["theta_index"], reg["path_index"], reg["tau_base_index"]
    u = aggregate(c, gidx, len(rides), ti)
    Nr, TPr, VHr, Mr = u["N"], u["TP"], u["VH"], u["M"]
    Z = fdp(Nr[:, path], TPr[:, path])                       # rides x K
    R_pool = Z.mean(0)
    P, K, N = reg["P_partitions"], reg["K_per_partition"], reg["N"]
    rng = np.random.default_rng(2026)
    res = {"anchor": anchor, "registration_sha256": reg["sha256"], "n_eval_rides": len(rides),
           "n_eval_images": len(c["names"]), "feasible": len(rides) >= 3 * N + 3 * 1,
           "R_pool_path": R_pool.tolist(), "path_alpha": reg["path_alpha"]}

    # ---------------- main arm (rides)
    r = run_ltt_rounds(Z, R_pool, N, P, K, rng)
    H0 = np.nonzero(R_pool > ALPHA)[0]
    res["T1_H0_pool_nonempty"] = bool(len(H0))
    res["jstar_pool"] = r["jstar"]
    res["jstar_alpha_k"] = reg["path_alpha"][r["jstar"]] if r["jstar"] is not None else None
    e = r["e"].astype(float)
    lo, hi = vr_ci(e)
    res["C2"] = {"VR": e.mean(), "CI": [lo, hi], "CP_on_B_ref": cp_ci(int(e.sum()), e.size),
                 "decision": three_zone(lo, hi, 0.065, 0.05) if len(H0) else "Khong kiem dinh duoc"}
    res["C2b"] = {"decision": "Khong kiem dinh duoc",
                  "reason": "no GPS/time metadata -> no super-block; held-out group = ride = unit"}
    nonempty = r["idx"] >= 0
    yield_ = nonempty.mean()
    res["C3"] = {"Yield": yield_, "CI": list(partition_bootstrap(nonempty.astype(float))),
                 "decision": "Dat" if yield_ >= 0.85 else "Khong dat"}
    res["D7"] = {"reach_jstar_share": r["reach"].mean(),
                 "selected_alpha_k_hist": {str(reg["path_alpha"][int(i)]) if i >= 0 else "empty": int(n)
                                           for i, n in zip(*np.unique(r["idx"], return_counts=True))}}
    # ---- C3b, D1, D3, D4 on the test side of each round
    rho, cov_t, base_fdp, micro, empty, spear, hal, rpool_sel = [], [], [], [], [], [], [], []
    for (p, k), mask in zip(np.ndindex(P, K), r["cal_sets"]):
        test = ~mask
        cov_base = TPr[test, base].sum() / Mr[test].sum()
        base_fdp.append(fdp(Nr[test, base], TPr[test, base]).mean())
        i = int(r["idx"][p, k])
        if i < 0:
            continue
        g = path[i]
        ct = TPr[test, g].sum() / Mr[test].sum()
        cov_t.append(ct); rho.append(ct / cov_base)
        nt, tpt = Nr[test, g], TPr[test, g]
        micro.append((nt - tpt).sum() / max(nt.sum(), 1)); empty.append((nt == 0).mean())
        ne = nt > 0
        spear.append(stats.spearmanr(nt[ne], fdp(nt[ne], tpt[ne])).correlation if ne.sum() > 2 else np.nan)
        v = (nt - tpt).sum(); hal.append(VHr[test, g].sum() / v if v else np.nan)
        rpool_sel.append(R_pool[i])
    rho_pk = np.full((P, K), np.nan)
    j = 0
    for (p, k) in np.ndindex(P, K):
        if r["idx"][p, k] >= 0:
            rho_pk[p, k] = rho[j]; j += 1
    rho = np.array(rho)
    med = float(np.median(rho)) if len(rho) else None
    med_ci = list(partition_bootstrap(rho_pk, stat=np.nanmedian)) if len(rho) else None
    res["C3b"] = {"median_rho_cov": med, "CI": med_ci, "decision": ("Dat" if med is not None and med >= 0.5 else "Khong dat")
                  if reg["tau_base_meets_alpha"] else "Khong xac dinh (baseline khong dat alpha tren Meta-Train)",
                  "median_cov_test_at_tau_hat": float(np.median(cov_t)) if cov_t else None}
    res["H1c"] = "DAT" if res["C3"]["decision"] == "Dat" and res["C3b"]["decision"] == "Dat" else "KHONG DAT"
    rp = np.array(rpool_sel) if rpool_sel else np.array([0.0])
    res["D1"] = {"median_R_pool_tau_hat": float(np.median(rp)), "p95_R_pool_tau_hat": float(np.percentile(rp, 95)),
                 "Delta_gap": ALPHA - float(np.median(rp)),
                 "median_R_test": float(np.median(r["Rb"][nonempty])) if nonempty.any() else None,
                 "p95_R_test": float(np.percentile(r["Rb"][nonempty], 95)) if nonempty.any() else None,
                 "share_R_test_gt_alpha": float((r["Rb"][nonempty] > ALPHA).mean()) if nonempty.any() else None}
    res["D3"] = {"micro_fdp_median": float(np.nanmedian(micro)) if micro else None,
                 "micro_fdp_p95": float(np.nanpercentile(micro, 95)) if micro else None,
                 "empty_unit_share_median": float(np.median(empty)) if empty else None,
                 "spearman_Nacc_FDP_median": float(np.nanmedian(spear)) if spear else None,
                 "tau_base": params[base], "macro_fdp_tau_base_test_median": float(np.median(base_fdp)),
                 "macro_fdp_tau_base_test_p95": float(np.percentile(base_fdp, 95)),
                 "tau_base_test_violation_share": float((np.array(base_fdp) > ALPHA).mean())}
    res["D4"] = {"halluc_share_of_FP_median": float(np.nanmedian(hal)) if hal else None}

    # ---------------- H1a combined (Section IX)
    c2, c2b = res["C2"]["decision"], res["C2b"]["decision"]
    if not res["T1_H0_pool_nonempty"]:
        h1a = "KHONG KIEM DINH DUOC (T1)"
    elif c2 == "Bac bo":
        h1a = "BAC BO"
    elif c2 == "Chap nhan" and c2b == "Chap nhan":
        h1a = "CHAP NHAN"
    else:
        h1a = "KHONG KET LUAN"
    res["H1a"] = h1a

    # ---------------- D9 positive controls on the real pool: procedures WITHOUT a valid guarantee
    def plugin_rounds(select):          # same calibration sets as the main arm (paired comparison)
        e_ = np.zeros((P, K))
        for (p, k), mask in zip(np.ndindex(P, K), r["cal_sets"]):
            i = select(Z[mask].mean(0))
            e_[p, k] = i >= 0 and R_pool[i] > ALPHA
        return e_
    def plugin(rh):                      # last path point whose empirical cal risk is <= alpha (no bound)
        ok = np.nonzero(rh <= ALPHA)[0]
        return int(ok[-1]) if len(ok) else -1
    d9 = {}
    e_pl = plugin_rounds(plugin)
    d9["plugin_no_bound"] = {"VR": e_pl.mean(), "CI": list(vr_ci(e_pl)), "C2_zone": three_zone(*vr_ci(e_pl), 0.065, 0.05)}
    for cfac in (4, 16, 64, 256):        # LTT that pretends the sample is c times larger
        e_c = plugin_rounds(lambda rh, c=cfac: fixed_sequence(rh, N * c, ALPHA)[0])
        d9[f"LTT_N_inflated_x{cfac}"] = {"VR": e_c.mean(), "CI": list(vr_ci(e_c)), "C2_zone": three_zone(*vr_ci(e_c), 0.065, 0.05)}
    res["D9_positive_controls"] = d9
    # T1b (v1.3): C2 is informative only if a procedure without guarantee is caught on this pool and path
    res["T1b_positive_control_caught"] = d9["plugin_no_bound"]["CI"][0] > 0.05
    if res["C2"]["decision"] == "Chap nhan" and not res["T1b_positive_control_caught"]:   # T1b only blocks acceptance
        res["C2"]["decision"] = "Khong kiem dinh duoc (T1b)"
        res["H1a"] = "KHONG KIEM DINH DUOC (T1b)"

    # ---------------- B-img: images as units, ignoring rides (descriptive)
    g_path = np.array(path)
    Zi = fdp(c["N"][:, ti][:, g_path], c["TP"][:, ti][:, g_path])
    Ri = Zi.mean(0)
    ri = run_ltt_rounds(Zi, Ri, N, P, K, np.random.default_rng(7))
    ei = ri["e"].astype(float)
    res["B_img"] = {"VR": ei.mean(), "CI": list(vr_ci(ei)), "Yield": float((ri["idx"] >= 0).mean()),
                    "reach_jstar_share": ri["reach"].mean(), "R_pool_img_path": Ri.tolist()}

    # ---------------- C2b instrument, one level down: images (units) inside rides (blocks)
    gA = reg["tau_A1_index"]
    zA = fdp(c["N"][:, ti, gA], c["TP"][:, ti, gA])
    sizes = np.bincount(gidx)
    members = [np.nonzero(gidx == j)[0] for j in range(len(rides))]
    rng2 = np.random.default_rng(11)
    var_rand, var_blk = np.zeros((P, K)), np.zeros((P, K))
    for p in range(P):
        perm = rng2.permutation(len(zA))
        rp_ = rng2.permutation(len(rides))
        cum = np.cumsum(sizes[rp_])
        start = 0
        for k in range(K):
            var_rand[p, k] = zA[perm[k * N:(k + 1) * N]].mean()
            stop = int(np.searchsorted(cum, cum[start - 1] + N if start else N)) + 1
            var_blk[p, k] = zA[np.concatenate([members[j] for j in rp_[start:stop]])].mean()
            start = stop
    bt = np.random.default_rng(3)
    ratios = []
    for _ in range(2000):
        a = var_blk[bt.integers(0, P, P)].ravel(); b = var_rand[bt.integers(0, P, P)].ravel()
        ratios.append(a.var(ddof=1) / b.var(ddof=1))
    icc_img = icc_anova(zA, gidx)
    m_t = float((sizes ** 2).sum() / sizes.sum())
    res["C2b_instrument_images_in_rides"] = {
        "var_ratio": float(var_blk.var(ddof=1) / var_rand.var(ddof=1)), "CI": np.percentile(ratios, [2.5, 97.5]).tolist(),
        "icc_image_anova": icc_img, "m_tilde": m_t, "Kish_Deff": 1 + (m_t - 1) * icc_img}

    # ---------------- stress (descriptive): subpopulations at tau_hat
    attrs = {}
    for n in c["names"]:
        a = json.load(open(os.path.join(DATA, "100k", "val", n + ".json"))).get("attributes", {})
        attrs[n] = a
    first = {}
    for n, gi in zip(c["names"], gidx):
        first.setdefault(gi, attrs[n])
    sub = {"night": np.array([first[j].get("timeofday") == "night" for j in range(len(rides))]),
           "rain_snow": np.array([first[j].get("weather") in ("rainy", "snowy") for j in range(len(rides))]),
           "highway": np.array([first[j].get("scene") == "highway" for j in range(len(rides))])}
    sel = r["idx"][nonempty].astype(int)
    res["stress"] = {k: {"rides": int(m.sum()), "median_risk_at_tau_hat": float(np.median(Z[m][:, sel].mean(0))) if len(sel) else None,
                         "share_rounds_risk_gt_alpha": float((Z[m][:, sel].mean(0) > ALPHA).mean()) if len(sel) else None}
                     for k, m in sub.items()}
    write_json(os.path.join(RESULTS, f"exp_b_{anchor}.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("R_pool_path", "path_alpha")}, indent=1, default=float))
    print("R_pool along path:", [f"{a:.4f}:{rv:.4f}" for a, rv in zip(reg["path_alpha"], R_pool)])


if __name__ == "__main__":
    main(sys.argv[1])
