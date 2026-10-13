"""Experiment C' — the Exp C protocol (alpha_C = 0.10, path Lambda_C, three arms) run on the only clustered
labelled data available: Val rides with >= 2 keyframes, M = 2 keyframes per ride (first two by name).
It is NOT the registered H1b test (that needs MOT clips with M = 200 labelled frames); it is a real-data
negative/weak-dependence control for the C4/C5 machinery: with M = 2 the frame-arm design effect is
1 + ICC <= 2, so a correct decision rule must not accept H1b here.
usage: python exp_c_prime.py <anchor_expert>
"""
import json
import os
import sys

import numpy as np

from h1common import ALPHA_C, CACHE, RESULTS, fixed_sequence, icc_anova, k_max, vr_ci, Z95, write_json
from h1units import fdp, load_counts, ride_index

M = 2


def main(anchor):
    reg = json.load(open(os.path.join(CACHE, f"registration_{anchor}.json")))
    c = load_counts("val", anchor)
    gidx, rides = ride_index("val", c["names"])
    ti, path = reg["theta_index"], np.array(reg["path_C_index"])
    img = fdp(c["N"][:, ti][:, path], c["TP"][:, ti][:, path])        # images x K_C
    clusters = []
    for j in range(len(rides)):
        mem = np.nonzero(gidx == j)[0]
        if len(mem) >= M:
            clusters.append(mem[:M])                                  # names are sorted -> first two
    clusters = np.array(clusters)                                       # n_cl x M
    F = img[clusters]                                                  # n_cl x M x K_C
    Zc = F.mean(1)
    R_pool = Zc.mean(0)                                                # = frame-level risk (equal M)
    n_cl = len(clusters)
    H0 = np.nonzero(R_pool > ALPHA_C)[0]
    jstar = int(H0[0]) if len(H0) else None
    # C4': design effect of the frame arm at tau_ref^C, bootstrap over clusters
    kref = reg["path_C_index"].index(reg["tau_ref_C_index"])
    v = F[:, :, kref]
    grp = np.repeat(np.arange(n_cl), M)
    icc = icc_anova(v.ravel(), grp)
    rng = np.random.default_rng(5)
    boots = []
    for _ in range(1000):
        s = rng.integers(0, n_cl, n_cl)
        boots.append(icc_anova(v[s].ravel(), grp))
    deff = 1 + (M - 1) * icc
    dlo, dhi = 1 + (M - 1) * np.nanpercentile(boots, 2.5), 1 + (M - 1) * np.nanpercentile(boots, 97.5)
    d_star_c = max(reg["D_star_C_by_Ncluster"].values()) if reg["D_star_C_by_Ncluster"] else 5.0
    c4 = "Chap nhan" if dlo >= max(5.0, d_star_c) else ("Bac bo" if dhi < 2 else "Khong ket luan")
    res = {"anchor": anchor, "clusters": n_cl, "M": M, "R_pool_C_path": R_pool.tolist(),
           "T2_H0_nonempty": bool(len(H0)), "jstar": jstar,
           "C4prime": {"icc": icc, "Deff_frame": deff, "CI": [dlo, dhi], "decision": c4}, "levels": {}}
    eligible = []
    for ncl in (100, 200, 400):
        K = int(0.6 * n_cl) // ncl
        P = int(np.ceil(1000 / K))
        rng = np.random.default_rng(100 + ncl)
        out = {a: {"idx": np.zeros((P, K), int), "e": np.zeros((P, K)), "reach": np.zeros((P, K))} for a in ("arm1", "arm2", "arm3")}
        e_plug = np.zeros((P, K))
        dleak = np.zeros((P, K))
        for p in range(P):
            perm = rng.permutation(n_cl)
            for k in range(K):
                cal = perm[k * ncl:(k + 1) * ncl]
                test = np.setdiff1d(np.arange(n_cl), cal)
                arms = {"arm1": (Zc[cal].mean(0), ncl),
                        "arm2": (F[cal].reshape(-1, F.shape[2]).mean(0), ncl * M)}
                half = rng.integers(0, M, ncl)                         # Arm 3: one frame cal, one leaked
                f_cal = F[cal, half]; f_leak = F[cal, 1 - half]
                arms["arm3"] = (f_cal.mean(0), ncl * M // 2)
                for a, (rhat, n) in arms.items():
                    idx, stop = fixed_sequence(rhat, n, ALPHA_C)
                    out[a]["idx"][p, k] = idx
                    out[a]["e"][p, k] = idx >= 0 and R_pool[idx] > ALPHA_C
                    out[a]["reach"][p, k] = jstar is not None and stop >= jstar
                ok = np.nonzero(arms["arm2"][0] <= ALPHA_C)[0]       # T2b positive control: plug-in, no bound
                e_plug[p, k] = len(ok) > 0 and R_pool[ok[-1]] > ALPHA_C
                i3 = out["arm3"]["idx"][p, k]
                dleak[p, k] = (Zc[test, i3].mean() - f_leak[:, i3].mean()) if i3 >= 0 else np.nan
        lv = {"K": K, "P": P}
        for a in out:
            e = out[a]["e"]
            lv[a] = {"VR": e.mean(), "CI": list(vr_ci(e)), "Yield": float((out[a]["idx"] >= 0).mean()),
                     "reach_jstar": out[a]["reach"].mean()}
        lv["plugin_frame"] = {"VR": e_plug.mean(), "CI": list(vr_ci(e_plug))}
        lv["arm3"]["Delta_leak_mean"] = float(np.nanmean(dleak))
        nn = ncl * M
        lv["kmax_cluster"], lv["kmax_frame"] = k_max(ncl, ALPHA_C), k_max(nn, ALPHA_C)
        res["levels"][ncl] = lv
        if lv["arm1"]["Yield"] >= 0.5:
            eligible.append(ncl)
    E = eligible
    if len(E) >= 2 and all(res["levels"][n]["arm2"]["CI"][0] > 0.05 and res["levels"][n]["arm1"]["CI"][1] <= 0.065 for n in E):
        c5 = "Chap nhan"
    elif len(E) >= 2 and sum(res["levels"][n]["arm2"]["CI"][1] <= 0.065 for n in E) >= 2:
        c5 = "Bac bo"
    else:
        c5 = "Khong ket luan"
    res["T2b_positive_control_caught"] = {n: res["levels"][n]["plugin_frame"]["CI"][0] > 0.05 for n in (100, 200, 400)}
    if not res["T2_H0_nonempty"]:
        c5 = "Khong kiem dinh duoc (T2)"
    elif c5 == "Bac bo" and sum(res["T2b_positive_control_caught"][n] for n in E) < 2:   # T2b only blocks rejection
        c5 = "Khong kiem dinh duoc (T2b)"
    res["E"] = E
    res["C5prime"] = c5
    res["H1b_prime"] = "CHAP NHAN" if (c4 == "Chap nhan" and c5 == "Chap nhan") else ("BAC BO" if c5 == "Bac bo" else "KHONG KET LUAN")
    write_json(os.path.join(RESULTS, f"exp_c_prime_{anchor}.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "R_pool_C_path"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
