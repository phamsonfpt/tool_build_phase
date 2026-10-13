"""D11 (descriptive, after registration) — what happens without the resolution rule.

Rebuilds Lambda_1D with the coarse t_cls ladder of the first draft of v1.3 (0.004–0.010 step 0.001, then
0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, and the 0.05–0.99 grid), same reachability rule, same path
algorithm, and reruns the T1/T1b quantities on the same Val pool and the same partitions as exp_b.py.
usage: python ablation_coarse_grid.py <anchor_expert>
"""
import json
import os
import sys

import numpy as np

from h1common import ALPHA, ALPHA_C, CACHE, RESULTS, TAU_CLS, fixed_sequence, vr_ci, write_json
from h1units import aggregate, build_path, fdp, grid_params, load_counts, ride_index
from p3_pilot import N_REG, core_ext_alphas

COARSE = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040] + \
         [float(t) for t in TAU_CLS if t >= 0.05 - 1e-12]


def main(anchor):
    reg = json.load(open(os.path.join(CACHE, f"registration_{anchor}.json")))
    ti = reg["theta_index"]
    cm = load_counts("meta", anchor)
    gm, rm = ride_index("meta", cm["names"])
    params = grid_params(cm["tau_fcd"])
    cls_of = np.array([p[2] for p in params])
    um = aggregate(cm, gm, len(rm), ti)
    f = fdp(um["N"], um["TP"]).mean(0)
    cov = um["TP"].sum(0) / um["M"].sum()
    coarse = np.isin(np.round(cls_of, 4), np.round(COARSE, 4))
    t_min = None
    for t in sorted({t for t in COARSE if t <= 0.05}, reverse=True):     # same reachability rule
        inG = coarse & (cls_of >= t - 1e-12)
        if f[inG].max() >= 2 * ALPHA_C:
            t_min = t
            break
    core, ext = core_ext_alphas(N_REG)
    path = [g for _, g in build_path(np.where(inG, f, np.inf), cov, np.concatenate([core, ext]))]
    alphas = [a for a, _ in build_path(np.where(inG, f, np.inf), cov, np.concatenate([core, ext]))]

    cv = load_counts("val", anchor)
    gv, rv = ride_index("val", cv["names"])
    uv = aggregate(cv, gv, len(rv), ti)
    Z = fdp(uv["N"][:, path], uv["TP"][:, path])
    R_pool = Z.mean(0)
    jstar = int(np.nonzero(R_pool > ALPHA)[0][0]) if (R_pool > ALPHA).any() else None
    P, K, N = reg["P_partitions"], reg["K_per_partition"], reg["N"]
    rng = np.random.default_rng(2026)                # same partitions as exp_b.py
    e_ltt, e_plug = np.zeros((P, K)), np.zeros((P, K))
    for p in range(P):
        perm = rng.permutation(len(rv))
        for k in range(K):
            rh = Z[perm[k * N:(k + 1) * N]].mean(0)
            i, _ = fixed_sequence(rh, N, ALPHA)
            e_ltt[p, k] = i >= 0 and R_pool[i] > ALPHA
            ok = np.nonzero(rh <= ALPHA)[0]
            e_plug[p, k] = len(ok) > 0 and R_pool[ok[-1]] > ALPHA
    res = {"anchor": anchor, "t_cls_min": t_min, "K": len(path), "path_alpha": alphas,
           "R_pool_path": R_pool.tolist(), "jstar_alpha_k": alphas[jstar] if jstar is not None else None,
           "R_pool_jstar": float(R_pool[jstar]) if jstar is not None else None,
           "last_below_alpha_pool": float(R_pool[R_pool <= ALPHA].max()),
           "LTT_VR": e_ltt.mean(), "plugin_VR": e_plug.mean(), "plugin_CI": list(vr_ci(e_plug)),
           "T1b_would_pass": vr_ci(e_plug)[0] > 0.05}
    write_json(os.path.join(RESULTS, f"ablation_coarse_grid_{anchor}.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("R_pool_path", "path_alpha")}, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
