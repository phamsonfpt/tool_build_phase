"""P3 — pilot on Meta-Train only; freezes everything Exp B / Exp C' need into registration_<anchor>.json.

usage: python p3_pilot.py <anchor_expert>
"""
import hashlib
import json
import os
import sys

import numpy as np

from h1common import ALPHA, ALPHA_C, BASE_GRID, CACHE, TAU_CLS, Z95, icc_anova, k_max, write_json
from h1units import aggregate, build_path, fdp, grid_params, load_counts, point_at, ride_index

N_REG = 1500


def core_ext_alphas(n):
    u = min(0.015, np.floor((k_max(n, ALPHA) / n + 0.003) * 1000) / 1000)
    core = np.round(np.arange(0.002, u + 1e-9, 0.001), 4)
    ext = np.concatenate([np.round(np.arange(u + 0.001, 0.030 + 1e-9, 0.001), 4),
                          np.round(np.arange(0.0325, 0.040 + 1e-9, 0.0025), 4)])
    return core, ext


ALPHAS_C = np.concatenate([np.round(np.arange(0.005, 0.090 + 1e-9, 0.005), 4),
                           np.round(np.arange(0.0925, 0.120 + 1e-9, 0.0025), 4),
                           np.round(np.arange(0.13, 0.200 + 1e-9, 0.01), 4)])


def main(anchor):
    c = load_counts("meta", anchor)
    gidx, rides = ride_index("meta", c["names"])
    params = grid_params(c["tau_fcd"])
    base_cols = [params.index((0.0, float("inf"), float(t))) for t in BASE_GRID]
    rep = {"anchor": anchor, "meta_rides": len(rides), "meta_images": len(c["names"]), "N": N_REG}
    kmax = k_max(N_REG, ALPHA)

    per_theta = {}
    for ti, th in enumerate(c["thetas"]):
        u = aggregate(c, gidx, len(rides), ti)
        Z = fdp(u["N"], u["TP"])                      # rides x G
        f_meta = Z.mean(0)
        cov = u["TP"].sum(0) / u["M"].sum()
        sd = Z.std(0, ddof=1)
        # baseline plug-in: smallest confidence on BASE_GRID (no loc / fcd filter) with Macro-FDP <= alpha
        ok = [j for j in base_cols if f_meta[j] <= ALPHA]
        base = min(ok, key=lambda j: params[j][2]) if ok else max(base_cols, key=lambda j: params[j][2])
        rho = cov / max(cov[base], 1e-12)
        r_star = kmax / N_REG - Z95 * sd / np.sqrt(N_REG)
        qual = np.nonzero((f_meta <= r_star) & (rho >= 0.6))[0]
        per_theta[float(th)] = dict(u=u, Z=Z, f=f_meta, cov=cov, base=base, base_ok=bool(ok), rho=rho,
                                    r_star=r_star, qual=qual, sd=sd)
        rep[f"theta_{th}"] = {
            "tau_base": params[base], "tau_base_meets_alpha": bool(ok), "fdp_base": f_meta[base],
            "cov_base": cov[base], "n_qualifying_tau": len(qual),
            "best_rho_among_qualifying": float(rho[qual].max()) if len(qual) else None,
            "r_star_range_at_feasible": [float(r_star[f_meta <= ALPHA].min()), float(r_star[f_meta <= ALPHA].max())]
            if (f_meta <= ALPHA).any() else None,
            "min_fdp_on_G": float(f_meta.min()),
        }

    theta = 0.75 if len(per_theta[0.75]["qual"]) else 0.5
    rep["theta_IoU"] = theta
    P = per_theta[theta]
    # reachability check (added in v1.3): G must contain points whose Meta-Train risk exceeds 2 alpha_C,
    # otherwise the path extensions cannot place true nulls and T1/T2 fail by construction
    rep["reachability"] = {"max_fdp_meta_on_G": float(P["f"].max()),
                           "reaches_2alpha": bool(P["f"].max() >= 2 * ALPHA),
                           "reaches_2alpha_C": bool(P["f"].max() >= 2 * ALPHA_C)}
    ti = list(map(float, c["thetas"])).index(theta)

    # provisional path (core only) -> tau_ref at alpha_k = 0.008
    core, ext = core_ext_alphas(N_REG)
    path0 = build_path(P["f"], P["cov"], core)
    ref = path0[point_at(path0, 0.008)][1]
    # Exp A tier 2 replacement: within-ride ICC of image-level FDP at tau_ref (super-blocks need GPS: n/a)
    img_fdp = fdp(c["N"][:, ti, ref], c["TP"][:, ti, ref])
    icc_img = icc_anova(img_fdp, gidx)
    rep["tau_ref"] = params[ref]
    rep["icc_image_within_ride_meta"] = icc_img
    rep["R_unit"] = ("ride: the ride (video-name prefix) is the coarsest grouping recoverable without GPS/time "
                     "metadata; block / super-block ICC cannot be computed (D6)")

    # final paths
    path = build_path(P["f"], P["cov"], np.concatenate([core, ext]))
    path_c = build_path(P["f"], P["cov"], ALPHAS_C)
    # image-level Meta-Train FDP along Lambda_C (for D*_C of the Exp C' frame arm)
    img_Z_C = fdp(c["N"][:, ti, [g for _, g in path_c]], c["TP"][:, ti, [g for _, g in path_c]])

    def first_above(pth, a):
        for k, (lv, _) in enumerate(pth):
            if lv > a + 1e-12:
                return k
        return None

    k1 = first_above(path, ALPHA)
    sig2 = float(P["Z"][:, path[k1][1]].var(ddof=1)) if k1 is not None else None
    d_star = N_REG * (ALPHA - kmax / N_REG) ** 2 / (Z95 ** 2 * sig2) if sig2 else None
    kc = first_above(path_c, ALPHA_C)
    d_star_c = {}
    if kc is not None:
        s2c = float(img_Z_C[:, kc].var(ddof=1))
        for ncl in (100, 200, 400):
            nn = 2 * ncl
            d_star_c[ncl] = nn * (ALPHA_C - k_max(nn, ALPHA_C) / nn) ** 2 / (Z95 ** 2 * s2c)
    rep.update({
        "path": [{"alpha_k": a, "tau": params[g], "fdp_meta": P["f"][g], "cov_meta": P["cov"][g],
                  "rho_meta": P["rho"][g]} for a, g in path],
        "path_C": [{"alpha_k": a, "tau": params[g], "fdp_meta": P["f"][g], "cov_meta": P["cov"][g]}
                   for a, g in path_c],
        "K": len(path), "K_core": sum(1 for a, _ in path if a <= core[-1] + 1e-12), "K_C": len(path_c),
        "sigma2_alpha": sig2, "D_star": d_star, "D_thr": min(2.0, d_star) if d_star else 2.0,
        "D_star_C_by_Ncluster": d_star_c,
        "var_ratio_Z_over_binary": sig2 / (ALPHA * (1 - ALPHA)) if sig2 else None,
        "empty_unit_share_along_path": [float((P["u"]["N"][:, g] == 0).mean()) for _, g in path],
        "halluc_share_along_path": [float(P["u"]["VH"][:, g].sum() / max((P["u"]["N"][:, g] - P["u"]["TP"][:, g]).sum(), 1))
                                    for _, g in path],
    })
    reg = {
        "anchor": anchor, "theta_IoU": theta, "theta_index": ti, "N": N_REG, "alpha": ALPHA, "alpha_C": ALPHA_C,
        "tau_fcd_grid": list(map(float, c["tau_fcd"])), "tau_cls_grid": TAU_CLS.tolist(),
        "tau_base_index": int(P["base"]), "tau_base": params[P["base"]], "tau_base_meets_alpha": P["base_ok"],
        "path_index": [g for _, g in path], "path_alpha": [a for a, _ in path],
        "path_C_index": [g for _, g in path_c], "path_C_alpha": [a for a, _ in path_c],
        "tau_ref_index": int(ref), "tau_ref_C_index": int(path_c[point_at(path_c, 0.05)][1]),
        "tau_A1_index": int(path[point_at(path, ALPHA)][1]),
        "D_star": d_star, "D_thr": rep["D_thr"], "D_star_C_by_Ncluster": d_star_c,
        "gamma_cov": 0.5, "P_partitions": 334, "K_per_partition": 3,
    }
    blob = json.dumps(reg, sort_keys=True, default=float).encode()
    reg["sha256"] = hashlib.sha256(blob).hexdigest()
    write_json(os.path.join(CACHE, f"registration_{anchor}.json"), reg)
    write_json(os.path.join(CACHE, f"p3_report_{anchor}.json"), rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("path", "path_C", "empty_unit_share_along_path",
                                                                 "halluc_share_along_path")}, indent=1, default=float))
    print("Lambda_1D:")
    for p in rep["path"]:
        print(f"  a={p['alpha_k']:.4f} tau={p['tau']} fdp={p['fdp_meta']:.4f} cov={p['cov_meta']:.3f} rho={p['rho_meta']:.2f}")
    print("Lambda_C:", [(p["alpha_k"], round(p["fdp_meta"], 4)) for p in rep["path_C"]])
    print("registration sha256:", reg["sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
