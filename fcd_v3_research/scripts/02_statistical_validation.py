"""Monte Carlo validation and power analysis for the H1-H3 statistical analysis plan.

Usage (from the repo root):
    python fcd_v3_research/scripts/02_statistical_validation.py            # full run
    python fcd_v3_research/scripts/02_statistical_validation.py --quick    # fewer repetitions

Writes docs/paper/results/statistical_validation.json.
No real detector output is used: every input distribution is a stated assumption
(see docs/paper/03_STATISTICAL_ANALYSIS_PLAN.md) to be replaced by Meta-Train pilot estimates.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from risk_stats import (  # noqa: E402
    clopper_pearson,
    clopper_pearson_upper,
    fixed_sequence,
    k_max,
    n_for_power_paired_t,
    power_binomial_exceeds,
    power_paired_t,
)

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO, "docs", "paper", "results")


# ---------------------------------------------------------------- H1: clip-level vs frame-level LTT
def risk_grid_c():
    """Exp C risk grid of the H1 document (alpha_C = 0.10, K_C = 38)."""
    core = np.round(np.arange(0.005, 0.0901, 0.005), 4)
    mid = np.round(np.arange(0.0925, 0.1201, 0.0025), 4)
    tail = np.round(np.arange(0.13, 0.2001, 0.01), 4)
    return np.concatenate([core, mid, tail])


def logit_offsets_for(targets, total_sd):
    """beta_k such that E[expit(beta_k + total_sd * Z)] = target_k, Z ~ N(0, 1)."""
    def mean_risk(beta):
        f = lambda z: special.expit(beta + total_sd * z) * stats.norm.pdf(z)
        return integrate.quad(f, -8, 8)[0]
    return np.array([optimize.brentq(lambda b: mean_risk(b) - t, -20, 5) for t in targets])


def simulate_h1(n_clips, frames, sigma_b, sigma_e, phi, rounds, alpha, delta, rng):
    """One scenario of the clip-vs-frame comparison (mechanism of H1b).

    Frame FDP: each frame has 1 + Poisson(3) accepted boxes, each box is FP with prob
    expit(beta_k + b_clip + e_frame); e is AR(1) within the clip. A shared uniform per box
    keeps FP counts nested along the path. True risk R_k is the grid value by construction.
    """
    targets = risk_grid_c()
    total_sd = math.sqrt(sigma_b ** 2 + sigma_e ** 2)
    beta = logit_offsets_for(targets, total_sd) if total_sd > 0 else special.logit(targets)
    k = targets.size
    unsafe = targets > alpha
    out = {"clip": [], "frame": []}
    ref = int(np.argmin(np.abs(targets - alpha / 2)))
    deff_list = []
    bmax = 10
    for _ in range(rounds):
        b = rng.normal(0, sigma_b, (n_clips, 1))
        e = np.empty((n_clips, frames))
        e[:, 0] = rng.normal(0, sigma_e, n_clips)
        innov_sd = sigma_e * math.sqrt(1 - phi ** 2)
        for t in range(1, frames):
            e[:, t] = phi * e[:, t - 1] + rng.normal(0, innov_sd, n_clips)
        n_box = np.minimum(1 + rng.poisson(3, (n_clips, frames)), bmax)
        q = special.expit(beta[None, None, :] + (b + e)[:, :, None])  # (C, T, K)
        u = rng.random((n_clips, frames, bmax))
        mask = np.arange(bmax)[None, None, :] < n_box[:, :, None]
        fp = ((u[:, :, :, None] < q[:, :, None, :]) & mask[:, :, :, None]).sum(2)
        fdp = fp / n_box[:, :, None]  # (C, T, K)
        z_clip = fdp.mean(1)
        j_clip = fixed_sequence(z_clip.mean(0), n_clips, alpha, delta)
        j_frame = fixed_sequence(fdp.reshape(-1, k).mean(0), n_clips * frames, alpha, delta)
        out["clip"].append(j_clip)
        out["frame"].append(j_frame)
        var_frame = fdp[:, :, ref].var()
        deff_list.append(frames * z_clip[:, ref].var() / var_frame if var_frame > 0 else np.nan)
    res = {}
    for arm, js in out.items():
        viol = sum(1 for j in js if j is not None and unsafe[j])
        nonempty = sum(1 for j in js if j is not None)
        sel = [targets[j] for j in js if j is not None]
        lo, hi = clopper_pearson(viol, rounds)
        res[arm] = {
            "violation_rate": viol / rounds,
            "vr_ci95": [lo, hi],
            "yield": nonempty / rounds,
            "mean_true_risk_selected": float(np.mean(sel)) if sel else None,
        }
    res["design_effect_frame_hat"] = float(np.nanmean(deff_list))
    return res


# ---------------------------------------------------------------- H2: power of the coverage comparison
def simulate_h2(n_units, delta_pts, base_cov, rho_obj, sigma_unit, reps, rng, alpha_test=0.0025,
                gamma=5.0, gamma0=2.0):
    """Paired comparison of two signals on the same GT objects.

    Each unit has 1 + Poisson(15) GT objects (about 1.4 BDD100K keyframes of 8 classes per ride).
    Unit difficulty shifts both signals' logit coverage by N(0, sigma_unit^2); per object, a
    Gaussian copula with correlation rho_obj links "covered by A" and "covered by B".
    Logit offsets are solved so that the population coverages are exactly base_cov and
    base_cov + delta_pts / 100 (g is independent of u, so Cov = E[expit(l + u)]).
    """
    from risk_stats import wilcoxon_greater

    lb, la = logit_offsets_for([base_cov, base_cov + delta_pts / 100], sigma_unit)
    acc = {"wilcoxon_only": 0, "old_rule": 0, "margin_rule": 0}
    deltas = []
    for _ in range(reps):
        g = 1 + rng.poisson(15, n_units)
        u = rng.normal(0, sigma_unit, n_units)
        tot = g.sum()
        unit_of = np.repeat(np.arange(n_units), g)
        z1 = rng.standard_normal(tot)
        z2 = rho_obj * z1 + math.sqrt(1 - rho_obj ** 2) * rng.standard_normal(tot)
        # per-unit coverage probability, then threshold the copula normals
        pa = special.expit(la + u)[unit_of]
        pb = special.expit(lb + u)[unit_of]
        cov_a = stats.norm.cdf(z1) < pa
        cov_b = stats.norm.cdf(z2) < pb
        tp_a = np.bincount(unit_of, cov_a, n_units)
        tp_b = np.bincount(unit_of, cov_b, n_units)
        d = tp_a - tp_b
        est = d.sum() / tot
        # ratio-estimator SE for sum(d) / sum(g)
        se = math.sqrt(n_units / (n_units - 1) * ((d - est * g) ** 2).sum()) / tot
        p = wilcoxon_greater(d)["p"]
        z_margin = (est - gamma0 / 100) / se
        deltas.append(est * 100)
        acc["wilcoxon_only"] += p < alpha_test
        acc["old_rule"] += (p < alpha_test) and (est * 100 >= gamma)
        acc["margin_rule"] += (p < alpha_test) and (stats.norm.sf(z_margin) < alpha_test)
    out = {k: v / reps for k, v in acc.items()}
    out["sd_delta_hat_pts"] = float(np.std(deltas))
    return out


# ---------------------------------------------------------------- H3: power across seeds and rounds
def simulate_h3_rules(delta, sd, n, rounds_corr, t_rounds, reps, rng, alpha=0.05, gamma=1.5):
    """Old rule (mean >= gamma and p < alpha) and the IUT over all rounds, by simulation."""
    cov = np.full((t_rounds, t_rounds), rounds_corr) + (1 - rounds_corr) * np.eye(t_rounds)
    chol = np.linalg.cholesky(cov)
    old_final, iut, final_only = 0, 0, 0
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    for _ in range(reps):
        d = delta + sd * (rng.standard_normal((n, t_rounds)) @ chol.T)
        m = d.mean(0)
        s = d.std(0, ddof=1)
        tstat = m / (s / math.sqrt(n))
        rej = tstat > tcrit
        final_only += rej[-1]
        old_final += rej[-1] and m[-1] >= gamma
        iut += rej.all()
    return {"final_round_test": final_only / reps, "old_rule_final": old_final / reps, "iut_all_rounds": iut / reps}


# ---------------------------------------------------------------- FCD aggregators (doc 02, D5.4)
def simulate_fcd_aggregators(reps, rng):
    rho, s0 = stats.chi.mean(64), stats.chi.std(64)
    aggs = {
        "anchor": lambda d: d[:, 0],
        "mean": lambda d: d.mean(1),
        "var": lambda d: d.var(1, ddof=1),
        "max": lambda d: d.max(1),
        "mean_plus_sd": lambda d: d.mean(1) + d.std(1, ddof=1),
    }
    normal = rng.normal(rho, s0, (reps, 3))
    q95 = {k: float(np.quantile(f(normal), 0.95)) for k, f in aggs.items()}
    table = {}
    for kind in ("partial", "consensus"):
        for shift in (1.5, 2.0, 3.0, 5.0):
            d = rng.normal(rho, s0, (reps, 3))
            if kind == "partial":
                d[np.arange(reps), rng.integers(0, 3, reps)] += shift
            else:
                d += shift
            table[f"{kind}_{shift}"] = {k: float((f(d) > q95[k]).mean()) for k, f in aggs.items()}
    return {"q95_normal": q95, "detection_rate": table}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scale = 0.2 if args.quick else 1.0
    results = {"seed": args.seed, "quick": args.quick}
    t0 = time.time()

    # S0. HB certification frontier (alpha = 0.02, delta = 0.05)
    results["hb_kmax_alpha_0.02"] = {n: k_max(n, 0.02, 0.05) for n in (150, 400, 800, 1000, 1500, 2000, 3000, 5000)}

    # S1. H1 mechanism (alpha_C = 0.10)
    h1 = {}
    scenarios = [
        ("independent_frames", dict(sigma_b=0.0, sigma_e=0.8, phi=0.0)),
        ("ar_only", dict(sigma_b=0.0, sigma_e=0.8, phi=0.9)),
        ("clip_effect_plus_ar", dict(sigma_b=0.8, sigma_e=0.5, phi=0.9)),
        ("strong_clip_effect", dict(sigma_b=1.2, sigma_e=0.5, phi=0.9)),
    ]
    for n_clips in (100, 400):
        for name, kw in scenarios:
            rounds = max(50, int(400 * scale))
            h1[f"{name}_C{n_clips}"] = simulate_h1(n_clips, 50, rounds=rounds, alpha=0.10, delta=0.05, rng=rng, **kw)
            print(f"[H1] {name} C={n_clips} done ({time.time() - t0:.0f}s)", flush=True)
    results["h1_clip_vs_frame"] = h1

    # S2. Number of Monte Carlo rounds for Exp B (violation-rate estimate)
    results["expB_rounds"] = {
        "cp_upper95": {b: {str(r): clopper_pearson_upper(int(round(r * b)), b) for r in (0.0, 0.01, 0.02, 0.03)}
                       for b in (100, 300, 500, 1000, 3000)},
        "power_detect_vr_0.10_vs_0.05": {b: power_binomial_exceeds(0.10, 0.05, b) for b in (100, 200, 300, 500, 1000)},
        "power_detect_vr_0.08_vs_0.05": {b: power_binomial_exceeds(0.08, 0.05, b) for b in (100, 200, 300, 500, 1000)},
    }

    # S3. H2 power
    h2 = {}
    reps = max(60, int(300 * scale))
    for rho_obj in (0.5, 0.8):
        for n_units in (500, 1000, 2000, 4800):
            for dpts in (0, 2, 3, 4, 5, 6, 7, 8, 10):
                h2[f"rho{rho_obj}_N{n_units}_d{dpts}"] = simulate_h2(n_units, dpts, 0.55, rho_obj, 0.8, reps, rng)
        print(f"[H2] rho={rho_obj} done ({time.time() - t0:.0f}s)", flush=True)
    results["h2_power"] = h2

    # S4. H3 power
    analytic = {}
    for sd in (0.5, 1.0, 1.5, 2.0):
        for delta in (1.5, 2.0, 3.0):
            analytic[f"sd{sd}_delta{delta}"] = {
                "power_n": {n: power_paired_t(delta, sd, n) for n in (3, 5, 8, 10, 15)},
                "n_for_80": n_for_power_paired_t(delta, sd, 0.8),
            }
    rules = {}
    for sd in (0.5, 1.0, 1.5):
        for n in (3, 5, 10):
            for delta in (1.5, 2.5):
                rules[f"sd{sd}_n{n}_delta{delta}"] = simulate_h3_rules(delta, sd, n, 0.5, 3, max(2000, int(20000 * scale)), rng)
    results["h3_power"] = {"analytic_paired_t": analytic, "decision_rules_sim": rules}

    # S5. Image-level false alarms of a per-box 95% gate
    results["image_false_alarm_per_box_q95"] = {n: 1 - 0.95 ** n for n in (1, 5, 10, 20, 40)}
    results["image_bonferroni_per_box_level"] = {n: 0.05 / n for n in (1, 5, 10, 20, 40)}

    # S6. FCD aggregators
    results["fcd_aggregators"] = simulate_fcd_aggregators(max(40000, int(200000 * scale)), rng)

    results["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "statistical_validation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved to {OUT_DIR} in {results['runtime_s']}s")


if __name__ == "__main__":
    main()
