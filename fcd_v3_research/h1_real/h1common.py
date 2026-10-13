"""Shared constants and the HB / LTT core used by every H1 real-data script."""
import json
import math
import os

import numpy as np
from scipy import stats

DATA = r"D:\tool_build_phase\ai_router\data"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")

ALPHA, ALPHA_C, DELTA = 0.02, 0.10, 0.05
Z95 = stats.norm.ppf(1 - DELTA)  # one-sided z_{1-delta} = 1.645

# 3-D grid G (registered in P3 before any Val label is read)
# low end added after the P3 reachability check failed on the first grid (max Macro-FDP 0.021 < 2 alpha_C)
TAU_CLS = np.array([0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025,
                    0.030, 0.040, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
TAU_LOC = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9])       # ensemble IoU agreement >= tau_loc
FCD_QUANTILES = [1.0, 0.90, 0.75, 0.50, 0.25]            # tau_fcd = Meta-Train quantile of fcd (1.0 = no filter)
BASE_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                      0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
SCORE_FLOOR = 0.004  # queries below this car score are never stored
EXPERTS = ["expert_1", "expert_2", "expert_3"]
CAR_LABEL = 3        # the experts were fine-tuned with category_id = 3 (HF id2label: 'motorbike')


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_np_default)


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- HB p-value (Section 5.2)
def kl_bern(a, b):
    a = np.clip(a, 1e-300, 1 - 1e-16)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def p_hb(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: R > alpha (vectorised over rhat)."""
    rhat = np.asarray(rhat, dtype=float)
    r = np.minimum(rhat, alpha)
    p_hoeff = np.where(rhat >= alpha, 1.0, np.exp(-n * kl_bern(r, alpha)))
    k = np.ceil(n * rhat - 1e-9)
    p_bent = np.minimum(1.0, math.e * stats.binom.cdf(k, n, alpha))
    return np.minimum(p_hoeff, p_bent)


def k_max(n, alpha, delta=DELTA):
    """Largest integer k with p_HB(k/n) <= delta (None if even k = 0 fails)."""
    ks = np.arange(0, int(alpha * n) + 2)
    ok = p_hb(ks / n, n, alpha) <= delta
    return int(ks[ok].max()) if ok.any() else None


def fixed_sequence(rhat_path, n, alpha, delta=DELTA):
    """Fixed-sequence LTT along a path. Returns index of tau_hat (-1 = empty) and #tests run."""
    p = p_hb(rhat_path, n, alpha)
    fail = np.nonzero(p > delta)[0]
    stop = fail[0] if len(fail) else len(p)
    return stop - 1, stop


# ---------------------------------------------------------------- CIs
def cp_ci(x, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else stats.beta.ppf(a, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a, x + 1, n - x)
    return float(lo), float(hi)


def partition_bootstrap(per_part, stat=np.mean, n_boot=4000, seed=0):
    """per_part: array (P, K) of per-round values. Resample partitions with replacement."""
    rng = np.random.default_rng(seed)
    P = per_part.shape[0]
    idx = rng.integers(0, P, size=(n_boot, P))
    vals = np.array([stat(per_part[i].ravel()) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def vr_ci(e, seed=0):
    """CI for a violation rate from a (P, K) indicator array.

    Percentile bootstrap over partitions degenerates to [0, 0] when no round violates, which overstates
    precision. The registered upper bound is therefore max(bootstrap upper, Clopper-Pearson upper with the
    partition count P as sample size), i.e. it never claims more precision than P independent rounds."""
    P = e.shape[0]
    lo, hi = partition_bootstrap(e.astype(float), seed=seed)
    x = int(round(e.mean() * P))
    cp_lo, cp_hi = cp_ci(x, P)
    return min(lo, cp_lo), max(hi, cp_hi)


# ---------------------------------------------------------------- ICC (ANOVA, unequal cluster sizes)
def icc_anova(values, groups):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    G, n = len(uniq), len(values)
    if G < 2 or n - G < 1:
        return float("nan")
    means = np.bincount(inv, weights=values) / counts
    grand = values.mean()
    ssb = np.sum(counts * (means - grand) ** 2)
    ssw = np.sum((values - means[inv]) ** 2)
    msb, msw = ssb / (G - 1), ssw / (n - G)
    n0 = (n - np.sum(counts ** 2) / n) / (G - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")
