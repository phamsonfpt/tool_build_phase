"""Hypothesis tests, confidence intervals and effect sizes used in the H1-H3 analysis plan."""
import numpy as np
from scipy import stats


def clopper_pearson(k, n, conf=0.95):
    """Exact two-sided binomial CI."""
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def clopper_pearson_upper(k, n, conf=0.95):
    """Exact one-sided upper bound."""
    return 1.0 if k == n else float(stats.beta.ppf(conf, k + 1, n - k))


def wilcoxon_greater(d):
    """One-sided paired Wilcoxon signed-rank (H1: median d > 0), zeros dropped.

    Returns statistic W+, p-value and the matched-pairs rank-biserial correlation
    r = (W+ - W-) / (W+ + W-), an effect size in [-1, 1].
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return {"w_plus": 0.0, "p": 1.0, "r_rank_biserial": 0.0, "n_nonzero": 0}
    res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return {
        "w_plus": float(w_plus),
        "p": float(res.pvalue),
        "r_rank_biserial": float((w_plus - w_minus) / (w_plus + w_minus)),
        "n_nonzero": int(d.size),
    }


def holm(pvals, alpha):
    """Holm step-down. Returns (reject mask, adjusted p-values) in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj <= alpha, adj


def paired_t_greater(d, margin=0.0):
    """One-sided paired t-test of H0: mean(d) <= margin. Also returns Cohen's d_z and a one-sided 95% lower bound."""
    d = np.asarray(d, dtype=float)
    n = d.size
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = (mean - margin) / se
    p = float(stats.t.sf(t, n - 1))
    lower = float(mean - stats.t.ppf(0.95, n - 1) * se)
    return {"mean": float(mean), "sd": float(sd), "t": float(t), "p": p, "d_z": float(mean / sd), "lower95": lower, "n": n}


def intersection_union(pvals, alpha):
    """IUT (Berger, 1982): reject the union null only if every component test rejects at alpha."""
    return bool(np.all(np.asarray(pvals) <= alpha))


def cluster_bootstrap_ratio_ci(num, den, b=2000, conf=0.95, rng=None):
    """Percentile CI for sum(num) / sum(den), resampling whole units (e.g. Coverage = sum TP / sum GT)."""
    rng = np.random.default_rng(rng)
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = num.size
    idx = rng.integers(0, n, size=(b, n))
    boots = num[idx].sum(1) / den[idx].sum(1)
    a = (1 - conf) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def cluster_bootstrap_diff_ci(num_a, num_b, den, b=2000, conf=0.95, rng=None):
    """Paired percentile CI for Cov_A - Cov_B on the same units."""
    rng = np.random.default_rng(rng)
    num_a, num_b, den = (np.asarray(x, float) for x in (num_a, num_b, den))
    n = den.size
    idx = rng.integers(0, n, size=(b, n))
    boots = (num_a[idx].sum(1) - num_b[idx].sum(1)) / den[idx].sum(1)
    a = (1 - conf) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def icc_oneway(groups):
    """ICC(1) from one-way random-effects ANOVA; groups is a list of 1D arrays (may be unbalanced)."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k = len(groups)
    sizes = np.array([g.size for g in groups])
    n_tot = sizes.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb = ssb / (k - 1)
    msw = ssw / (n_tot - k)
    m0 = (n_tot - (sizes ** 2).sum() / n_tot) / (k - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


def kish_design_effect(m, icc):
    return 1 + (m - 1) * icc
