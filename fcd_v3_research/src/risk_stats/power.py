"""Analytic power and sample-size helpers."""
import numpy as np
from scipy import stats

from .inference import clopper_pearson_upper


def power_paired_t(delta, sd, n, alpha=0.05, margin=0.0):
    """Power of the one-sided paired t-test of H0: mean <= margin when the true mean is delta (noncentral t)."""
    df = n - 1
    nc = (delta - margin) / (sd / np.sqrt(n))
    t_crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(t_crit, df, nc))


def n_for_power_paired_t(delta, sd, power=0.8, alpha=0.05, margin=0.0, n_max=200):
    for n in range(2, n_max + 1):
        if power_paired_t(delta, sd, n, alpha, margin) >= power:
            return n
    return None


def power_binomial_exceeds(p_true, p_null, n, alpha=0.05):
    """Power of the exact one-sided binomial test of H0: p <= p_null when the truth is p_true."""
    k_crit = int(stats.binom.isf(alpha, n, p_null)) + 1
    while stats.binom.sf(k_crit - 1, n, p_null) > alpha:
        k_crit += 1
    return float(stats.binom.sf(k_crit - 1, n, p_true))


def cp_upper_table(rounds, observed_rates, conf=0.95):
    """One-sided Clopper-Pearson upper bounds of a violation rate for several Monte Carlo sizes."""
    return {b: {r: clopper_pearson_upper(int(round(r * b)), b, conf) for r in observed_rates} for b in rounds}
