"""Hoeffding-Bentkus (HB) p-values for H0: R > alpha (Bates et al., 2021; Angelopoulos et al., 2021)."""
import math

import numpy as np
from scipy import stats


def bernoulli_kl(a, b):
    """kl(a || b) between Bernoulli(a) and Bernoulli(b), with 0*log(0) = 0."""
    t1 = 0.0 if a <= 0 else a * math.log(a / b)
    t2 = 0.0 if a >= 1 else (1 - a) * math.log((1 - a) / (1 - b))
    return t1 + t2


def p_hoeffding(r_hat, n, alpha):
    """Hoeffding KL form: exp(-n * kl(min(r_hat, alpha) || alpha))."""
    return math.exp(-n * bernoulli_kl(min(r_hat, alpha), alpha))


def p_bentkus(r_hat, n, alpha):
    """Bentkus: e * P(Bin(n, alpha) <= ceil(n * r_hat)), capped at 1."""
    # The 1e-9 guard keeps k = n * r_hat exact when r_hat was built as k / n.
    k = math.ceil(n * r_hat - 1e-9)
    return min(1.0, math.e * stats.binom.cdf(k, n, alpha))


def p_hb(r_hat, n, alpha):
    """Valid because both terms upper-bound the same lower-tail probability."""
    return min(p_hoeffding(r_hat, n, alpha), p_bentkus(r_hat, n, alpha))


def k_max(n, alpha, delta):
    """Largest integer k such that p_HB(k / n) <= delta, or None if nothing is certifiable."""
    best = None
    for k in range(int(math.ceil(alpha * n)) + 1):
        if p_hb(k / n, n, alpha) <= delta:
            best = k
        else:
            break
    return best


def min_n_certifiable(alpha, delta, n_cap=10**6):
    """Smallest n at which r_hat = 0 is certified."""
    n = 1
    while n <= n_cap:
        if p_hb(0.0, n, alpha) <= delta:
            return n
        n += 1
    return None


def ucb_hb(r_hat, n, delta, tol=1e-7):
    """inf{r : p_HB(r_hat; r) <= delta}; dual of the test (UCB <= alpha iff p(alpha) <= delta)."""
    lo, hi = r_hat, 1.0 - 1e-12
    if p_hb(r_hat, n, hi) > delta:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if p_hb(r_hat, n, mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def p_hb_vector(r_hats, n, alpha):
    return np.array([p_hb(float(r), n, alpha) for r in r_hats])
