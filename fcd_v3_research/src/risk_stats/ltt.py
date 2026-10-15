"""Threshold selection on a pre-registered 1D path (Learn-then-Test)."""
import numpy as np

from .hb import p_hb


def fixed_sequence(r_hats, n, alpha, delta, pfun=p_hb):
    """Walk the path in order, stop at the first non-rejection.

    Returns the index of the last rejected point, or None (empty set, accept nothing).
    """
    chosen = None
    for j, r in enumerate(r_hats):
        if pfun(float(r), n, alpha) <= delta:
            chosen = j
        else:
            break
    return chosen


def bonferroni_select(r_hats, n, alpha, delta, pfun=p_hb):
    """Most permissive point among those rejected at delta / K (valid for any order)."""
    k = len(r_hats)
    rejected = [j for j, r in enumerate(r_hats) if pfun(float(r), n, alpha) <= delta / k]
    return max(rejected) if rejected else None


def naive_select(r_hats, alpha):
    """Plug-in rule without a guarantee: last point with empirical risk <= alpha."""
    ok = [j for j, r in enumerate(r_hats) if r <= alpha]
    return max(ok) if ok else None


def ltt_from_losses(z, alpha, delta, rule="fixed_sequence"):
    """z: array (N units, K path points) of per-unit risks Z_i(lambda_k) in [0, 1]."""
    z = np.asarray(z, dtype=float)
    n = z.shape[0]
    r_hats = z.mean(axis=0)
    if rule == "fixed_sequence":
        return fixed_sequence(r_hats, n, alpha, delta)
    if rule == "bonferroni":
        return bonferroni_select(r_hats, n, alpha, delta)
    if rule == "naive":
        return naive_select(r_hats, alpha)
    raise ValueError(rule)
