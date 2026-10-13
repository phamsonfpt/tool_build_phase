"""Aggregate per-image grid counts into statistical units (rides) and build fixed 1-D paths."""
import json
import os

import numpy as np

from h1common import CACHE, TAU_CLS, TAU_LOC


def load_counts(which, anchor):
    z = np.load(os.path.join(CACHE, f"counts_{which}_{anchor}.npz"))
    d = {k: z[k] for k in z.files}
    n_img = len(d["names"])
    for k in ("N", "TP", "VH"):          # flatten grid: (img, theta, G) with G = loc x fcd x cls
        d[k] = d[k].reshape(n_img, d[k].shape[1], -1).astype(np.int32)
    d["M"] = d["M"].astype(np.int32)
    return d


def grid_params(tau_fcd):
    """Return list of (t_loc, t_fcd, t_cls) in flattened-G order."""
    return [(float(l), float(f), float(c)) for l in TAU_LOC for f in tau_fcd for c in TAU_CLS]


def ride_index(which, names):
    units = json.load(open(os.path.join(CACHE, "p1_units.json")))[which]
    r_of = {n: r for r, ns in units.items() for n in ns}
    rides = sorted(units)
    pos = {r: i for i, r in enumerate(rides)}
    return np.array([pos[r_of[n]] for n in names]), rides


def aggregate(counts, groups, n_groups, theta_i):
    """Sum image counts into groups. Returns N, TP, VH (groups x G) and M (groups)."""
    out = {}
    for k in ("N", "TP", "VH"):
        a = counts[k][:, theta_i, :]
        s = np.zeros((n_groups, a.shape[1]), np.int64)
        np.add.at(s, groups, a)
        out[k] = s
    out["M"] = np.bincount(groups, weights=counts["M"], minlength=n_groups).astype(np.int64)
    return out


def fdp(N, TP):
    return (N - TP) / np.maximum(N, 1)


def build_path(fdp_meta, cov_meta, alphas):
    """Pareto frontier tracing (Section 6.2): tau^(k) = argmax Cov s.t. FDP <= alpha_k; ties -> smaller FDP,
    then smaller grid index. Infeasible steps and consecutive duplicates are dropped."""
    path = []
    for a in alphas:
        feas = np.nonzero(fdp_meta <= a + 1e-12)[0]
        if len(feas) == 0:
            continue
        best = feas[np.lexsort((feas, fdp_meta[feas], -cov_meta[feas]))[0]]
        if path and path[-1][1] == best:
            continue
        path.append((float(a), int(best)))
    return path


def point_at(path, a):
    """Path point whose grid risk level is closest to a (ties -> lower level)."""
    lv = np.array([p[0] for p in path])
    return int(np.argmin(np.abs(lv - a) + 1e-9 * lv))
