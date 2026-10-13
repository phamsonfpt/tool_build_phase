"""Self-checks for the H1 real-data pipeline: python test_h1core.py"""
import numpy as np

from build_features import greedy_tp, iou_mat
from h1common import fixed_sequence, k_max, p_hb, vr_ci


def greedy_ref(iou, accept, theta):
    used, tp = set(), []
    for i in range(iou.shape[0]):
        if not accept[i]:
            tp.append(False); continue
        cand = [(iou[i, j], -j) for j in range(iou.shape[1]) if j not in used and iou[i, j] >= theta]
        if cand:
            used.add(-max(cand)[1]); tp.append(True)
        else:
            tp.append(False)
    return np.array(tp)


def rand_boxes(rng, n):
    xy = rng.uniform(0, 100, (n, 2)); wh = rng.uniform(5, 40, (n, 2))
    return np.hstack([xy, xy + wh]).astype(np.float32)


def test_greedy_and_prefix_property():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p, g = rand_boxes(rng, rng.integers(0, 12)), rand_boxes(rng, rng.integers(0, 8))
        sub = g[: rng.integers(0, len(g) + 1)]
        p = np.vstack([p, sub + rng.normal(0, 2, sub.shape).astype(np.float32)])   # near-duplicates of GT
        iou = iou_mat(p, g)
        acc = rng.random(len(p)) < 0.8
        for th in (0.5, 0.75):
            full = greedy_tp(iou, acc, th)
            assert (full == greedy_ref(iou, acc, th)).all()
            for cut in range(len(p) + 1):          # prefix (higher t_cls) gives the same flags
                pre = greedy_tp(iou[:cut], acc[:cut], th)
                assert (pre == full[:cut]).all()


def test_lemma_A4_monotone():
    rng = np.random.default_rng(1)
    for _ in range(500):
        p, g = rand_boxes(rng, rng.integers(1, 12)), rand_boxes(rng, rng.integers(1, 8))
        p = np.vstack([p, g + rng.normal(0, 1.5, g.shape).astype(np.float32)])
        acc = np.ones(len(p), bool)
        keep = rng.random(len(g)) < 0.6                 # S' subset of S
        fp_full = (~greedy_tp(iou_mat(p, g), acc, 0.5)).sum()
        fp_sub = (~greedy_tp(iou_mat(p, g[keep]), acc, 0.5)).sum()
        assert fp_sub >= fp_full


def test_hb_tables_and_validity():
    assert [k_max(n, 0.02) for n in (148, 149, 400, 800, 1500, 4000, 80000)] == [None, 0, 2, 7, 18, 61, 1517]
    assert [k_max(n, 0.10) for n in (28, 29, 100, 200, 400, 800)] == [None, 0, 3, 11, 27, 62]
    rng = np.random.default_rng(2)                      # worst case: Bernoulli(alpha * 1.001)
    rej = np.mean(p_hb(rng.binomial(149, 0.02002, 40000) / 149, 149, 0.02) <= 0.05)
    assert rej <= 0.055


def test_fixed_sequence_and_vr_ci():
    idx, stop = fixed_sequence(np.array([0.0, 0.005, 0.02, 0.0]), 1500, 0.02)
    assert (idx, stop) == (1, 2)                        # stops at first failure, never resumes
    lo, hi = vr_ci(np.zeros((334, 3)))
    assert lo == 0.0 and 0.010 < hi < 0.012             # no degenerate [0, 0] interval


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
