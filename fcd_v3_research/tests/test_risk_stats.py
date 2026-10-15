"""Tests for the statistical toolkit.

Run from the repo root:  python -m unittest discover -s fcd_v3_research/tests -v

Three kinds of checks:
  * regression against the tables already published in the H1 document (5.3, 5.4, 5.5),
  * exact small examples worked by hand in docs/paper/02_MATH_EXPLAINED.md,
  * Monte Carlo checks of the properties the paper relies on (type I error <= delta, CI coverage).
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from risk_stats import (  # noqa: E402
    bernoulli_kl,
    clopper_pearson,
    cluster_bootstrap_ratio_ci,
    fixed_sequence,
    holm,
    icc_oneway,
    intersection_union,
    k_max,
    kish_design_effect,
    ltt_from_losses,
    min_n_certifiable,
    n_for_power_paired_t,
    p_bentkus,
    p_hb,
    p_hoeffding,
    paired_t_greater,
    power_paired_t,
    ucb_hb,
    wilcoxon_greater,
)


class TestHB(unittest.TestCase):
    def test_kl_basic(self):
        self.assertAlmostEqual(bernoulli_kl(0.3, 0.3), 0.0)
        self.assertAlmostEqual(bernoulli_kl(0.0, 0.02), -math.log(0.98))
        self.assertGreater(bernoulli_kl(0.01, 0.02), 0)

    def test_table_5_3_values(self):
        n, a = 1500, 0.02
        self.assertAlmostEqual(p_hoeffding(18 / n, n, a), 0.0576, places=4)
        self.assertAlmostEqual(p_bentkus(18 / n, n, a), 0.0333, places=4)
        self.assertAlmostEqual(p_hb(19 / n, n, a), 0.0568, places=4)

    def test_table_5_4_kmax_alpha_002(self):
        expected = {400: 2, 800: 7, 1500: 18, 4000: 61, 80000: 1517}
        for n, k in expected.items():
            self.assertEqual(k_max(n, 0.02, 0.05), k, msg=f"N={n}")
        self.assertIsNone(k_max(100, 0.02, 0.05))
        self.assertEqual(k_max(150, 0.02, 0.05), 0)

    def test_table_5_5_kmax_alpha_010(self):
        expected = {100: 3, 200: 11, 400: 27, 800: 62, 2000: 171, 4000: 360}
        for n, k in expected.items():
            self.assertEqual(k_max(n, 0.10, 0.05), k, msg=f"N={n}")

    def test_minimum_n(self):
        self.assertEqual(min_n_certifiable(0.02, 0.05), 149)
        self.assertEqual(min_n_certifiable(0.10, 0.05), 29)

    def test_ucb_duality(self):
        n, a, d = 1500, 0.02, 0.05
        for k in (10, 18, 19, 25):
            r = k / n
            self.assertEqual(ucb_hb(r, n, d) <= a, p_hb(r, n, a) <= d, msg=f"k={k}")

    def test_type_one_error_binary_worst_case(self):
        """Under H0 (true risk slightly above alpha) the HB test rejects with probability <= delta."""
        rng = np.random.default_rng(1)
        n, a, d, reps = 400, 0.02, 0.05, 20000
        k = rng.binomial(n, a * 1.001, size=reps)
        kmax = k_max(n, a, d)
        rate = np.mean(k <= kmax)
        # 3 Monte Carlo standard errors of slack
        self.assertLessEqual(rate, d + 3 * math.sqrt(d * (1 - d) / reps))


class TestLTT(unittest.TestCase):
    def test_toy_path_from_doc_b44(self):
        n = 1500
        r_hats = np.array([3, 9, 14, 18, 21, 27]) / n
        self.assertEqual(fixed_sequence(r_hats, n, 0.02, 0.05), 3)  # 0-based index of step 4

    def test_empty_set(self):
        self.assertIsNone(fixed_sequence([0.05, 0.01], 1500, 0.02, 0.05))

    def test_fwer_with_unsafe_points_on_path(self):
        """End-to-end: path with unsafe points; P(selected point unsafe) <= delta."""
        rng = np.random.default_rng(2)
        true_risk = np.array([0.004, 0.008, 0.012, 0.016, 0.019, 0.021, 0.025, 0.03])
        n, a, d, reps = 1500, 0.02, 0.05, 4000
        bad = 0
        for _ in range(reps):
            z = rng.random((n, 1)) < true_risk  # binary worst case, nested by construction
            j = ltt_from_losses(z.astype(float), a, d)
            if j is not None and true_risk[j] > a:
                bad += 1
        self.assertLessEqual(bad / reps, d + 3 * math.sqrt(d * (1 - d) / reps))

    def test_naive_rule_violates(self):
        """Negative control: the plug-in rule picks unsafe thresholds far more often than delta."""
        rng = np.random.default_rng(3)
        true_risk = np.array([0.01, 0.015, 0.021, 0.023, 0.025])
        n, reps, bad = 1500, 3000, 0
        for _ in range(reps):
            z = (rng.random((n, 1)) < true_risk).astype(float)
            j = ltt_from_losses(z, 0.02, 0.05, rule="naive")
            bad += j is not None and true_risk[j] > 0.02
        self.assertGreater(bad / reps, 0.2)


class TestInference(unittest.TestCase):
    def test_wilcoxon_hand_example(self):
        res = wilcoxon_greater([5, 1, 3, -2, 4, 6])
        self.assertEqual(res["w_plus"], 19)
        self.assertAlmostEqual(res["p"], 3 / 64, places=10)
        self.assertAlmostEqual(res["r_rank_biserial"], (19 - 2) / 21)

    def test_holm_known(self):
        rej, adj = holm([0.001, 0.02, 0.004, 0.3], 0.01)
        np.testing.assert_allclose(adj, [0.004, 0.04, 0.012, 0.3])
        self.assertEqual(rej.tolist(), [True, False, False, False])

    def test_holm_dominates_bonferroni(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            p = rng.random(4) * 0.02
            self.assertTrue(np.all(holm(p, 0.01)[0] >= (p <= 0.01 / 4)))

    def test_clopper_pearson_coverage(self):
        rng = np.random.default_rng(5)
        n, p, reps = 200, 0.05, 3000
        ks = rng.binomial(n, p, reps)
        cover = np.mean([lo <= p <= hi for lo, hi in (clopper_pearson(k, n) for k in ks)])
        self.assertGreaterEqual(cover, 0.95 - 0.01)

    def test_paired_t_and_power(self):
        res = paired_t_greater([1.0, 2.0, 3.0])
        self.assertAlmostEqual(res["t"], 2.0 / (1.0 / math.sqrt(3)))
        self.assertAlmostEqual(power_paired_t(0.0, 1.0, 10), 0.05, places=6)
        self.assertEqual(n_for_power_paired_t(1.0, 1.0, power=0.8), 8)

    def test_iut(self):
        self.assertTrue(intersection_union([0.01, 0.04, 0.03], 0.05))
        self.assertFalse(intersection_union([0.01, 0.06, 0.03], 0.05))

    def test_icc_recovers_truth(self):
        rng = np.random.default_rng(6)
        rho, m, k = 0.3, 20, 400
        b = rng.normal(0, math.sqrt(rho), k)
        groups = [bi + rng.normal(0, math.sqrt(1 - rho), m) for bi in b]
        self.assertAlmostEqual(icc_oneway(groups), rho, delta=0.05)
        self.assertAlmostEqual(kish_design_effect(200, 0.6), 120.4)

    def test_cluster_bootstrap_ratio_coverage(self):
        rng = np.random.default_rng(7)
        hits, reps = 0, 300
        for _ in range(reps):
            gt = 1 + rng.poisson(15, 300)
            p_unit = np.clip(rng.normal(0.6, 0.15, 300), 0, 1)
            tp = rng.binomial(gt, p_unit)
            lo, hi = cluster_bootstrap_ratio_ci(tp, gt, b=500, rng=rng)
            # population coverage = E[gt * p] / E[gt] = 0.6 because gt and p are independent
            hits += lo <= 0.6 <= hi
        self.assertGreaterEqual(hits / reps, 0.90)


if __name__ == "__main__":
    unittest.main()
