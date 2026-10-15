"""Statistical toolkit for the H1-H3 paper: HB p-values, LTT selection, tests, CIs and power."""
from .hb import bernoulli_kl, k_max, min_n_certifiable, p_bentkus, p_hb, p_hoeffding, ucb_hb
from .inference import (
    clopper_pearson,
    clopper_pearson_upper,
    cluster_bootstrap_diff_ci,
    cluster_bootstrap_ratio_ci,
    holm,
    icc_oneway,
    intersection_union,
    kish_design_effect,
    paired_t_greater,
    wilcoxon_greater,
)
from .ltt import bonferroni_select, fixed_sequence, ltt_from_losses, naive_select
from .power import n_for_power_paired_t, power_binomial_exceeds, power_paired_t
