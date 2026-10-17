"""Path-level inference for one prospectively fixed synthetic policy contrast.

Bootstrap coverage and normal planning power are approximate. Matching an
admissible inventory envelope does not match realised variance or downside.
"""
from __future__ import annotations

import math
import numpy as np


def plan_sample(groups, materiality=1., planning_alternative=2., floor=1024, cap=4096):
    if planning_alternative <= materiality or not 2 <= floor <= cap:
        raise ValueError("invalid planning effect or bounds")
    arrays = [np.asarray(group, dtype=float) for group in groups]
    if not arrays or any(len(g) < 2 or not np.isfinite(g).all() for g in arrays):
        raise ValueError("finite independent pilot paths required in each stratum")
    variance = float(np.mean([np.var(g, ddof=1) for g in arrays]))
    gap = planning_alternative - materiality
    required = math.ceil((1.960 + .842) ** 2 * variance / gap ** 2)
    strata = len(arrays)
    if cap // strata * strata < floor:
        raise ValueError("cap cannot fit the minimum equal-stratum allocation")
    n = strata * math.ceil(max(floor, required) / strata)
    n = min(strata * (cap // strata), n)
    return dict(within_stratum_effective_variance=variance, required_independent_paths=required,
                final_paths=n, paths_per_scenario=n // strata,
                precision_cap_binds=required > n, materiality=materiality,
                planning_alternative=planning_alternative,
                planning_assumption="approximate 80% normal power at stated alternative; pilot variance may not transfer")


def path_interval(groups, seed, materiality=1., replicates=5000):
    arrays = [np.asarray(group, dtype=float) for group in groups]
    if not arrays or any(len(g) < 2 or not np.isfinite(g).all() for g in arrays):
        raise ValueError("finite independent paths required in every stratum")
    rng = np.random.default_rng(seed)
    bootstrap = sum(g[rng.integers(0, len(g), size=(replicates, len(g)))].mean(axis=1)
                    / len(arrays) for g in arrays)
    lower, upper = map(float, np.quantile(bootstrap, [.025, .975]))
    return dict(paired_mean_difference=float(np.mean([g.mean() for g in arrays])),
                stratified_path_bootstrap_95=[lower, upper],
                independent_paths=sum(map(len, arrays)), materiality=materiality,
                material_benefit_supported=lower > materiality,
                material_benefit_excluded=upper < materiality,
                mean_loss_supported=upper < 0.,
                practical_equivalence_supported=lower > -materiality and upper < materiality,
                assumptions="independent generated paths; fixed equal scenario weights; approximate bootstrap coverage")
