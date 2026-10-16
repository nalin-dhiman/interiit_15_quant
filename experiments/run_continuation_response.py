"""Known-family continuation-response diagnostic; no statistical refitting.

Run from the repository root with Python3.11 and pinned NumPy:
  .venv/bin/python -m experiments.run_continuation_response --out runs/.../continuation_response
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from resetmm.continuation import (BandPolicy, ImprovedPolicy, PolicyEvaluation,
                                  initial_weights, make_synthetic_model)
from resetmm.continuation_response import (LinearResponsePolicy, PrimitiveTiltFamily,
                                           TangentPolicyEvaluation,
                                           eligibility_tilt_directions)


def write(out, name, value):
    (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_outcomes(model):
    original = model.outcomes
    counts = dict(outcome_calls=0, atom_rows=0)
    def counted(q, b, action, *, reserve=True):
        rows = original(q, b, action, reserve=reserve)
        counts["outcome_calls"] += 1
        counts["atom_rows"] += len(rows)
        return rows
    model.outcomes = counted
    return counts


def moments(model, policy):
    """Exact expected ledger quantities, separate from design objective."""
    @lru_cache(maxsize=None)
    def value(t, q, b):
        if t == model.config.horizon:
            return np.array([model.terminal(q), 0., 0., 0., 0., 0., 0.])
        result = np.zeros(7)
        for outcome in model.outcomes(q, b, policy(t, q, b)):
            immediate = np.array([outcome.ledger_pnl, outcome.risk_penalty, outcome.fees,
                                  outcome.exchange_quantity, outcome.customer_orders,
                                  outcome.customer_quantity,
                                  (outcome.next_inventory*model.config.lot)**2*model.config.dt])
            future = sum(float(p)*value(t+1, outcome.next_inventory, nb)
                         for nb, p in enumerate(model.batch_transition[b]) if p)
            result += outcome.probability*(immediate+future)
        return result
    expected = sum(float(p)*value(0, 0, b) for b, p in enumerate(initial_weights(model, None)))
    names = ("raw_pnl", "risk_penalty", "fees", "exchange_quantity", "customer_orders",
             "customer_quantity", "inventory_sq_time")
    result = dict(zip(names, map(float, expected)))
    result["objective"] = result["raw_pnl"]-result["risk_penalty"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-results", default="data/finite-baselines.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"results.json").exists():
        raise FileExistsError("Use a new output directory; diagnostic results are not overwritten")
    baseline_path = Path(args.baseline_results)
    rows = json.loads(baseline_path.read_text())
    frozen = {row["model"]: row["methods"]["B3"]["parameters"] for row in rows}
    protocol = dict(stage="Fixed-support known-family response diagnostic",
                    protocol_utc=datetime.now(timezone.utc).isoformat(),
                    regimes=["balanced", "persistent", "informed", "expensive"],
                    horizon=6, limit_lots=3, risk_weight=0., fee_derivative=.02,
                    theta=[-.15, -.05, 0., .05, .15, .3], empirical_trust_radius=.15,
                    fd_magnitudes=[.1, .03, .01, .003, .001],
                    state_grid="t=0..5,q=-3..3,b=0..1 (84 states), every feasible R2 candidate",
                    tilt="p_theta(atom|batch) proportional p0*exp(theta*(mean_limit-.1)/.08)",
                    baseline_results=str(baseline_path), baseline_results_sha256=digest(baseline_path),
                    frozen_B3={k: v for k, v in frozen.items() if k.endswith("risk0")},
                    selection_methods=["frozen_R2_at0", "linear_R3", "known_theta_direct_Q"],
                    outside_region="linear_R3 returns conventional B3",
                    information="same exact known primitive family, no samples or fitted coefficients",
                    budget="same complete state/candidate grid for all selectors inside region; actual atom rows and elapsed CPU recorded separately",
                    validation_budget="direct Q error checks and fixed-policy FD are separate diagnostics, excluded from selector timing",
                    inference="exact synthetic expectations only; no CIs, no official score, no held-out estimation claim",
                    no_new_tuning=True, final_test_seeds_opened=False, deployment_enabled=False)
    # Freeze the protocol on disk before evaluating this family's outcomes.
    write(out, "config.json", protocol)
    protocol_hash = digest(out/"config.json")
    write(out, "environment.json", dict(python=sys.version, numpy=np.__version__,
          os=platform.platform(), executable=sys.executable, utc=datetime.now(timezone.utc).isoformat()))
    sources = [Path("resetmm/continuation.py"), Path("resetmm/continuation_response.py"),
               Path("tests/test_continuation_response.py"), Path(__file__), baseline_path]
    write(out, "inputs.json", {str(p): digest(p) for p in sources})
    results, derivatives, budgets = [], [], []
    started = time.perf_counter()
    for regime in protocol["regimes"]:
        base = make_synthetic_model(horizon=6, regime=regime, risk_weight=0)
        band = BandPolicy(**frozen[regime+"_risk0"])
        family = PrimitiveTiltFamily(base, eligibility_tilt_directions(base), .02)
        fixed = band.bind(base)
        tangent = TangentPolicyEvaluation(family, fixed)
        v0, dv = tangent.initial_value()
        action = ImprovedPolicy(base, band).candidates(0, 3, 0)[-1]
        # Pick explicit full-reset action, independently of candidate ordering.
        from resetmm.continuation import Action
        action = Action(0, band.quotes_at(base, 0, 0))
        aq, adq = tangent.action_value(0, 3, 0, action)
        incomplete = TangentPolicyEvaluation(family, fixed, include_payoff_derivative=False)
        _, adq_no_payoff = incomplete.action_value(0, 3, 0, action)
        for h in protocol["fd_magnitudes"]:
            before = time.perf_counter()
            plus, minus = family.model(h), family.model(-h)
            ep, em = PolicyEvaluation(plus, fixed), PolicyEvaluation(minus, fixed)
            fd = (ep.initial_value()-em.initial_value())/(2*h)
            fdq = (ep.action_value(0, 3, 0, action)-em.action_value(0, 3, 0, action))/(2*h)
            derivatives.append(dict(regime=regime, magnitude=h, tangent_initial=dv,
                                    fd_initial=fd, initial_error=abs(fd-dv),
                                    tangent_action=adq, fd_action=fdq, action_error=abs(fdq-adq),
                                    omitted_payoff_action_derivative=adq_no_payoff,
                                    omitted_payoff_error=abs(adq_no_payoff-adq),
                                    exact_family_model_constructions=2,
                                    elapsed_seconds=time.perf_counter()-before))
        grid = [(t, q, b) for t in range(6) for q in range(-3, 4) for b in range(2)]
        for theta in protocol["theta"]:
            actual = family.model(theta)
            # Separate cold selector instances; information and candidates match.
            rb = make_synthetic_model(horizon=6, regime=regime, risk_weight=0)
            rc = make_synthetic_model(horizon=6, regime=regime, risk_weight=0)
            response_family = PrimitiveTiltFamily(rc, eligibility_tilt_directions(rc), .02)
            direct_model = family.model(theta)
            counters = [count_outcomes(rb), count_outcomes(rc), count_outcomes(direct_model)]
            policies = [ImprovedPolicy(rb, band),
                        LinearResponsePolicy(response_family, band, theta, trust_radius=.15),
                        ImprovedPolicy(direct_model, band)]
            selected = []
            for method, policy, counter in zip(protocol["selection_methods"], policies, counters):
                before = time.perf_counter()
                actions = tuple(policy(*state) for state in grid)
                elapsed = time.perf_counter()-before
                selected.append(actions)
                candidate_count = (policy.candidate_evaluations if method == "linear_R3"
                                   else sum(len(policy.candidates(*state)) for state in grid))
                budgets.append(dict(regime=regime, theta=theta, method=method,
                                    state_grid_evaluations=len(grid), candidate_score_evaluations=candidate_count,
                                    **counter.copy(), selector_grid_seconds=elapsed,
                                    linear_derivative_atom_arithmetic=(policy.evaluation.atom_evaluations
                                         if method == "linear_R3" else 0),
                                    fallback=(method == "linear_R3" and not policy.supported)))
            metrics = {name: moments(actual, policy)
                       for name, policy in zip(protocol["selection_methods"], policies)}
            metrics["B3"] = moments(actual, band.bind(actual))
            before = time.perf_counter()
            max_error = 0.
            candidate_checks = 0
            for state in grid:
                for candidate in policies[2].candidates(*state):
                    q, dq = policies[1].evaluation.action_value(*state, candidate)
                    exact = policies[2].evaluation.action_value(*state, candidate)
                    max_error = max(max_error, abs(q+theta*dq-exact))
                    candidate_checks += 1
            results.append(dict(regime=regime, theta=theta, inside_empirical_radius=abs(theta)<=.15,
                                metrics=metrics,
                                R3_direct_action_agreement=sum(a==b for a, b in zip(selected[1], selected[2])),
                                R3_frozen_R2_action_changes=sum(a!=b for a, b in zip(selected[1], selected[0])),
                                state_grid_count=len(grid), max_linear_Q_error=max_error,
                                separate_Q_validation_candidate_evaluations=candidate_checks,
                                separate_Q_validation_seconds=time.perf_counter()-before))
    write(out, "results.json", results)
    write(out, "derivatives.json", derivatives)
    write(out, "budgets.json", budgets)
    elapsed = time.perf_counter()-started
    within = [row for row in results if row["inside_empirical_radius"]]
    summary = dict(protocol_sha256=protocol_hash, elapsed_seconds=elapsed,
                   finite_comparisons=len(results), fd_comparisons=len(derivatives),
                   max_fd_initial_error=max(row["initial_error"] for row in derivatives),
                   max_fd_action_error=max(row["action_error"] for row in derivatives),
                   max_inside_linear_Q_error=max(row["max_linear_Q_error"] for row in within),
                   min_inside_action_agreement=min(row["R3_direct_action_agreement"] for row in within),
                   inside_R3_R2_objective_differences=[dict(regime=row["regime"], theta=row["theta"],
                       difference=row["metrics"]["linear_R3"]["objective"]-row["metrics"]["frozen_R2_at0"]["objective"])
                       for row in within],
                   official_or_statistical_superiority_established=False)
    write(out, "summary.json", summary)
    with (out/"commands.log").open("a") as stream:
        stream.write(json.dumps(dict(command=[sys.executable, "-m", "experiments.run_continuation_response"]+sys.argv[1:],
                    utc_finished=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed, exit_code=0))+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
