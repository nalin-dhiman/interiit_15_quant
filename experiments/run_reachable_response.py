"""Predeclared exact fixed-support response diagnostic with reachable sizes.

This is synthetic known-model research. It does not change or calibrate the
sealed SDK dealer, open an official holdout, or promote response to deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from resetmm.continuation import (BandPolicy, Batch, FiniteRFQModel,
    ImprovedPolicy, ModelConfig, PrimitiveAtom, RFQ, initial_weights)
from resetmm.continuation_response import (LinearResponsePolicy,
    PrimitiveTiltFamily, TangentPolicyEvaluation, eligibility_tilt_directions)
from experiments.run_continuation_response import count_outcomes, moments

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "data/finite-baselines.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def sources():
    names = ["experiments/run_reachable_response.py",
             "experiments/run_continuation_response.py",
             "resetmm/continuation.py", "resetmm/continuation_response.py",
             "tests/test_reachable_response.py", "docs/model.md"]
    return complete_source_hashes({name: digest(ROOT / name) for name in names})


def declare(out):
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads(PRIOR.read_text())
    bands = {row["model"].removesuffix("_risk0"): row["methods"]["B3"]["parameters"]
             for row in previous if row["model"] in
             ("balanced_risk0", "persistent_risk0", "informed_risk0")}
    protocol = dict(stage="Reachable known-family response diagnostic",
        declared_utc=datetime.now(timezone.utc).isoformat(),
        model_config=asdict(ModelConfig(horizon=8, limit_lots=5, risk_weight=0.)),
        regimes=["balanced", "persistent", "informed"],
        rfqs=[[["BUY", 1]], [["SELL", 1]],
              [["BUY", 2], ["BUY", 1]], [["SELL", 2], ["SELL", 1]],
              [["BUY", 1], ["SELL", 2]]],
        atom_classes=[[.35, .18, .10], [.4, .10, .06], [.25, .04, .03]],
        tie_uniforms=[.1, .9], uninformed_price_changes=[-.02, .02],
        informed_price_change="sign(sum(customer_direction*size)) * (.03 if competitor>=.06 else -.02)",
        batch_transition="balanced/informed: uniform; persistent: .7*identity+.3*uniform",
        initial_inventory=0, initial_batch="uniform over five public batch types",
        theta=[-.15, -.05, 0., .05, .15, .3], empirical_trust_radius=.15,
        fee_derivative=.02,
        primitive_tilt="p_theta proportional p0*exp(theta*(mean_limit-.1)/.08)",
        fixed_support="RFQ sizes, hidden payloads, ties, tick, targets and batch transition fixed; all atoms positive",
        frozen_B3=bands, prior_baseline_file=str(PRIOR.relative_to(ROOT)),
        prior_baseline_sha256=digest(PRIOR),
        baseline_scope="previously selected conventional bands reused unchanged; not tuned for this new family",
        methods=["B3", "frozen_R2_at0", "linear_R3", "known_theta_direct_Q", "candidate_oracle"],
        oracle="optimal finite DP on exactly ImprovedPolicy.candidates; no extra joint quote grid",
        selection_grid="t=0..7,q=-5..5,b=0..4, including unreachable diagnostic states",
        primary_diagnostic="exact q0=0 values plus forward occupancy of action-change states",
        derivative_fd_magnitudes=[.03, .01, .003],
        information="same exact known primitive family and perturbation; no estimator/no samples",
        compute="cold whole-grid selectors and actual outcome/atom calls recorded; no equal-CPU or speed advantage claim",
        uncertainty="exact expectations to floating-point tolerance, no statistical CIs",
        no_post_result_search=True, no_seed_selection=True,
        official_holdout_opened=False, deployment_enabled=False,
        source_hashes=sources())
    write(out / "config.json", protocol)
    write(out / "environment.json", dict(python=sys.version, numpy=np.__version__,
        executable=sys.executable, platform=platform.platform()))
    return protocol


def build_model(protocol, regime):
    if regime not in protocol["regimes"]:
        raise ValueError("Regime outside declared family")
    config = dict(protocol["model_config"])
    config["quote_edges"] = tuple(config["quote_edges"])
    batches = []
    for rfq_rows in protocol["rfqs"]:
        rfqs = tuple(RFQ(side, int(size)) for side, size in rfq_rows)
        direction = int(math.copysign(1, sum(r.direction * r.quantity_lots for r in rfqs)))
        atoms = []
        for probability, limit, competitor in protocol["atom_classes"]:
            for tie in protocol["tie_uniforms"]:
                shocks = ([direction*(.03 if competitor >= .06 else -.02)]
                          if regime == "informed" else protocol["uninformed_price_changes"])
                for shock in shocks:
                    atoms.append(PrimitiveAtom(probability / (len(protocol["tie_uniforms"])*len(shocks)),
                        (limit,)*len(rfqs), ((competitor,),)*len(rfqs),
                        (tie,)*len(rfqs), shock))
        batches.append(Batch(rfqs, tuple(atoms)))
    n = len(batches)
    transition = np.full((n, n), 1/n)
    if regime == "persistent":
        transition = .7*np.eye(n)+.3*transition
    return FiniteRFQModel(ModelConfig(**config), tuple(batches), transition)


class CandidateOracle:
    """Optimal finite recursion on the common fixed-quote-map candidate set."""
    def __init__(self, model, band):
        self.model = model
        self.candidate_policy = ImprovedPolicy(model, band)
        self.selected = {}
        self.candidate_evaluations = 0
        self.value = lru_cache(maxsize=None)(self._value)

    def candidates(self, t, q, b):
        return self.candidate_policy.candidates(t, q, b)

    def _value(self, t, q, b):
        if t == self.model.config.horizon:
            return self.model.terminal(q)
        best, best_value = None, -math.inf
        for action in self.candidates(t, q, b):
            self.candidate_evaluations += 1
            value = sum(o.probability*(o.objective+sum(float(p)*self.value(t+1, o.next_inventory, nb)
                for nb, p in enumerate(self.model.batch_transition[b]) if p))
                for o in self.model.outcomes(q, b, action))
            if value > best_value+1e-12:
                best, best_value = action, value
        if best is None:
            raise AssertionError("No feasible incumbent")
        self.selected[t, q, b] = best
        return best_value

    def __call__(self, t, q, b):
        self.value(t, q, b)
        return self.selected[t, q, b]


def occupancy(model, policy):
    """Independent forward mass propagation, including terminal states."""
    levels = [{(0, 0, b): float(p) for b, p in enumerate(initial_weights(model, None)) if p}]
    for t in range(model.config.horizon):
        following = {}
        for (_, q, b), mass in levels[-1].items():
            for outcome in model.outcomes(q, b, policy(t, q, b)):
                for nb, probability in enumerate(model.batch_transition[b]):
                    if probability and outcome.probability:
                        key = (t+1, outcome.next_inventory, nb)
                        following[key] = following.get(key, 0.)+mass*outcome.probability*float(probability)
        if abs(sum(following.values())-1.) > 1e-10:
            raise AssertionError("Forward occupancy loses probability mass")
        levels.append(following)
    return levels


def occupancy_moments(model, policy, levels):
    result = np.zeros(7)
    for level in levels[:-1]:
        for (t, q, b), mass in level.items():
            for o in model.outcomes(q, b, policy(t, q, b)):
                result += mass*o.probability*np.array([o.ledger_pnl, o.risk_penalty, o.fees,
                    o.exchange_quantity, o.customer_orders, o.customer_quantity,
                    (o.next_inventory*model.config.lot)**2*model.config.dt])
    result[0] += sum(mass*model.terminal(q) for (_, q, _), mass in levels[-1].items())
    return result


def first_switch_probability(model, first, second):
    survivors = {(0, 0, b): float(p) for b, p in enumerate(initial_weights(model, None)) if p}
    for t in range(model.config.horizon):
        following = {}
        for (_, q, b), mass in survivors.items():
            if first(t, q, b) != second(t, q, b):
                continue
            for outcome in model.outcomes(q, b, first(t, q, b)):
                for nb, probability in enumerate(model.batch_transition[b]):
                    if probability and outcome.probability:
                        key = (t+1, outcome.next_inventory, nb)
                        following[key] = following.get(key, 0.)+mass*outcome.probability*float(probability)
        survivors = following
    return max(0., min(1., 1.-sum(survivors.values())))


def compare_actions(model, first, second, first_levels, second_levels, grid):
    changed = [state for state in grid if first(*state) != second(*state)]
    changed_set = set(changed)
    weighted = []
    for label, levels in (("first", first_levels), ("second", second_levels)):
        rows = [(state, mass) for level in levels[:-1] for state, mass in level.items()
                if state in changed_set and mass > 0]
        weighted.append(dict(policy_occupancy=label, reachable_changed_states=len(rows),
            expected_changed_decisions=sum(mass for _, mass in rows),
            state_masses=[dict(t=s[0], q=s[1], b=s[2], probability=mass) for s, mass in rows]))
    return dict(whole_grid_changed_states=len(changed),
        whole_grid_changes=[dict(t=s[0], q=s[1], b=s[2],
            first=asdict(first(*s)), second=asdict(second(*s))) for s in changed],
        occupancy=weighted,
        probability_of_at_least_one_switch_from_q0=first_switch_probability(model, first, second))


def execute(out):
    protocol = json.loads((out / "config.json").read_text())
    if protocol["source_hashes"] != sources() or digest(PRIOR) != protocol["prior_baseline_sha256"]:
        raise RuntimeError("Declared source/prior baseline changed before execution")
    if (out / "results.json").exists():
        raise FileExistsError("Never overwrite an executed diagnostic")
    started = time.perf_counter()
    results, budgets, derivatives = [], [], []
    c = protocol["model_config"]
    grid = [(t, q, b) for t in range(c["horizon"])
            for q in range(-c["limit_lots"], c["limit_lots"]+1) for b in range(len(protocol["rfqs"]))]
    for regime in protocol["regimes"]:
        base = build_model(protocol, regime)
        band = BandPolicy(**protocol["frozen_B3"][regime])
        family = PrimitiveTiltFamily(base, eligibility_tilt_directions(base), protocol["fee_derivative"])
        baseline = band.bind(base)
        tangent = TangentPolicyEvaluation(family, baseline)
        v0, derivative = tangent.initial_value()
        from resetmm.continuation import PolicyEvaluation
        for h in protocol["derivative_fd_magnitudes"]:
            plus, minus = family.model(h), family.model(-h)
            fd = (PolicyEvaluation(plus, baseline).initial_value()-
                  PolicyEvaluation(minus, baseline).initial_value())/(2*h)
            derivatives.append(dict(regime=regime, magnitude=h, initial_value=v0,
                tangent=derivative, finite_difference=fd, absolute_error=abs(fd-derivative)))
        for theta in protocol["theta"]:
            actual = family.model(theta)
            frozen_model = build_model(protocol, regime)
            response_model = build_model(protocol, regime)
            response_family = PrimitiveTiltFamily(response_model,
                eligibility_tilt_directions(response_model), protocol["fee_derivative"])
            direct_model, oracle_model = family.model(theta), family.model(theta)
            models = [actual, frozen_model, response_model, direct_model, oracle_model]
            counters = [count_outcomes(m) for m in models]
            policies = [band.bind(actual), ImprovedPolicy(frozen_model, band),
                LinearResponsePolicy(response_family, band, theta,
                    trust_radius=protocol["empirical_trust_radius"]),
                ImprovedPolicy(direct_model, band), CandidateOracle(oracle_model, band)]
            for name, policy, counter in zip(protocol["methods"], policies, counters):
                before = time.perf_counter()
                for state in grid:
                    policy(*state)
                elapsed = time.perf_counter()-before
                candidate_count = (0 if name == "B3" else policy.candidate_evaluations
                    if name in ("linear_R3", "candidate_oracle") else
                    sum(len(policy.candidates(*state)) for state in grid))
                budgets.append(dict(regime=regime, theta=theta, method=name,
                    whole_grid_states=len(grid), candidate_score_evaluations=candidate_count,
                    selector_seconds=elapsed, **counter.copy(),
                    linear_derivative_atom_arithmetic=(policy.evaluation.atom_evaluations
                        if name == "linear_R3" else 0)))
            levels = [occupancy(actual, policy) for policy in policies]
            metrics = {name: moments(actual, policy) for name, policy in zip(protocol["methods"], policies)}
            forward_error = 0.
            for name, policy, level in zip(protocol["methods"], policies, levels):
                backward = np.array([metrics[name][key] for key in
                    ("raw_pnl", "risk_penalty", "fees", "exchange_quantity", "customer_orders",
                     "customer_quantity", "inventory_sq_time")])
                forward_error = max(forward_error, float(np.max(np.abs(backward-occupancy_moments(actual, policy, level)))))
                if metrics[name]["objective"] > metrics["candidate_oracle"]["objective"]+1e-9:
                    raise AssertionError("Common-set oracle fails to dominate tested policy")
            if forward_error > 1e-9:
                raise AssertionError("Forward/backward exact expectations disagree")
            action_diagnostics = {
                "R3_vs_frozen_R2": compare_actions(actual, policies[2], policies[1], levels[2], levels[1], grid),
                "direct_vs_frozen_R2": compare_actions(actual, policies[3], policies[1], levels[3], levels[1], grid),
                "R3_vs_direct": compare_actions(actual, policies[2], policies[3], levels[2], levels[3], grid)}
            max_error = 0.
            for state in grid:
                for candidate in policies[3].candidates(*state):
                    q0, dq = policies[2].evaluation.action_value(*state, candidate)
                    direct_q = policies[3].evaluation.action_value(*state, candidate)
                    max_error = max(max_error, abs(q0+theta*dq-direct_q))
            row = dict(regime=regime, theta=theta,
                inside_empirical_radius=abs(theta)<=protocol["empirical_trust_radius"],
                metrics=metrics, action_diagnostics=action_diagnostics,
                max_candidate_linear_Q_error=max_error,
                forward_backward_max_absolute_error=forward_error,
                reachable_inventories={name: sorted({state[1] for level in ls[:-1] for state in level})
                    for name, ls in zip(protocol["methods"], levels)},
                R3_minus_frozen_R2=metrics["linear_R3"]["objective"]-metrics["frozen_R2_at0"]["objective"],
                direct_minus_frozen_R2=metrics["known_theta_direct_Q"]["objective"]-metrics["frozen_R2_at0"]["objective"],
                R3_minus_direct=metrics["linear_R3"]["objective"]-metrics["known_theta_direct_Q"]["objective"])
            results.append(row)
            write(out / "results_partial.json", results)
    write(out / "results.json", results)
    write(out / "budgets.json", budgets)
    write(out / "derivatives.json", derivatives)
    inside = [row for row in results if row["inside_empirical_radius"]]
    summary = dict(stage=protocol["stage"], config_sha256=digest(out / "config.json"),
        exact_comparisons=len(results), initial_inventory=0, elapsed_seconds=time.perf_counter()-started,
        in_radius_comparisons=len(inside),
        reachable_R3_switch_comparisons=sum(row["action_diagnostics"]["R3_vs_frozen_R2"]
            ["probability_of_at_least_one_switch_from_q0"]>1e-12 for row in inside),
        R3_positive_increment_count=sum(row["R3_minus_frozen_R2"]>1e-10 for row in inside),
        R3_negative_increment_count=sum(row["R3_minus_frozen_R2"] < -1e-10 for row in inside),
        R3_zero_increment_count=sum(abs(row["R3_minus_frozen_R2"])<=1e-10 for row in inside),
        direct_positive_increment_count=sum(row["direct_minus_frozen_R2"]>1e-10 for row in inside),
        max_fd_error=max(row["absolute_error"] for row in derivatives),
        max_forward_backward_error=max(row["forward_backward_max_absolute_error"] for row in results),
        rows=[{key: row[key] for key in ("regime", "theta", "inside_empirical_radius",
            "R3_minus_frozen_R2", "direct_minus_frozen_R2", "R3_minus_direct")}
            | {"R3_first_switch_probability": row["action_diagnostics"]["R3_vs_frozen_R2"]
                ["probability_of_at_least_one_switch_from_q0"]} for row in results],
        promotion_enabled=False, competition_response_identified=False,
        claims_status="NUMERICALLY_CHECKED in the declared fixed-support finite synthetic family")
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def manifest(out):
    write(out / "SHA256SUMS.json", {str(path.relative_to(out)): digest(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "SHA256SUMS.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["declare", "execute"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "declare":
        declare(args.out)
        print(f"Declared {args.out / 'config.json'} before family evaluation")
    else:
        execute(args.out)
    manifest(args.out)


if __name__ == "__main__":
    main()
