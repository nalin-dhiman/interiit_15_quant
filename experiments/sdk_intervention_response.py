"""Controlled primitive-history response using the unmodified organiser Engine.

All generated price/customer/opponent laws are synthetic. This research script
does I/O; it is not a submission and never promotes runtime adaptation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from experiments.sdk_environment import KIT, complete_source_hashes, require_sdk, source_key, source_path
require_sdk()
sys.path.insert(0, str(KIT / "rfq"))
from rfq.engine import Engine
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.store import Store
from experiments.sdk_calibration import logistic_fit, logistic_predict, monotone_decreasing, metrics

EDGES = (0., .02, .04, .06, .08, .10, .12, .16, .20, .24, .28, .32)
LIMITS = (.04, .08, .12, .16, .24, .32)
FAMILIES = ((.02, .04, .06, .08), (.10, .14, .18, .22))
SHIFTED = ((.01, .03, .05, .07), (.09, .13, .17, .21))
FEATURES = ("edge", "product_B", "customer_BUY", "quantity", "previous_increment")
CONFIG = dict(protocol_version=1, train_seed_start=710_000,
              validation_seed_start=720_000, family_validation_seed_start=730_000,
              sessions_per_split=100, intervals_per_session=120, rfqs_per_interval=1,
              theta=.5, epsilon=.05, trust_region=[.2, .8],
              edges=EDGES, limits=LIMITS, opponent_families=FAMILIES,
              shifted_families=SHIFTED, price_lag=.4,
              price_innovation_cents=[-4, -2, 0, 2, 4],
              idiosyncratic_innovation_cents=[-2, 0, 2],
              fee_intercept=.01, fee_derivative=.01,
              export_gates=dict(mean_calibration=.03, ece=.05,
                                supported_bin_error=.10, support_coverage=1.,
                                min_bin_n=100), runtime_enabled=False,
              labels="synthetic controlled SDK experiment; not organiser round law")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote_at(mid, edge, side):
    sign = 1 if side == "BUY" else -1
    return float(Decimal(str(mid)) + sign * Decimal(str(edge)))


def make_tape(seed, intervals=120):
    """Independent primitives. No callback or policy changes their draw count."""
    streams = [random.Random(seed*10+i) for i in range(4)]
    prices, customers, opponents, exploration = streams
    a, b, previous_a = 10_000, 20_000, 0
    rows, attributes = [dict(A=a/100, B=b/100)], []
    previous = dict(A=0., B=0.)
    for index in range(intervals):
        key = customers.choice(("A", "B"))
        side = customers.choice(("BUY", "SELL"))
        attributes.append(dict(product_id=key, side=side,
            quantity=customers.choice((2, 4, 10)), limit_offset=customers.choice(LIMITS),
            family_u=opponents.random(), opponent_indices=[opponents.randrange(4) for _ in range(2)],
            edge=exploration.choice(EDGES), previous_increment=previous[key],
            # This is an analysis label, never an input feature.
            expected_next_increment=(round(.4*previous_a)*(2 if key == "B" else 1))/100))
        da = round(.4*previous_a)+prices.choice((-4, -2, 0, 2, 4))
        db = 2*da+prices.choice((-2, 0, 2))
        a, b = a+da, b+db
        if a <= 100 or b <= 100:
            raise ValueError("Synthetic tape positivity condition failed")
        previous_a, previous = da, dict(A=da/100, B=db/100)
        rows.append(dict(A=a/100, B=b/100))
    return dict(seed=seed, price_rows=rows, attributes=attributes,
                stream_seeds={k: seed*10+i for i, k in enumerate(
                    ("price", "customer", "opponent", "quote_exploration"))},
                tie_seed=seed*10+7)


class TapeCustomers:
    def __init__(self, attributes):
        self.attributes = attributes

    def generate(self, interval_id, market, state, rng):
        a = self.attributes[interval_id-1]
        return interval_id, [CustomerOrder(f"r{interval_id}-1", "customer-1",
            a["product_id"], a["side"], a["quantity"], interval_id,
            limit_offset=a["limit_offset"])]


def checked_action(engine, name, message):
    events = engine.action(name, message)
    acks = [e for _, e in events if e["type"] == "ack"]
    if len(acks) != 1 or acks[0]["status"] != "accepted" or acks[0].get("errors"):
        raise AssertionError(f"Engine rejected experiment action: {acks}")
    return events


def run_tape(tape, theta=.5, shifted=False):
    if not .2 <= theta <= .8:
        raise ValueError("Intervention outside predeclared trust region")
    fee = .01+.01*theta
    products = [Product("A", 100., max_long=50, max_short=50,
                       breach_penalty_per_unit=1., fee_per_unit=fee),
                Product("B", 200., max_long=25, max_short=25,
                       breach_penalty_per_unit=2., fee_per_unit=fee)]
    config = HostConfig(products, ReplayPrices(tape["price_rows"],
                       {"A": "LAST_PRICE", "B": {"A": 2.}}),
                       TapeCustomers(tape["attributes"]), seed=tape["tie_seed"]-2,
                       price_ticks_per_interval=1, final_hedge_seconds=0.)
    store, records = Store(":memory:"), []
    engine = Engine(config, store)
    for name in ("dealer_1", "dealer_2", "dealer_3"):
        engine.register(name)
    family_grid = SHIFTED if shifted else FAMILIES
    action_no = 0
    for i, a in enumerate(tape["attributes"]):
        order = next(iter(engine.orders.values()))
        public = order.public()
        if "limit_offset" in public or "reservation_price" in public:
            raise AssertionError("Hidden customer state leaked")
        mid = engine.prices[a["product_id"]]
        z = int(a["family_u"] < theta)
        own_quote = quote_at(mid, a["edge"], a["side"])
        for j, name in enumerate(("dealer_1", "dealer_2", "dealer_3")):
            edge = a["edge"] if j == 0 else family_grid[z][a["opponent_indices"][j-1]]
            action_no += 1
            checked_action(engine, name, dict(type="quotes", action_id=f"q{action_no}",
                interval_id=engine.interval_id, quotes={order.order_id: quote_at(mid, edge, a["side"])}))
        events = engine.advance(final=i == len(tape["attributes"])-1)
        result = next(e for _, e in events if e["type"] == "customer_results")["results"][0]
        won = result["winner"] == "dealer_1"
        dealer_sign = -1 if a["side"] == "BUY" else 1
        increment = engine.prices[a["product_id"]]-mid
        # Physical opposite exchange transaction, fee charged by original Engine.
        if won:
            action_no += 1
            checked_action(engine, "dealer_1", dict(type="trades", action_id=f"h{action_no}",
                price_version=engine.price_version,
                trades=[dict(product_id=a["product_id"], side=a["side"], quantity=a["quantity"])]))
        reward = float(won)*a["quantity"]*(a["edge"]+dealer_sign*increment-fee)
        records.append(dict(session_seed=tape["seed"], interval=i+1, **public,
            own_quote=own_quote, edge=a["edge"], previous_increment=a["previous_increment"],
            won=won, public_status=result["status"], public_winner=result["winner"],
            public_winning_price=result["price"], matching_reference=mid,
            possible_public_tie=result["price"] == own_quote,
            signed_markout=dealer_sign*increment, reward=reward,
            fee_paid=float(won)*a["quantity"]*fee,
            explicit_reward_derivative=-.01*a["quantity"]*float(won),
            # Experimental primitive labels are used solely for response analysis.
            intervention_family=z, primitive_score=z/theta-(1-z)/(1-theta),
            expected_signed_markout=dealer_sign*a["expected_next_increment"],
            theta=theta, shifted_family=shifted))
    engine.finish()
    portfolio = engine.portfolio("dealer_1")
    reward_sum = sum(r["reward"] for r in records)
    if abs(portfolio["pnl"]-reward_sum) > 1e-7:
        raise AssertionError("Engine PnL does not reconcile with per-final-RFQ ledger")
    if any(abs(q)>1e-9 for q in portfolio["positions"].values()):
        raise AssertionError("Exchange transaction failed to flatten inventory")
    if abs(portfolio["fees_paid"]-sum(r["fee_paid"] for r in records)) > 1e-9:
        raise AssertionError("Engine fee total does not reconcile")
    store.close()
    return records, dict(seed=tape["seed"], pnl=portfolio["pnl"],
        fees=portfolio["fees_paid"], penalties=portfolio["penalties_paid"],
        customer_trades=portfolio["customer_trades"],
        exchange_trades=portfolio["exchange_trades"], cash=portfolio["cash"],
        inventory_flat=True, reconciliation_error=portfolio["pnl"]-reward_sum,
        theta=theta, shifted_family=shifted)


def features(records):
    return np.array([[r["edge"], r["product_id"] == "B", r["side"] == "BUY",
                      r["quantity"], r["previous_increment"]] for r in records], float)


def calibration(y, p, edges):
    out = metrics(y, p)
    bins = []
    for edge in EDGES:
        ix = np.isclose(edges, edge)
        bins.append(dict(edge=edge, n=int(ix.sum()), observed=float(y[ix].mean()) if ix.any() else None,
                         predicted=float(p[ix].mean()) if ix.any() else None,
                         error=float(abs(y[ix].mean()-p[ix].mean())) if ix.any() else None))
    prob_bins = np.minimum((p*10).astype(int), 9)
    out["ece"] = float(sum(np.sum(prob_bins==b)*abs(y[prob_bins==b].mean()-p[prob_bins==b].mean())
                           for b in range(10) if np.any(prob_bins==b))/len(y))
    out["absolute_mean_error"] = abs(out["observed_win_rate"]-out["predicted_win_rate"])
    out["edge_bins"] = bins
    supported = [b["error"] for b in bins if b["n"] >= 100]
    out["max_supported_bin_error"] = max(supported) if supported else 0.
    out["supported_bins"] = len(supported)
    return out


def markout_design(records):
    return np.array([[1., (-1 if r["side"] == "BUY" else 1)*r["previous_increment"],
                      r["product_id"] == "B"] for r in records], float)


def predictive_analysis(train, validations):
    x, y = features(train), np.array([r["won"] for r in train], float)
    started = time.perf_counter()
    fit = logistic_fit(x, y, ridge=1.)
    counts = np.array([sum(np.isclose(x[:,0], e)) for e in EDGES])
    wins = np.array([sum(y[np.isclose(x[:,0], e)]) for e in EDGES])
    pooled = (wins+.5)/(counts+1.)
    mono = monotone_decreasing(pooled, counts+1.)
    fills = [r for r in train if r["won"]]
    design = markout_design(fills)
    coeff = np.linalg.lstsq(design, [r["signed_markout"] for r in fills], rcond=None)[0]
    fitting_seconds = time.perf_counter()-started
    results, gates, markout_gates = {}, [], []
    joint_support = {(r["edge"],r["product_id"],r["side"],r["quantity"]) for r in train}
    for label, rows in validations.items():
        xv, yv = features(rows), np.array([r["won"] for r in rows], float)
        idx = np.array([EDGES.index(r["edge"]) for r in rows])
        estimates = dict(constant=np.full(len(rows), y.mean()), pooled=pooled[idx],
                         monotone=mono[idx], logistic=logistic_predict(fit, xv))
        item = {name: calibration(yv, p, xv[:,0]) for name, p in estimates.items()}
        support = float(np.mean([r["edge"] in EDGES for r in rows]))
        joint_coverage = float(np.mean([(r["edge"],r["product_id"],r["side"],r["quantity"]) in joint_support for r in rows]))
        lg = item["logistic"]
        gate = (support == 1. and lg["supported_bins"] == len(EDGES) and
                lg["brier"] < item["constant"]["brier"] and
                lg["absolute_mean_error"] <= .03 and lg["ece"] <= .05 and
                lg["max_supported_bin_error"] <= .10)
        filled = [r for r in rows if r["won"]]
        labels = np.array([r["signed_markout"] for r in filled])
        predicted = markout_design(filled)@coeff
        mse0, msefit = float(np.mean(labels**2)), float(np.mean((labels-predicted)**2))
        item.update(sessions=len({r["session_seed"] for r in rows}), edge_support_coverage=support,
                    observed_discrete_joint_support_coverage=joint_coverage,
                    fill_gate_passed=gate,
                    markout=dict(n=len(filled), zero_mse=mse0, fitted_mse=msefit,
                                 validation_mean=float(labels.mean()),
                                 gate_passed=msefit < mse0))
        # Validation-session intervals conditional on the frozen training fit.
        session_keys = sorted({r["session_seed"] for r in rows})
        indices = np.array([r["session_seed"] for r in rows])
        brier_differences = {}
        bootstrap_rng = np.random.default_rng(750000)
        bootstrap_ix = bootstrap_rng.integers(len(session_keys),size=(2000,len(session_keys)))
        for name in ("logistic","monotone"):
            losses = (estimates[name]-yv)**2-(estimates["constant"]-yv)**2
            session_loss = np.array([losses[indices == key].mean() for key in session_keys])
            brier_differences[name] = dict(mean=float(session_loss.mean()),
                session_bootstrap_95_interval=np.quantile(session_loss[bootstrap_ix].mean(1),[.025,.975]).tolist())
        item["brier_difference_from_constant"] = brier_differences
        filled_indices = np.array([r["session_seed"] for r in filled])
        markout_loss = (labels-predicted)**2-labels**2
        markout_keys = sorted(set(filled_indices))
        session_markout_loss = np.array([markout_loss[filled_indices == key].mean() for key in markout_keys])
        markout_ix = bootstrap_rng.integers(len(markout_keys),size=(2000,len(markout_keys)))
        item["markout"]["session_mse_difference_from_zero"] = dict(mean=float(session_markout_loss.mean()),
            session_bootstrap_95_interval=np.quantile(session_markout_loss[markout_ix].mean(1),[.025,.975]).tolist())
        results[label] = item
        gates.append(gate)
        markout_gates.append(msefit < mse0)
    return dict(training_rfqs=len(train), training_sessions=len({r["session_seed"] for r in train}),
        fitting_seconds=fitting_seconds, target="own winning AND eligibility",
        hidden_limits_identified=False, repeated_callbacks_counted=False,
        validation=results, fill_export_gate_passed=all(gates),
        markout_export_gate_passed=all(markout_gates),
        coefficients=fit, pooled_probabilities=pooled.tolist(),
        monotone_probabilities=mono.tolist(), training_edge_counts=counts.tolist(),
        markout_coefficients=coeff.tolist(),
        markout_feature_names=["intercept", "dealer_signed_previous_increment", "product_B"],
        limitations=["Empirical calibration gates are not uniform uncertainty certificates",
                     "Synthetic observable momentum does not establish predictability in competition data",
                     "Public losses censor hidden eligibility and competition separately"])


def exact_fill_probability(edge, family, shifted=False):
    """Ideal decimal-law finite sum; SDK float eligibility can differ at a limit."""
    grid = (SHIFTED if shifted else FAMILIES)[family]
    probability = 0.
    # Compare integer ticks, matching SDK Decimal quote arithmetic.
    e = round(100*edge)
    for limit in LIMITS:
        limit_tick = round(100*limit)
        if e > limit_tick:
            continue
        for a in grid:
            for b in grid:
                eligible = [round(100*x) for x in (a,b) if round(100*x) <= limit_tick]
                if eligible and min(eligible) < e:
                    continue
                tied = sum(x == e for x in eligible)
                probability += 1/(len(LIMITS)*len(grid)**2*(1+tied))
    return probability


@lru_cache(maxsize=100_000)
def sdk_fill_probability(edge, family, reference, side, shifted=False):
    """Finite primitive sum with the unmodified SDK's numerical comparisons.

    Quotes use Decimal arithmetic but Engine offset limits use float arithmetic.
    Reference and side therefore belong to this numerical reference state.
    """
    grid = (SHIFTED if shifted else FAMILIES)[family]
    own = quote_at(reference,edge,side)
    probability = 0.
    for offset in LIMITS:
        limit = reference+offset if side == "BUY" else reference-offset
        acceptable = (lambda q:q<=limit) if side == "BUY" else (lambda q:q>=limit)
        if not acceptable(own):
            continue
        for a in grid:
            for b in grid:
                eligible = [q for q in (quote_at(reference,a,side),quote_at(reference,b,side)) if acceptable(q)]
                if any(q<own if side == "BUY" else q>own for q in eligible):
                    continue
                probability += 1/(len(LIMITS)*len(grid)**2*(1+sum(q == own for q in eligible)))
    return probability


def session_summary(values):
    v = np.asarray(values, float)
    se = float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v)>1 else None,
                standard_error=se, normal_95_interval=[float(v.mean()-1.96*se),float(v.mean()+1.96*se)] if se is not None else None)


def response_analysis(baseline, plus, minus, plus_records, minus_records, theta=.5, epsilon=.05):
    started = time.perf_counter()
    grouped = {}
    for r in baseline:
        grouped.setdefault(r["session_seed"], []).append(r)
    plus_by = {r["seed"]:r for r in plus}
    minus_by = {r["seed"]:r for r in minus}
    # Shared observation budget: every estimated method receives all three arms.
    # Conditional family outcomes have the same law across theta; pool them.
    all_records = baseline+plus_records+minus_records
    arms = {}
    for r in all_records:
        arms.setdefault(r["session_seed"], []).append(r)
    fitting_started = time.perf_counter()
    fitted = {}
    for z in (0,1):
        for edge in EDGES:
            rows = [r for r in all_records if r["intervention_family"] == z and r["edge"] == edge]
            fitted[z,edge] = (sum(r["won"] for r in rows)+.5)/(len(rows)+1.)
    fitting_seconds = time.perf_counter()-fitting_started
    evaluation_started = time.perf_counter()
    estimates = {k: [] for k in ("local_likelihood", "full_path_likelihood", "paired_finite_difference",
                                 "known_tangent", "known_ideal_decimal_tangent", "estimated_finite_tangent", "direct_conditional_refit")}
    validation = []
    for seed, rows in grouped.items():
        explicit = sum(r["explicit_reward_derivative"] for r in rows)
        local = 0.
        for r in arms[seed]:
            z, source_theta = r["intervention_family"], r["theta"]
            weight = theta/source_theta if z else (1-theta)/(1-source_theta)
            target_reward = r["reward"] + .01*(source_theta-theta)*r["quantity"]*r["won"]
            score = z/theta-(1-z)/(1-theta)
            local += weight*(target_reward*score+r["explicit_reward_derivative"])/3
        estimates["local_likelihood"].append(local)
        estimates["full_path_likelihood"].append(sum(r["reward"] for r in rows)*sum(r["primitive_score"] for r in rows)+explicit)
        estimates["paired_finite_difference"].append((plus_by[seed]["pnl"]-minus_by[seed]["pnl"])/(2*epsilon))
        tangent, ideal_tangent, refit, refit_plus, refit_minus = 0., 0., 0., 0., 0.
        for r in rows:
            p0, p1 = (sdk_fill_probability(r["edge"], z,r["matching_reference"],r["side"]) for z in (0,1))
            d0,d1 = (exact_fill_probability(r["edge"], z) for z in (0,1))
            expected_net = r["edge"]+r["expected_signed_markout"]-(.01+.01*theta)
            tangent += r["quantity"]*((p1-p0)*expected_net-.01*((1-theta)*p0+theta*p1))
            ideal_tangent += r["quantity"]*((d1-d0)*expected_net-.01*((1-theta)*d0+theta*d1))
            f0, f1 = (fitted[z,r["edge"]] for z in (0,1))
            refit += r["quantity"]*((f1-f0)*expected_net-.01*((1-theta)*f0+theta*f1))
            for t, target in ((theta+epsilon, "plus"),(theta-epsilon,"minus")):
                predicted = r["quantity"]*((1-t)*f0+t*f1)*(r["edge"]+r["expected_signed_markout"]-(.01+.01*t))
                if target == "plus":
                    refit_plus += predicted
                else:
                    refit_minus += predicted
        estimates["known_tangent"].append(tangent)
        estimates["known_ideal_decimal_tangent"].append(ideal_tangent)
        estimates["estimated_finite_tangent"].append(refit)
        estimates["direct_conditional_refit"].append((refit_plus-refit_minus)/(2*epsilon))
    for edge in EDGES:
        for z in (0,1):
            observed = [r["won"] for r in baseline if r["edge"] == edge and r["intervention_family"] == z]
            corresponding = [r for r in baseline if r["edge"] == edge and r["intervention_family"] == z]
            validation.append(dict(edge=edge, family=z, n=len(observed), observed=float(np.mean(observed)) if observed else None,
                                   known_ideal_decimal_probability=exact_fill_probability(edge,z),
                                   known_sdk_probability=float(np.mean([sdk_fill_probability(edge,z,r["matching_reference"],r["side"]) for r in corresponding])) if corresponding else None))
    reference = np.array(estimates["known_tangent"])
    result = {k: dict(**session_summary(v), paired_error_against_known_tangent=session_summary(np.array(v)-reference))
              for k,v in estimates.items()}
    evaluation_seconds = time.perf_counter()-evaluation_started
    bootstrap_started = time.perf_counter()
    # A global conditional fit is shared across sessions. Its uncertainty cannot
    # be obtained by pretending fitted session predictions are independent.
    # Refit within paired session bootstrap samples, preserving all three arms.
    seeds = sorted(grouped)
    count = np.zeros((len(seeds),2,len(EDGES)))
    wins = np.zeros_like(count)
    a_coefficient = np.zeros((len(seeds),len(EDGES)))
    b_coefficient = np.zeros_like(a_coefficient)
    for si,seed in enumerate(seeds):
        for r in arms[seed]:
            z, ei = r["intervention_family"], EDGES.index(r["edge"])
            count[si,z,ei] += 1
            wins[si,z,ei] += r["won"]
        for r in grouped[seed]:
            ei = EDGES.index(r["edge"])
            a_coefficient[si,ei] += r["quantity"]*(r["edge"]+r["expected_signed_markout"]-(.01+.01*theta))
            b_coefficient[si,ei] += .01*r["quantity"]
    bootstrap = {name: [] for name in estimates}
    bootstrap_error = {name: [] for name in estimates}
    rng = np.random.default_rng(740_000)
    for _ in range(2000):
        ix = rng.integers(len(seeds),size=len(seeds))
        fitted_boot = (wins[ix].sum(0)+.5)/(count[ix].sum(0)+1.)
        f0,f1 = fitted_boot
        fitted_mean = np.sum((f1-f0)*a_coefficient[ix].mean(0)-
                             ((1-theta)*f0+theta*f1)*b_coefficient[ix].mean(0))
        for name, values in estimates.items():
            if name in ("estimated_finite_tangent","direct_conditional_refit"):
                value = fitted_mean
            else:
                value = np.asarray(values)[ix].mean()
            bootstrap[name].append(value)
            bootstrap_error[name].append(value-reference[ix].mean())
    for name,values in bootstrap.items():
        result[name]["paired_session_bootstrap_95_interval"] = np.quantile(values,[.025,.975]).tolist()
        result[name]["paired_session_bootstrap_standard_error"] = float(np.std(values,ddof=1))
        result[name]["paired_error_against_known_tangent"]["session_bootstrap_95_interval"] = np.quantile(bootstrap_error[name],[.025,.975]).tolist()
        if name in ("estimated_finite_tangent","direct_conditional_refit"):
            result[name]["normal_95_interval"] = None
            result[name]["standard_error"] = None
            result[name]["paired_error_against_known_tangent"]["standard_error"] = None
            result[name]["paired_error_against_known_tangent"]["normal_95_interval"] = None
            result[name]["conditional_session_spread_note"] = "Fitted predictions share coefficients; bootstrap refits preserve that dependence. Session-spread intervals are suppressed."
    return dict(estimators=result, engine_vs_finite_law=validation,
                analysis_seconds=time.perf_counter()-started,
                compute_seconds=dict(shared_conditional_fit=fitting_seconds,
                    combined_estimator_evaluation=evaluation_seconds,
                    paired_session_bootstrap=time.perf_counter()-bootstrap_started),
                bootstrap_replicates=2000, bootstrap_seed=740_000,
                independent_unit="fresh independently generated price/customer/opponent session",
                baseline_sessions=len(grouped), rfqs_per_baseline=len(baseline),
                intervention_parameter="known randomized opponent family mixture AND explicit exchange fee",
                known_tangent_uses_stronger_information=True, production_response_identified=False,
                known_tangent_matches_sdk_float_limit_arithmetic=True,
                known_ideal_decimal_tangent_is_not_exact_sdk_reference=True,
                fitted_tangent_is_estimated_fill_plus_known_synthetic_payoff_model=True,
                direct_refit_has_no_extra_engine_observations=True,
                bootstrap_intervals_are_empirical_not_uniform_certificates=True,
                runtime_response_enabled=False, conditional_refit_uses_shared_three_arm_rfqs=True,
                shared_engine_runs_per_session=3,
                shared_rfqs_per_estimated_method=len(all_records),
                comparison_budget_note="Every estimated competitor receives the identical baseline/plus/minus bundle and three-run Engine budget; methods may ignore observations. Full-path baseline score and known-law tangent are diagnostics.",
                tie_rng_note="Original Engine unchanged; conditional eligible sets may consume different numbers of tie draws",
                explicit_payoff_derivative_included=True, common_cash_path_density_assumed=False,
                singular_case_note="Zero own-fill probability is allowed; primitive family law stays strictly positive. No Bernoulli score is applied at p=0.")


def quantitative_report(prediction, response):
    lines = ["", "## Quantitative results", "",
        "Fill probabilities predict winning AND eligibility. These are held-out synthetic sessions;",
        "the original demo holdout is not used. Gates remain those frozen before execution.", "",
        "| Split | RFQs | Constant Brier | Monotone Brier | Logistic Brier | Logistic ECE | Max supported-bin error |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,r in prediction["validation"].items():
        lines.append(f"| {name} | {r['logistic']['n']} | {r['constant']['brier']:.6f} | {r['monotone']['brier']:.6f} | {r['logistic']['brier']:.6f} | {r['logistic']['ece']:.6f} | {r['logistic']['max_supported_bin_error']:.6f} |")
    lines += ["", "Logistic calibration fails the predeclared bin-error guard in both validation sets.",
        "The shifted-family ECE also exceeds 0.05. Monotone prediction performs well but was not substituted",
        "after inspecting validation results. This is a parametric/calibration gate failure, not proof",
        "that the synthetic fill law is unidentifiable. No runtime coefficients are exported.", "",
        "| Split | Zero markout MSE | Fitted markout MSE |",
        "|---|---:|---:|"]
    for name,r in prediction["validation"].items():
        lines.append(f"| {name} | {r['markout']['zero_mse']:.8f} | {r['markout']['fitted_mse']:.8f} |")
    lines += ["", "Markout improves because observable momentum was deliberately inserted into the synthetic price law.",
        "It is not evidence of predictable competition prices and is not enabled for runtime use.", "",
        "Response values below are PnL derivatives per 120-RFQ session with respect to a known",
        "experimental mixture/fee parameter, not strategy alpha or gains from adapting a policy.", "",
        "| Estimator | Mean derivative | Paired-session bootstrap 95% interval |",
        "|---|---:|---:|"]
    for name,r in response["estimators"].items():
        lo,hi = r["paired_session_bootstrap_95_interval"]
        lines.append(f"| {name} | {r['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "The fitted finite tangent and direct conditional refit agree to rounding; response algebra",
        "does not create extra information. The full-path score is much noisier than the local score.",
        "The finite-law reference and known synthetic payoff model use declared model information;",
        "their accuracy must not be presented as identification from SDK observations alone.",
        "Empirical bootstrap intervals assume independent sessions from the declared generator and are",
        "not simultaneous uniform error certificates. Session-level errors and full tables are in JSON.", ""]
    return "\n".join(lines)


def reanalyze(raw_directory, analysis_name="reanalysis"):
    """Repair analysis without rerunning, excluding or replacing raw outcomes."""
    raw_directory = Path(raw_directory)
    if analysis_name not in ("reanalysis","audited_analysis","sdk_numerical_audit"):
        raise ValueError("Unknown analysis output name")
    out = raw_directory/analysis_name
    out.mkdir(exist_ok=False)
    (out/"executed_source.py").write_bytes(Path(__file__).read_bytes())
    (out/"protocol_with_erratum.md").write_bytes((ROOT/"docs/SDK_INTERVENTION_PROTOCOL.md").read_bytes())
    started = time.perf_counter()
    def read_rows(name):
        return [json.loads(s) for s in (raw_directory/name).read_text().splitlines() if s]
    data = {label:read_rows(f"{label}_final_rfqs.jsonl") for label in
            ("train","validation","shifted_family_validation")}
    paired = read_rows("paired_interventions_final_rfqs.jsonl")
    totals = json.loads((raw_directory/"paired_interventions_session_totals.json").read_text())
    write_json(out/"analysis_config.json",dict(bootstrap_seed=740000,bootstrap_replicates=2000,
               change="Repair summary seed key, bootstrap shared fits, and preserve ideal-decimal reference beside exact SDK float eligibility reference; no outcome excluded or policy/criterion changed",
               source_sha256=sha(__file__), original_config_sha256=sha(raw_directory/"config.json"),
               original_source_manifest_sha256=sha(raw_directory/"source_manifest.json")))
    prediction = predictive_analysis(data["train"],{k:v for k,v in data.items() if k != "train"})
    response = response_analysis(data["train"],totals["plus"],totals["minus"],
                                 [r for r in paired if r["theta"] > .5],
                                 [r for r in paired if r["theta"] < .5])
    write_json(out/"predictive_analysis.json",prediction)
    write_json(out/"response_analysis.json",response)
    if prediction["fill_export_gate_passed"]:
        write_json(out/"validated_synthetic_coefficients.json",dict(domain="only predeclared synthetic controlled law",
            runtime_enabled=False, feature_names=FEATURES, fill=prediction["coefficients"],
            markout=prediction["markout_coefficients"] if prediction["markout_export_gate_passed"] else None,
            empirical_gate=True, uniform_error_certificate=False))
    sessions = len({r["session_seed"] for r in data["train"]})
    observation_count = sum(len(v) for v in data.values())+len(paired)
    report = ["# Controlled SDK response validation", "",
        f"Completed: {sessions*5} original Engine sessions, {observation_count} final RFQs including paired interventions.",
        "Every actor fill was flattened by a charged exchange transaction; cash/fees/final inventory reconcile.",
        "No supplied demo price CSV was read. One final observation per RFQ; hidden limits are not features.", "",
        f"Predeclared fill export gate: {prediction['fill_export_gate_passed']}.",
        f"Predeclared markout export gate: {prediction['markout_export_gate_passed']}.",
        "Runtime adaptation remains disabled regardless of these synthetic diagnostics.", "",
        "The first analysis failed on a summary-key mismatch after all raw data were generated.",
        "This repair reads all original raw outcomes. It also bootstraps paired sessions with conditional refitting",
        "to include shared fit uncertainty; no seed, outcome, policy or export criterion was changed.", "",
        "A subsequent numerical audit found that Engine hidden offset limits use binary float arithmetic",
        "while quotes are exact decimal ticks. A boundary quote can fail eligibility by one float ULP.",
        "The ideal-decimal tangent is retained but is not an exact SDK reference. known_tangent now",
        "enumerates the actual SDK numerical comparisons conditional on matching reference and side.", "",
        "Known finite-law tangent has stronger information. Estimated methods share the exact three-arm bundle",
        "and three-run Engine budget. Full-path baseline score is a diagnostic. Deterministic cash-path density",
        "compatibility is not assumed; forward response uses the common primitive family history and explicit fee derivative.", "",
        "Permitted claims: NUMERICALLY_CHECKED common-support response and exact SDK self-financing fixtures;",
        "EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT predictive performance only when listed gates pass.",
        "Production response identification: BLOCKED_BY_MISSING_INPUT. No trading superiority/R3 claim follows.", "",
        "Failed gates, detailed calibration and all estimator intervals are retained in the analysis JSON files.",
        "Markout momentum is deliberately present in the stated synthetic generator; it is not evidence of actual market predictability.", ""]
    (out/"REPORT.md").write_text("\n".join(report)+quantitative_report(prediction,response))
    write_json(out/"timing.json",dict(analysis_seconds=time.perf_counter()-started,
               shared_engine_sessions=sessions*3,validation_engine_sessions=sessions*2,source_raw_directory=str(raw_directory)))
    write_json(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"})
    print(json.dumps(dict(out=str(out),fill_gate=prediction["fill_export_gate_passed"],markout_gate=prediction["markout_export_gate_passed"])))


def execute(out, sessions=100, intervals=120):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(CONFIG, sessions_per_split=sessions, intervals_per_session=intervals)
    # Freeze configuration, protocol and inputs BEFORE computing any outcomes.
    write_json(out/"config.json", config)
    (out/"executed_source.py").write_bytes(Path(__file__).read_bytes())
    (out/"frozen_protocol.md").write_bytes((ROOT/"docs/SDK_INTERVENTION_PROTOCOL.md").read_bytes())
    sources = [Path(__file__), ROOT/"docs/SDK_INTERVENTION_PROTOCOL.md", ROOT/"experiments/sdk_calibration.py"]
    sources += sorted((KIT/"rfq/rfq").glob("*.py"))
    write_json(out/"source_manifest.json", complete_source_hashes({source_key(p):sha(p) for p in sources}))
    write_json(out/"environment.json", dict(python=sys.version, numpy=np.__version__, os=platform.platform(),
               executable=sys.executable, argv=sys.argv, utc_start=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    start = time.perf_counter()
    datasets, totals, timings = {}, {}, {}
    tape_handles = {}
    plus_totals, minus_totals, plus_records, minus_records = [], [], [], []
    for label, seed_start, shifted in (("train",710_000,False),("validation",720_000,False),
                                      ("shifted_family_validation",730_000,True)):
        rows, total = [], []
        before = time.perf_counter()
        with ((out/f"{label}_primitives.jsonl").open("w") as primitive,
              (out/f"{label}_final_rfqs.jsonl").open("w") as raw):
            for j in range(sessions):
                tape = make_tape(seed_start+j,intervals)
                primitive.write(json.dumps(tape, sort_keys=True)+"\n")
                records, summary = run_tape(tape,theta=.5,shifted=shifted)
                total.append(summary)
                rows.extend(records)
                raw.writelines(json.dumps(r,sort_keys=True)+"\n" for r in records)
        datasets[label], totals[label] = rows, total
        timings[label] = time.perf_counter()-before
        write_json(out/f"{label}_session_totals.json", total)
    before = time.perf_counter()
    with (out/"paired_interventions_final_rfqs.jsonl").open("w") as handle:
        for j in range(sessions):
            tape = make_tape(710_000+j,intervals)
            for theta, dest in ((.55,plus_totals),(.45,minus_totals)):
                rows, summary = run_tape(tape,theta=theta)
                dest.append(summary)
                (plus_records if theta > .5 else minus_records).extend(rows)
                handle.writelines(json.dumps(r,sort_keys=True)+"\n" for r in rows)
    timings["paired_interventions"] = time.perf_counter()-before
    write_json(out/"paired_interventions_session_totals.json",dict(plus=plus_totals,minus=minus_totals))
    prediction = predictive_analysis(datasets["train"],{k:v for k,v in datasets.items() if k != "train"})
    response = response_analysis(datasets["train"],plus_totals,minus_totals,plus_records,minus_records)
    write_json(out/"predictive_analysis.json",prediction)
    write_json(out/"response_analysis.json",response)
    if prediction["fill_export_gate_passed"]:
        write_json(out/"validated_synthetic_coefficients.json",dict(domain="only predeclared synthetic controlled law",
            runtime_enabled=False, feature_names=FEATURES, fill=prediction["coefficients"],
            markout=prediction["markout_coefficients"] if prediction["markout_export_gate_passed"] else None,
            empirical_gate=True, uniform_error_certificate=False))
    write_json(out/"timing.json",dict(stage_seconds=timings,total_seconds=time.perf_counter()-start,
                                     engine_sessions=sessions*5,total_final_rfqs=sessions*intervals*5))
    report = f"""# Controlled SDK response validation\n\nStatus: EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT for exact self-financing\nreconciliation and synthetic calibration only. Production identification: BLOCKED_BY_MISSING_INPUT.\n\nExecuted {sessions*5} original Engine sessions and {sessions*intervals*5} final-RFQ observations\n(including paired interventions). Every actor fill was flattened by a charged exchange transaction.\nAll cash, fees and final inventory reconciled. No supplied demo price CSV was read.\n\nFill export gate: {prediction['fill_export_gate_passed']}. Markout export gate:\n{prediction['markout_export_gate_passed']}. Runtime adaptation stays disabled.\n\nThe known-mixture forward likelihood includes explicit fee derivatives. The\nfinite law tangent has stronger information than fitted estimators. All estimated\ncompetitors receive the identical three-arm observation bundle and Engine budget;\nfitting times are disclosed. The baseline-only full-path score is a diagnostic.\n\nFailed checks and remaining work: inspect predictive_analysis.json for calibration\ntransport failures; finite-sample response intervals do not establish production\nresponse. Official later-round rules, empirical response estimation and runtime\nvalidation remain separate gates.\n\nPermitted claims: bounded fixed-support primitive response was numerically checked;\none-final-RFQ estimation was tested in this declared synthetic domain; no trading\nsuperiority, profitability, official-market representativeness or uniform safety\nclaim follows. No R3 was promoted and no external orders were sent.\n"""
    (out/"REPORT.md").write_text(report+quantitative_report(prediction,response))
    write_json(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"})
    print(json.dumps(dict(out=str(out),rfqs=sessions*intervals*5,
                         fill_gate=prediction["fill_export_gate_passed"],
                         markout_gate=prediction["markout_export_gate_passed"],seconds=time.perf_counter()-start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--reanalyze",action="store_true")
    parser.add_argument("--analysis-name",choices=("reanalysis","audited_analysis","sdk_numerical_audit"),default="reanalysis")
    parser.add_argument("--sessions",type=int,default=100)
    parser.add_argument("--intervals",type=int,default=120)
    args = parser.parse_args()
    if args.reanalyze:
        reanalyze(args.out,args.analysis_name)
    else:
        execute(args.out,args.sessions,args.intervals)
