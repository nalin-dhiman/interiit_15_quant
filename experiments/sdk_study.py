"""Prospective comparisons using the UNMODIFIED supplied organiser Engine.

This is a synchronous, zero-latency callback schedule, not a network emulator.
Research-only imports/file access are intentional; this file is not a submission.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from experiments.sdk_environment import KIT, complete_source_hashes, require_sdk, source_key, source_path
require_sdk()
sys.path.insert(0, str(KIT / "rfq"))
from rfq.engine import Engine
from rfq.models import HostConfig, Product, RandomCustomers, ReplayPrices
from rfq.store import Store


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_class(path):
    spec = importlib.util.spec_from_file_location("sdk_study_" + sha256(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dealer


def inputs_audit():
    result = {}
    for name in ("demo_data.csv", "demo_data_holdout.csv", "demo_settlement_rules.csv"):
        path = KIT / "demo" / name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header, rows = next(reader), sum(1 for _ in reader)
        result[name] = dict(sha256=sha256(path), bytes=path.stat().st_size,
                            columns=header, rows=rows)
    return result


def source_hashes():
    paths = list((KIT / "rfq/rfq").glob("*.py")) + [
        KIT / "demo/demo_participant.py", KIT / "examples/dealer.py",
        ROOT / "submission/sdk_dealer.py", Path(__file__),
        ROOT / "experiments/sdk_calibration.py", ROOT / "tests_sdk/test_sdk_study.py",
        ROOT / "docs/SDK_STUDY_PROTOCOL.md"]
    return complete_source_hashes({source_key(p): sha256(p) for p in paths if p.exists()})


def build_config(data_path, start_row, seed, rfqs=3, rows=None):
    rules = KIT / "demo/demo_settlement_rules.csv"
    prices = (ReplayPrices.from_csv(data_path, rules, start_row=start_row) if rows is None
              else ReplayPrices(rows, {"A": "LAST_PRICE", "B": {"A": 2.0}}, start_row))
    products = [Product("A", prices.rows[0]["A"], max_long=50, max_short=50,
                        breach_penalty_per_unit=1.0, fee_per_unit=.01),
                Product("B", prices.rows[0]["B"], max_long=25, max_short=25,
                        breach_penalty_per_unit=2.0, fee_per_unit=.02)]
    return HostConfig(products=products, price_model=prices,
                      customer_model=RandomCustomers(rfqs, quantities=(2, 4, 10),
                                                     limit_offset=(.03, .30)),
                      interval_seconds=5., price_ticks_per_interval=10,
                      final_hedge_seconds=2., seed=seed)


class SynchronousSession:
    """Actual Engine plus actual SDK Dealer callbacks, acknowledged synchronously.

    A new market is delivered with the previous portfolio, matching Runner.
    Customer fills and final portfolio follow, before the hedge callback. Every
    trade/quote is passed through Engine.action and its original validations.
    """
    def __init__(self, config, dealer, opponents=2, opponent_seed=0):
        self.store = Store(":memory:")
        self.engine = Engine(config, self.store)
        self.virtual_time = 1_000_000_000.0
        self.engine.closes_at = self.virtual_time + config.interval_seconds
        opponent_class = load_class(KIT / "examples/dealer.py")
        self.dealers = {"dealer_1": dealer}
        # The participant's time-dependent deadline feature must advance at
        # official simulated time, not at the speed of this offline loop.
        # Replace only the research-loaded participant module's `time` binding;
        # the original Engine, Runner source and Python time module are intact.
        globals_ = getattr(getattr(dealer, "hedge", None), "__func__", None)
        if globals_ is not None and "time" in globals_.__globals__:
            globals_.__globals__["time"] = SimpleNamespace(time=lambda: self.virtual_time)
        for index in range(opponents):
            self.dealers[f"dealer_{index+2}"] = opponent_class(
                edge_range=[.03, .20], seed=opponent_seed + index + 1)
        self.local = {}
        self.markets = {}
        self.action_no = 0
        self.rejections = []
        self.observations = []
        self.exposure = []
        self.callback_ns = {}
        self.price_tape = []
        self.customer_tape = []
        self.pending_results = {}
        self.final_quote_counts = 0
        for name in self.dealers:
            self.engine.register(name)
            self.local[name] = self.engine.portfolio(name)

    def callback(self, name, method, **kwargs):
        fn = getattr(self.dealers[name], method, None)
        if fn is None:
            return None
        before = time.perf_counter_ns()
        try:
            return fn(**copy.deepcopy(kwargs))
        finally:
            if name == "dealer_1":
                self.callback_ns.setdefault(method, []).append(time.perf_counter_ns() - before)

    def _action(self, name, kind, **payload):
        self.action_no += 1
        events = self.engine.action(name, dict(type=kind, action_id=f"a{self.action_no}", **payload))
        for recipient, event in events:
            if recipient != name:
                continue
            if event["type"] == "portfolio":
                self.local[name] = event
            elif event["type"] == "fill":
                self.callback(name, "on_fill", fill=event, portfolio=self.engine.portfolio(name))
            elif event["type"] == "ack":
                if event["status"] != "accepted" or event.get("errors"):
                    self.rejections.append(dict(dealer=name, **event))
                for fill in event.get("executions", []):
                    self.callback(name, "on_fill", fill=fill, portfolio=self.local[name])
                self.callback(name, "on_ack", ack=event)

    def quote(self, name):
        market = self.markets[name]
        if not market["active_orders"]:
            return
        quotes = self.callback(name, "quote", orders=market["active_orders"],
                               market=market, portfolio=self.local[name])
        if quotes:
            self._action(name, "quotes", interval_id=market["interval_id"], quotes=quotes)

    def hedge(self, name):
        trades = self.callback(name, "hedge", market=self.markets[name], portfolio=self.local[name])
        if trades:
            self._action(name, "trades", price_version=self.markets[name]["price_version"],
                         trades=[dict(product_id=key, side=side, quantity=quantity)
                                 for key, side, quantity in trades])

    def accept_market(self, name, market, after_match=False):
        self.markets[name] = {**market, "products": {
            key: asdict(product) for key, product in self.engine.products.items()}}
        self.callback(name, "on_market", market=self.markets[name], portfolio=self.local[name])
        for order in market["active_orders"]:
            self.callback(name, "on_customer_order", order=order,
                          market=self.markets[name], portfolio=self.local[name])
        self.quote(name)
        if not after_match:
            self.hedge(name)

    def record_state(self):
        self.price_tape.append(dict(version=self.engine.price_version,
                                    prices=self.engine.prices.copy()))
        self.exposure.append(self.engine.accounts["dealer_1"]["positions"].copy())

    def tick(self):
        self.virtual_time += self.engine.tick_seconds
        self.engine.tick()
        for name in self.dealers:
            self.markets[name].update(prices=self.engine.prices.copy(),
                                      price_version=self.engine.price_version)
            self.quote(name)
            self.hedge(name)
        self.record_state()

    def advance(self, final):
        self.virtual_time += self.engine.tick_seconds
        self.engine.closes_at = self.virtual_time + (
            self.engine.config.final_hedge_seconds if final else self.engine.config.interval_seconds)
        orders = {oid: o.public() for oid, o in self.engine.orders.items()}
        reference = self.engine.prices.copy()
        own_quotes = {oid: self.engine.quotes.get(oid, {}).get("dealer_1") for oid in orders}
        # Public RFQ tape deliberately excludes hidden reservation/limit fields.
        self.customer_tape.extend(orders.values())
        events = self.engine.advance(final=final)
        result_event = next(e for _, e in events if e["type"] == "customer_results")
        for item in result_event["results"]:
            order = orders[item["order_id"]]
            sign = -1 if order["side"] == "BUY" else 1
            quote = own_quotes[item["order_id"]]
            self.observations.append(dict(
                **order, own_quote=quote, own_quote_acknowledged=True,
                matching_reference=reference[order["product_id"]],
                edge=None if quote is None else ((quote-reference[order["product_id"]]) * -sign),
                won=item["winner"] == "dealer_1", outcome=item["status"],
                public_winner=item["winner"], public_winning_price=item["price"],
                # Label read only after the current decision; never supplied as a feature.
                next_reference=self.engine.prices[order["product_id"]],
                dealer_signed_one_tick_markout=sign*(self.engine.prices[order["product_id"]]
                                                    - reference[order["product_id"]])))
        for recipient, event in events:
            kind = event["type"]
            if kind == "market":
                for name in self.dealers:
                    self.accept_market(name, event, after_match=True)
            elif kind == "fill":
                name = recipient
                p = copy.deepcopy(self.local[name])
                signed = event["quantity"] if event["side"] == "BUY" else -event["quantity"]
                product = self.engine.products[event["product_id"]]
                p["cash"] -= signed * event["price"] * product.multiplier + event["fee"]
                p["positions"][event["product_id"]] += signed
                p["customer_trades"] += 1
                self.local[name] = p
                self.callback(name, "on_fill", fill=event, portfolio=p)
            elif kind == "customer_results":
                for item in event["results"]:
                    for name in self.dealers:
                        self.callback(name, "on_customer_result", result={
                            "type": "customer_result", **item, "traded": item["winner"] == name})
            elif kind == "portfolio" and recipient in self.dealers:
                # Use original boundary portfolio, not a later action's mutated account.
                self.local[recipient] = event
                self.hedge(recipient)
        self.record_state()

    def run(self, intervals=12):
        for name in self.dealers:
            self.accept_market(name, self.engine.market())
        self.record_state()
        for interval in range(intervals):
            for _ in range(self.engine.config.price_ticks_per_interval - 1):
                self.tick()
            self.advance(final=interval == intervals-1)
        # ExchangeService.step_clock checks deadline before a tick at that deadline.
        closing_ticks = max(0, math.ceil(self.engine.config.final_hedge_seconds /
                                       self.engine.tick_seconds) - 1)
        for _ in range(closing_ticks):
            self.tick()
        final_reference = self.engine.prices.copy()
        self.virtual_time = self.engine.closes_at
        self.engine.finish()
        account = self.engine.portfolio("dealer_1")
        ledger = reconcile(self.engine, "dealer_1")
        position_array = np.array([[p[key] for key in self.engine.products] for p in self.exposure])
        stats = dict(account=account, reconciliation=ledger,
                     final_reference=final_reference, final_settlement=self.engine.prices.copy(),
                     rfqs=len(self.observations), rejections=len(self.rejections),
                     max_abs_inventory=[max([abs(p[key]) for p in self.exposure] + [
                         abs(t["position_after"]) for t in self.engine.trades
                         if t["dealer_id"] == "dealer_1" and t["product_id"] == key])
                         for key in self.engine.products],
                     mean_squared_inventory=np.mean(position_array**2, axis=0).tolist(),
                     mean_squared_settlement_factor_exposure=float(np.mean(
                         (position_array[:, 0] + 2*position_array[:, 1])**2)),
                     price_rows_consumed=self.engine.price_state+1,
                     price_tape_sha256=hashlib.sha256(json.dumps(self.price_tape, sort_keys=True).encode()).hexdigest(),
                     public_customer_tape_sha256=hashlib.sha256(json.dumps(self.customer_tape, sort_keys=True).encode()).hexdigest(),
                     callback_seconds={key: dict(count=len(values), maximum=max(values)/1e9,
                                                mean=float(np.mean(values))/1e9)
                                       for key, values in self.callback_ns.items()})
        self.store.close()
        return stats


def reconcile(engine, name):
    """Independent arithmetic replay of actual executed trades and charged penalties."""
    cash = engine.config.starting_cash
    positions = {key: 0. for key in engine.products}
    fees = 0.
    for trade in engine.trades:
        if trade["dealer_id"] != name:
            continue
        signed = trade["quantity"] if trade["side"] == "BUY" else -trade["quantity"]
        product = engine.products[trade["product_id"]]
        cash -= signed * trade["price"] * product.multiplier + trade["fee"]
        positions[trade["product_id"]] += signed
        fees += trade["fee"]
    account = engine.accounts[name]
    cash -= account["penalties_paid"]
    pnl = cash + sum(positions[key]*engine.prices[key]*engine.products[key].multiplier
                     for key in positions) - engine.config.starting_cash
    difference = max(abs(cash-account["cash"]), abs(fees-account["fees_paid"]),
                     *(abs(positions[k]-account["positions"][k]) for k in positions),
                     abs(pnl-engine.portfolio(name)["pnl"]))
    if difference > 1e-7:
        raise AssertionError("authoritative ledger fails independent arithmetic replay")
    return dict(cash=cash, positions=positions, fees=fees, pnl=pnl,
                max_absolute_difference=difference,
                penalty_source="authoritative charged total; detailed penalty timing tested separately")


def shared_parameters():
    return dict(total_intervals=12, settlement_coefficients={"B": {"A": 2.}},
                settlement_constants={}, price_drift=None, forecast_weight=0.)


def candidate_grid(method):
    result = []
    for index, (edge, skew) in enumerate(itertools.product([.04, .08, .12, .16], [0., .002, .005])):
        result.append(dict(mode="B3" if method == "B3_factor" else method,
                           factor_hedge=method == "B3_factor", base_edge=edge, inventory_skew=skew,
                           band_fraction=[.2, .4, .6][index % 3],
                           target_fraction=[0., .5][(index // 3) % 2], **shared_parameters()))
    return result


def one_session(method, params, data, block, seed, collect=False, rows=None):
    factory = (load_class(KIT / "demo/demo_participant.py") if method == "B0"
               else load_class(ROOT / "submission/sdk_dealer.py"))
    dealer = factory(**({} if method == "B0" else params))
    runner = SynchronousSession(build_config(data, block*124, seed, rows=rows), dealer,
                                opponent_seed=seed+1)
    try:
        stats = runner.run()
        stats.update(method=method, block=block, seed=seed, parameters=params, crash=False)
        if collect:
            return stats, runner.observations, runner.engine.trades, runner.rejections
        return stats, [], [], []
    except Exception:
        runner.store.close()
        return dict(method=method, block=block, seed=seed, parameters=params, crash=True,
                    traceback=traceback.format_exc()), [], [], []


def mean_pnl(rows):
    if any(r["crash"] for r in rows):
        return -math.inf
    return float(np.mean([r["account"]["pnl"] for r in rows]))


def block_differences(rows, proposed, baseline):
    lookup = {(r["method"], r["block"], r["seed"]): r for r in rows}
    result = []
    for block in sorted({r["block"] for r in rows}):
        differences = []
        for seed in sorted({r["seed"] for r in rows if r["block"] == block}):
            a, b = lookup[proposed, block, seed], lookup[baseline, block, seed]
            if a["crash"] or b["crash"]:
                raise ValueError("A crash forbids an unqualified paired performance result")
            if (a["price_tape_sha256"], a["public_customer_tape_sha256"]) != (
                    b["price_tape_sha256"], b["public_customer_tape_sha256"]):
                raise AssertionError("paired policies have different exogenous price/RFQ histories")
            differences.append(a["account"]["pnl"]-b["account"]["pnl"])
        result.append(dict(block=block, mean_difference=float(np.mean(differences)),
                           seed_differences=differences))
    return result


def infer(differences):
    x = np.array([row["mean_difference"] for row in differences])
    null = np.array([np.mean(x*np.array(signs)) for signs in itertools.product([-1, 1], repeat=len(x))])
    p = float(np.mean(null >= x.mean()-1e-12))
    rng = np.random.default_rng(7301)
    boot = np.mean(rng.choice(x, (20000, len(x)), replace=True), axis=1)
    ci = np.quantile(boot, [.025, .975]).tolist()
    return dict(independent_price_blocks=len(x), paired_mean_difference=float(x.mean()),
                one_sided_block_sign_randomisation_p=p, descriptive_block_bootstrap_95=ci,
                worst_block_difference=float(x.min()), block_differences=differences,
                materiality_threshold=1., meets_predeclared_superiority_gate=bool(
                    x.mean() >= 1. and p <= .05 and ci[0] > 0),
                assumptions="independent representative blocks; sign exchangeability/symmetry; adjacent-path dependence untested")


def append_jsonl(handle, row):
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def run_many(method, params, data, blocks, seeds, raw, obs=None):
    result = []
    for block, seed in itertools.product(blocks, seeds):
        row, observations, _, _ = one_session(method, params, data, block, seed, collect=obs is not None)
        append_jsonl(raw, row)
        result.append(row)
        if obs is not None:
            for observation in observations:
                append_jsonl(obs, dict(method=method, block=block, seed=seed,
                    policy_id=hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12],
                    **observation))
    return result


def manifests(out):
    write_json(out / "SHA256SUMS.json", {str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"})


def pilot(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = KIT / "demo/demo_data.csv"
    write_json(out / "input_audit.json", inputs_audit())
    write_json(out / "source_hashes.json", source_hashes())
    write_json(out / "environment.json", dict(python=sys.version, numpy=np.__version__, platform=platform.platform()))
    write_json(out / "config.json", dict(protocol_sha256=sha256(ROOT / "docs/SDK_STUDY_PROTOCOL.md"),
        tuning_blocks=list(range(6)), validation_blocks=[6, 7, 8], tuning_seeds=[11, 29],
        validation_seeds=[101, 211, 307, 401], candidates_per_method=12,
        materiality=1., heldout_seeds=[1009, 2017, 3011, 4001],
        engine="unmodified supplied Engine", execution="synchronous zero latency with logical participant clock and official deadlines", grids={
            m: candidate_grid(m) for m in ("B1", "B2", "B3", "B3_factor", "R2")}))
    selected, tuning_summary, validation = {}, [], []
    with (out / "tuning.jsonl").open("w") as raw, (out / "rfq_train.jsonl").open("w") as obs:
        # B0 is a frozen reference and independent collection policy, not tuned.
        b0_train = run_many("B0", {}, data, range(6), [11, 29], raw, obs)
        for method in ("B1", "B2", "B3", "B3_factor", "R2"):
            trials = []
            for index, params in enumerate(candidate_grid(method)):
                rows = run_many(method, params, data, range(6), [11, 29], raw,
                                obs if method == "B1" else None)
                value = mean_pnl(rows)
                trials.append((value, -index, params))
                tuning_summary.append(dict(method=method, candidate=index, params=params,
                    mean_pnl=None if not math.isfinite(value) else value,
                    crashes=sum(r["crash"] for r in rows), sessions=len(rows)))
            value, _, selected[method] = max(trials, key=lambda x: (x[0], x[1]))
            if not math.isfinite(value):
                raise RuntimeError(f"all {method} candidates failed")
    write_json(out / "tuning_summary.json", tuning_summary)
    with (out / "validation.jsonl").open("w") as raw, (out / "rfq_validation.jsonl").open("w") as obs:
        validation.extend(run_many("B0", {}, data, [6, 7, 8], [101, 211, 307, 401], raw, obs))
        for method in ("B1", "B2", "B3", "B3_factor", "R2"):
            validation.extend(run_many(method, selected[method], data, [6, 7, 8], [101, 211, 307, 401], raw,
                                       obs if method == "B1" else None))
        fixed = {**selected["R2"], "base_edge": selected["B3"]["base_edge"],
                 "inventory_skew": selected["B3"]["inventory_skew"]}
        selected["R2_fixed_quotes"] = fixed
        validation.extend(run_many("R2_fixed_quotes", fixed, data, [6, 7, 8], [101, 211, 307, 401], raw))
    scores = {m: mean_pnl([r for r in validation if r["method"] == m]) for m in ("B1", "B2", "B3", "B3_factor")}
    from experiments.sdk_calibration import calibrate
    calibrate(out / "rfq_train.jsonl", out / "rfq_validation.jsonl", out / "calibration.json")
    fallback = max(scores, key=lambda m: scores[m])
    differences = block_differences(validation, "R2", fallback)
    sd = float(np.std([r["mean_difference"] for r in differences], ddof=1))
    required = max(1, math.ceil(((1.645+.842)*sd/1.)**2))
    seal = dict(selected=selected, fallback=fallback, validation_means=scores,
                pilot_paired_block_sd=sd, approximate_required_independent_blocks=required,
                available_heldout_blocks=9, pilot_indicates_insufficient_power=required>9,
                primary_comparison=f"R2 minus {fallback}", materiality=1.,
                source_hashes=source_hashes(), input_audit=inputs_audit(),
                protocol_sha256=sha256(ROOT / "docs/SDK_STUDY_PROTOCOL.md"),
                predictive_calibration_sha256=sha256(out / "calibration.json"),
                elapsed_seconds=time.perf_counter()-started,
                holdout_numerical_values_opened=False)
    write_json(out / "POLICY_SEAL.json", seal)
    write_json(out / "pilot_inference.json", infer(differences))
    manifests(out)
    print(json.dumps(dict(stage="pilot", fallback=fallback, required_blocks=required,
                         elapsed_seconds=seal["elapsed_seconds"]), indent=2))


def final(out, seal_path):
    seal = json.loads(seal_path.read_text())
    if seal["source_hashes"] != source_hashes():
        raise RuntimeError("Source changed after pilot seal; repeat prospective training before holdout")
    if seal["input_audit"] != inputs_audit():
        raise RuntimeError("Official inputs changed after seal")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "POLICY_SEAL.json", seal)
    write_json(out / "seal_verification.json", dict(seal_sha256=sha256(seal_path), verified=True,
               holdout_first_numerical_read_after_seal=True))
    data = KIT / "demo/demo_data_holdout.csv"
    rows = []
    with ((out / "heldout.jsonl").open("w") as raw,
          (out / "rfq_heldout.jsonl").open("w") as obs,
          (out / "trades_heldout.jsonl").open("w") as trades):
        for method in ("B0", "B1", "B2", "B3", "B3_factor", "R2", "R2_fixed_quotes"):
            params = {} if method == "B0" else seal["selected"][method]
            for block, seed in itertools.product(range(9), [1009, 2017, 3011, 4001]):
                row, observations, executions, rejects = one_session(method, params, data, block, seed, collect=True)
                rows.append(row)
                append_jsonl(raw, row)
                for observation in observations:
                    append_jsonl(obs, dict(method=method, block=block, seed=seed, **observation))
                for execution in executions:
                    append_jsonl(trades, dict(method=method, block=block, seed=seed, **execution))
                if rejects:
                    with (out / "rejections.jsonl").open("a") as errors:
                        for reject in rejects:
                            append_jsonl(errors, dict(method=method, block=block, seed=seed, **reject))
    summary = {m: dict(sessions=len(rs), crashes=sum(r["crash"] for r in rs),
        mean_pnl=mean_pnl(rs) if not any(r["crash"] for r in rs) else None,
        customer_orders=sum(r.get("account", {}).get("customer_trades", 0) for r in rs),
        customer_notional=sum(r.get("account", {}).get("customer_notional", 0) for r in rs),
        exchange_notional=sum(r.get("account", {}).get("exchange_notional", 0) for r in rs),
        fees=sum(r.get("account", {}).get("fees_paid", 0) for r in rs),
        penalties=sum(r.get("account", {}).get("penalties_paid", 0) for r in rs),
        rejections=sum(r.get("rejections", 0) for r in rs))
        for m in ("B0", "B1", "B2", "B3", "B3_factor", "R2", "R2_fixed_quotes")
        for rs in [[r for r in rows if r["method"] == m]]}
    for method, result in summary.items():
        valid = [r for r in rows if r["method"] == method and not r["crash"]]
        result["metrics_status"] = "complete" if len(valid) == result["sessions"] else "incomplete_no_crash_exclusion"
        if result["metrics_status"] == "complete":
            pnl = np.array([r["account"]["pnl"] for r in valid])
            result.update(worst_session_pnl=float(pnl.min()),
                          descriptive_session_pnl_quantiles=np.quantile(pnl, [.05, .5, .95]).tolist(),
                          negative_pnl_sessions=int(np.sum(pnl < 0)),
                          block_mean_pnl=[float(np.mean([r["account"]["pnl"] for r in valid
                                                       if r["block"] == block])) for block in range(9)],
                          mean_squared_inventory=np.mean([r["mean_squared_inventory"] for r in valid], axis=0).tolist(),
                          mean_squared_settlement_factor_exposure=float(np.mean([
                              r["mean_squared_settlement_factor_exposure"] for r in valid])))
    write_json(out / "summary.json", summary)
    if any(r["crash"] for r in rows):
        write_json(out / "primary_inference.json", dict(status="NOT_SUPPORTED", reason="crashes preserved; no excluded-session performance inference"))
        manifests(out)
        return
    write_json(out / "primary_inference.json", infer(block_differences(rows, "R2", seal["fallback"])))
    write_json(out / "ablation_inference.json", {
        "R2_minus_B3": infer(block_differences(rows, "R2", "B3")),
        "R2_fixed_quotes_minus_B3": infer(block_differences(rows, "R2_fixed_quotes", "B3"))})
    manifests(out)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "pilot", "final"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    if args.stage == "audit":
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "input_audit.json", inputs_audit())
    elif args.stage == "pilot":
        pilot(args.out)
    else:
        if args.seal is None:
            parser.error("final requires --seal")
        final(args.out, args.seal)


if __name__ == "__main__":
    main()
