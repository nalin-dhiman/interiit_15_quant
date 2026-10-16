"""Controlled boundary-controller contract replay through unchanged SDK methods.

Only transport and Worker are adapted; no claim of production latency is made.
Rejection/partial cases explicitly inject request faults and preserve them in
the trace. Those are robustness fixtures, not an invented exchange fill law.
"""
from __future__ import annotations
from experiments.sdk_environment import KIT, complete_source_hashes, source_key, source_path

import argparse
import asyncio
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

from experiments.sdk_async_replay import (ControlledRunner, ControlledSocket,
                                          Replay, plain)
from rfq.engine import Engine
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.store import Store

ROOT = Path(__file__).resolve().parents[1]


class PublicFixtureOrders:
    def __init__(self, product_ids, per_product=3):
        self.product_ids = tuple(product_ids)
        self.per_product = per_product

    def generate(self, interval_id, market, state, rng):
        orders = []
        for pid in self.product_ids:
            for index in range(self.per_product):
                orders.append(CustomerOrder(f"r{interval_id}-{pid}-{index}", "fixture", pid,
                                            "BUY" if index % 2 == 0 else "SELL", 4.,
                                            interval_id, limit_offset=1.))
        return state, orders


class BoundaryReplay(Replay):
    def __init__(self, dealer, products=1, initial_quantity=20., per_product=3):
        self.dealer = dealer
        self.store = Store(":memory:")
        names = [f"P{index:04d}" for index in range(products)]
        prices = {pid: 100. + index % 20 for index, pid in enumerate(names)}
        specs = [Product(pid, prices[pid], max_long=50., max_short=50.,
                         multiplier=float(1 + index % 3), fee_per_unit=.01,
                         fee_rate=.00001, breach_penalty_per_unit=1.)
                 for index, pid in enumerate(names)]
        config = HostConfig(products=specs, price_model=ReplayPrices([prices] * 100),
                            customer_model=PublicFixtureOrders(names, per_product), seed=173)
        self.engine = Engine(config, self.store)
        self.engine.register("dealer_1")
        if initial_quantity:
            for offset in range(0, products, 500):
                trades = [{"product_id": pid, "side": "BUY" if index % 2 else "SELL",
                           "quantity": float(initial_quantity)}
                          for index, pid in enumerate(names[offset:offset + 500], start=offset)]
                self.engine.action("dealer_1", dict(type="trades", action_id=f"initial-{offset}",
                                                     price_version=1, trades=trades))
        self.runner = ControlledRunner(dealer, self.engine)
        self.socket = ControlledSocket()
        self.trace, self.task = [], None
        self.cursor = 0

    async def bootstrap(self):
        await self.deliver([("dealer_1", self.engine.snapshot("dealer_1"))])

    def take_actions(self):
        actions = copy.deepcopy(self.socket.sent[self.cursor:])
        self.cursor = len(self.socket.sent)
        return actions

    async def drain_actions(self, limit=100):
        count = 0
        while True:
            actions = self.take_actions()
            if not actions:
                return count
            for action in actions:
                count += 1
                if count > limit:
                    raise AssertionError("unbounded action drain")
                await self.deliver(self.execute(action))

    async def tick(self):
        await self.deliver(self.engine.tick())

    def positive_quote_messages(self, start=0):
        return [action for action in self.socket.sent[start:]
                if action["type"] == "quotes" and any(
                    price is not None for price in action["quotes"].values())]

    def reconcile(self):
        result = super().reconcile()
        actual, observed = result["account"], self.runner.portfolio
        for key in ("cash", "fees_paid", "penalties_paid", "customer_trades", "exchange_trades"):
            result["errors"]["runner_" + key] = observed[key] - actual[key]
        for pid, quantity in actual["positions"].items():
            result["errors"]["runner_position_" + pid] = observed["positions"][pid] - quantity
        result["reconciled"] = all(abs(value) < 1e-8 for value in result["errors"].values())
        return result


def make_dealer(mode="B3", **kwargs):
    from submission.boundary_dealer import Dealer
    return Dealer(mode=mode, boundary_reset=True, band_fraction=.2,
                  target_fraction=.5, total_intervals=3, **kwargs)


async def run_scenarios():
    results = {}

    async def capture(name, replay, scenario):
        await replay.start()
        try:
            details = await scenario(replay)
            reconciliation = replay.reconcile()
            if not reconciliation["reconciled"]:
                raise AssertionError(name + ": account mismatch")
            results[name] = dict(details=details, reconciliation=reconciliation,
                                 trace=replay.trace, sent_actions=replay.socket.sent,
                                 calls=replay.runner.worker.calls)
        finally:
            await replay.stop()

    async def normal(r):
        await r.bootstrap()
        assert not r.positive_quote_messages()
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages()
        before = len(r.socket.sent)
        await r.deliver(r.engine.advance())
        # New interval quotes are withdrawn until the boundary hedge response.
        assert not r.positive_quote_messages(before)
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages(before)
        await r.deliver(r.engine.advance(final=True))
        await r.drain_actions()
        await r.deliver(r.engine.finish())
        return dict(ordinary_order=True, quote_after_confirmed_reset=True,
                    closing_no_customer_quotes=True)

    async def delayed(r):
        await r.bootstrap()
        pending = r.take_actions()
        trades = [a for a in pending if a["type"] == "trades"]
        assert len(trades) == 1
        for action in pending:
            if action["type"] != "trades":
                await r.deliver(r.execute(action))
        before = len(r.socket.sent)
        await r.tick()
        assert not r.positive_quote_messages(before)
        assert r.runner.trades_in_flight()
        await r.deliver(r.execute(trades[0]))
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages(before)
        return dict(delayed_reset_ack_withheld_capacity=True,
                    resumed_on_later_price_callback=True)

    async def stale(r):
        await r.bootstrap()
        pending = r.take_actions()
        trade = next(a for a in pending if a["type"] == "trades")
        for action in pending:
            if action["type"] != "trades":
                await r.deliver(r.execute(action))
        # Exogenous price versions advance while the correctly formed original
        # instruction is delayed; this is not a synthetic trade-price rewrite.
        for _ in range(r.engine.config.max_trade_lag_versions + 1):
            await r.tick()
        events = r.execute(trade)
        assert events[-1][1]["code"] == "STALE_PRICE"
        await r.deliver(events)
        before = len(r.socket.sent)
        await r.tick()
        await r.drain_actions()
        assert not r.positive_quote_messages(before)
        return dict(rejection="STALE_PRICE", uncorrelated_rejection_fails_closed=True)

    async def partial(r):
        await r.bootstrap()
        pending = r.take_actions()
        trade = next(a for a in pending if a["type"] == "trades")
        assert len(trade["trades"]) == 2
        # Explicit request fault: actual static demo has no depth-based partial
        # quantities. One invalid item exercises genuine indexed rejection.
        faulted = copy.deepcopy(trade)
        faulted["trades"][1]["quantity"] = .5
        for action in pending:
            if action["type"] != "trades":
                await r.deliver(r.execute(action))
        events = r.execute(faulted)
        ack = events[-1][1]
        assert ack["errors"] == {"1": "BAD_LOT"}
        assert len(ack["executions"]) == 1
        await r.deliver(events)
        before = len(r.socket.sent)
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages(before)
        return dict(injected_fault="second requested quantity changed to .5",
                    partial_batch_errors=ack["errors"],
                    admission_uses_actual_accepted_subset=True)

    async def duplicate(r):
        await r.bootstrap()
        pending = r.take_actions()
        trade = next(a for a in pending if a["type"] == "trades")
        for action in pending:
            await r.deliver(r.execute(action))
        cash = r.engine.portfolio("dealer_1")["cash"]
        duplicate_events = r.execute(trade)
        assert duplicate_events[-1][1]["duplicate"]
        await r.deliver(duplicate_events)
        assert r.engine.portfolio("dealer_1")["cash"] == cash
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages()
        return dict(duplicate_ack_books_cash_once=True)

    async def lost_results(r):
        await r.bootstrap()
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        assert r.positive_quote_messages()
        r.engine.advance()  # Committed boundary events lost during outage.
        r.engine.withdraw("dealer_1")
        r.runner.sent_quotes = {}
        before = len(r.socket.sent)
        await r.deliver([("dealer_1", r.engine.snapshot("dealer_1"))])
        await r.tick()
        await r.drain_actions()
        assert not r.positive_quote_messages(before)
        return dict(snapshot_has_current_inventory=True, missing_results_not_assumed_expired=True,
                    unsupported_reconnect_fails_closed=True)

    await capture("normal_BB3", BoundaryReplay(make_dealer()), normal)
    await capture("normal_BR2", BoundaryReplay(make_dealer("R2")), normal)
    await capture("delayed_reset_ack", BoundaryReplay(make_dealer()), delayed)
    await capture("stale_reset_rejection", BoundaryReplay(make_dealer()), stale)
    await capture("partial_reset_batch", BoundaryReplay(make_dealer(), products=2), partial)
    await capture("duplicate_ack", BoundaryReplay(make_dealer()), duplicate)
    await capture("reconnect_missing_results", BoundaryReplay(make_dealer()), lost_results)
    return results


def source_hashes():
    files = [Path(__file__), ROOT / "submission/boundary_dealer.py",
             ROOT / "experiments/sdk_async_replay.py"]
    files += [KIT / "rfq/rfq" / (name + ".py")
              for name in ("runner", "engine", "exchange", "sdk", "models", "store")]
    return complete_source_hashes({source_key(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before, log, started = source_hashes(), io.StringIO(), time.perf_counter()
    with contextlib.redirect_stdout(log):
        scenarios = asyncio.run(run_scenarios())
    result = dict(scope="unchanged SDK Runner receive/callback and Engine; controlled transport/Worker",
                  elapsed_seconds=time.perf_counter() - started,
                  environment=dict(python=sys.version, platform=platform.platform()),
                  source_hashes=before, source_unchanged=before == source_hashes(),
                  scenarios=scenarios, economic_performance_evidence=False)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "runner_stdout.log").write_text(log.getvalue())
    print(json.dumps(dict(scenarios=len(scenarios), source_unchanged=result["source_unchanged"],
                         all_reconciled=all(r["reconciliation"]["reconciled"] for r in scenarios.values()))))


if __name__ == "__main__":
    main()
