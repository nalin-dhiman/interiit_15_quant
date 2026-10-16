"""Controlled transport replay of the unchanged supplied Runner and Engine.

Only the network delivery and worker process are replaced by explicit adapters.
DealerRunner.receive/callback/send_action/hedge and Engine.action/advance/finish
are inherited unchanged. This is a deterministic contract test, not a latency
model or a substitute for real Worker/WebSocket certification.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
from experiments.sdk_environment import KIT, complete_source_hashes, require_sdk, source_key, source_path
require_sdk()
sys.path.insert(0, str(KIT / "rfq"))

from rfq.engine import Engine, MAX_BATCH
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.runner import CHUNK, DealerRunner
from rfq.sdk import Customers, Exchange, quotes_action, trades_action
from rfq.store import Store, encode


def plain(value):
    """Use the SDK serializer, removing shared-object aliases at the boundary."""
    return json.loads(encode(value))


class FixedOrders:
    """Explicit test orders; hidden eligibility stays inside the Engine."""
    def __init__(self, orders):
        self.orders = tuple(orders)

    def generate(self, interval_id, market, state, rng):
        return state, [CustomerOrder(oid, "fixture", "A", side, quantity,
                                     interval_id, limit_offset=1.)
                       for oid, side, quantity in self.orders]


class ScriptedDealer:
    """Fixture participant with callback outputs chosen before transport."""
    def __init__(self):
        self.quote_outputs = deque()
        self.hedge_outputs = deque()
        self.acks, self.fills, self.results, self.markets = [], [], [], []

    def quote(self, orders, market, portfolio):
        return self.quote_outputs.popleft() if self.quote_outputs else {}

    def hedge(self, market, portfolio):
        return self.hedge_outputs.popleft() if self.hedge_outputs else []

    def on_market(self, market, portfolio, exchange):
        self.markets.append(plain(market))

    def on_fill(self, fill, portfolio, exchange):
        self.fills.append({"fill": plain(fill), "portfolio": plain(portfolio)})

    def on_customer_result(self, result):
        self.results.append(plain(result))

    def on_ack(self, ack):
        self.acks.append(plain(ack))


class LocalWorker:
    """Worker interface adapter with the SDK's exact output conversions.

    No process sandbox, timeout enforcement, or multiprocessing performance is
    claimed. Parent execution separately runs the actual Worker/WebSocket path.
    """
    def __init__(self, dealer):
        self.dealer, self.timeout = dealer, 1.
        self.methods = {name for name in ("quote", "hedge", "on_market", "on_fill",
                                         "on_customer_result", "on_ack", "on_timer")
                        if callable(getattr(dealer, name, None))}
        self.calls = []

    def call(self, method, data):
        started = time.perf_counter_ns()
        actions = []
        data = plain(data)
        fn = getattr(self.dealer, method)
        exchange, customers = Exchange(actions), Customers(actions)
        if method == "quote":
            returned = quotes_action(fn(data["orders"], data["market"], data["portfolio"]))
        elif method == "hedge":
            returned = trades_action(fn(data["market"], data["portfolio"]))
        elif method == "on_market":
            returned = fn(data["market"], data["portfolio"], exchange)
        elif method == "on_fill":
            returned = fn(data["fill"], data["portfolio"], exchange)
        elif method == "on_customer_result":
            returned = fn(data["result"])
        elif method == "on_ack":
            returned = fn(data["ack"])
        elif method == "on_timer":
            returned = fn(data["market"], data["portfolio"], exchange, customers)
        else:
            raise AssertionError(method)
        if returned:
            actions.extend(returned)
        # Real strategy_worker validates with encode(), then sends through Pipe;
        # dict insertion order survives until Runner chunks it. Do not sort here.
        encode(actions)
        actions = copy.deepcopy(actions)
        self.calls.append({"method": method, "data": data, "actions": actions,
                           "elapsed_ns": time.perf_counter_ns() - started})
        return actions


class ControlledSocket:
    def __init__(self):
        self.sent, self.incoming = [], asyncio.Queue()

    async def send(self, raw):
        self.sent.append(json.loads(raw))


class ControlledRunner(DealerRunner):
    """Override construction and messages delivery, preserving callback logic."""
    def __init__(self, dealer, engine):
        self.args = SimpleNamespace(id="dealer_1")
        self.worker = LocalWorker(dealer)
        snapshot = engine.snapshot("dealer_1")
        self.products, self.portfolio = snapshot["products"], snapshot["portfolio"]
        self.market = {**snapshot["market"], "products": self.products}
        self.pending, self.sent_quotes = {}, {}
        self.hedge_wanted = self.hedge_after_fills = False
        self.slow_warnings = self.skipped_updates = 0
        self.session_id = snapshot["session_id"]
        self.last_reported_interval, self.running = None, True
        self.callback_lock = asyncio.Lock()
        self.inbox = deque()

    async def messages(self, websocket):
        while True:
            event, completed = await websocket.incoming.get()
            try:
                yield event
            finally:
                if not completed.done():
                    completed.set_result(None)


class Replay:
    def __init__(self, dealer=None, orders=(("x", "BUY", 2.), ("y", "SELL", 3.)),
                 limit=20., multiplier=3., fee_rate=.001):
        self.dealer = dealer or ScriptedDealer()
        self.store = Store(":memory:")
        config = HostConfig(
            products=[Product("A", 100., multiplier=multiplier, max_long=limit,
                              max_short=limit, fee_per_unit=.02, fee_rate=fee_rate,
                              breach_penalty_per_unit=2.)],
            price_model=ReplayPrices([{"A": 100.}] * 8),
            customer_model=FixedOrders(orders), seed=123)
        self.engine = Engine(config, self.store)
        self.engine.register("dealer_1")
        self.runner = ControlledRunner(self.dealer, self.engine)
        self.socket = ControlledSocket()
        self.trace = []
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.runner.receive(self.socket))
        return self

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
        self.store.close()

    async def deliver(self, events):
        for recipient, event in events:
            if recipient not in ("*", "dealer_1"):
                continue
            self.trace.append({"delivery": plain(event)})
            completed = asyncio.get_running_loop().create_future()
            await self.socket.incoming.put((plain(event), completed))
            done, _ = await asyncio.wait((completed, self.task), timeout=2.,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError("controlled delivery did not complete")
            if self.task in done:
                self.task.result()

    async def quote(self, quotes):
        self.dealer.quote_outputs.append(quotes)
        before = len(self.socket.sent)
        await self.runner.callback(self.socket, "quote", orders=self.runner.market["active_orders"],
                                   market=self.runner.market, portfolio=self.runner.portfolio)
        messages = copy.deepcopy(self.socket.sent[before:])
        self.trace.append({"quote_request": plain(quotes), "messages": messages})
        return messages

    async def hedge(self, trades):
        self.dealer.hedge_outputs.append(trades)
        before = len(self.socket.sent)
        await self.runner.hedge(self.socket)
        messages = copy.deepcopy(self.socket.sent[before:])
        self.trace.append({"hedge_request": plain(trades), "messages": messages})
        return messages

    def execute(self, action):
        self.trace.append({"exchange_receives": plain(action)})
        return self.engine.action("dealer_1", plain(action))

    def reconcile(self):
        trades = [t for t in self.engine.trades if t["dealer_id"] == "dealer_1"]
        cash = self.engine.config.starting_cash
        positions = {pid: 0. for pid in self.engine.products}
        fees = 0.
        for trade in trades:
            signed = trade["quantity"] * (1 if trade["side"] == "BUY" else -1)
            multiplier = self.engine.products[trade["product_id"]].multiplier
            cash -= signed * trade["price"] * multiplier + trade["fee"]
            positions[trade["product_id"]] += signed
            fees += trade["fee"]
        penalties = sum(json.loads(payload)["charge"] for (payload,) in self.store.db.execute(
            "SELECT payload FROM events WHERE recipient='dealer_1' AND json_extract(payload,'$.type')='penalty'"))
        cash -= penalties
        account = self.engine.portfolio("dealer_1")
        errors = {"cash": cash - account["cash"], "fees": fees - account["fees_paid"],
                  "penalties": penalties - account["penalties_paid"]}
        errors.update({"position_" + pid: positions[pid] - account["positions"][pid]
                       for pid in positions})
        equity = cash + sum(positions[p] * self.engine.prices[p] * self.engine.products[p].multiplier
                            for p in positions)
        errors["equity"] = equity - account["equity"]
        return {"errors": errors, "reconciled": all(abs(e) < 1e-8 for e in errors.values()),
                "account": account, "trade_count": len(trades)}


async def run_scenarios():
    results = {}

    async def capture(name, replay, scenario):
        await replay.start()
        try:
            details = await scenario(replay)
            result = {"details": details, "reconciliation": replay.reconcile(),
                      "trace": replay.trace, "callback_calls": replay.runner.worker.calls}
            if not result["reconciliation"]["reconciled"]:
                raise AssertionError(name + ": cash/inventory discrepancy")
            results[name] = result
        finally:
            await replay.stop()

    async def partial(r):
        first = (await r.quote({"x": 100.10, "y": 99.90}))[0]
        await r.deliver(r.execute(first))
        replacement = (await r.quote({"x": 100.20, "y": 99.905}))[0]
        events = r.execute(replacement)
        ack = events[-1][1]
        assert ack["errors"] == {"y": "BAD_TICK"}
        assert r.engine.quotes["y"]["dealer_1"] == 99.90
        assert r.engine.quotes["x"]["dealer_1"] == 100.20
        await r.deliver(events)
        repeated = await r.quote({"x": 100.20, "y": 99.905})
        assert repeated == []
        await r.deliver(r.engine.advance(final=True))
        await r.deliver(r.engine.finish())
        return {"partial_errors": ack["errors"], "rejected_request_repeat_suppressed": True,
                "actual_customer_prices": [t["price"] for t in r.engine.trades],
                "ack_has_original_quotes": "quotes" in ack}

    async def delayed(r):
        initial = (await r.quote({"x": 100.10}))[0]
        await r.deliver(r.execute(initial))
        pending = (await r.quote({"x": 100.20}))[0]
        # Delay request delivery itself until after the matching boundary.
        await r.deliver(r.engine.advance())
        events = r.execute(pending)
        assert events[-1][1]["code"] == "STALE_QUOTE"
        await r.deliver(events)
        assert r.engine.trades[0]["price"] == 100.10
        await r.deliver(r.engine.finish())
        return {"matched_old_price": r.engine.trades[0]["price"],
                "late_replacement_code": events[-1][1]["code"]}

    async def withdrawal(r):
        initial = (await r.quote({"x": 100.10, "y": 99.90}))[0]
        await r.deliver(r.execute(initial))
        assert await r.quote({}) == []
        pending = (await r.quote({"x": None}))[0]
        assert "dealer_1" in r.engine.quotes["x"]
        await r.deliver(r.execute(pending))
        assert "dealer_1" not in r.engine.quotes["x"]
        assert "dealer_1" in r.engine.quotes["y"]
        await r.deliver(r.engine.advance(final=True))
        await r.deliver(r.engine.finish())
        return {"omission_preserved_quotes": True, "withdrawal_effective_after_exchange_processing": True,
                "filled_order_ids": [t["order_id"] for t in r.engine.trades]}

    async def reconnect_duplicate(r):
        action = (await r.quote({"x": 100.10}))[0]
        lost_ack = r.execute(action)
        assert "dealer_1" in r.engine.quotes["x"]
        # Explicit source-equivalent disconnect/replay, not actual run() networking.
        r.engine.withdraw("dealer_1")
        r.runner.sent_quotes = {}
        duplicate_events = r.execute(action)
        ack = duplicate_events[0][1]
        assert ack["duplicate"] and ack["status"] == "accepted"
        assert "dealer_1" not in r.engine.quotes["x"]
        await r.deliver(duplicate_events)
        await r.deliver(r.engine.advance(final=True))
        await r.deliver(r.engine.finish())
        return {"quote_ack_lost": True, "duplicate_accepted_ack_restored_quote": False,
                "customer_trades": len(r.engine.trades), "duplicate_ack": ack}

    async def trades(r):
        action = (await r.hedge([("A", "BUY", 2), ("A", "BUY", .5)]))[0]
        assert await r.hedge([]) == []
        events = r.execute(action)
        ack = events[-1][1]
        assert ack["errors"] == {"1": "BAD_LOT"}
        await r.deliver(events)
        before = len(r.dealer.fills)
        await r.deliver(r.execute(action))
        assert len(r.dealer.fills) == before
        await r.deliver(r.engine.finish())
        return {"partial_trade_errors": ack["errors"], "duplicate_fills": 0,
                "on_fill_post_batch_position": r.dealer.fills[0]["portfolio"]["positions"]["A"],
                "fee": r.engine.trades[0]["fee"]}

    async def overshoot(r):
        action = (await r.quote({"x": 99.9, "y": 99.9}))[0]
        await r.deliver(r.execute(action))
        await r.deliver(r.engine.advance(final=True))
        pre_settlement = r.engine.portfolio("dealer_1")
        assert pre_settlement["positions"]["A"] == 24
        assert pre_settlement["penalties_paid"] == 0
        await r.deliver(r.engine.finish())
        assert r.engine.portfolio("dealer_1")["penalties_paid"] == 8
        return {"uncapped_customer_position": 24., "settlement_penalty": 8.}

    await capture("partial_replacement_and_suppressed_retry", Replay(), partial)
    await capture("delayed_replacement_boundary", Replay(), delayed)
    await capture("omission_and_withdrawal", Replay(), withdrawal)
    await capture("lost_ack_disconnect_duplicate", Replay(), reconnect_duplicate)
    await capture("partial_trade_batch_and_duplicate", Replay(), trades)
    await capture("customer_batch_overshoot", Replay(orders=(("x", "SELL", 12.), ("y", "SELL", 12.))), overshoot)
    return results


def source_hashes():
    paths = [KIT / "rfq/rfq" / (name + ".py") for name in
             ("runner", "engine", "exchange", "sdk", "store", "models")]
    paths += [Path(__file__), ROOT / "submission/sdk_dealer.py"]
    return complete_source_hashes({source_key(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stdout = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        results = asyncio.run(run_scenarios())
    result = {"scope": "controlled transport; unchanged Runner callback/receive and Engine methods; local worker adapter",
              "runtime": {"python": sys.version, "platform": platform.platform()},
              "runner_chunk": CHUNK, "engine_batch_limit": MAX_BATCH,
              "elapsed_seconds": time.perf_counter() - started,
              "source_hashes": source_hashes(), "scenarios": results}
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "runner_stdout.log").write_text(stdout.getvalue())
    print(json.dumps({"scenario_count": len(results), "all_reconciled": all(
        item["reconciliation"]["reconciled"] for item in results.values()),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
