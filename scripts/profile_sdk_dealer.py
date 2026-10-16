"""Profile complete public Dealer calls on schema-valid synthetic load fixtures.

These fixtures exercise computation and all-subset capacity only; they are not
official later-round data or an impact simulation. End-to-end WebSocket tests
are separate and include the real Runner and its subprocess/IPC overhead.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sdk_environment import require_sdk
require_sdk()
POLICY_SHA256 = hashlib.sha256((ROOT/"submission/sdk_dealer.py").read_bytes()).hexdigest()

import numpy as np
from submission.sdk_dealer import Dealer


def fixture(count):
    products, prices, positions, orders = {}, {}, {}, []
    for i in range(count):
        key = f"P{i:04d}"
        products[key] = dict(product_id=key, tick_size=0.01, lot_size=1.0,
            multiplier=float(1 + i % 3), tradable_against_exchange=True,
            max_long=50.0, max_short=50.0, breach_penalty_per_unit=1.0,
            fee_per_unit=0.01, fee_rate=0.00001, initial_price=100.0)
        prices[key] = 100.0 + i % 20
        positions[key] = float(20 if i % 2 else -20)
        for j, side in enumerate(("BUY", "SELL", "BUY")):
            orders.append(dict(order_id=f"r1-{i}-{j}", product_id=key,
                customer_id=f"c{j}", side=side, quantity=4.0, expires_interval=1))
    market = dict(interval_id=1, price_version=1, prices=prices, products=products,
                  interval_seconds=5.0, closes_at=time.time() + 5,
                  closing=False, active_orders=orders)
    portfolio = dict(cash=100000.0, positions=positions, interval_id=1, price_version=1)
    return market, portfolio, orders


def profile(count, mode, repeats):
    market, portfolio, orders = fixture(count)
    dealer = Dealer(mode=mode, observation_limit=0, diagnostic_limit=0)
    timings, invalid = [], []
    total = 0.0
    for repeat in range(repeats):
        market["price_version"] = repeat + 1
        portfolio["price_version"] = repeat + 1
        market["closes_at"] = time.time() + 5
        begin = time.perf_counter()
        dealer.on_market(market, portfolio)
        on_market = time.perf_counter() - begin
        begin = time.perf_counter()
        quotes = dealer.quote(orders, market, portfolio)
        quote = time.perf_counter() - begin
        begin = time.perf_counter()
        trades = dealer.hedge(market, portfolio)
        hedge = time.perf_counter() - begin
        total = on_market + quote + hedge
        # Every possible subset lies between these two endpoints. This also
        # catches accidentally netting opposite-side orders as certain fills.
        lower = dict(portfolio["positions"])
        upper = dict(lower)
        by_id = {order["order_id"]: order for order in orders}
        for oid, price in quotes.items():
            if price is None:
                continue
            order = by_id[oid]
            if Decimal(str(price)) % Decimal("0.01"):
                invalid.append(dict(reason="BAD_TICK", order_id=oid, price=price))
            if order["side"] == "BUY":
                lower[order["product_id"]] -= order["quantity"]
            else:
                upper[order["product_id"]] += order["quantity"]
        for key in lower:
            if lower[key] < -50 or upper[key] > 50:
                invalid.append(dict(reason="subset_limit", product=key, lower=lower[key], upper=upper[key]))
        timings.append(dict(repeat=repeat, on_market_seconds=on_market, quote_seconds=quote,
                            hedge_seconds=hedge, combined_seconds=total, quotes=len(quotes), trades=len(trades)))
        # Mimic synchronous completion so the next callback is not given a
        # fabricated in-flight hedge. Portfolio itself stays this load fixture.
        dealer.on_ack({"type": "ack", "status": "accepted"})
        for pid, side, quantity in trades:
            dealer.on_fill(dict(source="exchange", product_id=pid, side=side,
                                quantity=quantity), portfolio)
    market.update(closing=True, active_orders=[])
    begin = time.perf_counter()
    closing_trades = dealer.hedge(market, portfolio)
    closing_seconds = time.perf_counter() - begin
    quote_times = [r["quote_seconds"] for r in timings]
    hedge_times = [r["hedge_seconds"] for r in timings] + [closing_seconds]
    return dict(mode=mode, products=count, rfqs=len(orders), repeats=repeats,
                complete_callback_timings=timings, closing_hedge_seconds=closing_seconds,
                closing_trades=len(closing_trades),
                quote_max_seconds=max(quote_times), hedge_max_seconds=max(hedge_times),
                combined_max_seconds=max(r["combined_seconds"] for r in timings),
                callback_errors=dealer.callback_errors, validation_errors=invalid,
                each_callback_under_one_second=max(quote_times + hedge_times)<1,
                each_callback_under_internal_quarter_second=max(quote_times + hedge_times)<0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new profile output path")
    rows = [profile(count, mode, args.repeats) for count in (2, 50, 250, 251, 1000) for mode in ("B3", "R2")]
    result = dict(scope="whole Python public callbacks, synthetic load, excludes Runner IPC/network",
                  policy_sha256=POLICY_SHA256,
                  policy_source_unchanged=POLICY_SHA256 == hashlib.sha256(
                      (ROOT/"submission/sdk_dealer.py").read_bytes()).hexdigest(),
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  rows=rows, official_later_round_certification=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: r[key] for key in ("mode", "products", "rfqs", "quote_max_seconds",
          "hedge_max_seconds", "combined_max_seconds", "callback_errors",
          "each_callback_under_one_second")} for r in rows], indent=2))
    raise SystemExit(0 if result["policy_source_unchanged"] and all(r["each_callback_under_one_second"] and not r["callback_errors"]
                            and not r["validation_errors"] for r in rows) else 1)


if __name__ == "__main__":
    main()
