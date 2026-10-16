"""Independently reconcile an organiser demo audit database, without SDK imports.

This is offline research tooling, never part of a submitted Dealer. Read-only
SQLite access reconstructs cash, positions, fees, notional and terminal wealth
from fill/penalty events and verifies every authoritative portfolio message.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3


def dec(value):
    return Decimal(str(value))


def reconcile(directory: Path, tolerance=Decimal("0.00000001")):
    database = directory / "audit.sqlite"
    summary = json.loads((directory / "results.json").read_text())
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    rows = [(seq, recipient, json.loads(payload)) for seq, recipient, payload in
            connection.execute("SELECT seq,recipient,payload FROM events ORDER BY seq")]
    accounts, products, prices = {}, {}, {}
    counts, errors, rejected = Counter(), [], []
    checked = 0
    terminal_marks = {}
    for index, (seq, recipient, item) in enumerate(rows):
        if item["type"] == "session_end":
            # finish() emits exactly one final portfolio per registered
            # account immediately before session_end. Earlier adjacent
            # portfolios may belong to advance() and retain intraday marks.
            final_rows = rows[index-len(item["results"]):index]
            if ({r for _, r, p in final_rows if p["type"] == "portfolio"}
                    != set(item["results"])):
                errors.append(dict(seq=seq, error="missing final settlement portfolios"))
            for final_seq, _, portfolio in final_rows:
                if portfolio["type"] == "portfolio":
                    terminal_marks[final_seq] = item["prices"]

    def check(seq, name, actual):
        nonlocal checked
        if name not in accounts:
            errors.append(dict(seq=seq, dealer=name, error="portfolio before initial snapshot"))
            return
        expected = accounts[name]
        comparisons = {key: expected[key] for key in
                       ("cash", "fees_paid", "penalties_paid", "customer_notional",
                        "exchange_notional", "customer_trades", "exchange_trades")}
        for key, quantity in expected["positions"].items():
            comparisons["position:" + key] = quantity
        comparisons["equity"] = expected["cash"] + sum(
            quantity * dec(prices[key]) * dec(products[key]["multiplier"])
            for key, quantity in expected["positions"].items())
        comparisons["pnl"] = comparisons["equity"] - expected["initial_cash"]
        for key, value in comparisons.items():
            observed = (actual["positions"].get(key[9:], 0) if key.startswith("position:")
                        else actual[key])
            difference = abs(value - dec(observed))
            if difference > tolerance:
                errors.append(dict(seq=seq, dealer=name, field=key, expected=str(value),
                                   actual=str(observed), absolute_difference=str(difference)))
        checked += 1

    def book_fill(seq, recipient, item):
        account = accounts[recipient]
        key, quantity, price = item["product_id"], dec(item["quantity"]), dec(item["price"])
        product = products[key]
        signed = quantity if item["side"] == "BUY" else -quantity
        principal = signed * price * dec(product["multiplier"])
        notional = quantity * price * dec(product["multiplier"])
        source = item["counterparty"]
        expected_fee = (quantity * dec(product["fee_per_unit"]) + notional * dec(product["fee_rate"])
                        if source == "exchange" else Decimal(0))
        if abs(expected_fee - dec(item["fee"])) > tolerance:
            errors.append(dict(seq=seq, dealer=recipient, error="fill fee does not reconcile",
                               expected=str(expected_fee), actual=str(item["fee"])))
        account["cash"] -= principal + expected_fee
        account["positions"][key] = account["positions"].get(key, Decimal(0)) + signed
        account["fees_paid"] += expected_fee
        account[source + "_notional"] += notional
        account[source + "_trades"] += 1

    embedded_executions = 0
    for index, (seq, recipient, item) in enumerate(rows):
        kind = item["type"]
        counts[kind] += 1
        if kind == "session_config":
            products = {p["product_id"]: p for p in item["products"]}
        elif kind == "snapshot":
            products.update(item["products"])
            prices = item["market"]["prices"]
            portfolio = item["portfolio"]
            if recipient not in accounts:
                # Demo sessions start flat; never infer starting wealth from a
                # later cash balance or silently absorb unlogged trades.
                if any(portfolio["positions"].values()) or portfolio["pnl"] != 0:
                    errors.append(dict(seq=seq, dealer=recipient,
                                       error="non-flat/nonzero-PnL initial snapshot"))
                accounts[recipient] = {key: dec(portfolio[key]) for key in
                    ("cash", "fees_paid", "penalties_paid", "customer_notional",
                     "exchange_notional", "customer_trades", "exchange_trades")}
                accounts[recipient]["positions"] = {key: dec(q) for key, q in portfolio["positions"].items()}
                accounts[recipient]["initial_cash"] = dec(portfolio["cash"])
            check(seq, recipient, portfolio)
        elif kind in ("market", "prices"):
            prices = item["prices"]
        elif kind == "fill":
            book_fill(seq, recipient, item)
        elif kind == "penalty":
            account = accounts[recipient]
            # Penalty cash is independently derived from current inventory,
            # the published physical-unit limit and per-unit penalty.
            key = item["product_id"]
            product = products[key]
            quantity = account["positions"].get(key, Decimal(0))
            long = product["max_long"]
            short = product["max_short"]
            excess = max(Decimal(0), quantity-dec(long) if long is not None else Decimal(0),
                         -quantity-dec(short) if short is not None else Decimal(0))
            charge = excess * dec(product["breach_penalty_per_unit"])
            actual = item.get("charge")
            if actual is None or abs(charge-dec(actual)) > tolerance:
                errors.append(dict(seq=seq, dealer=recipient, error="penalty does not reconcile",
                                   expected=str(charge), actual=actual))
            account["cash"] -= charge
            account["penalties_paid"] += charge
        elif kind == "portfolio":
            # finish() marks settlement before emitting final portfolios, then
            # publishes those prices in session_end in the SAME atomic commit.
            # No intermediate prices message announces this settlement mark.
            if seq in terminal_marks:
                prices = terminal_marks[seq]
            # A trades action is one atomic Store.commit: portfolio first,
            # then ack containing every execution. The latter are not also
            # standalone fill events. Single-trade execution IS standalone.
            if index + 1 < len(rows):
                next_seq, next_recipient, following = rows[index + 1]
                if next_recipient == recipient and following["type"] == "ack":
                    for fill in following.get("executions", []):
                        book_fill(next_seq, recipient, fill)
                        embedded_executions += 1
            check(seq, recipient, item)
        elif kind == "ack" and (item.get("status") == "rejected" or item.get("errors")):
            rejected.append(dict(seq=seq, dealer=recipient, acknowledgement=item))
        elif kind == "session_end":
            prices = item["prices"]
            for name, portfolio in item["results"].items():
                check(seq, name, portfolio)
    connection.close()
    if summary.get("status") != "completed" or counts["session_end"] != 1 or not accounts:
        errors.append(dict(error="no unique completed session with registered accounts",
                           status=summary.get("status"), session_ends=counts["session_end"]))
    for name, portfolio in summary.get("results", {}).items():
        check("results.json", name, portfolio)
    return dict(scope="independent Decimal reconstruction of supplied organiser demo events",
                tolerance_currency=str(tolerance), portfolio_messages_checked=checked,
                event_counts=dict(counts), embedded_batch_executions=embedded_executions,
                dealer_count=len(accounts),
                status=summary.get("status"), cash_reconciles=not errors,
                discrepancies=errors, rejected_acknowledgements=rejected,
                final_results=summary.get("results", {}),
                input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (database, directory / "results.json", directory / "trades.csv")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Reconciliation outputs are not overwritten")
    result = reconcile(args.directory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("cash_reconciles", "portfolio_messages_checked", "event_counts", "discrepancies")}, indent=2))
    raise SystemExit(0 if result["cash_reconciles"] else 1)


if __name__ == "__main__":
    main()
