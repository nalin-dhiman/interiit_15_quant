"""Independent post-seal monetary and innovation-exposure decomposition.

Uses already opened archived trades and exogenous price history. No policy work.
"""
import argparse
from experiments.sdk_environment import KIT
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def session_key(row):
    return row["method"], row["block"], row["seed"]


def load_prices(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    products = [key for key in rows[0] if key not in ("time_s", "round", "price_version")]
    return [{pid: float(row[pid]) for pid in products} for row in rows]


def decompose_session(row, trades, prices, multipliers, initial_cash=100000., tolerance=1e-8):
    """Marked-wealth identity with customer fills immediately before next jump."""
    products = list(multipliers)
    n = row["account"]["price_version"]
    prices = prices[:n]
    if len(prices) != n:
        raise ValueError("Insufficient exact price rows")
    own = sorted((trade for trade in trades if trade["dealer_id"] == "dealer_1"),
                 key=lambda trade: trade["trade_no"])
    grouped = {}
    cash, fees = initial_cash, 0.
    positions = {pid: 0. for pid in products}
    cash_checks, position_checks, source_checks = [], [], []
    customer_spread = exchange_edge = 0.
    for trade in own:
        pid, version = trade["product_id"], trade["price_version"]
        reference = prices[version - 1][pid]
        signed = trade["quantity"] * (1 if trade["side"] == "BUY" else -1)
        edge = signed * multipliers[pid] * (reference - trade["price"])
        source = "customer" if trade.get("counterparty") == "customer" else "exchange"
        if source == "customer":
            customer_spread += edge
        else:
            exchange_edge += edge
        fees += trade["fee"]
        cash -= signed * trade["price"] * multipliers[pid] + trade["fee"]
        positions[pid] += signed
        if "cash_after" in trade and row["account"]["penalties_paid"] == 0:
            cash_checks.append(abs(cash - trade["cash_after"]))
        if "position_after" in trade:
            position_checks.append(abs(positions[pid] - trade["position_after"]))
        if "reference_price" in trade:
            source_checks.append(abs(reference - trade["reference_price"]))
        if "notional" in trade:
            source_checks.append(abs(trade["notional"] - trade["quantity"] * trade["price"] * multipliers[pid]))
        grouped.setdefault(version, {"customer": [], "exchange": []})[source].append((pid, signed))
    # Monetary replay above follows original global trade_no. Exposure below
    # separates exchange at v from customer matching at v immediately pre-jump.
    q = {pid: 0. for pid in products}
    callback_samples, innovation_samples, steps = [], [], []
    price_move_pnl = 0.
    for version in range(1, n + 1):
        events = grouped.get(version, {"customer": [], "exchange": []})
        for pid, signed in events["exchange"]:
            q[pid] += signed
        callback_samples.append([q[pid] for pid in products])
        after_callback = q.copy()
        for pid, signed in events["customer"]:
            q[pid] += signed
        if version < n:
            price_delta = {pid: prices[version][pid] - prices[version - 1][pid] for pid in products}
            innovation_pnl = sum(q[pid] * multipliers[pid] * price_delta[pid] for pid in products)
            price_move_pnl += innovation_pnl
            innovation_samples.append([q[pid] for pid in products])
            steps.append({"price_version_before": version, "price_version_after": version + 1,
                          "after_callback_positions": after_callback, "postmatch_prehedge_positions": q.copy(),
                          "price_change": price_delta, "exogenous_price_move_pnl": innovation_pnl})
    terminal_residual = sum(q[pid] * multipliers[pid] *
                            (row["final_settlement"][pid] - prices[-1][pid]) for pid in products)
    penalties = row["account"]["penalties_paid"]
    decomposition = customer_spread + exchange_edge - fees + price_move_pnl + terminal_residual - penalties
    cash -= penalties
    cash_settlement_pnl = cash + sum(positions[pid] * multipliers[pid] * row["final_settlement"][pid]
                                     for pid in products) - initial_cash
    tape = [{"version": i + 1, "prices": price} for i, price in enumerate(prices)]
    tape_hash = hashlib.sha256(json.dumps(tape, sort_keys=True).encode()).hexdigest()
    differences = {"decomposition_pnl": abs(decomposition - row["account"]["pnl"]),
                   "cash_replay_pnl": abs(cash_settlement_pnl - row["account"]["pnl"]),
                   "cash": abs(cash - row["account"]["cash"]),
                   "fees": abs(fees - row["account"]["fees_paid"]),
                   "final_position": max(abs(positions[p] - row["account"]["positions"][p]) for p in products),
                   "exposure_final_position": max(abs(q[p] - positions[p]) for p in products),
                   "every_trade_position": max(position_checks, default=0),
                   "every_trade_cash_when_no_penalty": max(cash_checks, default=0),
                   "trade_reference_and_notional": max(source_checks, default=0)}
    callback_array, innovation_array = np.asarray(callback_samples), np.asarray(innovation_samples)
    callback_q2 = np.mean(callback_array ** 2, axis=0)
    innovation_q2 = np.mean(innovation_array ** 2, axis=0)
    multipliers_array = np.array([multipliers[p] for p in products])
    factor = np.array([1., 2.]) * multipliers_array
    exposure = {"products": products,
                "mean_q_squared_at_price_innovation": innovation_q2.tolist(),
                "mean_multiplier_exposure_squared_at_price_innovation":
                    np.mean((innovation_array * multipliers_array) ** 2, axis=0).tolist(),
                "mean_terminal_factor_squared_at_price_innovation": float(np.mean((innovation_array @ factor) ** 2)),
                "price_innovations_with_nonzero_inventory": int(np.sum(np.any(innovation_array != 0, axis=1))),
                "price_innovations": n - 1,
                "mean_q_squared_after_callback": callback_q2.tolist(),
                "mean_terminal_factor_squared_after_callback": float(np.mean((callback_array @ factor) ** 2)),
                "after_callback_grid_proxy_qty2_seconds": (callback_array ** 2).sum(axis=0).tolist(),
                "proxy_note": "The preceding proxy array must be multiplied by configured tick seconds. It is not exact holding time. Innovation q² is never multiplied by .5 seconds and called holding time."}
    if "mean_squared_inventory" in row:
        differences["after_callback_q2_against_original"] = float(np.max(np.abs(callback_q2 - row["mean_squared_inventory"])))
    if "mean_squared_settlement_factor_exposure" in row:
        differences["after_callback_factor_q2_against_original"] = abs(exposure["mean_terminal_factor_squared_after_callback"] - row["mean_squared_settlement_factor_exposure"])
    failed = {key: value for key, value in differences.items() if value > tolerance}
    if row.get("price_tape_sha256") not in (None, tape_hash):
        failed["price_tape_hash"] = {"expected": row["price_tape_sha256"], "reconstructed": tape_hash}
    return {"method": row["method"], "block": row["block"], "seed": row["seed"],
            "official_pnl": row["account"]["pnl"], "customer_spread": customer_spread,
            "exchange_execution_edge": exchange_edge, "fees": fees,
            "exogenous_price_move_pnl": price_move_pnl, "terminal_settlement_residual": terminal_residual,
            "penalties": penalties, "decomposed_pnl": decomposition, "cash_replayed_pnl": cash_settlement_pnl,
            "differences": differences, "failed_checks": failed, "exposure": exposure,
            "trade_count": len(own), "price_tape_sha256_reconstructed": tape_hash}, steps


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/20261015T141142Z_sdk_heldout")
    parser.add_argument("--data", default=str(KIT / "demo/demo_data_holdout.csv"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    utc = datetime.now(timezone.utc)
    out = Path(args.out or ("runs/" + utc.strftime("%Y%m%dT%H%M%SZ") + "_sdk_pnl_decomposition"))
    out.mkdir(parents=True, exist_ok=False)
    study = Path(args.study)
    rows = jsonl(study / "heldout.jsonl")
    trades = jsonl(study / "trades_heldout.jsonl")
    grouped = {}
    for trade in trades:
        if trade["dealer_id"] == "dealer_1":
            grouped.setdefault(session_key(trade), []).append(trade)
    prices = load_prices(args.data)
    config = {"stage": "Post-seal monetary/exposure analysis", "study": args.study, "data": args.data,
              "expected_raw_sessions": 252, "multipliers": {"A": 1., "B": 1.}, "starting_cash": 100000.,
              "tick_seconds": .5, "session_start_row": "block * 124", "absolute_reconciliation_tolerance": 1e-8,
              "count_as_new_performance_evidence": False, "no_policy_retuning": True,
              "penalty_timing": "authoritative charged total; no reconstructed check times"}
    dump(out / "config.json", config)
    dump(out / "environment.json", {"python": sys.version, "numpy": np.__version__, "os": platform.platform()})
    dump(out / "input_hashes.json", {str(path): sha(path) for path in
        (study / "heldout.jsonl", study / "trades_heldout.jsonl", study / "POLICY_SEAL.json", Path(args.data), Path(__file__))})
    result_rows = []
    failures = []
    with (out / "decomposition.jsonl").open("w") as raw, gzip.open(out / "innovation_exposure.jsonl.gz", "wt") as detail:
        for row in rows:
            if row.get("crash") or "account" not in row:
                failures.append({"session": session_key(row), "reason": "crash/missing official account", "raw": row})
                continue
            record, steps = decompose_session(row, grouped.get(session_key(row), []),
                                              prices[row["block"] * 124:], config["multipliers"])
            record["exposure"]["after_callback_grid_proxy_qty2_seconds"] = [
                value * config["tick_seconds"] for value in record["exposure"]["after_callback_grid_proxy_qty2_seconds"]]
            record["exposure"]["proxy_note"] = "After-callback q² times .5-second grid is a holding-time proxy only. Price-innovation q² is a separate point-shock exposure; it is not exact holding time."
            raw.write(json.dumps(record, allow_nan=False) + "\n")
            result_rows.append(record)
            if record["failed_checks"]:
                failures.append({"session": session_key(row), "checks": record["failed_checks"]})
            for step in steps:
                detail.write(json.dumps({"method": row["method"], "block": row["block"], "seed": row["seed"], **step}) + "\n")
    if len(rows) != config["expected_raw_sessions"]:
        failures.append({"reason": "unexpected raw session count", "actual": len(rows)})
    summary = {}
    money = ("official_pnl", "customer_spread", "exchange_execution_edge", "fees",
             "exogenous_price_move_pnl", "terminal_settlement_residual", "penalties")
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in result_rows if row["method"] == method]
        summary[method] = {"sessions": len(selected),
                           "mean_monetary_components": {key: float(np.mean([row[key] for row in selected])) for key in money},
                           "mean_q_squared_at_price_innovation": np.mean([row["exposure"]["mean_q_squared_at_price_innovation"] for row in selected], axis=0).tolist(),
                           "mean_q_squared_after_callback": np.mean([row["exposure"]["mean_q_squared_after_callback"] for row in selected], axis=0).tolist(),
                           "mean_factor_squared_at_price_innovation": float(np.mean([row["exposure"]["mean_terminal_factor_squared_at_price_innovation"] for row in selected])),
                           "mean_innovations_with_inventory": float(np.mean([row["exposure"]["price_innovations_with_nonzero_inventory"] for row in selected])),
                           "max_reconciliation_difference": max(max(row["differences"].values()) for row in selected)}
    dump(out / "summary.json", summary)
    dump(out / "failures.json", failures)
    maximum = max(max(row["differences"].values()) for row in result_rows)
    rows_text = "\n".join(f"| {method} | {r['mean_monetary_components']['official_pnl']:.6f} | {r['mean_monetary_components']['customer_spread']:.6f} | {r['mean_monetary_components']['exogenous_price_move_pnl']:.6f} | {r['mean_monetary_components']['fees']:.6f} | {r['mean_monetary_components']['terminal_settlement_residual']:.6f} |" for method, r in summary.items())
    exposure_text = "\n".join(f"| {method} | {r['mean_q_squared_at_price_innovation']} | {r['mean_q_squared_after_callback']} | {r['mean_innovations_with_inventory']:.3f} |" for method, r in summary.items())
    (out / "REPORT.md").write_text(f"""# Post-seal SDK PnL and price-innovation exposure audit

All {len(rows)} raw held-out records were inspected; {len(result_rows)} were decomposed. Failed checks: {len(failures)}. Maximum absolute reconciliation difference: {maximum:.12g}; tolerance 1e-8. Source trades, account cash, fees, positions and exact price-tape hashes are independently checked. Any failed checks are retained in failures.json; none are silently removed from performance statistics. These are the already opened sealed-study sessions, not additional performance evidence, and no policy or SDK source changed.

The identity is customer spread plus exchange execution edge, minus fees, plus exogenous price-move PnL, plus retained-position terminal settlement residual, minus actual charged penalties. Principal is replayed separately in cash; risk preferences are absent from this money decomposition. Exchange execution edge and penalties are also retained in the raw and summary outputs.

| Policy | Mean official PnL | Customer spread | Price-move PnL | Fees | Terminal residual |
|---|---:|---:|---:|---:|---:|
{rows_text}

| Policy | Mean q² at price innovations [A,B] | Mean q² after callback [A,B] | Mean price jumps with inventory |
|---|---|---|---:|
{exposure_text}

The original after-callback inventory statistic samples positions after hedging. B0/B2 can therefore report zero while still carrying customer inventory across the required post-matching price jump. At version v exchange trades execute at current reference; customer fills then execute at matching reference v immediately before the price move v→v+1. The innovation exposure uses that post-match/pre-hedge inventory. Only after observing v+1 can the dealer hedge it. Inventory q² at a point shock is not multiplied by .5 seconds and called exact holding time. A separately named after-callback grid holding-time proxy is archived and matched to the original statistic.

Final B=2*A settlement residual uses final retained inventory times multiplier times settlement less final trading reference. It does not force liquidation. Penalties use the authoritative charged total; their intermediate timing cannot be reconstructed from these trade files alone. Cash-after checks are enabled where total penalties are zero.

Permitted claim I/NUMERICALLY_CHECKED: the marked-wealth identity and point-shock exposure reconcile these archived official-Engine sessions. No new performance inference, causal response identification, holding-time theorem or actual-market validation follows. Hand fixtures are tested separately.
""")
    dump(out / "commands.log", {"argv": [sys.executable, "-m", "experiments.sdk_pnl_decomposition"] + list(sys.argv[1:]),
                                 "started_utc": utc.isoformat(), "elapsed_seconds": time.perf_counter()-started,
                                 "exit_code": 0 if not failures else 1})
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(sha(path) + "  " + str(path.relative_to(out)) + "\n" for path in files))
    print(json.dumps({"out": str(out), "raw_sessions": len(rows), "decomposed_sessions": len(result_rows),
                      "failed_checks": len(failures), "max_absolute_difference": maximum}))
    if failures:
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main()
