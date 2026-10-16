"""Independent post-seal integrity and descriptive economic audit.

Reads every completed raw session. It never changes the seal or excludes cases.
No data are read while the sealed experiment is running.
"""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def signature(row):
    """Customer executions exclude unrelated exchange action IDs/trade numbers."""
    return [(t["order_id"], t["product_id"], t["side"], t["quantity"],
             t["price"], t["reference_price"])
            for t in row["own_trades"] if t["counterparty"] == "customer"]


def check_group(group):
    if len(group) != 8 or len({r["method"] for r in group}) != 8:
        raise AssertionError("Missing or duplicate scheduled policy session")
    for key in ("price_tape_sha256", "public_customer_tape_sha256"):
        if len({r[key] for r in group}) != 1:
            raise AssertionError("Paired policies saw different primitive tapes")
    methods = {r["method"]: r for r in group}
    left, right = (methods[m] for m in ("R2_fixed_quotes", "B3_frozen_quotes"))
    for row in group:
        observations = row["final_rfq_observations"]
        if len(observations) != 36 or len({r["order_id"] for r in observations}) != 36:
            raise AssertionError("One-final-observation-per-RFQ failed")
        if row["crash"] or row["decomposition_max_error"] > 1e-8:
            raise AssertionError("Crash or unreconciled session retained")
    offered = lambda r: [(x["order_id"], x["own_quote"]) for x in r["final_rfq_observations"]]
    if offered(left) != offered(right):
        raise AssertionError("Matched-book ablation has different prices/admission")
    if signature(left) != signature(right):
        raise AssertionError("Matched-book ablation has different customer executions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    execution = json.loads((args.study / "execution.json").read_text())
    seal = json.loads((args.study / "POLICY_SEAL.json").read_text())
    groups = defaultdict(list)
    methods = defaultdict(list)
    reference = []
    total = 0
    with gzip.open(args.study / "sessions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            key = (row["scenario"], row["path"], row["family"])
            groups[key].append(row)
            if len(groups[key]) == 8:
                check_group(groups[key])
                for completed in groups[key]:
                    if completed["path"] == 0:
                        reference.append(copy.deepcopy(completed))
                    # Retain full records until all eight policies are checked.
                    for field in ("own_trades", "final_rfq_observations"):
                        completed.pop(field)
                    methods[completed["method"]].append(completed)
                del groups[key]
    if groups or total != execution["scheduled_sessions"] or total != 16 * seal["final_paths"]:
        raise AssertionError("Incomplete raw experiment")
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        scheduled_sessions=total, independent_price_paths=seal["final_paths"],
        matched_price_admission_and_customer_fill_pairs=2 * seal["final_paths"],
        final_rfq_records=36 * total, failures_excluded=0,
        all_paired_primitive_hashes_match=True, all_matched_book_customer_trades_match=True,
        all_final_rfq_records_unique_within_session=True,
        inference_unit="price path; two opponent replicates are grouped",
        tail_note="Quantiles and tail means are descriptive session outcomes; no powered tail or simultaneous-risk certificate.",
        methods={})
    for method, rows in methods.items():
        pnl = np.array([r["account"]["pnl"] for r in rows])
        q05 = float(np.quantile(pnl, .05))
        result["methods"][method] = dict(
            sessions=len(rows), mean_pnl=float(pnl.mean()), median_pnl=float(np.median(pnl)),
            pnl_quantile_05=q05, mean_pnl_at_or_below_q05=float(pnl[pnl <= q05].mean()),
            negative_fraction=float(np.mean(pnl < 0)),
            mean_customer_orders=float(np.mean([r["account"]["customer_trades"] for r in rows])),
            mean_fees=float(np.mean([r["account"]["fees_paid"] for r in rows])),
            mean_exchange_notional=float(np.mean([r["account"]["exchange_notional"] for r in rows])),
            mean_innovation_q_squared=np.mean([r["innovation_exposure"]["mean_q_squared_at_price_innovation"] for r in rows], axis=0).tolist(),
            mean_terminal_factor_squared_at_innovation=float(np.mean([r["innovation_exposure"]["mean_terminal_factor_squared_at_price_innovation"] for r in rows])),
            mean_monetary_components={k:float(np.mean([r["monetary_components"][k] for r in rows]))
                                      for k in rows[0]["monetary_components"]})
    result["elapsed_seconds"] = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(args.out / "postrepair_replay_reference.json.gz", "wt") as stream:
        json.dump(reference, stream, sort_keys=True)
    (args.out / "config.json").write_text(json.dumps(dict(study=str(args.study),
        changed_policy=False, outcome_exclusion=False, inference="descriptive economic audit"), indent=2) + "\n")
    (args.out / "environment.json").write_text(json.dumps(dict(numpy=np.__version__), indent=2) + "\n")
    (args.out / "REPORT.md").write_text(
        "# Independent fresh-study integrity and economic audit\n\n"
        f"All {total:,} scheduled raw records audited. No case excluded. "
        "Every fixed-book pair has identical offered prices, admission and customer executions. "
        "Every paired policy has matching exogenous price/customer tape hashes. "
        "Each session has exactly 36 unique final RFQ records and reconciles cash.\n\n"
        "Raw PnL components, fees, price-innovation exposure and descriptive downside outcomes "
        "are in analysis.json. Opponent repetitions are not independent price paths. "
        "This audit does not establish unknown-market value-error bounds or competition profitability.\n")
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "methods"}, indent=2))


if __name__ == "__main__":
    main()
