"""Reproduce archived integer-lot sessions after the separate quantity repair.

These are consumed-path regression replays, never new statistical evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from experiments.fresh_sdk_comparison import run_case


def ledger_signature(row):
    keys=("counterparty","order_id","product_id","side","quantity","price","fee",
          "position_after","cash_after","reference_price")
    return [tuple(t.get(k) for k in keys) for t in row["own_trades"]]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    with gzip.open(args.reference,"rt") as stream:reference=json.load(stream)
    results=[]
    with gzip.open(args.out/"replayed_sessions.jsonl.gz","wt") as stream:
        for old in reference:
            new=run_case(old["method"],old["parameters"],old["phase"],old["scenario"],old["path"],old["family"],detail=True)
            checks={key:new[key]==old[key] for key in
                    ("account","price_tape_sha256","public_customer_tape_sha256",
                     "final_rfq_observations","monetary_components","innovation_exposure","counter")}
            checks["every_financial_trade"]=ledger_signature(new)==ledger_signature(old)
            results.append(dict(method=old["method"],scenario=old["scenario"],path=old["path"],family=old["family"],checks=checks))
            stream.write(json.dumps(new,sort_keys=True)+"\n")
    failures=[r for r in results if not all(r["checks"].values())]
    summary=dict(utc=datetime.now(timezone.utc).isoformat(),sessions=len(results),
        economic_and_primitive_history_identical=not failures,failures=failures,
        independent_new_samples=0,interpretation="Consumed-path postrepair regression replay; not fresh confirmation",
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(Path("submission/sdk_dealer.py").read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,checks=results)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.out/"environment.json").write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,os=platform.platform()),indent=2)+"\n")
    (args.out/"config.json").write_text(json.dumps(dict(reference=str(args.reference),selection="path0 in each scenario, both opponents, all8methods",exclusions=False),indent=2)+"\n")
    (args.out/"REPORT.md").write_text(f"# Postrepair integer-lot regression replay\n\n{len(results)} archived sessions replayed. Economic and primitive histories identical: {not failures}. New independent samples: **0**. Wall-clock timings/timestamps and transport action IDs are not required to match. Full financial trades, accounts, observed RFQs, tape hashes, decomposition, exposure and counters are compared.\n")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out/"SHA256SUMS.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="checks"},indent=2))
    raise SystemExit(1 if failures else 0)


if __name__=="__main__":main()
