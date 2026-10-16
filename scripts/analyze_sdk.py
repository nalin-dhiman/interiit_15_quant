"""Build descriptive SDK-demo figures from sealed results, without refitting."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resetmm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot",type=Path,default=Path("runs/20261015T140952Z_sdk_pilot"))
    parser.add_argument("--heldout",type=Path,default=Path("runs/20261015T141142Z_sdk_heldout"))
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh descriptive analysis directory")
    args.out.mkdir(parents=True)
    summary=load(args.heldout/"summary.json")
    primary=load(args.heldout/"primary_inference.json")
    seal=load(args.pilot/"POLICY_SEAL.json")
    rows=[json.loads(line) for line in (args.heldout/"heldout.jsonl").read_text().splitlines()]
    methods=list(summary)
    keys=[(r["method"],r["block"],r["seed"]) for r in rows]
    if len(keys)!=len(set(keys)) or len(rows)!=252:
        raise ValueError("Expected 252 unique scheduled method/block/seed results")
    paired={}
    for row in rows:
        paired.setdefault((row["block"],row["seed"]),[]).append(row)
    if len(paired)!=36 or any(len(group)!=7 for group in paired.values()):
        raise ValueError("Incomplete paired experiment")
    for group in paired.values():
        for field in ("price_tape_sha256","public_customer_tape_sha256"):
            if len({r[field] for r in group})!=1:
                raise ValueError("Exogenous histories differ between paired policies")
    for method in methods:
        observed=[r["account"]["pnl"] for r in rows if r["method"]==method]
        if not np.isclose(np.mean(observed),summary[method]["mean_pnl"],atol=1e-10,rtol=0):
            raise ValueError("Summary mean does not reconcile with raw sessions")
    config=dict(stage="descriptive analysis of sealed demo results; no new confirmation",
        created_utc=datetime.now(timezone.utc).isoformat(),bootstrap_seed=8301,
        bootstrap_draws=10000,inference_unit="9 adjacent 124-row price blocks, independence assumed",
        no_refitting=True,no_policy_selection=True)
    (args.out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    block_values=np.array([summary[m]["block_mean_pnl"] for m in methods])
    rng=np.random.default_rng(config["bootstrap_seed"])
    indices=rng.integers(0,9,size=(config["bootstrap_draws"],9))
    intervals=np.quantile(block_values[:,indices].mean(axis=2),[.025,.975],axis=1).T
    means=block_values.mean(axis=1)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    labels=[m.replace("_factor"," factor").replace("_fixed_quotes"," fixed quotes") for m in methods]
    fig,ax=plt.subplots(figsize=(8,4.4),layout="constrained")
    ax.errorbar(means,np.arange(len(methods)),xerr=np.array([means-intervals[:,0],intervals[:,1]-means]),
                fmt="o",color="#17646b",capsize=4)
    ax.set_yticks(np.arange(len(methods)),labels)
    ax.axvline(0,color="black",lw=.8)
    ax.set_xlabel("Mean raw PnL per 12-interval session (currency)")
    ax.set_title("Supplied demo holdout: descriptive block bootstrap intervals")
    ax.grid(axis="x",alpha=.2)
    fig.savefig(args.out/"heldout_pnl.pdf")
    fig.savefig(args.out/"heldout_pnl.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    differences=np.array([r["mean_difference"] for r in primary["block_differences"]])
    ax.bar(np.arange(1,10),differences,color=np.where(differences>=0,"#17646b","#ac5144"))
    ax.axhline(0,color="black",lw=.8)
    ax.axhline(primary["paired_mean_difference"],color="#495867",ls="--",label="paired mean")
    ax.set(xlabel="Held-out price block (four paired RFQ seeds each)",ylabel="R2 minus training-selected B1 PnL",
           title="Sealed primary demo diagnostic: superiority gate failed")
    ax.set_xticks(range(1,10));ax.legend(frameon=False)
    fig.savefig(args.out/"primary_blocks.pdf")
    fig.savefig(args.out/"primary_blocks.png",dpi=180)
    plt.close(fig)
    data=dict(scope="supplied development demo, unchanged organiser Engine, synchronous zero-latency ACKs",
        unique_scheduled_sessions=len(rows),paired_session_tapes=len(paired),price_blocks=9,
        repeated_reproduction_increases_sample_size=False,summary=summary,primary=primary,
        strongest_realised_heldout_mean=max(methods,key=lambda m:summary[m]["mean_pnl"]),
        baseline_preselected_on_training=seal["fallback"],
        required_pilot_blocks=seal["approximate_required_independent_blocks"],
        descriptive_policy_bootstrap_95={m:list(map(float,intervals[i])) for i,m in enumerate(methods)},
        cash_reconciliation_max=max(r["reconciliation"]["max_absolute_difference"] for r in rows),
        after_callback_exposure_is_not_prehedge_price_jump_exposure=True,
        score_implemented=False,response_promoted=False,superiority_established=False)
    (args.out/"analysis.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    sources=[args.heldout/n for n in ("summary.json","primary_inference.json","heldout.jsonl")]
    sources += [args.pilot/"POLICY_SEAL.json",Path(__file__)]
    (args.out/"source_hashes.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in sources},indent=2)+"\n")
    (args.out/"environment.json").write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,
        matplotlib=matplotlib.__version__,os=platform.platform(),scope="offline report tooling"),indent=2)+"\n")
    print(json.dumps({k:data[k] for k in ("unique_scheduled_sessions","price_blocks","cash_reconciliation_max",
         "baseline_preselected_on_training","strongest_realised_heldout_mean","superiority_established")},indent=2))


if __name__=="__main__":main()
