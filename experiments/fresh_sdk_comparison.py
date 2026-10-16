"""Prospective independent-path study; research price laws, original Engine.

The consumed demo holdout is never read. See docs/FRESH_ACK_STUDY_PROTOCOL.md.
"""
from __future__ import annotations
from experiments.sdk_environment import KIT, complete_source_hashes, source_key, source_path
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from experiments.sdk_study import (KIT, ROOT, SynchronousSession, build_config,
    candidate_grid, load_class, sha256, shared_parameters, write_json)
from experiments.sdk_pnl_decomposition import decompose_session

SCENARIOS=((.08,.03,.75),(.20,.06,.90),(.35,.10,.95),(.12,.18,.98))
METHODS=("B1","B2","B3","B3_factor","R2")
EXTRAS=("B0","R2_fixed_quotes","B3_frozen_quotes")
PHASES={"tuning":610001,"validation":620003,"final":630017,"test":699991}


def seed_for(phase, scenario, path, stream):
    raw=f"ack-study-v1/{PHASES[phase]}/{scenario}/{path}/{stream}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest(),"big")


def generate_prices(phase, scenario, path):
    sigma,basis_sigma,rho=SCENARIOS[scenario]
    factor_rng=np.random.default_rng(seed_for(phase,scenario,path,"factor"))
    basis_rng=np.random.default_rng(seed_for(phase,scenario,path,"basis"))
    factor,basis=100.,0.
    rows=[{"A":factor,"B":2*factor+basis}]
    log_sigma=sigma/100*math.sqrt(.5)
    for _ in range(123):
        factor*=math.exp(-.5*log_sigma**2+log_sigma*float(factor_rng.normal()))
        basis=rho*basis+basis_sigma*float(basis_rng.normal())
        row={"A":float(factor),"B":float(2*factor+basis)}
        if not all(math.isfinite(x) and x>0 for x in row.values()):
            raise ValueError("Invalid generated price tape; no clipping or reroll permitted")
        rows.append(row)
    return rows


class InventoryOpponent:
    """Declared research extension; observes only its own public callback data."""
    def __init__(self, example_class, seed):
        self.base=example_class(edge_range=[.03,.16],seed=seed)

    def quote(self, orders, market, portfolio):
        from rfq.sdk import round_quote
        result={}
        for order in orders:
            pid=order["product_id"]
            fair=market["prices"][pid]-.003*portfolio["positions"].get(pid,0)
            edge=self.base.edge_for(order["order_id"])
            value=fair+(edge if order["side"]=="BUY" else -edge)
            result[order["order_id"]]=round_quote(value,market["products"][pid]["tick_size"],order["side"])
        return result

    def hedge(self, market, portfolio):
        from rfq.sdk import round_lot
        result=[]
        for pid,q in portfolio["positions"].items():
            product=market["products"][pid]
            limit=product["max_long"] if q>=0 else product["max_short"]
            if abs(q)>=.5*limit:
                qty=round_lot(abs(q)-.25*limit,product["lot_size"])
                if qty:result.append((pid,"SELL" if q>0 else "BUY",qty))
        return result


class FreshSession(SynchronousSession):
    def __init__(self, config, dealer, family, opponent_seed):
        super().__init__(config,dealer,opponents=0)
        example=load_class(KIT/"examples/dealer.py")
        ranges={"standard":[.03,.20],"wide":[.08,.28],"tight":[.015,.09]}
        for index in range(2):
            name=f"dealer_{index+2}"
            self.dealers[name]=(InventoryOpponent(example,opponent_seed+index) if family=="inventory"
                                else example(edge_range=ranges[family],seed=opponent_seed+index))
            self.engine.register(name)
            self.local[name]=self.engine.portfolio(name)


FACTORIES={}
FAILURE_DIR=ROOT/"runs/sdk-comparison-failures"


def fixed_price_factory(base):
    class FixedPublicQuotes(base):
        """Common predetermined RFQ prices; inventory affects capacity only."""
        def on_market(self,market,portfolio,exchange=None):
            super().on_market(market,portfolio,exchange)
            if getattr(self,"_fixed_interval",None)!=market["interval_id"]:
                self._fixed_interval=market["interval_id"]
                self._fixed_references=dict(market["prices"])
            if not hasattr(self,"_public_admission"):
                self._public_admission={}
                self._public_budget={pid:min(p["max_long"],p["max_short"])
                                     for pid,p in market["products"].items()}

        def quote(self,orders,market,portfolio):
            eligible=[];withdrawals={}
            for order in orders:
                key=(market["interval_id"],order["order_id"])
                pid,quantity=order["product_id"],order["quantity"]
                if key not in self._public_admission:
                    admitted=quantity<=self._public_budget[pid]
                    self._public_admission[key]=admitted
                    if admitted:self._public_budget[pid]-=quantity
                if self._public_admission[key]:eligible.append(order)
                else:withdrawals[order["order_id"]]=None
            return {**super().quote(eligible,market,portfolio),**withdrawals}

        def _edge(self,side,q,price,tick,product_id,market):
            from rfq.sdk import round_quote
            reference=(price if getattr(self,"_hypothetical_quotes",False) else
                       getattr(self,"_fixed_references",{}).get(product_id,price))
            return round_quote(reference+(self.base_edge if side=="BUY" else -self.base_edge),tick,side)

        def _continuation_values(self,*args,**kwargs):
            previous=getattr(self,"_hypothetical_quotes",False)
            self._hypothetical_quotes=True
            try:return super()._continuation_values(*args,**kwargs)
            finally:self._hypothetical_quotes=previous
    return FixedPublicQuotes


def run_case(method,params,phase,scenario,path,family,detail=False):
    try:
        return _run_case(method,params,phase,scenario,path,family,detail)
    except Exception:
        FAILURE_DIR.mkdir(parents=True,exist_ok=True)
        failed=dict(method=method,parameters=params,phase=phase,scenario=scenario,path=path,
                    family=family,crash=True,traceback=traceback.format_exc(),
                    price_rows=generate_prices(phase,scenario,path),source_hashes=sources())
        filename=f"{phase}_{scenario}_{path}_{family}_{method}_{time.time_ns()}.json"
        write_json(FAILURE_DIR/filename,failed)
        raise


def _run_case(method,params,phase,scenario,path,family,detail=False):
    key="B0" if method=="B0" else "candidate"
    if key not in FACTORIES:
        FACTORIES[key]=load_class(KIT/"demo/demo_participant.py" if key=="B0" else ROOT/"submission/sdk_dealer.py")
    factory=(fixed_price_factory(FACTORIES[key]) if method in
             ("R2_fixed_quotes","B3_frozen_quotes") else FACTORIES[key])
    dealer=factory(**({} if method=="B0" else copy.deepcopy(params)))
    prices=generate_prices(phase,scenario,path)
    seed=seed_for(phase,scenario,path,"customers")
    config=build_config(None,0,seed,rows=prices)
    runner=FreshSession(config,dealer,family,seed_for(phase,scenario,path,"opponents/"+family))
    stats=runner.run()
    stats.update(method=method,phase=phase,scenario=scenario,path=path,family=family,
                 block=0,seed=seed,parameters=params,crash=False)
    own=[t for t in runner.engine.trades if t["dealer_id"]=="dealer_1"]
    decomposition,_=decompose_session(stats,own,prices,{"A":1.,"B":1.},tolerance=1e-8)
    if decomposition["failed_checks"]:raise AssertionError(decomposition["failed_checks"])
    stats["monetary_components"]={k:decomposition[k] for k in
        ("customer_spread","exchange_execution_edge","fees","exogenous_price_move_pnl",
         "terminal_settlement_residual","penalties","decomposed_pnl")}
    stats["innovation_exposure"]=decomposition["exposure"]
    stats["counter"]={name:int(getattr(dealer,name,0)) for name in
        ("quote_ambiguity_decisions","robust_r2_decisions","robust_r2_improvements",
         "quote_fallback_decisions","fill_state_cap_fallbacks","callback_errors")}
    stats["decomposition_max_error"]=max(decomposition["differences"].values())
    if detail:
        stats["own_trades"]=own
        stats["final_rfq_observations"]=runner.observations
    return stats


def sources():
    paths=[Path(__file__),ROOT/"submission/sdk_dealer.py",ROOT/"experiments/sdk_study.py",
           ROOT/"experiments/sdk_pnl_decomposition.py",ROOT/"docs/FRESH_ACK_STUDY_PROTOCOL.md",
           KIT/"examples/dealer.py",KIT/"demo/demo_participant.py"]
    paths+=list((KIT/"rfq/rfq").glob("*.py"))
    return complete_source_hashes({source_key(p):sha256(p) for p in paths})


def init(out,stage):
    global FAILURE_DIR
    FAILURE_DIR=out/"failed_cases"
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"config.json",dict(stage=stage,protocol="docs/FRESH_ACK_STUDY_PROTOCOL.md",
        protocol_sha256=sha256(ROOT/"docs/FRESH_ACK_STUDY_PROTOCOL.md"),scenarios=SCENARIOS,
        phase_namespaces=PHASES,price_law_is_official=False,unchanged_official_engine=True,
        old_holdout_read=False,ack_schedule="synchronous; independent delayed transport validation separate"))
    write_json(out/"environment.json",dict(python=sys.version,numpy=np.__version__,os=platform.platform()))
    write_json(out/"source_hashes.json",sources())


def record(stream,row):
    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")


def means(rows):
    return float(np.mean([r["account"]["pnl"] for r in rows]))


def path_pairs(rows,proposed,baseline):
    grouped={}
    for row in rows:
        key=(row["scenario"],row["path"])
        grouped.setdefault(key,{}).setdefault(row["method"],[]).append(row)
    result=[]
    for (scenario,path),methods in sorted(grouped.items()):
        if proposed not in methods or baseline not in methods:continue
        a,b=methods[proposed],methods[baseline]
        for left in a:
            right=next(r for r in b if r["family"]==left["family"])
            for field in ("price_tape_sha256","public_customer_tape_sha256"):
                if left[field]!=right[field]:raise ValueError("Different exogenous histories in paired policies")
        result.append(dict(scenario=scenario,path=path,difference=means(a)-means(b)))
    return result


def inference(pairs,replicates=4000):
    rng=np.random.default_rng(640007)
    groups=[np.array([r["difference"] for r in pairs if r["scenario"]==scenario]) for scenario in range(4)]
    if any(len(g)<2 for g in groups):raise ValueError("At least two paths per scenario required")
    mean=float(np.mean([g.mean() for g in groups]))
    boot=sum(g[rng.integers(0,len(g),size=(replicates,len(g)))].mean(axis=1)/4 for g in groups)
    low,high=map(float,np.quantile(boot,[.025,.975]))
    se=math.sqrt(sum(float(g.var(ddof=1))/len(g) for g in groups)/16)
    p=.5*math.erfc((mean-1)/(math.sqrt(2)*se)) if se else (0. if mean>1 else 1.)
    return dict(paired_mean_difference=mean,descriptive_stratified_path_bootstrap_95=[low,high],
                normal_approx_one_sided_p_against_materiality=p,independent_price_paths=sum(map(len,groups)),
                opponent_repetitions_not_independent=True,materiality=1.,
                superiority_gate=low>1,assumptions="independent generated paths; equal scenario mixture; finite variance; synthetic laws only")


def seal_manifest(out):
    write_json(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*'))
                                   if p.is_file() and p.name!="SHA256SUMS.json"})


def pilot(out):
    started=time.perf_counter();init(out,"prospective tuning and sample-size pilot")
    selected={};all_trials=[]
    with gzip.open(out/"tuning.jsonl.gz","wt") as raw:
        for method in METHODS:
            candidates=candidate_grid(method);scores=[]
            for index,params in enumerate(candidates):
                result=[]
                for scenario in range(4):
                    for path in range(3):
                        family=("standard","wide")[path%2]
                        row=run_case(method,params,"tuning",scenario,path,family)
                        row["candidate"]=index;record(raw,row);result.append(row)
                scores.append(means(result));all_trials.append(dict(method=method,candidate=index,params=params,mean_pnl=scores[-1]))
            chosen=max(range(len(scores)),key=lambda i:(scores[i],-i));selected[method]=candidates[chosen]
            write_json(out/"progress.json",dict(phase="tuning",completed_method=method))
    write_json(out/"tuning_summary.json",all_trials)
    validation=[]
    with gzip.open(out/"validation.jsonl.gz","wt") as raw:
        for method in METHODS:
            for scenario in range(4):
                for path in range(20):
                    for family in ("standard","wide"):
                        row=run_case(method,selected[method],"validation",scenario,path,family)
                        record(raw,row);validation.append(row)
    conventional=METHODS[:-1]
    scores={m:means([r for r in validation if r["method"]==m]) for m in conventional}
    fallback=max(conventional,key=lambda m:scores[m])
    pairs=path_pairs(validation,"R2",fallback)
    sd=float(np.std([r["difference"] for r in pairs],ddof=1))
    required=math.ceil(((1.960+.842)*sd/(2.-1.))**2)
    paths=min(4096,4*math.ceil(max(128,required)/4))
    selected["R2_fixed_quotes"]={**selected["R2"],"base_edge":selected["B3"]["base_edge"],
                                 "inventory_skew":0.,"freeze_current_quotes":True}
    selected["B3_frozen_quotes"]={**selected["B3"],"inventory_skew":0.,"freeze_current_quotes":True}
    seal=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,fallback=fallback,
              validation_means=scores,validation_paired_sd=sd,required_independent_paths=required,
              final_paths=paths,paths_per_scenario=paths//4,precision_cap_binds=required>4096,
              source_hashes=sources(),final_generated=False,primary="R2 minus "+fallback,
              R3_promoted=False,elapsed_seconds=time.perf_counter()-started)
    all_seeds=[seed_for(phase,s,path,stream) for phase,count in
        (("tuning",3),("validation",20),("final",paths//4)) for s in range(4)
        for path in range(count) for stream in
        ("factor","basis","customers","opponents/standard","opponents/wide","opponents/tight","opponents/inventory")]
    if len(all_seeds)!=len(set(all_seeds)):raise RuntimeError("Primitive seed collision before seal")
    write_json(out/"PRIMITIVE_SEED_AUDIT.json",dict(seeds_checked=len(all_seeds),collisions=0,hash_bits=256,
        final_prices_generated=False,independence_is_PRNG_assumption=True))
    write_json(out/"POLICY_SEAL.json",seal)
    write_json(out/"pilot_inference.json",inference(pairs))
    seal_manifest(out);print(json.dumps(seal,indent=2))


def final(out,seal_path):
    started=time.perf_counter();seal=json.loads(seal_path.read_text())
    if seal["source_hashes"]!=sources():raise RuntimeError("Source changed after sealing; final generation prohibited")
    init(out,"sealed fresh synthetic comparison")
    write_json(out/"POLICY_SEAL.json",seal)
    rows=[];tapes={}
    with gzip.open(out/"sessions.jsonl.gz","wt") as raw:
        for scenario in range(4):
            for path in range(seal["paths_per_scenario"]):
                prices=generate_prices("final",scenario,path)
                tapes[f"{scenario}/{path}"]=prices
                for family in ("tight","inventory"):
                    ablation_quotes={}
                    for method in METHODS+EXTRAS:
                        params={} if method=="B0" else seal["selected"][method]
                        row=run_case(method,params,"final",scenario,path,family,detail=True)
                        if method in ("R2_fixed_quotes","B3_frozen_quotes"):
                            ablation_quotes[method]={r["order_id"]:r["own_quote"] for r in row["final_rfq_observations"]}
                        record(raw,row)
                        row.pop("own_trades");row.pop("final_rfq_observations");rows.append(row)
                    a,b=(ablation_quotes[m] for m in ("R2_fixed_quotes","B3_frozen_quotes"))
                    if a!=b:
                        raise AssertionError("Fixed-quote ablation differs in offered prices or admission")
            write_json(out/"progress.json",dict(phase="final",completed_scenarios=scenario+1,scheduled_sessions=len(rows)))
    with gzip.open(out/"price_tapes.json.gz","wt") as stream:json.dump(tapes,stream,sort_keys=True)
    write_json(out/"primary_inference.json",inference(path_pairs(rows,"R2",seal["fallback"])))
    write_json(out/"ablation_inference.json",{name:inference(path_pairs(rows,left,right)) for name,left,right in
        (("R2_minus_B2","R2","B2"),("R2_minus_B3","R2","B3"),
         ("fixed_quotes_minus_frozen_B3","R2_fixed_quotes","B3_frozen_quotes"))})
    summary={}
    for method in METHODS+EXTRAS:
        values=[r for r in rows if r["method"]==method];pnl=[r["account"]["pnl"] for r in values]
        summary[method]=dict(sessions=len(values),mean_pnl=float(np.mean(pnl)),worst_pnl=min(pnl),
            negative_sessions=sum(x<0 for x in pnl),crashes=sum(r["crash"] for r in values),
            rejections=sum(r["rejections"] for r in values),
            customer_orders=sum(r["account"]["customer_trades"] for r in values),
            customer_notional=sum(r["account"]["customer_notional"] for r in values),
            exchange_notional=sum(r["account"]["exchange_notional"] for r in values),
            fees=sum(r["account"]["fees_paid"] for r in values),
            penalties=sum(r["account"]["penalties_paid"] for r in values),
            counters={k:sum(r["counter"][k] for r in values) for k in values[0]["counter"]},
            max_decomposition_error=max(r["decomposition_max_error"] for r in values),
            maximum_callback_seconds=max(c["maximum"] for r in values for c in r["callback_seconds"].values()),
            scenario_means={str(s):means([r for r in values if r["scenario"]==s]) for s in range(4)})
    write_json(out/"summary.json",summary)
    write_json(out/"execution.json",dict(elapsed_seconds=time.perf_counter()-started,scheduled_sessions=len(rows),
        independent_paths=seal["final_paths"],source_unchanged=sources()==seal["source_hashes"],
        R3_vs_B3_programme_executed=False,price_law_official=False))
    seal_manifest(out);print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("pilot","final","smoke"))
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--seal",type=Path)
    args=parser.parse_args()
    if args.stage=="pilot":pilot(args.out)
    elif args.stage=="final":
        if args.seal is None:parser.error("final requires --seal")
        final(args.out,args.seal)
    else:
        init(args.out,"test-namespace smoke, not statistical evidence")
        rows=[run_case(method,candidate_grid(method)[0],"test",0,0,"standard") for method in METHODS]
        write_json(args.out/"smoke.json",rows);seal_manifest(args.out)
        print({r["method"]:r["account"]["pnl"] for r in rows})


if __name__=="__main__":main()
