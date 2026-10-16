"""Regenerate transparent foundation evidence; no competition claims."""
from __future__ import annotations
import argparse, csv, json, platform, sys, time
from pathlib import Path
import numpy as np
from resetmm.analytic import ResetCosts,optimal_poisson_rate,optimal_threshold,poisson_cost,threshold_cost,reset_exit_time
from resetmm.mdp import solve,evaluate,fixed_policy_derivative
from resetmm.inventory_mdp import build_inventory_mdp
from resetmm.response import response_fixture
from resetmm.policies import InventoryQuote,ImmediateReset,ThresholdReset,PoissonReset,CostAwareReset
from resetmm.toy_exchange import make_scenario,run
from resetmm.statistics import paired_bootstrap_interval,lower_tail_mean


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="evidence"); ap.add_argument("--sessions",type=int,default=12)
    args=ap.parse_args()
    if args.sessions<2:
        ap.error("At least two sessions required")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    costs=ResetCosts(D=1,a=.02,K=.1,c=0)
    r=optimal_poisson_rate(costs); b=optimal_threshold(costs)
    analytic=dict(parameters=costs.__dict__,optimal_poisson_rate=r,optimal_threshold=b,
                  poisson_cost=poisson_cost(r,costs),threshold_cost=threshold_cost(b,costs),
                  cost_ratio=threshold_cost(b,costs)/poisson_cost(r,costs),
                  expected_ratio=1/np.sqrt(6),
                  risk_exit_time_without_reset=reset_exit_time(1,0,2),
                  risk_exit_time_with_reset=reset_exit_time(1,.5,2))
    # Exact finite-MDP comparison (known model; not RFQ simulator performance).
    m,states,actions=build_inventory_mdp()
    v,pi,Q=solve(m)
    base,bs,ba=build_inventory_mdp(allow_reset=False)
    bv,_,_=solve(base)
    start=[states.index((0,-1)),states.index((0,1))]
    mdp=dict(states=len(states),actions=len(actions),horizon=m.P.shape[0],
             reset_value=float(v[0,start].mean()),no_reset_value=float(bv[0,start].mean()),
             policy_evaluation_residual=float(np.max(abs(v-evaluate(m,pi)))),
             action_value_residual=float(np.max(abs(v[:-1]-np.max(Q,axis=-1)))))
    rm,rpi,dP,dg=response_fixture(); rv,dv=fixed_policy_derivative(rm,rpi,dP,dg)
    response=[]
    for eps in (.1,.03,.01,.003,.001,.0003):
        plus,pp,_,_=response_fixture(eps); minus,mp,_,_=response_fixture(-eps)
        fd=(evaluate(plus,pp)[0,0]-evaluate(minus,mp)[0,0])/(2*eps)
        response.append(dict(epsilon=eps,exact_derivative=float(dv[0,0]),
                             central_difference=float(fd),absolute_error=float(abs(fd-dv[0,0]))))
    records=[]
    for regime in ("stationary","stress","informed"):
        for seed in range(100,100+args.sessions):
            tape=make_scenario(seed,regime=regime)
            policies={"inventory_only":InventoryQuote(),"immediate_reset":ImmediateReset(),
                      "poisson_reset":PoissonReset(seed=seed+10_000),
                      "threshold_reset":ThresholdReset(),"cost_aware_proxy":CostAwareReset()}
            for name,policy in policies.items():
                result=run(tape,policy)
                result.pop("terminal_positions")
                records.append(dict(regime=regime,seed=seed,policy=name,**result))
    with (out/"toy_sessions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for regime in ("stationary","stress","informed"):
        baseline=[r["pnl"] for r in records if r["regime"]==regime and r["policy"]=="inventory_only"]
        for name in ("inventory_only","immediate_reset","poisson_reset","threshold_reset","cost_aware_proxy"):
            subset=[r for r in records if r["regime"]==regime and r["policy"]==name]
            vals=[r["pnl"] for r in subset]
            d,lo,hi=paired_bootstrap_interval(vals,baseline,replicates=2000)
            summary.append(dict(regime=regime,policy=name,n=len(vals),mean_pnl=float(np.mean(vals)),
                                mean_orders=float(np.mean([r["customer_orders"] for r in subset])),
                                mean_fees=float(np.mean([r["fees"] for r in subset])),
                                lower_10pct_mean=lower_tail_mean(vals),
                                paired_difference=d,ci_low=lo,ci_high=hi))
    report=dict(status="FOUNDATION_ONLY_NOT_COMPETITION_VALIDATED",python=sys.version,numpy=np.__version__,
                platform=platform.platform(),sessions_per_regime=args.sessions,
                toy_runs=len(records),elapsed_seconds=time.perf_counter()-started,
                analytic=analytic,finite_mdp=mdp,response=response,toy_summary=summary,
                limitations=["Synthetic parameters fixed a priori; no organizer data or opponent strategies used.",
                 "Short paired smoke experiment, not a powered superiority study.",
                 "Risk-adjusted finite-MDP values are NOT monetary PnL.",
                 "Bootstrap intervals are descriptive, unadjusted for multiple comparisons.",
                 "All toy policies share quoting; cost-aware proxy is NOT the proposed full planner."])
    (out/"foundation_results.json").write_text(json.dumps(report,indent=2))
    lines=[r"\begin{tabular}{lrr}",r"\toprule",r"Diagnostic & Value & Reference\\",r"\midrule",
           rf"Optimal Poisson rate & {r:.6f} & $\sqrt{{2aD/K}}$\\",
           rf"Optimal zero-reset threshold & {b:.6f} & $(12DK/a)^{{1/4}}$\\",
           rf"Threshold/Poisson cost & {analytic['cost_ratio']:.6f} & $1/\sqrt{{6}}$\\",
           rf"Finite-MDP policy residual & {mdp['policy_evaluation_residual']:.2e} & numerical zero\\",
           r"\bottomrule",r"\end{tabular}"]
    (out/"analytic_table.tex").write_text("\n".join(lines))
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Regime & Policy & PnL & Orders & Fees & $\Delta$PnL\\",r"\midrule"]
    aliases={"inventory_only":"Inventory","immediate_reset":"Immediate","poisson_reset":"Poisson",
             "threshold_reset":"Threshold","cost_aware_proxy":"Cost proxy"}
    for row in summary:
        lines.append(f"{row['regime']} & {aliases[row['policy']]} & {row['mean_pnl']:.2f} & {row['mean_orders']:.1f} & {row['mean_fees']:.2f} & {row['paired_difference']:.2f}\\\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (out/"toy_table.tex").write_text("\n".join(lines))
    print(json.dumps({"toy_runs":len(records),"analytic":analytic,"finite_mdp":mdp,
                      "elapsed_seconds":report['elapsed_seconds']},indent=2))

if __name__=="__main__":
    main()
