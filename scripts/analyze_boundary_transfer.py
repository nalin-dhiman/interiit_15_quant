"""Post-seal descriptive transfer and declared-law PnL attribution; no retuning."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np

RHO=(.75,.90,.95,.98)
MONEY=('customer_spread','fees','exogenous_price_move_pnl','terminal_settlement_residual','penalties','decomposed_pnl')


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictive_price_pnl(row,prices):
    n=row['account']['price_version'];deltas=defaultdict(float)
    for trade in row['own_trades']:
        if trade['product_id']=='B':deltas[trade['price_version']]+=trade['quantity']*(1 if trade['side']=='BUY' else -1)
    q=0.;expected=0.;rho=RHO[row['scenario']]
    for version in range(1,n):
        q+=deltas[version]
        basis=prices[version-1]['B']-2*prices[version-1]['A']
        expected+=q*(rho-1)*basis
    return expected


def summarize(rows):
    fields=tuple(rows[0]['money'])
    return dict(sessions=len(rows),mean_pnl=float(np.mean([r['pnl'] for r in rows])),
        mean_customer_fills=float(np.mean([r['fills'] for r in rows])),
        mean_monetary_components={k:float(np.mean([r['money'][k] for r in rows])) for k in fields},
        declared_law_expected_price_pnl=float(np.mean([r['predictable_price_pnl'] for r in rows])),
        innovation_noise_price_pnl=float(np.mean([r['money']['exogenous_price_move_pnl']-r['predictable_price_pnl'] for r in rows])))


def contrast(left,right):
    return dict(mean_pnl=left['mean_pnl']-right['mean_pnl'],
        mean_customer_fills=left['mean_customer_fills']-right['mean_customer_fills'],
        mean_monetary_components={k:left['mean_monetary_components'][k]-right['mean_monetary_components'][k] for k in left['mean_monetary_components']},
        declared_law_expected_price_pnl=left['declared_law_expected_price_pnl']-right['declared_law_expected_price_pnl'],
        innovation_noise_price_pnl=left['innovation_noise_price_pnl']-right['innovation_noise_price_pnl'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();started=time.perf_counter()
    execution=json.loads((a.study/'execution.json').read_text());audit=json.loads(a.audit.read_text())
    if audit['scheduled_sessions']!=execution['scheduled_sessions'] or audit['failed_checks'] or audit['failures_excluded']:
        raise AssertionError('Independent complete raw audit prerequisite failed')
    a.out.mkdir(parents=True,exist_ok=False)
    with gzip.open(a.study/'price_tapes.json.gz','rt') as f:tapes=json.load(f)
    groups=defaultdict(list);total=0
    with gzip.open(a.study/'sessions.jsonl.gz','rt') as raw:
        for line in raw:
            row=json.loads(line);total+=1;prices=tapes[f"{row['scenario']}/{row['path']}"]
            compact=dict(pnl=row['account']['pnl'],fills=row['account']['customer_trades'],money=row['monetary_components'],
                predictable_price_pnl=predictive_price_pnl(row,prices))
            groups['all',row['method']].append(compact)
            groups['family/'+row['family'],row['method']].append(compact)
            groups['scenario/'+str(row['scenario']),row['method']].append(compact)
            groups[f"scenario/{row['scenario']}/family/{row['family']}",row['method']].append(compact)
    if total!=execution['scheduled_sessions']:raise AssertionError('Incomplete transfer analysis')
    summarized={}
    for (scope,method),rows in groups.items():summarized.setdefault(scope,{})[method]=summarize(rows)
    contrasts={scope:contrast(methods['BR2'],methods['B3']) for scope,methods in summarized.items()}
    for scope,value in contrasts.items():
        components=value['mean_monetary_components']
        gap=(components['customer_spread']+components['exchange_execution_edge']-components['fees']+
             components['exogenous_price_move_pnl']+components['terminal_settlement_residual']-components['penalties'])
        if abs(gap-value['mean_pnl'])>1e-8:raise AssertionError('Transfer monetary difference does not sum')
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        scheduled_sessions=total,analysis_scope='post-seal descriptive only; no additional confirmatory tests, policies or samples',
        all_scopes=summarized,BR2_minus_B3_by_scope=contrasts,
        declared_price_law=dict(factor='E[A_next-A|history]=0 under mean-one independent lognormal innovation',
            basis='E[basis_next-basis|history]=(rho-1)*basis',rho_by_scenario=RHO,
            conditional_expected_price_pnl='sum pre-innovation dealer B inventory*(rho-1)*(B-2*A)',
            assumptions='bounded finite-horizon exposures; independent primitive future price innovations; declared laws only',
            terminal_residual='separate exact retained-settlement jump; no forced liquidation',
            market_law_verified=False),R3_enabled=False)
    write(a.out/'analysis.json',result)
    write(a.out/'config.json',dict(study=str(a.study),complete_independent_audit=str(a.audit),posthoc_descriptive=True,
        claim_of_customer_spread_improvement=False,scientific_sources_changed=False))
    write(a.out/'environment.json',dict(python=sys.version,numpy=np.__version__,os=platform.platform()))
    write(a.out/'input_hashes.json',{str(p):sha(p) for p in (a.study/'sessions.jsonl.gz',a.study/'price_tapes.json.gz',a.audit,Path(__file__))})
    primary=contrasts['all'];text='\n'.join(f"| {scope} | {r['mean_pnl']:.6f} | {r['mean_monetary_components']['customer_spread']:.6f} | {r['mean_monetary_components']['fees']:.6f} | {r['declared_law_expected_price_pnl']:.6f} | {r['innovation_noise_price_pnl']:.6f} |" for scope,r in sorted(contrasts.items()))
    (a.out/'REPORT.md').write_text(f'''# Descriptive post-seal transfer and declared-law attribution\n\nAll {total:,} final raw records were reused after their complete independent audit. These are neither new samples nor new confirmatory tests. No controller, quote model, opponent, seal or inference gate changed.\n\n| BR2 minus B3 scope | PnL | Customer spread | Additional fees | Declared-law predictable price PnL | Innovation-noise price PnL |\n|---|---:|---:|---:|---:|---:|\n{text}\n\nPositive fee difference is a monetary cost. Total benefit has customer spread contribution {primary['mean_monetary_components']['customer_spread']:.6f}, additional fees {primary['mean_monetary_components']['fees']:.6f}, actual price-move contribution {primary['mean_monetary_components']['exogenous_price_move_pnl']:.6f}, and terminal residual {primary['mean_monetary_components']['terminal_settlement_residual']:.6f}. Its actual price-move difference separates into declared-law predictable component {primary['declared_law_expected_price_pnl']:.6f} plus realised innovation-noise difference {primary['innovation_noise_price_pnl']:.6f}. This is exact arithmetic conditional on the declared stochastic-law formula, not a fitted causal attribution in a real market.\n\nUnder the declared independent mean-one lognormal factor, conditional expected factor change is zero. For the AR(1) basis b=B-2A, expected next basis change is (rho-1)b. Pre-innovation B inventory times that change gives predictable price PnL; remaining actual price PnL is an innovation term. These expectations require the declared integrable finite-horizon adapted exposure and independent future primitive shocks. Known final B=2A settlement creates a separately retained terminal residual. Zero empirical directional forecast does not remove this synthetic mean-reversion/settlement opportunity.\n\nFamily/scenario results are descriptive; no simultaneous transfer, tail or competition guarantee is inferred. Support remains scoped to the one sealed whole-policy comparison and declared opponents/laws. R3, model calibration and actual-round certification remain separate gates.\n''')
    write(a.out/'SHA256SUMS.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps(dict(total=total,BR2_minus_B3=contrasts),indent=2))

if __name__=='__main__':main()
