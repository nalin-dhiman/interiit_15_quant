"""One prospective boundary redesign; original Engine, new synthetic histories."""
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
from experiments.sdk_study import ROOT,KIT,build_config,load_class,sha256,write_json
from experiments.fresh_sdk_comparison import SCENARIOS,candidate_grid
from experiments.freeze_diagnostic import DiagnosticSession
from experiments.sdk_pnl_decomposition import decompose_session

DOMAIN='boundary-policy-v1/20261016T103800Z'
METHODS=('B1','B2','B3','B3_factor','BB3','BR2')
EXTRAS=('B0','BR2_BB3_quotes')
CONVENTIONAL=METHODS[:-1]
DEVELOPMENT=('standard','wide')
FINAL_FAMILIES=('volatility','sticky')
FACTORIES={}
FAILURE_DIR=ROOT/'runs/sdk-boundary-failures'


def seed_for(phase,scenario,path,stream):
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}/{phase}/{scenario}/{path}/{stream}'.encode()).digest(),'big')


def generate_prices(phase,scenario,path):
    sigma,basis_sigma,rho=SCENARIOS[scenario]
    factor_rng=np.random.default_rng(seed_for(phase,scenario,path,'factor'))
    basis_rng=np.random.default_rng(seed_for(phase,scenario,path,'basis'))
    factor,basis=100.,0.;rows=[{'A':factor,'B':2*factor}]
    log_sigma=sigma/100*math.sqrt(.5)
    for _ in range(123):
        factor*=math.exp(-.5*log_sigma**2+log_sigma*float(factor_rng.normal()))
        basis=rho*basis+basis_sigma*float(basis_rng.normal())
        row={'A':factor,'B':2*factor+basis}
        if not all(math.isfinite(p) and p>0 for p in row.values()):
            raise ValueError('Invalid generated tape; no reroll/clipping')
        rows.append(row)
    return rows


class VolatilityOpponent:
    """Declared research family using only latest public price changes."""
    def __init__(self,base,seed):
        self.base=base(edge_range=[.025,.16],seed=seed)
        self.last_prices={};self.delta={};self.version=None

    def quote(self,orders,market,portfolio):
        from rfq.sdk import round_quote
        version=market['price_version']
        if version!=self.version:
            self.delta={pid:abs(price-self.last_prices.get(pid,price)) for pid,price in market['prices'].items()}
            self.last_prices=dict(market['prices']);self.version=version
        result={}
        for order in orders:
            pid=order['product_id'];edge=self.base.edge_for(order['order_id'])+min(.12,.5*self.delta.get(pid,0.))
            result[order['order_id']]=round_quote(market['prices'][pid]+(edge if order['side']=='BUY' else -edge),
                                                  market['products'][pid]['tick_size'],order['side'])
        return result

    def hedge(self,market,portfolio):return self.base.hedge(market,portfolio)


class StickyOpponent:
    """New RFQs quote immediately; other replacements follow a public clock."""
    def __init__(self,base,seed):
        self.base=base(edge_range=[.03,.20],seed=seed);self.seen=set()

    def quote(self,orders,market,portfolio):
        selected={o['order_id'] for o in orders if o['order_id'] not in self.seen or (market['price_version']-1)%3==0}
        self.seen.update(o['order_id'] for o in orders)
        # Keep the original dealer's active-RFQ edge cache even on omitted
        # cadence callbacks; omission must not make it redraw the same RFQ.
        all_quotes=self.base.quote(orders,market,portfolio)
        return {oid:price for oid,price in all_quotes.items() if oid in selected}

    def hedge(self,market,portfolio):return self.base.hedge(market,portfolio)


class StudySession(DiagnosticSession):
    def __init__(self,config,dealer,family,opponent_seed):
        self.first_seen={};self.first_admitted={}
        super().__init__(config,dealer,family if family in DEVELOPMENT else 'standard',opponent_seed)
        if family in FINAL_FAMILIES:
            example=load_class(KIT/'examples/dealer.py')
            factory=VolatilityOpponent if family=='volatility' else StickyOpponent
            for index in range(2):self.dealers[f'dealer_{index+2}']=factory(example,opponent_seed+index)
        elif family not in DEVELOPMENT:raise ValueError(family)

    def callback(self,name,method,**kwargs):
        if name=='dealer_1' and method=='quote':
            for order in kwargs['orders']:
                self.first_seen.setdefault((kwargs['market']['interval_id'],order['order_id']),self.virtual_time)
        return super().callback(name,method,**kwargs)

    def _action(self,name,kind,**payload):
        super()._action(name,kind,**payload)
        if name=='dealer_1' and kind=='quotes':
            for oid in payload.get('quotes',{}):
                if self.engine.quotes.get(oid,{}).get(name) is not None:
                    self.first_admitted.setdefault((self.engine.interval_id,oid),self.virtual_time)

    def advance(self,final):
        start=len(self.observations);super().advance(final)
        for row in self.observations[start:]:
            key=(row['interval_id'],row['order_id']);first=self.first_admitted.get(key)
            row.update(first_seen_virtual_time=self.first_seen[key],first_positive_virtual_time=first,
                       admission_delay_seconds=None if first is None else first-self.first_seen[key])


def parameters(method):
    base='R2' if method.startswith('BR2') else 'B3' if method=='BB3' else method
    grid=candidate_grid(base)
    if method in ('BB3','BR2'):
        grid=[dict(p,boundary_reset=True,freeze_current_quotes=False) for p in grid]
    return grid


def sources():
    files=[Path(__file__),ROOT/'docs/BOUNDARY_STUDY_PROTOCOL.md',ROOT/'submission/sdk_dealer.py',
           ROOT/'submission/boundary_dealer.py',ROOT/'experiments/sdk_study.py',
           ROOT/'experiments/fresh_sdk_comparison.py',ROOT/'experiments/freeze_diagnostic.py',
           ROOT/'experiments/sdk_pnl_decomposition.py',KIT/'examples/dealer.py',KIT/'demo/demo_participant.py']
    files+=sorted((KIT/'rfq/rfq').glob('*.py'))
    return complete_source_hashes({source_key(p):sha256(p) for p in files})


def run_case(method,params,phase,scenario,path,family,detail=False):
    try:return _run_case(method,params,phase,scenario,path,family,detail)
    except Exception:
        FAILURE_DIR.mkdir(parents=True,exist_ok=True)
        failure=dict(method=method,parameters=params,phase=phase,scenario=scenario,path=path,
            family=family,crash=True,traceback=traceback.format_exc(),source_hashes=sources())
        write_json(FAILURE_DIR/f'{phase}_{scenario}_{path}_{family}_{method}_{time.time_ns()}.json',failure)
        raise


def _run_case(method,params,phase,scenario,path,family,detail=False):
    key='B0' if method=='B0' else 'boundary' if method in ('BB3','BR2','BR2_BB3_quotes') else 'core'
    if key not in FACTORIES:
        file=KIT/'demo/demo_participant.py' if key=='B0' else ROOT/'submission'/('boundary_dealer.py' if key=='boundary' else 'sdk_dealer.py')
        FACTORIES[key]=load_class(file)
    dealer=FACTORIES[key](**copy.deepcopy(params))
    prices=generate_prices(phase,scenario,path);seed=seed_for(phase,scenario,path,'customers')
    runner=StudySession(build_config(None,0,seed,rows=prices),dealer,family,
                        seed_for(phase,scenario,path,'opponents/'+family))
    row=runner.run();row.update(method=method,phase=phase,scenario=scenario,path=path,family=family,
                              seed=seed,block=0,parameters=params,crash=False)
    own=[t for t in runner.engine.trades if t['dealer_id']=='dealer_1']
    decomp,_=decompose_session(row,own,prices,{'A':1.,'B':1.},tolerance=1e-8)
    if decomp['failed_checks']:
        FAILURE_DIR.mkdir(parents=True,exist_ok=True)
        write_json(FAILURE_DIR/f'ledger_{phase}_{scenario}_{path}_{family}_{method}.json',
                   dict(row=row,own_trades=own,price_rows=prices,failed_checks=decomp['failed_checks']))
        raise AssertionError(decomp['failed_checks'])
    row['monetary_components']={k:decomp[k] for k in ('customer_spread','exchange_execution_edge','fees',
        'exogenous_price_move_pnl','terminal_settlement_residual','penalties','decomposed_pnl')}
    row['innovation_exposure']=decomp['exposure'];row['decomposition_max_error']=max(decomp['differences'].values())
    row['own_rejections']=[r for r in runner.rejections if r['dealer']=='dealer_1']
    row['callback_errors']=int(getattr(dealer,'callback_errors',0))
    row['boundary_counters']=copy.deepcopy(getattr(dealer,'boundary_counters',{}))
    observations=runner.observations
    operational_failure=(row['callback_errors'] or row['own_rejections'] or
                         row['boundary_counters'].get('unsupported_histories',0))
    if operational_failure:
        FAILURE_DIR.mkdir(parents=True,exist_ok=True)
        write_json(FAILURE_DIR/f'operational_{phase}_{scenario}_{path}_{family}_{method}.json',
                   dict(row=row,own_trades=own,final_rfq_observations=observations,price_rows=prices,
                        boundary_failures=list(getattr(dealer,'boundary_failures',()))))
        raise AssertionError('Unexpected callback/rejection/unsupported history in ordered study schedule')
    if len(observations)!=36 or len({(r['interval_id'],r['order_id']) for r in observations})!=36:
        raise AssertionError('One final observation per RFQ failed')
    ages=[r['accepted_quote_age_seconds'] for r in observations if r['own_quote'] is not None]
    delays=[r['admission_delay_seconds'] for r in observations if r['admission_delay_seconds'] is not None]
    row['admission']=dict(ever_admitted=len(delays),never_admitted=36-len(delays),
                         sum_delay_seconds=sum(delays),maximum_delay_seconds=max(delays,default=0.))
    row['quote_age']=dict(positive_final_quotes=len(ages),sum_age_seconds=sum(ages),maximum_age_seconds=max(ages,default=0.))
    if detail:row.update(own_trades=own,final_rfq_observations=observations)
    return row


def init(out,stage):
    global FAILURE_DIR
    out.mkdir(parents=True,exist_ok=False);FAILURE_DIR=out/'failed_cases'
    write_json(out/'config.json',dict(stage=stage,protocol='docs/BOUNDARY_STUDY_PROTOCOL.md',namespace=DOMAIN,
        scenarios=SCENARIOS,methods=METHODS,diagnostics=EXTRAS,development_opponents=DEVELOPMENT,
        final_opponents=FINAL_FAMILIES,ack_schedule='synchronous immediate ACK; logical time; transport tested separately',
        old_holdouts_read=False,price_law_official=False,SDK_unchanged=True,R3_enabled=False))
    write_json(out/'environment.json',dict(python=sys.version,numpy=np.__version__,os=platform.platform(),executable=sys.executable))
    write_json(out/'source_hashes.json',sources())
    for rel in sources():
        target=out/'executed_sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source_path(rel).read_bytes())


def manifest(out):
    write_json(out/'SHA256SUMS.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*'))
                                    if p.is_file() and p.name!='SHA256SUMS.json'})


def mean(rows):return float(np.mean([r['account']['pnl'] for r in rows]))


def pairs(rows,left,right):
    grouped={};result=[]
    for row in rows:
        grouped.setdefault((row['scenario'],row['path']),{}).setdefault(row['family'],{})[row['method']]=row
    for (scenario,path),families in sorted(grouped.items()):
        differences=[]
        for family,methods in sorted(families.items()):
            a,b=methods[left],methods[right]
            for key in ('price_tape_sha256','public_customer_tape_sha256'):
                if a[key]!=b[key]:raise AssertionError('Paired primitive histories differ')
            differences.append(a['account']['pnl']-b['account']['pnl'])
        result.append(dict(scenario=scenario,path=path,difference=float(np.mean(differences))))
    return result


def inference(values,replicates=5000):
    groups=[np.array([r['difference'] for r in values if r['scenario']==s]) for s in range(4)]
    if any(len(g)<2 for g in groups):raise ValueError('At least two paths per scenario required')
    rng=np.random.default_rng(seed_for('analysis',0,0,'bootstrap'))
    boot=sum(g[rng.integers(0,len(g),size=(replicates,len(g)))].mean(axis=1)/4 for g in groups)
    lo,hi=map(float,np.quantile(boot,[.025,.975]));gap=float(np.mean([g.mean() for g in groups]))
    se=math.sqrt(sum(float(g.var(ddof=1))/len(g) for g in groups)/16)
    p=.5*math.erfc((gap-1)/(math.sqrt(2)*se)) if se else float(gap<=1)
    return dict(paired_mean_difference=gap,stratified_path_bootstrap_95=[lo,hi],materiality=1.,
        independent_paths=sum(map(len,groups)),opponent_repetitions_not_independent=True,
        approximate_one_sided_materiality_p=p,superiority_gate=lo>1,
        assumptions='independent generated path units; equal scenario mixture; finite variance; bootstrap approximate; synthetic scope')


def pilot(out):
    started=time.perf_counter();init(out,'boundary tuning and validation before final seal')
    initial_sources=sources()
    selected={};trials=[]
    known=dict(base_edge=.08,inventory_skew=.005,band_fraction=.6,target_fraction=.5)
    assert any(all(p[k]==v for k,v in known.items()) for p in parameters('B3_factor'))
    with gzip.open(out/'tuning.jsonl.gz','wt') as stream:
        for method in METHODS:
            scores=[]
            for index,params in enumerate(parameters(method)):
                cases=[]
                for scenario in range(4):
                    for path in range(3):
                        row=run_case(method,params,'tuning',scenario,path,DEVELOPMENT[path%2]);cases.append(row)
                        stream.write(json.dumps(dict(candidate=index,**row),sort_keys=True)+'\n')
                score=mean(cases);scores.append(score);trials.append(dict(method=method,candidate=index,parameters=params,mean_pnl=score,sessions=len(cases)))
            selected[method]=parameters(method)[int(np.argmax(scores))]
            write_json(out/'progress.json',dict(stage='tuning',completed_method=method,trials=len(trials)))
    write_json(out/'tuning_summary.json',trials)
    validation=[]
    with gzip.open(out/'validation.jsonl.gz','wt') as stream:
        for scenario in range(4):
            for path in range(20):
                for family in DEVELOPMENT:
                    for method in METHODS:
                        row=run_case(method,selected[method],'validation',scenario,path,family);validation.append(row)
                        stream.write(json.dumps(row,sort_keys=True)+'\n')
    conventional_scores={m:mean([r for r in validation if r['method']==m]) for m in CONVENTIONAL}
    fallback=max(CONVENTIONAL,key=lambda m:conventional_scores[m]);values=pairs(validation,'BR2',fallback)
    sd=float(np.std([r['difference'] for r in values],ddof=1));required=math.ceil(((1.960+.842)*sd)**2)
    paths=min(4096,4*math.ceil(max(128,required)/4))
    selected['BR2_BB3_quotes']={**selected['BR2'],'base_edge':selected['BB3']['base_edge'],
                              'inventory_skew':selected['BB3']['inventory_skew']}
    write_json(out/'pilot_inference.json',inference(values))
    primitives=[]
    for phase,count in (('tuning',3),('validation',20),('final',paths//4)):
        for scenario in range(4):
            for path in range(count):
                names=('factor','basis','customers','opponents/standard','opponents/wide','opponents/volatility','opponents/sticky')
                primitives.extend(seed_for(phase,scenario,path,name) for name in names)
                primitives.extend(seed_for(phase,scenario,path,'customers')+j for j in (1,2))
                # Original opponent pair uses seed and seed+1; both are audited.
                primitives.extend(seed_for(phase,scenario,path,'opponents/'+f)+1 for f in DEVELOPMENT+FINAL_FAMILIES)
    if len(primitives)!=len(set(primitives)):raise AssertionError('Derived primitive seed collision')
    write_json(out/'PRIMITIVE_SEED_AUDIT.json',dict(checked=len(primitives),digest_bits=256,collisions=0,
                                                final_prices_generated=False,independence_is_PRNG_assumption=True))
    if sources()!=initial_sources:raise RuntimeError('Source changed during development; no policy seal produced')
    seal=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,fallback=fallback,
        validation_conventional_means=conventional_scores,validation_paired_pooled_sd=sd,
        required_independent_paths=required,final_paths=paths,paths_per_scenario=paths//4,
        precision_cap_binds=required>4096,primary='BR2 minus '+fallback,materiality=1.,
        source_hashes=sources(),final_generated=False,elapsed_seconds=time.perf_counter()-started,
        programme_R3_vs_B3_executed=False)
    write_json(out/'POLICY_SEAL.json',seal)
    for rel in sources():
        target=out/'executed_sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source_path(rel).read_bytes())
    manifest(out);print(json.dumps(seal,indent=2))


def summaries(rows):
    result={}
    for method in METHODS+EXTRAS:
        selected=[r for r in rows if r['method']==method];pnls=np.array([r['account']['pnl'] for r in selected])
        ages=sum(r['quote_age']['positive_final_quotes'] for r in selected)
        admitted=sum(r['admission']['ever_admitted'] for r in selected)
        counter_names={k for r in selected for k in r['boundary_counters']}
        result[method]=dict(sessions=len(selected),mean_pnl=float(pnls.mean()),median_pnl=float(np.median(pnls)),
            worst_pnl=float(pnls.min()),pnl_quantile_05=float(np.quantile(pnls,.05)),negative_sessions=int(np.sum(pnls<0)),
            zero_pnl_sessions=int(np.sum(pnls==0)),customer_orders=sum(r['account']['customer_trades'] for r in selected),
            customer_notional=sum(r['account']['customer_notional'] for r in selected),
            exchange_notional=sum(r['account']['exchange_notional'] for r in selected),fees=sum(r['account']['fees_paid'] for r in selected),
            penalties=sum(r['account']['penalties_paid'] for r in selected),crashes=sum(r['crash'] for r in selected),
            own_rejections=sum(len(r['own_rejections']) for r in selected),callback_errors=sum(r['callback_errors'] for r in selected),
            max_decomposition_error=max(r['decomposition_max_error'] for r in selected),
            maximum_callback_seconds=max(c['maximum'] for r in selected for c in r['callback_seconds'].values()),
            mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in selected])),
            mean_quote_age_seconds=sum(r['quote_age']['sum_age_seconds'] for r in selected)/ages if ages else None,
            mean_admission_delay_seconds=sum(r['admission']['sum_delay_seconds'] for r in selected)/admitted if admitted else None,
            never_admitted_rfqs=sum(r['admission']['never_admitted'] for r in selected),
            boundary_counters={k:sum(r['boundary_counters'].get(k,0) for r in selected) for k in counter_names},
            mean_monetary_components={k:float(np.mean([r['monetary_components'][k] for r in selected])) for k in selected[0]['monetary_components']},
            mean_innovation_factor_q2=float(np.mean([r['innovation_exposure']['mean_terminal_factor_squared_at_price_innovation'] for r in selected])),
            scenario_means={str(s):mean([r for r in selected if r['scenario']==s]) for s in range(4)})
    return result


def final(out,seal_path):
    started=time.perf_counter();seal=json.loads(seal_path.read_text())
    if seal['source_hashes']!=sources():raise RuntimeError('Source changed after seal; final prohibited')
    init(out,'sealed boundary comparison and retirement decision');write_json(out/'POLICY_SEAL.json',seal)
    rows=[];tapes={}
    with gzip.open(out/'sessions.jsonl.gz','wt') as stream:
        for scenario in range(4):
            for path in range(seal['paths_per_scenario']):
                tapes[f'{scenario}/{path}']=generate_prices('final',scenario,path)
                for family in FINAL_FAMILIES:
                    for method in METHODS+EXTRAS:
                        params={} if method=='B0' else seal['selected'][method]
                        row=run_case(method,params,'final',scenario,path,family,detail=True)
                        stream.write(json.dumps(row,sort_keys=True)+'\n')
                        row.pop('own_trades');row.pop('final_rfq_observations');rows.append(row)
            write_json(out/'progress.json',dict(completed_scenarios=scenario+1,scheduled_sessions=len(rows)))
    with gzip.open(out/'price_tapes.json.gz','wt') as stream:json.dump(tapes,stream,sort_keys=True)
    if sources()!=seal['source_hashes']:
        write_json(out/'SOURCE_CHANGE_FAILURE.json',dict(sealed=seal['source_hashes'],current=sources()))
        raise RuntimeError('Source changed during final generation; raw output preserved, inference prohibited')
    write_json(out/'summary.json',summaries(rows))
    primary=inference(pairs(rows,'BR2',seal['fallback']));write_json(out/'primary_inference.json',primary)
    write_json(out/'ablation_inference.json',{name:inference(pairs(rows,a,b)) for name,a,b in
        (('BR2_minus_BB3','BR2','BB3'),('BB3_minus_B3','BB3','B3'),('BR2_minus_B3','BR2','B3'),
         ('BB3_quote_BR2_minus_BB3','BR2_BB3_quotes','BB3'))})
    write_json(out/'execution.json',dict(elapsed_seconds=time.perf_counter()-started,scheduled_sessions=len(rows),
        independent_paths=seal['final_paths'],source_unchanged=seal['source_hashes']==sources(),
        R3_vs_B3_programme_executed=False,price_law_official=False,
        boundary_branch_decision='SYNTHETIC_PRIMARY_PASS' if primary['superiority_gate'] else 'RETIRE_TESTED_BOUNDARY_R2'))
    manifest(out);print(json.dumps(dict(summary=summaries(rows),primary=primary),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('pilot','final','smoke'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seal',type=Path)
    a=p.parse_args()
    if a.stage=='pilot':pilot(a.out)
    elif a.stage=='final':
        if a.seal is None:p.error('final requires --seal')
        final(a.out,a.seal)
    else:
        init(a.out,'test namespace smoke; no statistical evidence')
        rows=[run_case(m,parameters(m)[0],'test',0,0,'standard',detail=True) for m in METHODS]
        write_json(a.out/'smoke.json',rows);manifest(a.out);print({r['method']:r['account']['pnl'] for r in rows})


if __name__=='__main__':main()
