"""Prospectively sealed scalar-B3 quote freeze mechanism study (research only)."""
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
from experiments.sdk_study import ROOT, KIT, build_config, load_class, sha256, write_json
from experiments.fresh_sdk_comparison import FreshSession, SCENARIOS
from experiments.sdk_pnl_decomposition import decompose_session

PROTOCOL=ROOT/'docs/FREEZE_DIAGNOSTIC_PROTOCOL.md'
PRIOR_SEAL=ROOT/'data/sdk-freeze-parameters.json'
METHODS=('B3_normal','B3_freeze')
FAMILIES=('tight','inventory')
FACTORY=None
DOMAIN='freeze-mechanism-v1/20261016T103800Z'


def seed_for(phase, scenario, path, stream):
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
        if not all(math.isfinite(x) and x>0 for x in row.values()):
            raise ValueError('Invalid generated tape; no reroll or clipping')
        rows.append(row)
    return rows


def policy_parameters(method):
    params=copy.deepcopy(json.loads(PRIOR_SEAL.read_text())['selected']['B3'])
    assert params['mode']=='B3' and not params['factor_hedge']
    params['freeze_current_quotes']=method=='B3_freeze'
    if method not in METHODS:raise ValueError(method)
    return params


class DiagnosticSession(FreshSession):
    """Original callbacks/actions plus offline accepted-live-price instrumentation."""
    def __init__(self,*args,**kwargs):
        self.live_quote_changes={}
        self.accepted_price_changes=0
        self.accepted_withdrawals=0
        self.accepted_unchanged_positive_requests=0
        self.frozen_seen=set()
        self.frozen_omissions=0
        super().__init__(*args,**kwargs)

    def callback(self,name,method,**kwargs):
        out=super().callback(name,method,**kwargs)
        if name=='dealer_1':
            dealer=self.dealers[name]
            self.frozen_seen.update(getattr(dealer,'_frozen_products',set()))
            if method=='quote' and isinstance(out,dict):
                interval=kwargs['market']['interval_id']
                self.frozen_omissions+=sum(order['order_id'] not in out and
                    (interval,order['product_id']) in getattr(dealer,'_frozen_products',set())
                    for order in kwargs['orders'])
        return out

    def _action(self,name,kind,**payload):
        interval=self.engine.interval_id
        requested=payload.get('quotes',{}) if name=='dealer_1' and kind=='quotes' else {}
        before={oid:self.engine.quotes.get(oid,{}).get(name) for oid in requested}
        super()._action(name,kind,**payload)
        for oid in requested:
            old=before[oid];new=self.engine.quotes.get(oid,{}).get(name)
            key=(interval,oid)
            if old!=new:
                if new is None:
                    self.accepted_withdrawals+=1
                    self.live_quote_changes.pop(key,None)
                else:
                    self.accepted_price_changes+=1
                    self.live_quote_changes[key]=dict(time=self.virtual_time,price=new,
                                                     version=self.engine.price_version)
            elif new is not None:
                self.accepted_unchanged_positive_requests+=1
            if new is not None and key not in self.live_quote_changes:
                raise AssertionError('Positive live quote lacks accepted-change history')

    def advance(self,final):
        interval=self.engine.interval_id
        matching_time=self.virtual_time+self.engine.tick_seconds
        ages={}
        for oid in self.engine.orders:
            value=self.engine.quotes.get(oid,{}).get('dealer_1')
            if value is None:continue
            installed=self.live_quote_changes[(interval,oid)]
            if installed['price']!=value:raise AssertionError('Live price and installed age disagree')
            ages[oid]=dict(accepted_quote_age_seconds=matching_time-installed['time'],
                accepted_quote_price_version=installed['version'],matching_virtual_time=matching_time,
                quote_installation_virtual_time=installed['time'])
        start=len(self.observations)
        super().advance(final)
        for observation in self.observations[start:]:
            observation['interval_id']=interval
            observation.update(ages.get(observation['order_id'],dict(accepted_quote_age_seconds=None,
                accepted_quote_price_version=None,matching_virtual_time=matching_time,
                quote_installation_virtual_time=None)))
            if (observation['own_quote'] is None)!=(observation['accepted_quote_age_seconds'] is None):
                raise AssertionError('Final live quote and age missingness disagree')
        self.live_quote_changes={k:v for k,v in self.live_quote_changes.items() if k[0]!=interval}


def run_case(method,scenario,path,family,phase='diagnosis'):
    global FACTORY
    if FACTORY is None:FACTORY=load_class(ROOT/'submission/sdk_dealer.py')
    params=policy_parameters(method);dealer=FACTORY(**params)
    prices=generate_prices(phase,scenario,path)
    seed=seed_for(phase,scenario,path,'customers')
    runner=DiagnosticSession(build_config(None,0,seed,rows=prices),dealer,family,
        seed_for(phase,scenario,path,'opponents/'+family))
    stats=runner.run()
    stats.update(method=method,scenario=scenario,path=path,family=family,phase=phase,
                 seed=seed,block=0,parameters=params,crash=False)
    own=[t for t in runner.engine.trades if t['dealer_id']=='dealer_1']
    decomposition,_=decompose_session(stats,own,prices,{'A':1.,'B':1.},tolerance=1e-8)
    if decomposition['failed_checks']:raise AssertionError(decomposition['failed_checks'])
    stats['monetary_components']={k:decomposition[k] for k in ('customer_spread',
        'exchange_execution_edge','fees','exogenous_price_move_pnl','terminal_settlement_residual',
        'penalties','decomposed_pnl')}
    stats['innovation_exposure']=decomposition['exposure']
    stats['decomposition_max_error']=max(decomposition['differences'].values())
    stats['own_rejections']=[r for r in runner.rejections if r['dealer']=='dealer_1']
    stats['callback_errors']=dealer.callback_errors
    stats['freeze_instrumentation']=dict(activated_product_intervals=len(runner.frozen_seen),
        omitted_replacements=runner.frozen_omissions,
        accepted_live_price_changes=runner.accepted_price_changes,
        accepted_withdrawals=runner.accepted_withdrawals,
        accepted_unchanged_positive_requests=runner.accepted_unchanged_positive_requests)
    observations=runner.observations
    if len(observations)!=36 or len({(r['interval_id'],r['order_id']) for r in observations})!=36:
        raise AssertionError('Expected exactly one final record per each of 36 RFQs')
    stats['final_rfq_observations']=observations
    stats['own_trades']=own
    positive=[r['accepted_quote_age_seconds'] for r in observations if r['own_quote'] is not None]
    won=[r['accepted_quote_age_seconds'] for r in observations if r['won']]
    stats['quote_age']=dict(positive_final_quotes=len(positive),customer_fills=len(won),
        sum_all_positive_age_seconds=sum(positive),sum_filled_age_seconds=sum(won),
        mean_all_positive_age_seconds=float(np.mean(positive)) if positive else None,
        mean_filled_age_seconds=float(np.mean(won)) if won else None,
        maximum_age_seconds=max(positive,default=None))
    stats['exchange_turnover']=sum(t['notional'] for t in own if t['counterparty']!='customer')
    return stats,prices


def sources():
    files=[Path(__file__),PROTOCOL,PRIOR_SEAL,ROOT/'submission/sdk_dealer.py',
        ROOT/'experiments/fresh_sdk_comparison.py',ROOT/'experiments/sdk_study.py',
        ROOT/'experiments/sdk_pnl_decomposition.py',KIT/'examples/dealer.py']
    files+=sorted((KIT/'rfq/rfq').glob('*.py'))
    return complete_source_hashes({source_key(p):sha256(p) for p in files})


def seal(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SEAL.json').exists():raise FileExistsError('Seal already exists')
    seeds=[dict(scenario=s,path=p,stream=stream,seed=seed_for('diagnosis',s,p,stream))
           for s in range(4) for p in range(50)
           for stream in ('factor','basis','customers','opponents/tight','opponents/inventory')]
    if len({r['seed'] for r in seeds})!=len(seeds):raise AssertionError('Primitive seed collision')
    config=dict(stage='prospective quote-freezing diagnosis',scenarios=SCENARIOS,
        paths_per_scenario=50,independent_paths=200,opponent_families=FAMILIES,
        sessions=800,methods=METHODS,parameters={m:policy_parameters(m) for m in METHODS},
        ack_schedule='immediate synchronous; no transport or computation latency',
        price_law_is_official=False,bootstrap_replicates=6000,materiality=1.,
        source_hashes=sources(),primitive_seed_namespace=DOMAIN,seed_collision_count=0,
        predeclared_gate='lower95(normal-freeze)>1 AND normal_mean_pnl>0 AND mean_customer_spread_gap>0 AND freeze/accounting checks')
    write_json(out/'config.json',config)
    write_json(out/'primitive_seeds.json',seeds)
    write_json(out/'environment.json',dict(python=sys.version,numpy=np.__version__,os=platform.platform(),
                                         executable=sys.executable))
    write_json(out/'SEAL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=sources(),config_sha256=sha256(out/'config.json'),
        seed_audit_sha256=sha256(out/'primitive_seeds.json'),diagnosis_generated=False))
    copies=out/'executed_sources'
    for rel in sources():
        target=copies/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source_path(rel).read_bytes())
    print(json.dumps(dict(seal=str(out/'SEAL.json'),sessions=800)))


def fingerprint(row):
    keep=('account','reconciliation','price_tape_sha256','public_customer_tape_sha256',
          'monetary_components','innovation_exposure','decomposition_max_error','own_rejections',
          'callback_errors','freeze_instrumentation','final_rfq_observations','quote_age','exchange_turnover')
    return {k:row[k] for k in keep}


def paired_inference(rows,replicates=6000):
    pairs=[]
    lookup={(r['scenario'],r['path'],r['family'],r['method']):r for r in rows}
    for scenario,path in sorted({(r['scenario'],r['path']) for r in rows}):
        diffs=[]
        for family in FAMILIES:
            a,b=(lookup[scenario,path,family,m] for m in METHODS)
            for key in ('price_tape_sha256','public_customer_tape_sha256'):
                if a[key]!=b[key]:raise AssertionError('Paired primitive histories differ')
            diffs.append(a['account']['pnl']-b['account']['pnl'])
        pairs.append(dict(scenario=scenario,path=path,difference=float(np.mean(diffs))))
    groups=[np.array([r['difference'] for r in pairs if r['scenario']==s]) for s in range(4)]
    if any(len(g)<2 for g in groups):raise ValueError('Need >=2 paths in each stratum')
    rng=np.random.default_rng(seed_for('bootstrap',0,0,'analysis'))
    bootstrap=sum(g[rng.integers(0,len(g),size=(replicates,len(g)))].mean(axis=1)/4 for g in groups)
    lo,hi=map(float,np.quantile(bootstrap,[.025,.975]))
    return dict(normal_minus_freeze=float(np.mean([g.mean() for g in groups])),
        stratified_path_bootstrap_95=[lo,hi],independent_price_customer_paths=sum(map(len,groups)),
        opponent_repeats_are_not_independent=True,bootstrap_replicates=replicates,
        materiality=1.,paths=pairs)


def summaries(rows):
    output={}
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        output[method]=dict(sessions=len(selected),
            mean_pnl=float(np.mean([r['account']['pnl'] for r in selected])),
            mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in selected])),
            mean_customer_notional=float(np.mean([r['account']['customer_notional'] for r in selected])),
            mean_exchange_turnover=float(np.mean([r['exchange_turnover'] for r in selected])),
            mean_monetary_components={k:float(np.mean([r['monetary_components'][k] for r in selected]))
                for k in selected[0]['monetary_components']},
            mean_innovation_q2=np.mean([r['innovation_exposure']['mean_q_squared_at_price_innovation'] for r in selected],axis=0).tolist(),
            mean_innovation_factor_q2=float(np.mean([r['innovation_exposure']['mean_terminal_factor_squared_at_price_innovation'] for r in selected])),
            activated_product_intervals=sum(r['freeze_instrumentation']['activated_product_intervals'] for r in selected),
            omitted_replacements=sum(r['freeze_instrumentation']['omitted_replacements'] for r in selected),
            accepted_live_price_changes=sum(r['freeze_instrumentation']['accepted_live_price_changes'] for r in selected),
            own_rejections=sum(len(r['own_rejections']) for r in selected),
            callback_errors=sum(r['callback_errors'] for r in selected),
            max_decomposition_error=max(r['decomposition_max_error'] for r in selected),
            all_positive_quote_age_seconds=sum(r['quote_age']['sum_all_positive_age_seconds'] for r in selected)/
                max(1,sum(r['quote_age']['positive_final_quotes'] for r in selected)),
            filled_quote_age_seconds=sum(r['quote_age']['sum_filled_age_seconds'] for r in selected)/
                max(1,sum(r['quote_age']['customer_fills'] for r in selected)))
    return output


def run(out):
    started=time.perf_counter();seal_=json.loads((out/'SEAL.json').read_text())
    if seal_['source_hashes']!=sources():raise ValueError('Frozen scientific source changed')
    if seal_['config_sha256']!=sha256(out/'config.json'):raise ValueError('Frozen config changed')
    if (out/'sessions.jsonl.gz').exists():raise FileExistsError('Refuse overwrite')
    rows=[];prices_written=set();failures=[]
    with gzip.open(out/'sessions.jsonl.gz','wt') as raw,gzip.open(out/'price_tapes.jsonl.gz','wt') as tapes:
        for scenario in range(4):
            for path in range(50):
                for family in FAMILIES:
                    for method in METHODS:
                        try:row,prices=run_case(method,scenario,path,family)
                        except BaseException:
                            failures.append(dict(scenario=scenario,path=path,family=family,method=method,
                                traceback=traceback.format_exc()))
                            write_json(out/'failed_checks.json',failures)
                            raise
                        raw.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');rows.append(row)
                        if (scenario,path) not in prices_written:
                            tapes.write(json.dumps(dict(scenario=scenario,path=path,prices=prices),sort_keys=True)+'\n')
                            prices_written.add((scenario,path))
                if path%10==0:print(json.dumps(dict(completed_sessions=len(rows),scenario=scenario,path=path)),flush=True)
    inference=paired_inference(rows);summary=summaries(rows)
    normal,frozen=(summary[m] for m in METHODS)
    checks=dict(lower95_pnl_gap_exceeds_1=inference['stratified_path_bootstrap_95'][0]>1,
        normal_mean_pnl_positive=normal['mean_pnl']>0,
        normal_mean_customer_spread_improves=normal['mean_monetary_components']['customer_spread']>frozen['mean_monetary_components']['customer_spread'],
        scalar_freeze_activated=frozen['activated_product_intervals']>0 and frozen['omitted_replacements']>0,
        normal_did_not_freeze=normal['activated_product_intervals']==normal['omitted_replacements']==0,
        accounting_reconciles=max(normal['max_decomposition_error'],frozen['max_decomposition_error'])<1e-8,
        no_rejections_or_callback_errors=all(v['own_rejections']==v['callback_errors']==0 for v in summary.values()))
    inference['mechanism_checks']=checks
    inference['mechanism_gate']='GO' if all(checks.values()) else 'NO_GO'
    write_json(out/'summary.json',summary);write_json(out/'inference.json',inference)
    reproduce=[]
    with gzip.open(out/'reproduction.jsonl.gz','wt') as repeated:
        for original in rows:
            if original['path']!=0:continue
            replay,_=run_case(original['method'],original['scenario'],0,original['family'])
            equal=fingerprint(replay)==fingerprint(original)
            repeated.write(json.dumps(replay,sort_keys=True,allow_nan=False)+'\n')
            reproduce.append(dict(method=original['method'],scenario=original['scenario'],family=original['family'],equal=equal))
    write_json(out/'reproduction_audit.json',dict(repeats=reproduce,all_scientific_outcomes_identical=all(r['equal'] for r in reproduce),count_as_new_samples=False))
    if not all(r['equal'] for r in reproduce):raise AssertionError('Reproduction discrepancy')
    write_json(out/'failed_checks.json',failures)
    elapsed=time.perf_counter()-started
    write_json(out/'execution.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=elapsed,
        sessions=len(rows),reproduction_sessions=len(reproduce),exit_code=0,source_hashes_unchanged=sources()==seal_['source_hashes']))
    report=f'''# Bounded quote-freezing diagnostic\n\nStage: prospective scalar-B3 mechanism diagnosis. The 800 scheduled sessions and 16 reproduction sessions completed, over 200 independent synthetic price/customer paths. Source/config/primitive seeds were frozen before diagnosis. Cash, fees, inventory and decomposed PnL reconcile (max error {max(v['max_decomposition_error'] for v in summary.values()):.12g}); all 16 scientific replay outcomes are identical. Failed checks/crashes: {len(failures)}.\n\nNormal B3 mean PnL {normal['mean_pnl']:.6f}; frozen B3 {frozen['mean_pnl']:.6f}. Paired normal-minus-freeze {inference['normal_minus_freeze']:.6f}, stratified path-bootstrap 95% interval [{inference['stratified_path_bootstrap_95'][0]:.6f}, {inference['stratified_path_bootstrap_95'][1]:.6f}]. Predeclared mechanism decision: **{inference['mechanism_gate']}**. All gate components are retained in inference.json.\n\nNormal/frozen customer fills per session {normal['mean_customer_fills']:.6f}/{frozen['mean_customer_fills']:.6f}; customer spread {normal['mean_monetary_components']['customer_spread']:.6f}/{frozen['mean_monetary_components']['customer_spread']:.6f}; all-positive accepted quote age {normal['all_positive_quote_age_seconds']:.6f}/{frozen['all_positive_quote_age_seconds']:.6f} seconds; fill quote age {normal['filled_quote_age_seconds']:.6f}/{frozen['filled_quote_age_seconds']:.6f} seconds. Exact quote-age histories and final RFQ observations are archived. Same-price accepted requests do not reset age. Normal/frozen fees, settlement and innovation exposure remain separate in summary.json.\n\nThis compares identical scalar B3 hedge **rules**, normal capacity admission and all parameters except the freeze flag; realised actions differ endogenously. No lifetime admission budget is applied. Freeze activation was checked, rather than inferred from constructor arguments. Original SDK Engine is unchanged, with logical-time callbacks and immediate acknowledgements. The prior scalar B3 parameters were fixed, not selected on these new paths. Opponent families are previously studied development families. Their repetitions are averaged within path before inference.\n\nPermitted claim: EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT only for a freeze mechanism effect if the declared gate is GO, otherwise INCONCLUSIVE/NOT_SUPPORTED as appropriate. This is no claim of reset superiority, online market calibration, production latency safety, profitable competition deployment, or universally harmful frozen quotes. R3 remains disabled. A GO permits one separately frozen redesign and comparison; a NO_GO does not. Actual round/scoring/production clarifications remain unavailable.\n'''
    (out/'REPORT.md').write_text(report)
    write_json(out/'SHA256SUMS.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps(dict(gate=inference['mechanism_gate'],summary=summary,inference={k:v for k,v in inference.items() if k!='paths'},elapsed_seconds=elapsed),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','run']);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    (seal if args.stage=='seal' else run)(out)

if __name__=='__main__':main()
