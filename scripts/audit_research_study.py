"""Independent streaming post-completion audit; no policy/source modifications."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np

METHODS=('B2','B3','B3_factor','B4','B4_factor','BR2','R2B','B0','B4_reduce','B4_zeroquote','R2B_matched','B4_fixedprice','R2B_fixedprice')
CONVENTIONAL=('B2','B3','B3_factor','B4','B4_factor')
BOUNDARY=set(METHODS)-{'B2','B3','B3_factor','B0'}
DOMAIN='research-completion-v1/20261016T134827Z'
FAMILIES=('volatility','sticky')
PUBLIC_ORDER=('order_id','customer_id','product_id','side','quantity','expires_interval')
MONEY=('customer_spread','exchange_execution_edge','fees','exogenous_price_move_pnl',
       'terminal_settlement_residual','penalties','decomposed_pnl')
TOLERANCE=1e-8
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def near(actual,expected,label,maximum):
    difference=abs(float(actual)-float(expected))
    maximum[label]=max(maximum.get(label,0.),difference)
    if not math.isfinite(difference) or difference>TOLERANCE:
        raise AssertionError(f'{label}: actual={actual!r}, expected={expected!r}, difference={difference!r}')


def independent_ledger(row,prices):
    """Independent float replay of this declared unit-multiplier A/B study."""
    maximum={};account=row['account'];n=account['price_version']
    if n>len(prices) or n<2:raise AssertionError('Insufficient matching price tape')
    prices=prices[:n];q={'A':0.,'B':0.};cash=100000.;fees=0.;customer_edge=0.;exchange_edge=0.
    customer_count=exchange_count=0;customer_notional=exchange_notional=0.
    by_version=defaultdict(lambda:defaultdict(list));previous_trade=0
    for trade in row['own_trades']:
        if trade['dealer_id']!='dealer_1' or trade['trade_no']<=previous_trade:
            raise AssertionError('Trade identity/order mismatch')
        previous_trade=trade['trade_no'];pid=trade['product_id'];v=trade['price_version']
        if pid not in q or v<1 or v>n:raise AssertionError('Trade domain mismatch')
        quantity=trade['quantity'];price=trade['price'];fee=trade['fee']
        if quantity<=0 or price<=0 or fee<0 or trade['side'] not in ('BUY','SELL'):
            raise AssertionError('Invalid trade')
        if trade['counterparty'] not in ('customer','exchange') or not float(quantity).is_integer():
            raise AssertionError('Unsupported trade source or non-lot quantity')
        signed=quantity if trade['side']=='BUY' else -quantity
        customer=trade['counterparty']=='customer';kind='customer' if customer else 'exchange'
        expected_fee=0. if customer else quantity*(.01 if pid=='A' else .02)
        near(fee,expected_fee,'trade_fee',maximum)
        near(trade['reference_price'],prices[v-1][pid],'reference_price',maximum)
        notional=quantity*price
        near(trade['notional'],notional,'trade_notional',maximum)
        edge=-signed*(price-prices[v-1][pid])
        if customer:
            ticks=Decimal(str(price))/Decimal('.01')
            if ticks!=ticks.to_integral_value():raise AssertionError('Customer execution not on quote tick')
            customer_edge+=edge;customer_count+=1;customer_notional+=notional
        else:
            near(price,prices[v-1][pid],'exchange_execution_price',maximum)
            exchange_edge+=edge;exchange_count+=1;exchange_notional+=notional
        cash-=signed*price+fee;q[pid]+=signed;fees+=fee
        near(trade['position_after'],q[pid],'position_after_each_trade',maximum)
        if not account['penalties_paid']:near(trade['cash_after'],cash,'cash_after_each_trade',maximum)
        by_version[v][kind].append((pid,signed))
    cash-=account['penalties_paid']
    near(account['cash'],cash,'account_cash',maximum)
    near(account['fees_paid'],fees,'account_fees',maximum)
    near(account['customer_trades'],customer_count,'customer_trade_count',maximum)
    near(account['exchange_trades'],exchange_count,'exchange_trade_count',maximum)
    near(account['customer_notional'],customer_notional,'customer_notional',maximum)
    near(account['exchange_notional'],exchange_notional,'exchange_notional',maximum)
    for pid in q:near(account['positions'][pid],q[pid],'final_position_'+pid,maximum)
    settlement={'A':prices[-1]['A'],'B':prices[-1]['B'] if row['scenario']==7 else 2*prices[-1]['A']}
    if row['settlement_graph']!=('LAST_PRICE' if row['scenario']==7 else 'B=2A'):
        raise AssertionError('Wrong declared settlement graph')
    for pid in q:
        near(row['final_reference'][pid],prices[-1][pid],'final_reference_'+pid,maximum)
        near(row['final_settlement'][pid],settlement[pid],'final_settlement_'+pid,maximum)
    pnl=cash+sum(q[pid]*settlement[pid] for pid in q)-100000.
    near(account['pnl'],pnl,'cash_settlement_pnl',maximum)
    innovation_positions=[];holding_positions=[];exposure_q={'A':0.,'B':0.};price_move=0.
    for version in range(1,n+1):
        for pid,signed in by_version[version]['exchange']:exposure_q[pid]+=signed
        holding_positions.append([exposure_q['A'],exposure_q['B']])
        for pid,signed in by_version[version]['customer']:exposure_q[pid]+=signed
        if version<n:
            innovation_positions.append([exposure_q['A'],exposure_q['B']])
            price_move+=sum(exposure_q[pid]*(prices[version][pid]-prices[version-1][pid]) for pid in exposure_q)
    residual=sum(q[pid]*(settlement[pid]-prices[-1][pid]) for pid in q)
    decomposition=customer_edge+exchange_edge-fees+price_move+residual-account['penalties_paid']
    near(pnl,decomposition,'independent_money_identity',maximum)
    components=dict(customer_spread=customer_edge,exchange_execution_edge=exchange_edge,fees=fees,
        exogenous_price_move_pnl=price_move,terminal_settlement_residual=residual,
        penalties=account['penalties_paid'],decomposed_pnl=decomposition)
    for name,value in components.items():near(row['monetary_components'][name],value,'component_'+name,maximum)
    arr=np.array(innovation_positions);q2=np.mean(arr**2,axis=0);factor2=float(np.mean((arr[:,0]+2*arr[:,1])**2))
    exposure=row['innovation_exposure']
    for index,pid in enumerate(('A','B')):
        near(exposure['mean_q_squared_at_price_innovation'][index],q2[index],'innovation_q2_'+pid,maximum)
        near(exposure['mean_multiplier_exposure_squared_at_price_innovation'][index],q2[index],'multiplier_q2_'+pid,maximum)
    near(exposure['mean_terminal_factor_squared_at_price_innovation'],factor2,'innovation_factor2',maximum)
    near(exposure['price_innovations'],n-1,'innovation_count',maximum)
    near(exposure['price_innovations_with_nonzero_inventory'],int(np.sum(np.any(arr!=0,axis=1))),'nonzero_innovation_count',maximum)
    return dict(components=components,maximum_errors=maximum,innovation_q2=q2.tolist(),innovation_factor2=factor2)


def audit_observations(row,prices):
    obs=row['final_rfq_observations'];maximum={}
    keys=[(o['interval_id'],o['order_id']) for o in obs]
    if len(keys)!=36 or len(set(keys))!=36:raise AssertionError('Missing/duplicate final RFQ')
    public=[{k:o[k] for k in PUBLIC_ORDER} for o in obs]
    public_hash=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    if public_hash!=row['public_customer_tape_sha256']:raise AssertionError('Reconstructed public RFQ tape hash mismatch')
    customer_trades=[t for t in row['own_trades'] if t['counterparty']=='customer']
    own_customers={(t['interval_id'],t['order_id']):t for t in customer_trades}
    if len(own_customers)!=len(customer_trades):raise AssertionError('Duplicate customer trade for one RFQ')
    winning_keys={(o['interval_id'],o['order_id']) for o in obs if o['won']}
    if set(own_customers)!=winning_keys:raise AssertionError('Customer-trade/winning-observation bijection failed')
    positive_ages=[];admitted_delays=[];nonzero_delays=0;ever_but_absent=0
    for o in obs:
        oid=o['order_id'];live=o['own_quote'];age=o['accepted_quote_age_seconds'];first=o['first_positive_virtual_time']
        matching_version=10*o['interval_id']
        if not 1<=matching_version<len(prices):raise AssertionError('RFQ matching version outside declared schedule')
        pid=o['product_id'];reference=prices[matching_version-1][pid];next_reference=prices[matching_version][pid]
        near(o['matching_reference'],reference,'every_rfq_matching_reference',maximum)
        near(o['next_reference'],next_reference,'every_rfq_next_reference',maximum)
        sign=-1 if o['side']=='BUY' else 1
        near(o['dealer_signed_one_tick_markout'],sign*(next_reference-reference),'every_rfq_markout',maximum)
        if (live is None)!=(age is None):raise AssertionError('Quote/age missingness mismatch')
        if live is not None:
            ticks=Decimal(str(live))/Decimal('.01')
            if ticks!=ticks.to_integral_value():raise AssertionError('Live quote not on product tick')
            if age<0 or o['quote_installation_virtual_time']>o['matching_virtual_time']:
                raise AssertionError('Invalid quote age')
            near(age,o['matching_virtual_time']-o['quote_installation_virtual_time'],'quote_age_arithmetic',maximum)
            direction=1 if o['side']=='BUY' else -1
            near(o['edge'],direction*(live-o['matching_reference']),'customer_edge_arithmetic',maximum)
            positive_ages.append(age)
        if first is None:
            if o['admission_delay_seconds'] is not None or live is not None:raise AssertionError('Unadmitted positive quote')
        else:
            delay=first-o['first_seen_virtual_time']
            near(o['admission_delay_seconds'],delay,'admission_delay_arithmetic',maximum)
            if delay<0 or first>o['matching_virtual_time']:raise AssertionError('Invalid admission delay')
            admitted_delays.append(delay);nonzero_delays+=delay>0;ever_but_absent+=live is None
        traded=own_customers.get((o['interval_id'],oid))
        if bool(o['won'])!=(traded is not None):raise AssertionError('Fill and final RFQ winning label differ')
        if 'public_winner' in o and bool(o['won'])!=(o['public_winner']=='dealer_1'):
            raise AssertionError('Own winning label disagrees with public result')
        if traded:
            if traded['product_id']!=o['product_id'] or traded['interval_id']!=o['interval_id']:
                raise AssertionError('Winning RFQ/trade identity mismatch')
            if traded['side']!=('SELL' if o['side']=='BUY' else 'BUY'):raise AssertionError('Customer/dealer signs differ')
            near(traded['quantity'],o['quantity'],'customer_rfq_quantity',maximum)
            near(traded['price'],live,'customer_rfq_execution_price',maximum)
            if 'public_winning_price' in o:
                near(o['public_winning_price'],live,'customer_public_winning_price',maximum)
            near(traded['reference_price'],o['matching_reference'],'customer_matching_reference',maximum)
    age=row['quote_age'];adm=row['admission']
    near(age['positive_final_quotes'],len(positive_ages),'positive_final_quotes',maximum)
    near(age['sum_age_seconds'],sum(positive_ages),'quote_age_sum',maximum)
    near(age['maximum_age_seconds'],max(positive_ages,default=0.),'quote_age_max',maximum)
    near(adm['ever_admitted'],len(admitted_delays),'ever_admitted',maximum)
    near(adm['never_admitted'],36-len(admitted_delays),'never_admitted',maximum)
    near(adm['sum_delay_seconds'],sum(admitted_delays),'admission_delay_sum',maximum)
    near(adm['maximum_delay_seconds'],max(admitted_delays,default=0.),'admission_delay_max',maximum)
    return dict(maximum_errors=maximum,positive_quotes=len(positive_ages),age_sum=sum(positive_ages),
        admitted=len(admitted_delays),admission_delay_sum=sum(admitted_delays),
        nonzero_admission_delay_rfqs=nonzero_delays,ever_admitted_but_absent_at_matching=ever_but_absent)


def fixed_price_quote(observation, prices, parameters):
    """Independent Decimal implementation of the declared causal price rule."""
    version=observation['accepted_quote_price_version']
    if type(version) is not int or not 1<=version<=len(prices):
        raise AssertionError('Invalid installed-quote public price version')
    pid=observation['product_id'];public=prices[version-1]
    coefficients=parameters.get('settlement_coefficients',{}).get(pid)
    basis=0.
    if coefficients is not None:
        basis=float(parameters.get('settlement_constants',{}).get(pid,0.))
        basis+=sum(float(coefficient)*public[root] for root,coefficient in coefficients.items())
        basis-=public[pid]
    fair=public[pid]+float(parameters.get('forecast_weight',0.))*basis
    direction=1 if observation['side']=='BUY' else -1
    unrounded=fair+direction*float(parameters.get('base_edge',.08))
    ticks=(Decimal(str(float(unrounded)))/Decimal('.01')).to_integral_value(
        rounding=ROUND_CEILING if direction==1 else ROUND_FLOOR)
    return float(ticks*Decimal('.01'))


def audit_fixed_prices(row,prices):
    if row['method'] not in ('B4_fixedprice','R2B_fixedprice'):
        return dict(checked=0,observations={},maximum_errors={})
    maximum={};observations={}
    for observation in row['final_rfq_observations']:
        key=(observation['interval_id'],observation['order_id'])
        if observation['own_quote'] is not None:
            expected=fixed_price_quote(observation,prices,row['parameters'])
            near(observation['own_quote'],expected,'fixed_public_quote_rule',maximum)
        observations[key]=observation['own_quote']
    return dict(checked=sum(p is not None for p in observations.values()),
                observations=observations,maximum_errors=maximum)


def recompute_inference(path_differences):
    """Independent implementation of the eight-stratum sealed estimator."""
    primitive=f'{DOMAIN}/analysis/0/0/bootstrap'.encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(primitive).digest(),'big'))
    groups=[np.asarray([value for (s,_),value in sorted(path_differences.items()) if s==scenario])
            for scenario in range(8)]
    if len({len(group) for group in groups})!=1 or any(len(group)<2 for group in groups):
        raise AssertionError('Missing/unbalanced independent path strata')
    bootstrap=np.zeros(5000)
    for group in groups:
        indices=rng.integers(0,len(group),size=(5000,len(group)))
        bootstrap+=np.mean(group[indices],axis=1)/8.
    lower,upper=map(float,np.quantile(bootstrap,[.025,.975]))
    return dict(paired_mean_difference=float(np.mean([float(group.mean()) for group in groups])),
        stratified_path_bootstrap_95=[lower,upper],independent_paths=sum(len(group) for group in groups),
        materiality=1.,superiority_gate=lower>1.)


def audit_pilot(pilot,seal):
    """Inspect all archived pilot accounts and reconstruct selection; no trades exist there."""
    scores=defaultdict(list);parameters={};keys=set();counts=defaultdict(int)
    with gzip.open(pilot/'tuning.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);method=row['method'];candidate=row['candidate']
            key=(method,candidate,row['scenario'],row['path'])
            if (method not in METHODS[:7] or candidate not in range(24)
                    or row['scenario'] not in range(4) or row['path'] not in range(3)
                    or row['family']!=('standard' if row['path']%2==0 else 'wide')
                    or row['phase']!='tuning' or row['crash'] or key in keys):
                raise AssertionError('Invalid/duplicate tuning record')
            if row['own_rejections'] or row['callback_errors']:
                raise AssertionError('Unexplained pilot operational failure')
            if not math.isfinite(row['account']['pnl']):raise AssertionError('Invalid tuning account')
            keys.add(key);scores[(method,candidate)].append(row['account']['pnl'])
            parameters.setdefault((method,candidate),row['parameters'])
            if parameters[(method,candidate)]!=row['parameters']:raise AssertionError('Candidate parameters changed')
            counts[method]+=1
    if any(counts[method]!=24*12 for method in METHODS[:7]) or len(scores)!=7*24:
        raise AssertionError('Unequal/incomplete tuning budgets')
    for method in METHODS[:7]:
        best=max(range(24),key=lambda candidate:np.mean(scores[(method,candidate)]))
        if seal['selected'][method]!=parameters[(method,best)]:
            raise AssertionError('Policy seal differs from raw tuning argmax')
    validation=defaultdict(list);seen=set();per_method=defaultdict(list)
    with gzip.open(pilot/'validation.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);method=row['method'];key=(row['scenario'],row['path'],row['family'],method)
            if (method not in METHODS[:7] or row['scenario'] not in range(4)
                    or row['path'] not in range(12) or row['family'] not in ('standard','wide')
                    or row['phase']!='validation' or row['crash'] or key in seen
                    or row['parameters']!=seal['selected'][method]):
                raise AssertionError('Invalid/duplicate/unsealed validation record')
            if row['own_rejections'] or row['callback_errors']:raise AssertionError('Pilot operational failure')
            seen.add(key);validation[key[:3]].append((method,row['account']['pnl']))
            per_method[method].append(row['account']['pnl'])
    if len(seen)!=4*12*2*7 or any(len(per_method[m])!=96 for m in METHODS[:7]):
        raise AssertionError('Incomplete validation accounts')
    conventional_means={m:float(np.mean(per_method[m])) for m in CONVENTIONAL}
    fallback=max(CONVENTIONAL,key=lambda m:conventional_means[m])
    if fallback!=seal['fallback']:raise AssertionError('Fallback differs from validation argmax')
    path_diffs=defaultdict(list)
    for key,items in validation.items():
        lookup=dict(items)
        if len(lookup)!=7:raise AssertionError('Missing paired validation method')
        path_diffs[key[:2]].append(lookup['R2B']-lookup[fallback])
    sd=float(np.std([np.mean(v) for _,v in sorted(path_diffs.items())],ddof=1))
    required=math.ceil(((1.960+.842)*sd)**2)
    n=min(2048,8*math.ceil(max(512,required)/8))
    maximum={}
    near(seal['validation_paired_pooled_sd'],sd,'pilot_sd',maximum)
    if (seal['required_independent_paths']!=required or seal['final_paths']!=n
            or seal['paths_per_scenario']!=n//8 or seal['precision_cap_binds']!=(required>2048)):
        raise AssertionError('Prospective sample-size plan differs from raw pilot')
    for method,value in conventional_means.items():
        near(seal['validation_conventional_means'][method],value,'pilot_validation_mean',maximum)
    return dict(tuning_sessions=len(keys),validation_sessions=len(seen),
        per_method_tuning_sessions=dict(counts),fallback=fallback,
        conventional_validation_means=conventional_means,
        required_independent_paths=required,final_paths=n,precision_cap_binds=required>2048,
        maximum_errors=maximum,pilot_trade_replay_available=False,
        note='Pilot archives accounts/parameters, but not all trades/RFQs; complete ledger replay below covers final only.')


def check_parameters(row,seal):
    expected={} if row['method']=='B0' else dict(seal['selected'][row['method']])
    if row['scenario']==7 and row['method']!='B0':expected['settlement_coefficients']={}
    if row['parameters']!=expected:raise AssertionError('Unsealed or wrong scenario-local parameters')


def economic_fingerprint(row):
    """Hash all archived economics, excluding documented host timing only."""
    stable={key:value for key,value in row.items() if key!='callback_seconds'}
    stable['own_trades']=[{key:value for key,value in trade.items() if key!='time'}
                          for trade in row['own_trades']]
    return hashlib.sha256(json.dumps(stable,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_sequential_prefix(study,seal):
    path=study.parent/'study_final_sequential_prefix'
    if not path.exists():return {},None
    prefix_seal=json.loads((path/'POLICY_SEAL.json').read_text())
    if prefix_seal!=seal:raise AssertionError('Interrupted prefix used a different economic seal')
    hashes={}
    with gzip.open(path/'sessions.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            key=(row['scenario'],row['path'],row['family'],row['method'])
            if key in hashes:raise AssertionError('Duplicate interrupted-prefix record')
            hashes[key]=economic_fingerprint(row)
    return hashes,path


def audit_orchestration(study,tapes):
    original_path=study/'ORCHESTRATION_SEAL.json'
    if not original_path.exists():return dict(parallel_orchestration=False)
    original=json.loads(original_path.read_text())
    if sha(ROOT/'scripts/run_research_final_parallel.py')!=original['driver_sha256']:
        raise AssertionError('Original archived parallel source changed')
    repair_path=study/'ORCHESTRATION_REPAIR_SEAL.json'
    if not repair_path.exists():raise AssertionError('Timestamp repair is not explicitly sealed')
    repair=json.loads(repair_path.read_text())
    if (repair['assembler_sha256']!=sha(ROOT/'scripts/assemble_research_final.py')
            or repair['original_driver_sha256']!=original['driver_sha256']
            or repair['excluded_fields']!=['callback_seconds','own_trades[*].time']
            or repair['outcome_exclusions']!=0 or repair['new_independent_N_from_replay']!=0
            or not repair['economic_policy_seal_unchanged']):
        raise AssertionError('Unsealed/substantive orchestration amendment')
    failure=json.loads((study/'REPLAY_FAILURE.json').read_text())
    if economic_fingerprint(failure['sequential'])!=economic_fingerprint(failure['parallel']):
        raise AssertionError('Original orchestration failure changed economic information')
    shard_records=0;record_hashes={}
    for scenario in range(8):
        path=study/f'shard_{scenario}.jsonl.gz'
        if sha(path)!=repair['shard_sha256'][str(scenario)]:
            raise AssertionError('Parallel raw shard changed after repair seal')
        with gzip.open(study/f'tapes_{scenario}.json.gz','rt') as stream:shard_tapes=json.load(stream)
        expected={key for key in tapes if key.startswith(str(scenario)+'/')}
        if set(shard_tapes)!=expected or any(shard_tapes[key]!=tapes[key] for key in expected):
            raise AssertionError('Merged price tape differs from completed shard')
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['scenario']!=scenario:raise AssertionError('Wrong scenario in sealed shard')
                key=(row['scenario'],row['path'],row['family'],row['method'])
                if key in record_hashes:raise AssertionError('Duplicate raw record across parallel shards')
                record_hashes[key]=hashlib.sha256(line.encode()).hexdigest()
                shard_records+=1
    return dict(parallel_orchestration=True,original_driver_preserved=True,
        timestamp_only_initial_failure_confirmed=True,all_shard_hashes_match=True,
        all_price_tape_shards_match=True,sealed_shard_records=shard_records,
        _record_hashes=record_hashes,
        no_new_independent_N=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pilot',type=Path)
    args=parser.parse_args();args.study=args.study.resolve();args.out=args.out.resolve()
    args.pilot=(args.pilot or args.study.parent/'study_pilot').resolve()
    started=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
    complete=json.loads((args.study/'execution.json').read_text())
    seal=json.loads((args.study/'POLICY_SEAL.json').read_text())
    expected_sessions=seal['final_paths']*len(METHODS)*len(FAMILIES)
    if (complete['scheduled_sessions']!=expected_sessions or not complete['source_unchanged']
            or complete['competition_scope'] or complete['R3_enabled']):
        raise AssertionError('Experiment completion/source/scope guard failed')
    if args.out.exists():raise FileExistsError('Refuse audit overwrite')
    args.out.mkdir(parents=True)
    dump(args.out/'config.json',dict(study=str(args.study),pilot=str(args.pilot),no_policy_retuning=True,
        count_as_new_performance_evidence=False,all_final_records_required=True,tolerance=TOLERANCE,
        methods=METHODS,families=FAMILIES,scenario_count=8,settlement_last_price_scenario=7,
        multiplier_scope={'A':1.,'B':1.},starting_cash=100000.,
        independence='cash/quote/inference formulas reimplemented without importing experiment or dealer',
        penalty_timing='authoritative total retained; intermediate checkpoint timestamps not archived'))
    dump(args.out/'environment.json',dict(python=sys.version,numpy=np.__version__,os=platform.platform(),executable=sys.executable))
    failures=[];maximum={};method_rows=defaultdict(list);groups={};path_groups=defaultdict(list)
    total=trades=customer_trades=prefix_checked=0;common_fixed_quotes=one_fixed_quote=both_fixed_absent=fixed_rule_checks=0
    try:
        pilot=audit_pilot(args.pilot,seal)
        prefix_hashes,prefix_path=load_sequential_prefix(args.study,seal)
        with gzip.open(args.study/'price_tapes.json.gz','rt') as stream:tapes=json.load(stream)
        expected_keys={f'{s}/{p}' for s in range(8) for p in range(seal['paths_per_scenario'])}
        if set(tapes)!=expected_keys:raise AssertionError('Missing/extra archived price paths')
        for key,prices in tapes.items():
            if len(prices)!=124 or any(set(p)!=set(('A','B')) or
                    any(not math.isfinite(v) or v<=0 for v in p.values()) for p in prices):
                raise AssertionError('Invalid/short/extra price tape')
        orchestration=audit_orchestration(args.study,tapes)
        shard_record_hashes=orchestration.pop('_record_hashes',{})
        pricehashes={key:hashlib.sha256(json.dumps([dict(version=i+1,prices=p)
            for i,p in enumerate(prices)],sort_keys=True).encode()).hexdigest() for key,prices in tapes.items()}
        with gzip.open(args.study/'sessions.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);total+=1;method=row['method']
                key=(row['scenario'],row['path'],row['family'])
                if (method not in METHODS or row['family'] not in FAMILIES or row['phase']!='final'
                        or row['crash'] or row['scenario'] not in range(8)
                        or row['path'] not in range(seal['paths_per_scenario'])):
                    raise AssertionError('Invalid scheduled final record')
                full_key=key+(method,)
                if shard_record_hashes and shard_record_hashes.get(full_key)!=hashlib.sha256(line.encode()).hexdigest():
                    raise AssertionError('Merged raw session differs from sealed completed shard')
                if full_key in prefix_hashes:
                    if prefix_hashes[full_key]!=economic_fingerprint(row):
                        raise AssertionError('Parallel replay changed an interrupted-prefix economic record')
                    prefix_checked+=1
                check_parameters(row,seal)
                expected_seed=int.from_bytes(hashlib.sha256(
                    f'{DOMAIN}/final/{key[0]}/{key[1]}/customers'.encode()).digest(),'big')
                if row['seed']!=expected_seed:raise AssertionError('Wrong final customer primitive seed')
                if row['own_rejections'] or row['callback_errors']:raise AssertionError('Unexplained final operational failure')
                counters=row['boundary_counters']
                if method in BOUNDARY:
                    if counters['epochs']!=13:raise AssertionError('Missing boundary epoch')
                    if any(counters[k] for k in ('unsupported_histories','partial_batches','duplicate_events','deferred_items')):
                        raise AssertionError('Unexpected boundary refusal/transport/action state')
                    if counters['execution_items']!=row['account']['exchange_trades']:
                        raise AssertionError('Boundary actual executions differ from account')
                    if (counters['actions']>counters['epochs'] or counters['actions']>counters['execution_items']
                            or counters['nonbaseline_decisions']>counters['model_decisions']):
                        raise AssertionError('Impossible boundary action/model counters')
                elif counters:raise AssertionError('Ordinary control has boundary counters')
                pricekey=f'{key[0]}/{key[1]}';prices=tapes[pricekey]
                if row['account']['price_version']!=124 or row['price_tape_sha256']!=pricehashes[pricekey]:
                    raise AssertionError('Final window/price tape hash mismatch')
                monetary=independent_ledger(row,prices);obs=audit_observations(row,prices)
                fixed=audit_fixed_prices(row,prices);fixed_rule_checks+=fixed['checked']
                for name,value in {**monetary['maximum_errors'],**obs['maximum_errors'],**fixed['maximum_errors']}.items():
                    maximum[name]=max(maximum.get(name,0.),value)
                trades+=len(row['own_trades']);customer_trades+=row['account']['customer_trades']
                compact=dict(account=row['account'],components=monetary['components'],
                    innovation_q2=monetary['innovation_q2'],innovation_factor2=monetary['innovation_factor2'],
                    observations=obs,scenario=key[0],path=key[1],family=key[2],counters=counters,
                    price_hash=row['price_tape_sha256'],customer_hash=row['public_customer_tape_sha256'],
                    fixed_prices=fixed['observations'],basis_opening_actions=row['basis_opening_actions'],
                    callback_maximum=max(c['maximum'] for c in row['callback_seconds'].values()))
                methods=groups.setdefault(key,{})
                if method in methods:raise AssertionError('Duplicate raw session')
                methods[method]=compact;method_rows[method].append(compact)
                if len(methods)==len(METHODS):
                    if set(methods)!=set(METHODS):raise AssertionError('Missing policy family')
                    if (len({r['price_hash'] for r in methods.values()})!=1
                            or len({r['customer_hash'] for r in methods.values()})!=1):
                        raise AssertionError('Paired primitive histories differ')
                    a,b=methods['B4_fixedprice']['fixed_prices'],methods['R2B_fixedprice']['fixed_prices']
                    if set(a)!=set(b):raise AssertionError('Fixed-price arms have different public RFQ keys')
                    for oid in a:
                        if a[oid] is not None and b[oid] is not None:
                            near(a[oid],b[oid],'paired_fixed_price_when_both_live',maximum);common_fixed_quotes+=1
                        elif a[oid] is not None or b[oid] is not None:one_fixed_quote+=1
                        else:both_fixed_absent+=1
                    path_groups[key[:2]].append((key[2],{m:r['account']['pnl'] for m,r in methods.items()}))
                    del groups[key]
                if total%8000==0:print(json.dumps(dict(audited_sessions=total)),flush=True)
        if groups or total!=expected_sessions or len(path_groups)!=seal['final_paths']:
            raise AssertionError('Incomplete scheduled final experiment')
        if orchestration.get('sealed_shard_records',total)!=total:
            raise AssertionError('Merged record count differs from completed raw shards')
        if prefix_checked!=len(prefix_hashes):raise AssertionError('Some interrupted-prefix records were not replayed')
        if prefix_hashes and (complete.get('sequential_prefix_reproduced')!=prefix_checked
                or not complete.get('economic_replay_exact') or not complete.get('no_new_N_from_prefix')):
            raise AssertionError('Driver replay metadata disagrees with independent prefix comparison')
        for key,repeats in path_groups.items():
            if len(repeats)!=2 or {family for family,_ in repeats}!=set(FAMILIES):
                raise AssertionError('Incomplete opponent grouping')
        comparisons={'primary':('R2B',seal['fallback']),
            'R2B_minus_B4':('R2B','B4'),'R2B_matched_minus_B4':('R2B_matched','B4'),
            'B4_minus_reduce':('B4','B4_reduce'),'B4_minus_zeroquote':('B4','B4_zeroquote'),
            'B4_minus_B3':('B4','B3'),'BR2_minus_B4':('BR2','B4'),
            'R2B_fixedprice_minus_B4_fixedprice':('R2B_fixedprice','B4_fixedprice')}
        recorded_ablations=json.loads((args.study/'descriptive_ablations.json').read_text());inferences={}
        for name,(left,right) in comparisons.items():
            differences={key:float(np.mean([pnls[left]-pnls[right] for _,pnls in repeats]))
                         for key,repeats in path_groups.items()}
            independent=recompute_inference(differences);inferences[name]=independent
            reported=json.loads((args.study/'primary_inference.json').read_text()) if name=='primary' else recorded_ablations[name]
            for field in ('paired_mean_difference','independent_paths','materiality'):
                near(reported[field],independent[field],'inference_'+field,maximum)
            for index in (0,1):near(reported['stratified_path_bootstrap_95'][index],
                independent['stratified_path_bootstrap_95'][index],'inference_endpoint_'+str(index),maximum)
            if reported['superiority_gate']!=independent['superiority_gate']:raise AssertionError('Inference gate differs')
        reported=json.loads((args.study/'summary.json').read_text());summaries={}
        for method,rows in method_rows.items():
            pnls=np.asarray([r['account']['pnl'] for r in rows])
            result=dict(sessions=len(rows),mean_pnl=float(pnls.mean()),median_pnl=float(np.median(pnls)),
                worst_pnl=float(pnls.min()),pnl_quantile_05=float(np.quantile(pnls,.05)),
                negative_sessions=int((pnls<0).sum()),zero_pnl_sessions=int((pnls==0).sum()),
                mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in rows])),
                customer_notional=sum(r['account']['customer_notional'] for r in rows),
                exchange_notional=sum(r['account']['exchange_notional'] for r in rows),
                fees=sum(r['account']['fees_paid'] for r in rows),penalties=sum(r['account']['penalties_paid'] for r in rows),
                mean_monetary_components={name:float(np.mean([r['components'][name] for r in rows])) for name in MONEY},
                mean_q2_at_innovation=np.mean([r['innovation_q2'] for r in rows],axis=0).tolist(),
                scenario_means={str(s):float(np.mean([r['account']['pnl'] for r in rows if r['scenario']==s])) for s in range(8)},
                maximum_callback_seconds=max(r['callback_maximum'] for r in rows),
                boundary_counters={name:sum(r['counters'].get(name,0) for r in rows) for name in {k for r in rows for k in r['counters']}},
                proposed_basis_opening_actions=sum(r['basis_opening_actions'] for r in rows),
                descriptive_session_downside_mean_at_or_below_q05=float(pnls[pnls<=np.quantile(pnls,.05)].mean()))
            for name in ('sessions','mean_pnl','median_pnl','worst_pnl','pnl_quantile_05','negative_sessions',
                         'mean_customer_fills','customer_notional','exchange_notional','fees','penalties','maximum_callback_seconds'):
                near(reported[method][name],result[name],'summary_'+name,maximum)
            for name,value in result['mean_monetary_components'].items():
                near(reported[method]['mean_monetary_components'][name],value,'summary_component_'+name,maximum)
            for name,value in result['scenario_means'].items():
                near(reported[method]['scenario_means'][name],value,'summary_scenario_'+name,maximum)
            for index,value in enumerate(result['mean_q2_at_innovation']):
                near(reported[method]['mean_q2_at_innovation'][index],value,'summary_q2_'+str(index),maximum)
            summaries[method]=result
        for rel,digest in seal['source_hashes'].items():
            if sha(args.study/'executed_sources'/rel)!=digest:raise AssertionError('Executed source differs from seal')
        result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            final_sessions=total,independent_paths=seal['final_paths'],final_rfqs=36*total,own_trades=trades,
            customer_trades=customer_trades,records_excluded=0,pilot=pilot,failed_checks=[],
            maximum_absolute_errors=maximum,inferences=inferences,methods=summaries,
            fixed_price_rule_checks=fixed_rule_checks,paired_fixed_prices_both_live=common_fixed_quotes,
            paired_fixed_prices_one_live=one_fixed_quote,paired_fixed_prices_both_absent=both_fixed_absent,
            interrupted_prefix_economic_records_reproduced=prefix_checked,
            prefix_replay_adds_independent_N=False,
            economic_fingerprint_excludes=['callback_seconds','own_trades[*].time'],
            orchestration=orchestration,
            source_snapshots_match_seal=True,competition_scope=False,R3_enabled=False,
            permitted_claim='I/NUMERICALLY_CHECKED: archived research-account, RFQ, quote-rule and inference integrity only')
        dump(args.out/'analysis.json',result)
        dump(args.out/'summary.json',dict(passed=True,failed_checks=[],final_sessions=total,
            independent_paths=seal['final_paths'],final_rfqs=36*total,own_trades=trades,
            interrupted_prefix_reproduced=prefix_checked,records_excluded=0,
            primary=inferences['primary'],details='analysis.json'))
        input_paths=[args.study/name for name in ('sessions.jsonl.gz','price_tapes.json.gz','POLICY_SEAL.json',
            'summary.json','primary_inference.json','descriptive_ablations.json','execution.json')]
        input_paths += [args.pilot/'tuning.jsonl.gz',args.pilot/'validation.jsonl.gz',Path(__file__)]
        if prefix_path is not None:
            input_paths += [prefix_path/'sessions.jsonl.gz',prefix_path/'POLICY_SEAL.json']
        input_paths += sorted(args.study.glob('ORCHESTRATION*.json'))
        if orchestration.get('parallel_orchestration'):
            input_paths += [ROOT/'scripts/run_research_final_parallel.py',ROOT/'scripts/assemble_research_final.py',
                            args.study/'REPLAY_FAILURE.json']
            input_paths += [args.study/f'shard_{s}.jsonl.gz' for s in range(8)]
            input_paths += [args.study/f'tapes_{s}.json.gz' for s in range(8)]
        dump(args.out/'input_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in input_paths})
        primary=inferences['primary']
        (args.out/'REPORT.md').write_text(f'''# Independent research completion audit\n\nAudited all {total:,} final sessions, {36*total:,} final RFQs and {trades:,} own trades over {seal['final_paths']:,} independent price/customer paths. No record was excluded. All archived price tapes, unique RFQs, dealer/customer signs, actual principal, fees, cash, inventory, matched references, terminal settlement, monetary decompositions and price-innovation exposures reconcile. Scenario 7 independently uses LAST_PRICE for both products; the other seven use B=2A. Maximum discrepancy {max(maximum.values()):.12g}, tolerance 1e-8.\n\nPilot selection was separately reconstructed from all {pilot['tuning_sessions']:,} tuning and {pilot['validation_sessions']:,} validation accounts. Equal family budgets, chosen candidate, validation fallback and predeclared sample-size calculation agree. Pilot trade/RFQ histories were not archived in those files, so complete independent trade-ledger replay is claimed only for final sessions. The deliberately interrupted sequential prefix contains {prefix_checked:,} records. Every account, trade, RFQ, parameter and price/exposure field reproduces exactly in the parallel execution after excluding callback runtime and raw exchange wall-clock timestamps. These duplicate replay records add zero independent paths and are not excluded losses.\n\nThe primary raw-account result is R2B minus {seal['fallback']}: {primary['paired_mean_difference']:.6f}, stratified path-bootstrap 95% interval {primary['stratified_path_bootstrap_95']}; materiality gate {'PASS' if primary['superiority_gate'] else 'FAIL'}. Both opponent repetitions are averaged inside each independent path. All seven descriptive comparisons also reproduce. Approximate bootstrap coverage is an empirical assumption, not a safety certificate.\n\nIndependently checked {fixed_rule_checks:,} positive fixed-arm quotes using only their archived installation-time public state and Decimal tick rounding. Both arms were live for {common_fixed_quotes:,} paired RFQs and had equal prices; only one was live for {one_fixed_quote:,}. Different capacity/admission remains an intervention outcome. This does not freeze executions. Quote age/admission arithmetic uses archived instrumentation rather than a complete independent accepted-action log. Penalty totals are retained; intermediate checkpoint timing cannot be independently reconstructed.\n\nStatus: accounting/inference integrity NUMERICALLY_CHECKED in the declared supplied SDK synthetic experiment. Performance, novelty and response claims require their separate evidence and cannot be inferred from this audit. Competition participation is outside scope. No policy was retuned and these replayed archived cases add no independent performance sample.\n''')
    except BaseException:
        import traceback
        failures.append(dict(records_read=total,traceback=traceback.format_exc()))
        dump(args.out/'FAILED_CHECKS.json',failures)
        dump(args.out/'summary.json',dict(passed=False,failed_checks=failures,records_read=total))
        raise
    dump(args.out/'commands.log',dict(utc=utc,argv=[sys.executable]+sys.argv,elapsed_seconds=time.perf_counter()-started,exit_code=0))
    dump(args.out/'SHA256SUMS.json',{str(p.relative_to(args.out)):sha(p) for p in sorted(args.out.rglob('*'))
                                  if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('methods','maximum_absolute_errors','pilot')},indent=2))


if __name__=='__main__':main()
