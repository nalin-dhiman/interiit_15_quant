"""Independent streaming post-completion audit; no policy/source modifications."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np

METHODS=('B1','B2','B3','B3_factor','BB3','BR2','B0','BR2_BB3_quotes')
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
        signed=quantity if trade['side']=='BUY' else -quantity
        customer=trade['counterparty']=='customer';kind='customer' if customer else 'exchange'
        expected_fee=0. if customer else quantity*(.01 if pid=='A' else .02)
        near(fee,expected_fee,'trade_fee',maximum)
        near(trade['reference_price'],prices[v-1][pid],'reference_price',maximum)
        notional=quantity*price
        near(trade['notional'],notional,'trade_notional',maximum)
        edge=-signed*(price-prices[v-1][pid])
        if customer:
            customer_edge+=edge;customer_count+=1;customer_notional+=notional
        else:
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
    settlement={'A':prices[-1]['A'],'B':2*prices[-1]['A']}
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
        if traded:
            if traded['product_id']!=o['product_id'] or traded['interval_id']!=o['interval_id']:
                raise AssertionError('Winning RFQ/trade identity mismatch')
            if traded['side']!=('SELL' if o['side']=='BUY' else 'BUY'):raise AssertionError('Customer/dealer signs differ')
            near(traded['quantity'],o['quantity'],'customer_rfq_quantity',maximum)
            near(traded['price'],live,'customer_rfq_execution_price',maximum)
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


def recompute_inference(path_differences):
    # Reimplement the sealed algorithm without importing the experiment's estimator.
    seed=int.from_bytes(hashlib.sha256(b'boundary-policy-v1/20261016T103800Z/analysis/0/0/bootstrap').digest(),'big')
    rng=np.random.default_rng(seed)
    groups=[np.asarray([value for (s,_),value in sorted(path_differences.items()) if s==scenario]) for scenario in range(4)]
    if len({len(g) for g in groups})!=1 or any(len(g)<2 for g in groups):raise AssertionError('Unbalanced path strata')
    bootstrap=np.zeros(5000)
    for group in groups:
        sampled=rng.integers(0,len(group),size=(5000,len(group)))
        bootstrap+=np.mean(group[sampled],axis=1)*.25
    lower,upper=map(float,np.quantile(bootstrap,[.025,.975]))
    mean=float(np.mean([float(g.mean()) for g in groups]))
    standard_error=math.sqrt(sum(float(np.var(g,ddof=1))/len(g) for g in groups)/16.)
    p=.5*math.erfc((mean-1.)/(math.sqrt(2.)*standard_error)) if standard_error else float(mean<=1.)
    return dict(paired_mean_difference=mean,stratified_path_bootstrap_95=[lower,upper],
        independent_paths=sum(len(g) for g in groups),approximate_one_sided_materiality_p=p,
        superiority_gate=lower>1.,materiality=1.,standard_error=standard_error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.study=args.study.resolve();args.out=args.out.resolve();started=time.perf_counter()
    # This file is not created until the complete scheduled comparison and its inference finish.
    complete=json.loads((args.study/'execution.json').read_text())
    seal=json.loads((args.study/'POLICY_SEAL.json').read_text())
    if complete['scheduled_sessions']!=seal['final_paths']*16 or not complete['source_unchanged']:
        raise AssertionError('Experiment completion/source guard failed')
    if args.out.exists():raise FileExistsError('Refuse audit overwrite')
    args.out.mkdir(parents=True);dump(args.out/'config.json',dict(study=str(args.study),policy_changed=False,
        all_raw_sessions_required=True,tolerance=TOLERANCE,independent_arithmetic=True,read_only_after_complete=True,
        actual_SDK_rule_scope='supplied A/B demo, unit multipliers, no impact, matching then exogenous innovation',
        inference_scope='stratified independent path bootstrap; opponent repeats grouped; empirical approximation'))
    dump(args.out/'environment.json',dict(python=sys.version,numpy=np.__version__,os=platform.platform(),executable=sys.executable))
    failures=[];max_errors={};method_rows=defaultdict(list);groups={};path_groups=defaultdict(list);total=0
    with gzip.open(args.study/'price_tapes.json.gz','rt') as stream:tapes=json.load(stream)
    if len(tapes)!=seal['final_paths']:raise AssertionError('Missing/duplicate archived price tape')
    pricehashes={}
    try:
        with gzip.open(args.study/'sessions.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);total+=1;method=row['method'];key=(row['scenario'],row['path'],row['family'])
                if method not in METHODS or row['family'] not in FAMILIES or row['phase']!='final' or row['crash']:
                    raise AssertionError('Unexpected phase/method/family/crash')
                expected_params={} if method=='B0' else seal['selected'][method]
                if row['parameters']!=expected_params:raise AssertionError('Unsealed policy parameters')
                if row['own_rejections'] or row['callback_errors']:raise AssertionError('Unexpected operational failure')
                counters=row['boundary_counters']
                if method in ('BB3','BR2','BR2_BB3_quotes'):
                    if counters['epochs']!=13:raise AssertionError('Missing boundary epoch')
                    if any(counters[k] for k in ('unsupported_histories','partial_batches','duplicate_events','deferred_items')):
                        raise AssertionError('Unexpected boundary state counter')
                    if counters['execution_items']!=row['account']['exchange_trades']:
                        raise AssertionError('Boundary executions/account mismatch')
                    if counters['actions']>counters['epochs'] or counters['actions']>counters['execution_items']:
                        raise AssertionError('Boundary action/event count impossible')
                    if counters['nonbaseline_decisions']>counters['model_decisions']:
                        raise AssertionError('Boundary model decisions mismatch')
                elif counters:raise AssertionError('Ordinary policy unexpectedly has boundary counters')
                pricekey=f"{row['scenario']}/{row['path']}";prices=tapes[pricekey];n=row['account']['price_version']
                hkey=(pricekey,n)
                if hkey not in pricehashes:
                    pricehashes[hkey]=hashlib.sha256(json.dumps([dict(version=i+1,prices=p) for i,p in enumerate(prices[:n])],sort_keys=True).encode()).hexdigest()
                if pricehashes[hkey]!=row['price_tape_sha256']:raise AssertionError('Exact archived price tape hash mismatch')
                monetary=independent_ledger(row,prices);observations=audit_observations(row,prices)
                for name,value in {**monetary['maximum_errors'],**observations['maximum_errors']}.items():max_errors[name]=max(max_errors.get(name,0.),value)
                compact=dict(account=row['account'],components=monetary['components'],innovation_q2=monetary['innovation_q2'],
                    innovation_factor2=monetary['innovation_factor2'],observations=observations,
                    scenario=row['scenario'],path=row['path'],family=row['family'],counters=counters,
                    quote_age=row['quote_age'],admission=row['admission'],
                    price_hash=row['price_tape_sha256'],customer_hash=row['public_customer_tape_sha256'])
                methods=groups.setdefault(key,{})
                if method in methods:raise AssertionError('Duplicate scheduled raw policy session')
                methods[method]=compact;method_rows[method].append(compact)
                if len(methods)==8:
                    if set(methods)!=set(METHODS):raise AssertionError('Incomplete method family')
                    if len({r['price_hash'] for r in methods.values()})!=1 or len({r['customer_hash'] for r in methods.values()})!=1:
                        raise AssertionError('Paired histories differ')
                    difference=methods['BR2']['account']['pnl']-methods[seal['fallback']]['account']['pnl']
                    path_groups[key[:2]].append((key[2],difference));del groups[key]
                if total%8000==0:print(json.dumps(dict(audited_sessions=total)),flush=True)
        if groups or total!=complete['scheduled_sessions'] or len(path_groups)!=seal['final_paths']:
            raise AssertionError('Incomplete raw scheduled experiment')
        if any(set(fam for fam,_ in repetitions)!=set(FAMILIES) or len(repetitions)!=2 for repetitions in path_groups.values()):
            raise AssertionError('Incomplete independent path opponent grouping')
        diffs={key:float(np.mean([value for _,value in repetitions])) for key,repetitions in path_groups.items()}
        independent=recompute_inference(diffs);recorded=json.loads((args.study/'primary_inference.json').read_text())
        for field in ('paired_mean_difference','approximate_one_sided_materiality_p','independent_paths'):
            near(recorded[field],independent[field],'inference_'+field,max_errors)
        for index in (0,1):near(recorded['stratified_path_bootstrap_95'][index],independent['stratified_path_bootstrap_95'][index],f'inference_endpoint_{index}',max_errors)
        if recorded['superiority_gate']!=independent['superiority_gate']:raise AssertionError('Primary decision mismatch')
        reported=json.loads((args.study/'summary.json').read_text());summary={}
        for method,rows in method_rows.items():
            pnls=np.array([r['account']['pnl'] for r in rows]);quote_count=sum(r['observations']['positive_quotes'] for r in rows)
            admitted=sum(r['observations']['admitted'] for r in rows);names={k for r in rows for k in r['counters']}
            result=dict(sessions=len(rows),mean_pnl=float(pnls.mean()),median_pnl=float(np.median(pnls)),
                worst_pnl=float(pnls.min()),pnl_quantile_05=float(np.quantile(pnls,.05)),negative_sessions=int(np.sum(pnls<0)),
                zero_pnl_sessions=int(np.sum(pnls==0)),customer_orders=sum(r['account']['customer_trades'] for r in rows),
                customer_notional=sum(r['account']['customer_notional'] for r in rows),exchange_notional=sum(r['account']['exchange_notional'] for r in rows),
                fees=sum(r['account']['fees_paid'] for r in rows),penalties=sum(r['account']['penalties_paid'] for r in rows),
                mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in rows])),
                mean_quote_age_seconds=sum(r['observations']['age_sum'] for r in rows)/quote_count if quote_count else None,
                mean_admission_delay_seconds=sum(r['observations']['admission_delay_sum'] for r in rows)/admitted if admitted else None,
                never_admitted_rfqs=36*len(rows)-admitted,boundary_counters={name:sum(r['counters'].get(name,0) for r in rows) for name in names},
                mean_monetary_components={name:float(np.mean([r['components'][name] for r in rows])) for name in MONEY},
                mean_innovation_factor_q2=float(np.mean([r['innovation_factor2'] for r in rows])),
                scenario_means={str(s):float(np.mean([r['account']['pnl'] for r in rows if r['scenario']==s])) for s in range(4)})
            for name in ('sessions','mean_pnl','median_pnl','worst_pnl','pnl_quantile_05','negative_sessions','zero_pnl_sessions',
                'customer_orders','customer_notional','exchange_notional','fees','penalties','mean_customer_fills','mean_quote_age_seconds',
                'mean_admission_delay_seconds','never_admitted_rfqs','mean_innovation_factor_q2'):
                if result[name] is None:
                    if reported[method][name] is not None:raise AssertionError('Missing summary mismatch')
                else:near(reported[method][name],result[name],'summary_'+name,max_errors)
            for name,value in result['mean_monetary_components'].items():near(reported[method]['mean_monetary_components'][name],value,'summary_component_'+name,max_errors)
            for name,value in result['scenario_means'].items():near(reported[method]['scenario_means'][name],value,'summary_scenario_'+name,max_errors)
            if result['boundary_counters']!=reported[method]['boundary_counters']:raise AssertionError('Aggregated state counters differ')
            result.update(mean_innovation_q2=np.mean([r['innovation_q2'] for r in rows],axis=0).tolist(),
                nonzero_admission_delay_rfqs=sum(r['observations']['nonzero_admission_delay_rfqs'] for r in rows),
                ever_admitted_but_absent_at_matching=sum(r['observations']['ever_admitted_but_absent_at_matching'] for r in rows),
                descriptive_session_downside_mean_at_or_below_q05=float(pnls[pnls<=np.quantile(pnls,.05)].mean()))
            summary[method]=result
        snapshot_hashes={}
        for rel,expected in seal['source_hashes'].items():
            archived=args.study/'executed_sources'/rel
            if not archived.exists() or sha(archived)!=expected:raise AssertionError('Executed source snapshot disagrees with seal')
            snapshot_hashes[rel]=sha(archived)
        result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            scheduled_sessions=total,independent_price_customer_paths=seal['final_paths'],
            opponent_repeats_grouped=True,final_rfq_records=36*total,own_trade_and_money_replay='all sessions',
            failures_excluded=0,failed_checks=failures,maximum_absolute_errors=max_errors,
            primary_independent_recomputation=independent,methods=summary,source_snapshots_match_seal=True,
            claims=dict(raw_account_and_inference_integrity='NUMERICALLY_CHECKED',
                boundary_reset_superiority='EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT' if independent['superiority_gate'] else 'NOT_SUPPORTED',
                competition_certification='BLOCKED_BY_MISSING_INPUT',R3_superiority='NOT_SUPPORTED'))
        dump(args.out/'analysis.json',result)
        dump(args.out/'input_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in
            (args.study/'sessions.jsonl.gz',args.study/'price_tapes.json.gz',args.study/'POLICY_SEAL.json',args.study/'summary.json',
             args.study/'primary_inference.json',args.study/'execution.json',Path(__file__))})
        (args.out/'REPORT.md').write_text(f'''# Independent complete boundary-study audit\n\nAudited all {total:,} scheduled raw sessions and {36*total:,} final RFQ records, over {seal['final_paths']:,} independent price/customer paths. No raw case was excluded. Price/customer histories match paired policies; all unique RFQs, trade/customer signs, actual fees, principal/cash, inventories, matching references, retained settlement, monetary components and innovation exposures reconcile independently. Maximum numerical discrepancy across checks: {max(max_errors.values()):.12g}, tolerance1e-8. Archived executed scientific sources match the policy seal.\n\nRecomputed the sealed primary from raw accounts, averaging both opponent repetitions within each path and resampling independent paths within the four strata. Mean BR2-minus-{seal['fallback']} {independent['paired_mean_difference']:.6f}; 95% interval {independent['stratified_path_bootstrap_95']}; materiality gate {'PASS' if independent['superiority_gate'] else 'FAIL'}. Numerical result agrees with the original analysis. This approximate bootstrap is empirical inference under the declared synthetic sampling law, not a mathematical safety certificate.\n\nQuote age and admission delay were independently checked against archived logical installation/matching/first-seen times, not inferred from callback counts. Boundary execution items, epochs, actions, model decisions and unsupported-state counters were checked against executed accounts and aggregated summaries. Complete per-method money, exposure, delay, activity and descriptive downside are in analysis.json. Admission delay and an absent quote do not identify a lost profitable fill. This audit cannot validate transport latency or missing actual round rules, and does not equate risk penalties with money PnL.\n\nStatus: raw accounting and inference integrity NUMERICALLY_CHECKED; reset superiority {'EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT' if independent['superiority_gate'] else 'NOT_SUPPORTED'} for this exact sealed comparison only. R3 remains disabled and its programme primary was not executed. Actual competition certification BLOCKED_BY_MISSING_INPUT. The penalty input is the authoritative charged account total; intermediate penalty checkpoint timing is not independently reconstructed from these trade records. Quote-age arithmetic is checked against archived instrumentation, rather than independently recovered from a complete accepted-action event log. No policy, seal or original SDK source changed during this audit.\n''')
    except BaseException:
        import traceback
        failure=dict(total_records_read=total,traceback=traceback.format_exc())
        failures.append(failure);dump(args.out/'FAILED_CHECKS.json',failures)
        raise
    dump(args.out/'SHA256SUMS.json',{str(p.relative_to(args.out)):sha(p) for p in sorted(args.out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('methods','maximum_absolute_errors')},indent=2))

if __name__=='__main__':main()
