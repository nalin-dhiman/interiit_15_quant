"""Frozen shared-envelope RFQ attribution benchmark; offline SDK research only."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
from experiments.sdk_environment import ROOT, KIT, complete_source_hashes, source_key, source_path
from experiments.sdk_study import build_config, load_class, write_json, sha256
from experiments.boundary_study import StudySession
from experiments import research_study as legacy
from experiments.sdk_pnl_decomposition import decompose_session
from resetmm.attribution import path_interval, plan_sample
from resetmm.common_risk import CommonRiskBandDealer, CommonRiskResetDealer

DOMAIN = 'constrained-attribution-v1/20261017T051731Z'
BUDGETS = {'tight': {'A': 10., 'B': 5.}, 'wide': {'A': 20., 'B': 10.}}
MODES = ('fixed', 'inventory')
CONTROLLERS = ('B4', 'R2B')
DEVELOPMENT = ('standard', 'wide')
HELD_OUT = ('size_mix', 'public_basis')
PROTOCOL = ROOT / 'docs/RISK_MATCHED_ATTRIBUTION_PROTOCOL.md'


def seed_for(phase, scenario, path, stream):
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}/{phase}/{scenario}/{path}/{stream}'.encode()).digest(), 'big')


def prices_for(phase, scenario, path):
    if scenario not in range(8): raise ValueError('unknown price law')
    sigma, basis_sigma, rho = legacy.REGULAR[scenario] if scenario < 4 else (.20, .10, 1.)
    factor_rng = np.random.default_rng(seed_for(phase, scenario, path, 'factor'))
    basis_rng = np.random.default_rng(seed_for(phase, scenario, path, 'basis'))
    factor, basis = 100., 0.
    rows = [{'A': factor, 'B': 2 * factor}]
    log_sigma = sigma / 100 * math.sqrt(.5)
    for index in range(123):
        factor *= math.exp(-.5 * log_sigma ** 2 + log_sigma * float(factor_rng.normal()))
        local_rho = -.35 if scenario == 5 else (.95 if (index // 30) % 2 == 0 else .2) if scenario == 6 else rho
        local_sigma = basis_sigma * (6. if scenario == 6 and index % 20 == 0 else 1.)
        basis = local_rho * basis + local_sigma * float(basis_rng.normal())
        row = {'A': factor, 'B': 2 * factor + basis}
        if not all(math.isfinite(v) and v > 0 for v in row.values()):
            error = ValueError('invalid tape; no clipping or rerolls')
            error.partial_tape = rows + [row]
            raise error
        rows.append(row)
    return rows


class FixedPriceMixin:
    def _edge(self, side, q, price, tick, pid, market):
        return super()._edge(side, 0., price, tick, pid, market)


class FixedBand(FixedPriceMixin, CommonRiskBandDealer): pass
class FixedReset(FixedPriceMixin, CommonRiskResetDealer): pass


class EventTies:
    """Uniform SDK choices with a primitive event key, not policy draw count."""
    def __init__(self, identities, phase, scenario, path):
        self.identities = list(identities)
        self.phase, self.scenario, self.path = phase, scenario, path
        self.used = 0

    def choice(self, choices):
        if self.used >= len(self.identities): raise AssertionError('unexpected SDK tie choice')
        interval, oid = self.identities[self.used]
        self.used += 1
        return random.Random(seed_for(self.phase, self.scenario, self.path,
                                      f'ties/{interval}/{oid}')).choice(choices)


class EventOpponent:
    """Two predeclared held-out research structures; no latent features."""
    def __init__(self, family, seed):
        self.family, self.seed = family, seed

    def quote(self, orders, market, portfolio):
        from rfq.sdk import round_quote
        result = {}
        for order in orders:
            pid, side = order['product_id'], order['side']
            draw = random.Random(f'{self.seed}/{order["order_id"]}').random()
            price = market['prices'][pid]
            if self.family == 'size_mix':
                edge = .02 + .06 * draw if order['quantity'] >= 4 else .08 + .16 * draw
                fair = price
            elif self.family == 'public_basis':
                edge = .025 + .14 * draw
                basis = 2 * market['prices']['A'] - price if pid == 'B' else 0.
                fair = price + .30 * basis
            else: raise ValueError('unknown opponent family')
            result[order['order_id']] = round_quote(fair + (edge if side == 'BUY' else -edge),
                                                   market['products'][pid]['tick_size'], side)
        return result

    def hedge(self, market, portfolio): return []


class AttributionSession(StudySession):
    def __init__(self, config, dealer, family, phase, scenario, path):
        self.phase, self.scenario, self.path = phase, scenario, path
        opponent_seed = seed_for(phase, scenario, path, 'opponents/' + family)
        super().__init__(config, dealer, family if family in DEVELOPMENT else 'standard', opponent_seed)
        if family in HELD_OUT:
            for index in range(2):
                self.dealers[f'dealer_{index+2}'] = EventOpponent(family, opponent_seed + index)
        elif family not in DEVELOPMENT: raise ValueError(family)
        self.tie_events = 0

    def advance(self, final):
        identities = [(self.engine.interval_id, oid) for oid, order in self.engine.orders.items()
                      if self.engine.matches(order)]
        ties = EventTies(identities, self.phase, self.scenario, self.path)
        self.engine.tie_rng = ties
        super().advance(final)
        if ties.used != len(identities): raise AssertionError('SDK eligible/tie queue mismatch')
        self.tie_events += ties.used


def cell_name(controller, package, budget, mode):
    return '/'.join((controller, package, budget, mode))


def sources():
    files = []
    for directory in ('resetmm', 'submission', 'experiments', 'tests', 'tests_sdk'):
        files += sorted((ROOT / directory).rglob('*.py'))
    files += [PROTOCOL, ROOT / 'scripts/audit_risk_matched.py', ROOT / 'data/sdk-contract-hashes.json']
    files += sorted((KIT / 'rfq/rfq').glob('*.py'))
    files += [KIT / 'examples/dealer.py', KIT / 'demo/demo_participant.py']
    return complete_source_hashes({source_key(p): sha256(p) for p in files})


def init(out, stage):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', dict(stage=stage, namespace=DOMAIN, budgets=BUDGETS,
        quote_modes=MODES, controllers=CONTROLLERS, development_opponents=DEVELOPMENT,
        final_opponents=HELD_OUT, primary='R2B-B4 / B4-selected shared package / wide / fixed',
        scope='same ex ante rectangle, not equal realised risk; synthetic offline only', R3_enabled=False))
    write_json(out / 'environment.json', dict(python=sys.version, numpy=np.__version__,
        platform=platform.platform(), executable=sys.executable))


def manifest(out):
    write_json(out / 'SHA256SUMS.json', {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*'))
                                      if p.is_file() and p.name != 'SHA256SUMS.json'})


def audit_seed_keys(final_paths):
    seeds = []
    for phase, scenarios, count in [('tuning',4,3),('validation',8,12),('final',8,final_paths//8)]:
        for scenario in range(scenarios):
            for path in range(count):
                seeds += [seed_for(phase,scenario,path,name) for name in ('factor','basis')]
                seeds.append(seed_for(phase,scenario,path,'customers')+1)
                for family in DEVELOPMENT + HELD_OUT:
                    base=seed_for(phase,scenario,path,'opponents/'+family)
                    seeds += [base+1,base+2]
                seeds += [seed_for(phase,scenario,path,f'ties/{interval}/r{interval}-{i}')
                          for interval in range(1,13) for i in range(1,4)]
    if len(seeds)!=len(set(seeds)): raise AssertionError('primitive seed collision')
    return dict(checked_keys=len(seeds),collisions=0,final_prices_generated=False,
                independence_is_prng_assumption=True,event_keyed_tie_primitives=True)


def declare(out):
    init(out, 'prospective declaration before any pilot executions')
    hashes = sources()
    write_json(out / 'source_hashes.json', hashes)
    for key in hashes:
        target = out / 'executed_sources' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path(key).read_bytes())
    write_json(out / 'DECLARATION.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        namespace=DOMAIN, source_hashes=hashes, final_prices_generated=False,
        tuning_or_validation_executed=False, primary_materiality=1., floor_paths=1024, cap_paths=4096))
    manifest(out)


def run_case(controller, parameters, phase, scenario, path, family, budget, mode, failed_dir):
    runner = dealer = None
    prices = []
    try:
        prices = prices_for(phase, scenario, path)
        params = copy.deepcopy(parameters)
        if scenario == 7: params['settlement_coefficients'] = {}
        cls = (FixedBand if controller == 'B4' else FixedReset) if mode == 'fixed' else (
            CommonRiskBandDealer if controller == 'B4' else CommonRiskResetDealer)
        dealer = cls(inventory_caps=BUDGETS[budget], risk_weights={'A': 1., 'B': 2.}, **params)
        config = build_config(None, 0, seed_for(phase, scenario, path, 'customers'), rows=prices)
        if scenario == 7:
            from rfq.models import ReplayPrices
            config.price_model = ReplayPrices(prices, {'A': 'LAST_PRICE', 'B': 'LAST_PRICE'}, 0)
        runner = AttributionSession(config, dealer, family, phase, scenario, path)
        row = runner.run()
        own = [t for t in runner.engine.trades if t['dealer_id'] == 'dealer_1']
        row.update(controller=controller, method=controller, phase=phase, scenario=scenario, path=path,
                   family=family, budget=budget, quote_mode=mode, parameters=params, crash=False,
                   settlement_graph='LAST_PRICE' if scenario == 7 else 'B=2A',
                   seed=seed_for(phase, scenario, path, 'customers'), block=0)
        decomp, _ = decompose_session(row, own, prices, {'A': 1., 'B': 1.}, tolerance=1e-8)
        if decomp['failed_checks']: raise ArithmeticError(decomp['failed_checks'])
        row['monetary_components'] = {k: decomp[k] for k in (
            'customer_spread', 'exchange_execution_edge', 'fees', 'exogenous_price_move_pnl',
            'terminal_settlement_residual', 'penalties', 'decomposed_pnl')}
        row['innovation_exposure'] = decomp['exposure']
        row['decomposition_max_error'] = max(decomp['differences'].values())
        row['own_trades'] = own
        row['final_rfq_observations'] = runner.observations
        row['risk_budget_trace'] = copy.deepcopy(list(dealer.risk_budget_trace))
        row['risk_budget_counters'] = copy.deepcopy(dealer.risk_budget_counters)
        if any(row['risk_budget_counters'][k] for k in ('actual_breaches','possible_breaches','trace_dropped')):
            raise AssertionError('risk-envelope failure or incomplete risk trace')
        row['callback_errors'] = dealer.callback_errors
        row['boundary_counters'] = copy.deepcopy(dealer.boundary_counters)
        row['own_rejections'] = [r for r in runner.rejections if r['dealer'] == 'dealer_1']
        row['tie_events'] = runner.tie_events
        row['decomposition_max_error'] = max(decomp['differences'].values())
        if (row['callback_errors'] or row['own_rejections'] or
                row['boundary_counters'].get('unsupported_histories', 0)):
            raise AssertionError('callback, rejected action or unsupported history')
        if len(runner.observations) != 36: raise AssertionError('missing final RFQ observations')
        for pid, cap in BUDGETS[budget].items():
            if row['max_abs_inventory'][('A', 'B').index(pid)] > cap + 1e-8:
                raise AssertionError('actual inventory envelope breached')
        return row
    except Exception as error:
        prices = getattr(error, 'partial_tape', prices)
        Path(failed_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(failed_dir) / f'{phase}_{scenario}_{path}_{family}_{controller}_{budget}_{mode}_{time.time_ns()}.json',
            dict(controller=controller, parameters=parameters, phase=phase, scenario=scenario, path=path,
                 family=family, budget=budget, quote_mode=mode, price_rows=prices,
                 traceback=traceback.format_exc(), own_trades=[] if runner is None else runner.engine.trades,
                 boundary_failures=[] if dealer is None else list(dealer.boundary_failures),
                 risk_trace=[] if dealer is None else dealer.risk_budget_trace))
        raise


def paired_groups(rows, package, budget, mode):
    groups = {}
    for row in rows:
        if (row['package'], row['budget'], row['quote_mode']) != (package, budget, mode): continue
        key = (row['scenario'], row['path'])
        groups.setdefault(key, {}).setdefault(row['family'], {})[row['controller']] = row
    result = [[] for _ in range(8)]
    for (scenario, _), families in sorted(groups.items()):
        differences = []
        for cells in families.values():
            a, b = cells['R2B'], cells['B4']
            if any(a[k] != b[k] for k in ('price_tape_sha256', 'public_customer_tape_sha256', 'parameters')):
                raise AssertionError('paired information/package mismatch')
            differences.append(a['account']['pnl'] - b['account']['pnl'])
        if len(differences) != 2: raise AssertionError('two opponent repeats required')
        result[scenario].append(float(np.mean(differences)))
    return result


def write_row(stream, row):
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    return {k: v for k, v in row.items() if k not in ('own_trades', 'final_rfq_observations', 'risk_budget_trace')}


def execute_case_task(task):
    row = run_case(*task[:-1])
    row.update(task[-1])
    return row


def pilot(out, declaration):
    declared = json.loads(declaration.read_text())
    if declared['source_hashes'] != sources(): raise RuntimeError('source changed after declaration')
    init(out, 'equal-budget tuning and shared-package validation')
    start = time.perf_counter()
    grid = legacy.parameters('B4')
    nominated = {}
    trials = []
    context = multiprocessing.get_context('spawn')
    tasks = [(controller, params, 'tuning', scenario, path, DEVELOPMENT[path % 2], budget, mode,
              str(out/'failed_cases'), dict(candidate=index))
             for budget in BUDGETS for mode in MODES for controller in CONTROLLERS
             for index, params in enumerate(grid) for scenario in range(4) for path in range(3)]
    training_rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        with gzip.open(out/'tuning.jsonl.gz','wt') as stream:
            for index, row in enumerate(pool.map(execute_case_task, tasks, chunksize=4)):
                training_rows.append(write_row(stream,row))
                if index % 96 == 0:
                    write_json(out/'progress.json',dict(stage='tuning',completed_sessions=index+1))
        for budget in BUDGETS:
            for mode in MODES:
                nominated[budget+'/'+mode] = {}
                for controller in CONTROLLERS:
                    scores = []
                    for index, params in enumerate(grid):
                        values = [r['account']['pnl'] for r in training_rows if
                                  (r['budget'],r['quote_mode'],r['controller'],r['candidate']) ==
                                  (budget,mode,controller,index)]
                        if len(values)!=12: raise AssertionError('unequal candidate tuning budget')
                        score=float(np.mean(values));scores.append(score)
                        trials.append(dict(controller=controller,budget=budget,mode=mode,
                                           candidate=index,mean_pnl=score,sessions=len(values)))
                    nominated[budget+'/'+mode][controller]=int(np.argmax(scores))
        write_json(out/'tuning_summary.json',trials)
        rows=[]
        with gzip.open(out/'price_tapes.json.gz','wt') as stream:
            json.dump({f'{phase}/{scenario}/{path}':prices_for(phase,scenario,path)
                       for phase,scenarios,count in [('tuning',4,3),('validation',8,12)]
                       for scenario in range(scenarios) for path in range(count)},stream,sort_keys=True)
        tasks=[(controller,grid[nominated[budget+'/'+mode][package]],'validation',scenario,path,
                family,budget,mode,str(out/'failed_cases'),
                dict(package=package,candidate=nominated[budget+'/'+mode][package]))
               for scenario in range(8) for path in range(12) for family in DEVELOPMENT
               for budget in BUDGETS for mode in MODES for package in CONTROLLERS
               for controller in CONTROLLERS]
        with gzip.open(out/'validation.jsonl.gz','wt') as stream:
            for index,row in enumerate(pool.map(execute_case_task,tasks,chunksize=4)):
                rows.append(write_row(stream,row))
                if index % 96 == 0:
                    write_json(out/'progress.json',dict(stage='validation',completed_sessions=index+1))
    selected, selection = {}, {}
    for budget in BUDGETS:
        for mode in MODES:
            key = budget + '/' + mode
            selected[key], selection[key] = {}, {}
            for controller in CONTROLLERS:
                scores = {label: float(np.mean([r['account']['pnl'] for r in rows if
                    (r['budget'], r['quote_mode'], r['package'], r['controller']) ==
                    (budget, mode, label, controller)])) for label in CONTROLLERS}
                best = max(CONTROLLERS, key=lambda label: (scores[label], -nominated[key][label]))
                index = nominated[key][best]
                selected[key][controller] = grid[index]
                selection[key][controller] = dict(chosen_nomination=best, candidate=index, validation_means=scores)
    # Re-label validation rows with the final shared primary B4 nomination.
    primary_label = selection['wide/fixed']['B4']['chosen_nomination']
    groups = paired_groups(rows, primary_label, 'wide', 'fixed')
    plan = plan_sample(groups)
    write_json(out/'PRIMITIVE_SEED_AUDIT.json',audit_seed_keys(plan['final_paths']))
    if sources() != declared['source_hashes']: raise RuntimeError('source changed during pilot')
    seal = dict(created_utc=datetime.now(timezone.utc).isoformat(), namespace=DOMAIN,
        selected=selected, selection=selection, nominated_candidates=nominated,
        source_hashes=sources(), declaration_sha256=sha256(declaration), final_generated=False,
        tuning_sessions=2304, validation_sessions=len(rows), **plan,
        primary='R2B-B4 using common B4-validation-selected package/wide/fixed',
        final_opponents=HELD_OUT, elapsed_seconds=time.perf_counter()-start)
    write_json(out / 'POLICY_SEAL.json', seal)
    manifest(out)
    print(json.dumps(seal, indent=2))


def execute_path(task):
    phase, scenario, path, selected, failed_dir = task
    rows = []
    for family in HELD_OUT:
        for budget in BUDGETS:
            for mode in MODES:
                for package in CONTROLLERS:
                    for controller in CONTROLLERS:
                        row = run_case(controller, selected[budget + '/' + mode][package], phase,
                                       scenario, path, family, budget, mode, failed_dir)
                        row['package'] = package
                        rows.append(row)
    if len({(r['price_tape_sha256'], r['public_customer_tape_sha256']) for r in rows}) != 1:
        raise AssertionError('path primitive tapes changed across policies/cells')
    return rows


def analyze(rows):
    summaries = {}
    for budget in BUDGETS:
        for mode in MODES:
            for package in CONTROLLERS:
                for controller in CONTROLLERS:
                    own = [r for r in rows if (r['budget'], r['quote_mode'], r['package'], r['controller']) ==
                           (budget, mode, package, controller)]
                    pnl = np.asarray([r['account']['pnl'] for r in own])
                    name = cell_name(controller, package, budget, mode)
                    summaries[name] = dict(sessions=len(own), mean_pnl=float(pnl.mean()),
                        pnl_quantile_05=float(np.quantile(pnl, .05)), worst_pnl=float(pnl.min()),
                        negative_sessions=int(np.sum(pnl < 0)),
                        mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in own])),
                        mean_fees=float(np.mean([r['account']['fees_paid'] for r in own])),
                        mean_exchange_notional=float(np.mean([r['account']['exchange_notional'] for r in own])),
                        mean_q2_at_innovation=np.mean([r['innovation_exposure']['mean_q_squared_at_price_innovation']
                                                       for r in own], axis=0).tolist(),
                        mean_factor2_at_innovation=float(np.mean([r['innovation_exposure']['mean_terminal_factor_squared_at_price_innovation'] for r in own])),
                        mean_monetary_components={k: float(np.mean([r['monetary_components'][k] for r in own]))
                                                 for k in own[0]['monetary_components']},
                        scenario_means={str(s):float(np.mean([r['account']['pnl'] for r in own if r['scenario']==s])) for s in range(8)})
    contrasts = {}
    for budget in BUDGETS:
        for mode in MODES:
            for package in CONTROLLERS:
                key = '/'.join((package, budget, mode))
                groups = paired_groups(rows, package, budget, mode)
                result = path_interval(groups, seed_for('analysis', 0, 0, 'bootstrap/' + key))
                result.update(confirmatory=key == 'B4/wide/fixed',
                              multiplicity_adjusted=False if key != 'B4/wide/fixed' else None,
                              scenario_mean_differences={str(i):float(np.mean(g)) for i,g in enumerate(groups)})
                contrasts[key] = result
    return summaries, contrasts


def final(out, seal_path, workers):
    seal = json.loads(seal_path.read_text())
    if seal['source_hashes'] != sources(): raise RuntimeError('source changed after policy seal')
    init(out, 'sealed fresh constrained-attribution final study')
    write_json(out / 'POLICY_SEAL.json', seal)
    started = time.perf_counter()
    tasks = [('final', s, p, seal['selected'], str(out/'failed_cases')) for s in range(8)
             for p in range(seal['paths_per_scenario'])]
    prefix = execute_path(tasks[0])
    with gzip.open(out/'sequential_prefix.jsonl.gz', 'wt') as stream:
        for row in prefix: write_row(stream, row)
    # A duplicate prefix is a reproducibility check, not extra independent N.
    from scripts.audit_research_study import economic_fingerprint
    prefix_fingerprints = [economic_fingerprint(r) for r in prefix]
    compact = []
    context = multiprocessing.get_context('spawn')
    with gzip.open(out/'sessions.jsonl.gz','wt') as stream, ProcessPoolExecutor(
            max_workers=workers, mp_context=context) as pool:
        for index, path_rows in enumerate(pool.map(execute_path, tasks, chunksize=1)):
            if index == 0 and [economic_fingerprint(r) for r in path_rows] != prefix_fingerprints:
                raise AssertionError('sequential/parallel economic prefix differs')
            for row in path_rows: compact.append(write_row(stream,row))
            if index % 16 == 0:
                write_json(out/'progress.json',dict(completed_paths=index+1,completed_sessions=len(compact)))
    if sources() != seal['source_hashes']: raise RuntimeError('source changed during final; inference stopped')
    summaries, contrasts = analyze(compact)
    write_json(out/'summary.json',summaries);write_json(out/'contrasts.json',contrasts)
    write_json(out/'primary_inference.json',contrasts['B4/wide/fixed'])
    with gzip.open(out/'price_tapes.json.gz','wt') as stream:
        json.dump({f'{s}/{p}':prices_for('final',s,p) for s in range(8)
                   for p in range(seal['paths_per_scenario'])},stream,sort_keys=True)
    write_json(out/'execution.json',dict(elapsed_seconds=time.perf_counter()-started,
        independent_paths=len(tasks),sessions=len(compact),workers=workers,
        sequential_parallel_prefix_sessions=len(prefix),prefix_identical=True,
        source_unchanged=True,precision_cap_binds=seal['precision_cap_binds'],
        all_failures_retained=True,realised_risk_matched=False,common_ex_ante_envelope=True))
    manifest(out)
    print(json.dumps(contrasts['B4/wide/fixed'],indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('declare','pilot','final','smoke'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--declaration',type=Path)
    parser.add_argument('--seal',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.stage=='declare':declare(args.out)
    elif args.stage=='pilot':pilot(args.out,args.declaration)
    elif args.stage=='final':final(args.out,args.seal,args.workers)
    else:
        init(args.out,'predeclaration smoke; not pilot or final data')
        params=legacy.parameters('B4')[0]
        rows=execute_path(('smoke',0,0,{b+'/'+m:{c:params for c in CONTROLLERS}
            for b in BUDGETS for m in MODES},str(args.out/'failed_cases')))
        with gzip.open(args.out/'sessions.jsonl.gz','wt') as stream:
            for row in rows:write_row(stream,row)
        write_json(args.out/'smoke.json',dict(sessions=len(rows),primitive_pairs_match=True,
                   callback_errors=sum(r['callback_errors'] for r in rows)))
        manifest(args.out)


if __name__=='__main__':main()
