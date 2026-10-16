"""Prospective basis-matched research comparison; original SDK execution rules."""
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
from experiments.boundary_study import StudySession
from experiments.sdk_pnl_decomposition import decompose_session
from resetmm.basis_baseline import BasisBandDealer, BasisResetDealer, basis_candidate_grid

DOMAIN = 'research-completion-v1/20261016T134827Z'
METHODS = ('B2', 'B3', 'B3_factor', 'B4', 'B4_factor', 'BR2', 'R2B')
CONVENTIONAL = METHODS[:5]
EXTRAS = ('B0', 'B4_reduce', 'B4_zeroquote', 'R2B_matched', 'B4_fixedprice', 'R2B_fixedprice')
DEVELOPMENT = ('standard', 'wide')
FINAL_FAMILIES = ('volatility', 'sticky')
REGULAR = ((.08, .03, .75), (.20, .06, .90), (.35, .10, .95), (.12, .18, .98))
FAILURE_DIR = ROOT / 'runs/sdk-study-failures'
FACTORIES = {}


class FixedPriceMixin:
    """Same causal quote-price function; inventory still constrains admission."""
    def _edge(self, side, q, price, tick, pid, market):
        return super()._edge(side, 0., price, tick, pid, market)


class FixedPriceBand(FixedPriceMixin, BasisBandDealer):
    pass


class FixedPriceReset(FixedPriceMixin, BasisResetDealer):
    pass


def seed_for(phase, scenario, path, stream):
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}/{phase}/{scenario}/{path}/{stream}'.encode()).digest(), 'big')


def generate_prices(phase, scenario, path):
    if scenario not in range(8):
        raise ValueError('unknown scenario')
    sigma, basis_sigma, rho = REGULAR[scenario] if scenario < 4 else (.20, .10, 1.)
    factor_rng = np.random.default_rng(seed_for(phase, scenario, path, 'factor'))
    basis_rng = np.random.default_rng(seed_for(phase, scenario, path, 'basis'))
    factor, basis = 100., 0.
    rows = [{'A': factor, 'B': 2 * factor}]
    log_sigma = sigma / 100 * math.sqrt(.5)
    for index in range(123):
        factor *= math.exp(-.5 * log_sigma**2 + log_sigma * float(factor_rng.normal()))
        local_rho = -.35 if scenario == 5 else (.95 if (index // 30) % 2 == 0 else .2) if scenario == 6 else rho
        local_sigma = basis_sigma * (6. if scenario == 6 and index % 20 == 0 else 1.)
        basis = local_rho * basis + local_sigma * float(basis_rng.normal())
        row = {'A': factor, 'B': 2 * factor + basis}
        if not all(math.isfinite(value) and value > 0 for value in row.values()):
            error = ValueError('invalid price tape; no reroll/clipping')
            error.partial_tape = rows + [row]
            raise error
        rows.append(row)
    return rows


def parameters(method):
    common = dict(total_intervals=12, settlement_coefficients={'B': {'A': 2.}},
                  freeze_current_quotes=False)
    if method == 'B2':
        return [dict(common, mode='B2', base_edge=edge, inventory_skew=skew,
                     forecast_weight=forecast)
                for forecast in (0., .25, .5, 1.)
                for edge in (.04, .08, .12) for skew in (0., .005)]
    result = []
    grid = basis_candidate_grid(method == 'B4_factor')
    # Preserve previously selected serious controls prospectively. The last
    # two packages in EACH forecast block are shared historical anchors,
    # replacing exploratory packages without increasing any family's budget.
    for start in range(0, 24, 6):
        for offset, values in ((4, (.08, .002, .4, .5)), (5, (.08, .005, .6, .5))):
            edge, skew, band, target = values
            grid[start + offset].update(base_edge=edge, inventory_skew=skew,
                band_fraction=band, target_fraction=target, risk_multiplier=1.)
    for package in grid:
        p = dict(common, **package)
        if method not in ('B4', 'B4_factor', 'R2B'):
            p.pop('factor_risk')
            p['risk_weight'] = .0002 * p.pop('risk_multiplier')
            p.update(mode='R2' if method == 'BR2' else 'B3',
                     factor_hedge=method == 'B3_factor')
            if method == 'BR2':
                p['boundary_reset'] = True
        result.append(p)
    return result


def factory(method):
    if method == 'B4_fixedprice':
        return FixedPriceBand
    if method == 'R2B_fixedprice':
        return FixedPriceReset
    if method in ('B4', 'B4_factor', 'B4_reduce', 'B4_zeroquote'):
        return BasisBandDealer
    if method in ('R2B', 'R2B_matched'):
        return BasisResetDealer
    key = 'B0' if method == 'B0' else 'boundary' if method == 'BR2' else 'core'
    if key not in FACTORIES:
        file = KIT / 'demo/demo_participant.py' if key == 'B0' else ROOT / 'submission' / (
            'boundary_dealer.py' if key == 'boundary' else 'sdk_dealer.py')
        FACTORIES[key] = load_class(file)
    return FACTORIES[key]


def sources():
    files = [Path(__file__), ROOT / 'docs/RESEARCH_COMPLETION_PROTOCOL.md',
             ROOT / 'resetmm/basis_baseline.py', ROOT / 'submission/sdk_dealer.py',
             ROOT / 'submission/boundary_dealer.py', ROOT / 'experiments/boundary_study.py',
             ROOT / 'experiments/sdk_study.py', ROOT / 'experiments/fresh_sdk_comparison.py',
             ROOT / 'experiments/freeze_diagnostic.py', ROOT / 'experiments/sdk_pnl_decomposition.py',
             KIT / 'examples/dealer.py', KIT / 'demo/demo_participant.py']
    files += sorted((KIT / 'rfq/rfq').glob('*.py'))
    return complete_source_hashes({source_key(p): sha256(p) for p in files})


def run_case(method, params, phase, scenario, path, family, detail=False):
    dealer = runner = None
    prices = []
    try:
        prices = generate_prices(phase, scenario, path)
        local_params = copy.deepcopy(params)
        if scenario == 7 and method != 'B0':
            local_params['settlement_coefficients'] = {}
        dealer = factory(method)(**local_params)
        seed = seed_for(phase, scenario, path, 'customers')
        config = build_config(None, 0, seed, rows=prices)
        if scenario == 7:
            from rfq.models import ReplayPrices
            config.price_model = ReplayPrices(prices, {'A': 'LAST_PRICE', 'B': 'LAST_PRICE'}, 0)
        runner = StudySession(config, dealer, family, seed_for(phase, scenario, path, 'opponents/' + family))
        row = runner.run()
        row.update(method=method, phase=phase, scenario=scenario, path=path, family=family,
                   seed=seed, block=0, parameters=local_params, crash=False,
                   settlement_graph='LAST_PRICE' if scenario == 7 else 'B=2A')
        own = [t for t in runner.engine.trades if t['dealer_id'] == 'dealer_1']
        decomp, _ = decompose_session(row, own, prices, {'A': 1., 'B': 1.}, tolerance=1e-8)
        if decomp['failed_checks']:
            raise ArithmeticError(decomp['failed_checks'])
        row['monetary_components'] = {k: decomp[k] for k in (
            'customer_spread', 'exchange_execution_edge', 'fees', 'exogenous_price_move_pnl',
            'terminal_settlement_residual', 'penalties', 'decomposed_pnl')}
        row['innovation_exposure'] = decomp['exposure']
        row['innovation_exposure']['factor_note'] = (
            'qA+2qB is a declared loading proxy; for LAST_PRICE it is not exact settlement exposure')
        row['decomposition_max_error'] = max(decomp['differences'].values())
        row['own_rejections'] = [r for r in runner.rejections if r['dealer'] == 'dealer_1']
        row['callback_errors'] = int(getattr(dealer, 'callback_errors', 0))
        row['boundary_counters'] = copy.deepcopy(getattr(dealer, 'boundary_counters', {}))
        row['basis_opening_actions'] = int(getattr(dealer, 'basis_opening_actions', 0))
        observations = runner.observations
        if (row['callback_errors'] or row['own_rejections'] or
                row['boundary_counters'].get('unsupported_histories', 0)):
            raise AssertionError('callback/rejection/unsupported boundary history')
        if len(observations) != 36 or len({(r['interval_id'], r['order_id']) for r in observations}) != 36:
            raise AssertionError('one final observation per RFQ failed')
        ages = [r['accepted_quote_age_seconds'] for r in observations if r['own_quote'] is not None]
        delays = [r['admission_delay_seconds'] for r in observations if r['admission_delay_seconds'] is not None]
        row['admission'] = dict(ever_admitted=len(delays), never_admitted=36-len(delays),
                               sum_delay_seconds=sum(delays), maximum_delay_seconds=max(delays, default=0.))
        row['quote_age'] = dict(positive_final_quotes=len(ages), sum_age_seconds=sum(ages),
                               maximum_age_seconds=max(ages, default=0.))
        if detail:
            row.update(own_trades=own, final_rfq_observations=observations)
        return row
    except Exception as error:
        FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        write_json(FAILURE_DIR / f'{phase}_{scenario}_{path}_{family}_{method}_{time.time_ns()}.json',
                   dict(method=method, parameters=params, phase=phase, scenario=scenario, path=path,
                        family=family, price_rows=getattr(error, 'partial_tape', prices),
                        primitive_seeds={stream: seed_for(phase, scenario, path, stream)
                                         for stream in ('factor', 'basis', 'customers', 'opponents/' + family)},
                        traceback=traceback.format_exc(),
                        own_trades=[] if runner is None else runner.engine.trades,
                        boundary_failures=[] if dealer is None else list(getattr(dealer, 'boundary_failures', ()))))
        if runner is not None:
            runner.store.close()
        raise


def init(out, stage):
    global FAILURE_DIR
    out.mkdir(parents=True, exist_ok=False)
    FAILURE_DIR = out / 'failed_cases'
    write_json(out / 'config.json', dict(stage=stage, namespace=DOMAIN, methods=METHODS,
        extras=EXTRAS, protocol='docs/RESEARCH_COMPLETION_PROTOCOL.md', regular=REGULAR,
        development_opponents=DEVELOPMENT, final_opponents=FINAL_FAMILIES, competition_scope=False,
        acknowledgement_schedule='synchronous immediate ACK; earlier network regressions separate', R3_enabled=False))
    write_json(out / 'environment.json', dict(python=sys.version, numpy=np.__version__,
                                            os=platform.platform(), executable=sys.executable))
    write_json(out / 'source_hashes.json', sources())
    for rel in sources():
        destination = out / 'executed_sources' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path(rel).read_bytes())


def manifest(out):
    write_json(out / 'SHA256SUMS.json', {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name != 'SHA256SUMS.json'})


def mean(rows):
    return float(np.mean([r['account']['pnl'] for r in rows]))


def pairs(rows, left, right):
    grouped, result = {}, []
    for row in rows:
        grouped.setdefault((row['scenario'], row['path']), {}).setdefault(row['family'], {})[row['method']] = row
    for (scenario, path), families in sorted(grouped.items()):
        differences = []
        for family, methods in sorted(families.items()):
            a, b = methods[left], methods[right]
            for key in ('price_tape_sha256', 'public_customer_tape_sha256'):
                if a[key] != b[key]:
                    raise AssertionError('paired primitive tapes differ')
            differences.append(a['account']['pnl'] - b['account']['pnl'])
        result.append(dict(scenario=scenario, path=path, difference=float(np.mean(differences))))
    return result


def inference(values, replicates=5000):
    groups = [np.array([r['difference'] for r in values if r['scenario'] == s])
              for s in sorted({r['scenario'] for r in values})]
    if any(len(g) < 2 for g in groups):
        raise ValueError('at least two independent paths per stratum')
    rng = np.random.default_rng(seed_for('analysis', 0, 0, 'bootstrap'))
    boot = sum(g[rng.integers(0, len(g), size=(replicates, len(g)))].mean(axis=1) / len(groups) for g in groups)
    lo, hi = map(float, np.quantile(boot, [.025, .975]))
    return dict(paired_mean_difference=float(np.mean([g.mean() for g in groups])),
                stratified_path_bootstrap_95=[lo, hi], independent_paths=sum(map(len, groups)),
                materiality=1., superiority_gate=lo > 1., opponent_repetitions_not_independent=True,
                assumptions='independent synthetic path units; equal scenario mixture; approximate bootstrap; no tail-power claim')


def pilot(out):
    started = time.perf_counter()
    init(out, 'prospective tuning and validation')
    initial_sources = sources()
    selected, trials = {}, []
    with gzip.open(out / 'tuning.jsonl.gz', 'wt') as stream:
        for method in METHODS:
            scores = []
            for index, params in enumerate(parameters(method)):
                cases = []
                for scenario in range(4):
                    for path in range(3):
                        row = run_case(method, params, 'tuning', scenario, path, DEVELOPMENT[path % 2])
                        cases.append(row)
                        stream.write(json.dumps(dict(candidate=index, **row), sort_keys=True) + '\n')
                score = mean(cases)
                scores.append(score)
                trials.append(dict(method=method, candidate=index, parameters=params, mean_pnl=score, sessions=len(cases)))
            selected[method] = parameters(method)[int(np.argmax(scores))]
            write_json(out / 'progress.json', dict(stage='tuning', completed_method=method, trials=len(trials)))
    write_json(out / 'tuning_summary.json', trials)
    validation = []
    with gzip.open(out / 'validation.jsonl.gz', 'wt') as stream:
        for scenario in range(4):
            for path in range(12):
                for family in DEVELOPMENT:
                    for method in METHODS:
                        row = run_case(method, selected[method], 'validation', scenario, path, family)
                        validation.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
    scores = {m: mean([r for r in validation if r['method'] == m]) for m in CONVENTIONAL}
    fallback = max(CONVENTIONAL, key=lambda m: scores[m])
    values = pairs(validation, 'R2B', fallback)
    sd = float(np.std([r['difference'] for r in values], ddof=1))
    required = math.ceil(((1.960 + .842) * sd)**2)
    paths = min(2048, 8 * math.ceil(max(512, required) / 8))
    selected['B4_reduce'] = dict(selected['B4'], reduce_only=True)
    selected['B4_zeroquote'] = dict(selected['B4'], forecast_weight=0.)
    selected['R2B_matched'] = copy.deepcopy(selected['B4'])
    selected['B4_fixedprice'] = copy.deepcopy(selected['B4'])
    selected['R2B_fixedprice'] = copy.deepcopy(selected['B4'])
    primitives = []
    for phase, scenario_count, count in (('tuning', 4, 3), ('validation', 4, 12), ('final', 8, paths // 8)):
        for scenario in range(scenario_count):
            for path in range(count):
                names = ('factor', 'basis', 'customers') + tuple('opponents/' + f for f in DEVELOPMENT + FINAL_FAMILIES)
                primitives.extend(seed_for(phase, scenario, path, name) for name in names)
                primitives.extend(seed_for(phase, scenario, path, 'customers') + j for j in (1, 2))
                primitives.extend(seed_for(phase, scenario, path, 'opponents/' + f) + 1 for f in DEVELOPMENT + FINAL_FAMILIES)
    if len(primitives) != len(set(primitives)):
        raise AssertionError('primitive seed collision')
    write_json(out / 'PRIMITIVE_SEED_AUDIT.json', dict(checked=len(primitives), collisions=0, digest_bits=256,
        final_prices_generated=False, independence_is_PRNG_assumption=True))
    if sources() != initial_sources:
        raise RuntimeError('source changed in pilot; no final seal')
    seal = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected=selected, fallback=fallback,
        validation_conventional_means=scores, validation_paired_pooled_sd=sd,
        required_independent_paths=required, final_paths=paths, paths_per_scenario=paths // 8,
        precision_cap_binds=required > 2048, primary='R2B minus ' + fallback, materiality=1.,
        source_hashes=sources(), final_generated=False, elapsed_seconds=time.perf_counter() - started,
        R3_enabled=False, competition_scope=False)
    write_json(out / 'pilot_inference.json', inference(values))
    write_json(out / 'POLICY_SEAL.json', seal)
    manifest(out)
    print(json.dumps(seal, indent=2))


def summaries(rows):
    result = {}
    for method in METHODS + EXTRAS:
        own = [r for r in rows if r['method'] == method]
        pnl = np.array([r['account']['pnl'] for r in own])
        result[method] = dict(sessions=len(own), mean_pnl=float(pnl.mean()), median_pnl=float(np.median(pnl)),
            worst_pnl=float(pnl.min()), pnl_quantile_05=float(np.quantile(pnl, .05)),
            negative_sessions=int((pnl < 0).sum()), mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in own])),
            customer_notional=sum(r['account']['customer_notional'] for r in own),
            exchange_notional=sum(r['account']['exchange_notional'] for r in own),
            fees=sum(r['account']['fees_paid'] for r in own), penalties=sum(r['account']['penalties_paid'] for r in own),
            crashes=sum(r['crash'] for r in own), own_rejections=sum(len(r['own_rejections']) for r in own),
            callback_errors=sum(r['callback_errors'] for r in own),
            max_decomposition_error=max(r['decomposition_max_error'] for r in own),
            maximum_callback_seconds=max(c['maximum'] for r in own for c in r['callback_seconds'].values()),
            mean_q2_at_innovation=np.mean([r['innovation_exposure']['mean_q_squared_at_price_innovation'] for r in own], axis=0).tolist(),
            mean_monetary_components={k: float(np.mean([r['monetary_components'][k] for r in own])) for k in own[0]['monetary_components']},
            scenario_means={str(s): mean([r for r in own if r['scenario'] == s]) for s in range(8)})
    return result


def final(out, seal_path):
    started = time.perf_counter()
    seal = json.loads(seal_path.read_text())
    if seal['source_hashes'] != sources():
        raise RuntimeError('source changed after seal')
    init(out, 'sealed research comparison and branch decision')
    write_json(out / 'POLICY_SEAL.json', seal)
    rows, tapes = [], {}
    with gzip.open(out / 'sessions.jsonl.gz', 'wt') as stream:
        for scenario in range(8):
            for path in range(seal['paths_per_scenario']):
                tapes[f'{scenario}/{path}'] = generate_prices('final', scenario, path)
                for family in FINAL_FAMILIES:
                    for method in METHODS + EXTRAS:
                        row = run_case(method, {} if method == 'B0' else seal['selected'][method],
                                       'final', scenario, path, family, detail=True)
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
                        row.pop('own_trades')
                        row.pop('final_rfq_observations')
                        rows.append(row)
            write_json(out / 'progress.json', dict(completed_scenarios=scenario + 1, scheduled_sessions=len(rows)))
    with gzip.open(out / 'price_tapes.json.gz', 'wt') as stream:
        json.dump(tapes, stream, sort_keys=True)
    if sources() != seal['source_hashes']:
        write_json(out / 'SOURCE_CHANGE_FAILURE.json', dict(sealed=seal['source_hashes'], current=sources()))
        raise RuntimeError('source changed during final; inference prohibited')
    summary = summaries(rows)
    primary = inference(pairs(rows, 'R2B', seal['fallback']))
    write_json(out / 'summary.json', summary)
    write_json(out / 'primary_inference.json', primary)
    comparisons = (('R2B_minus_B4', 'R2B', 'B4'), ('R2B_matched_minus_B4', 'R2B_matched', 'B4'),
                   ('B4_minus_reduce', 'B4', 'B4_reduce'), ('B4_minus_zeroquote', 'B4', 'B4_zeroquote'),
                   ('B4_minus_B3', 'B4', 'B3'), ('BR2_minus_B4', 'BR2', 'B4'),
                   ('R2B_fixedprice_minus_B4_fixedprice', 'R2B_fixedprice', 'B4_fixedprice'))
    write_json(out / 'descriptive_ablations.json', {name: inference(pairs(rows, a, b)) for name, a, b in comparisons})
    write_json(out / 'execution.json', dict(elapsed_seconds=time.perf_counter() - started,
        scheduled_sessions=len(rows), independent_paths=seal['final_paths'], source_unchanged=True,
        branch_decision='SCOPED_SYNTHETIC_PRIMARY_PASS' if primary['superiority_gate'] else 'INCREMENTAL_SUPERIORITY_NOT_SUPPORTED',
        R3_enabled=False, competition_scope=False))
    manifest(out)
    print(json.dumps(dict(primary=primary, method_means={m: v['mean_pnl'] for m, v in summary.items()}), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('smoke', 'pilot', 'final'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seal', type=Path)
    args = p.parse_args()
    if args.stage == 'pilot':
        pilot(args.out)
    elif args.stage == 'final':
        if args.seal is None:
            p.error('final requires --seal')
        final(args.out, args.seal)
    else:
        init(args.out, 'test namespace only; no performance inference')
        rows = [run_case(m, parameters(m)[0], 'test', s, 0, 'standard', detail=True)
                for s in (0, 7) for m in METHODS]
        write_json(args.out / 'smoke.json', rows)
        manifest(args.out)
        print({f'{r["scenario"]}/{r["method"]}': r['account']['pnl'] for r in rows})


if __name__ == '__main__':
    main()
