"""Predeclared exploratory continuation study; no competition claims."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np

from resetmm.continuation import (BandPolicy, FiniteRFQModel, PolicyEvaluation,
    evaluate_policy, improve_policy, make_synthetic_model, make_primitive_tape,
    simulate_policy, solve_optimal, tune_band)
from resetmm.statistics import paired_bootstrap_interval, lower_tail_mean


def write_json(path, value):
    def encode(obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=encode) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/continuation_diagnostic.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', cfg)
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('resetmm', 'experiments', 'configs', 'tests')
                     for p in sorted((root/folder).glob('*')) if p.is_file()}
    write_json(out/'source_hashes.json', source_hashes)
    write_json(out/'environment.json', dict(python=sys.version, numpy=np.__version__,
               platform=platform.platform(), sdk=False))
    grid_cfg = cfg['band_grid']
    grid = [BandPolicy(*values) for values in product(*(grid_cfg[k] for k in
            ('edge', 'skew', 'threshold_lots', 'target_fraction')))]
    assert len(grid) == cfg['candidate_budget_per_method_per_training_model']
    validation_seeds = range(cfg['validation_seeds']['first'], cfg['validation_seeds']['last']+1)
    stress_seeds = range(cfg['stress_seeds']['first'], cfg['stress_seeds']['last']+1)
    records, summaries, tuning, exact, decisions, timings, tapes_archive = [], [], [], [], [], [], {}
    trained = {}
    start = time.perf_counter()
    try:
        for risk in cfg['risk_weights']:
            for regime in cfg['regimes']:
                model = make_synthetic_model(horizon=cfg['horizon'], limit_lots=cfg['limit_lots'],
                                             regime=regime, risk_weight=risk)
                label = f'{regime}_risk{risk:g}'
                bands, policies = {}, {}
                for method in ('B1', 'B2', 'B3', 'R2_joint'):
                    method_grid = [replace(b, threshold_lots=None) for b in grid] if method == 'B1' else (
                                  [replace(b, threshold_lots=1, target_fraction=0.) for b in grid]
                                  if method == 'B2' else grid)
                    tick = time.perf_counter()
                    band, trials = tune_band(model, method_grid, improved=method == 'R2_joint')
                    timings.append(dict(model=label, method=method, phase='tuning',
                                        elapsed_seconds=time.perf_counter()-tick, trials=len(trials)))
                    bands[method] = band
                    policies[method] = improve_policy(model, band) if method == 'R2_joint' else band.bind(model)
                    tuning.extend(dict(model=label, method=method, trial=i, **row) for i, row in enumerate(trials))
                policies['R2_fixed_quotes'] = improve_policy(model, bands['B3'])
                trained[risk, regime] = (model, bands, policies)
                raw_model = FiniteRFQModel(replace(model.config, risk_weight=0.), model.batches, model.batch_transition)
                # Include every quote produced by the tuning grid at any feasible inventory.
                edges = set(model.config.quote_edges)
                for band in grid:
                    for q in range(-model.config.limit_lots, model.config.limit_lots+1):
                        for d in (-1, 1):
                            edges.add(model.round_edge(band.edge-d*band.skew*q))
                oracle_model = FiniteRFQModel(replace(model.config, quote_edges=tuple(sorted(edges))),
                                              model.batches, model.batch_transition)
                tick = time.perf_counter()
                oracle = solve_optimal(oracle_model)
                oracle_value = evaluate_policy(oracle_model, oracle)
                timings.append(dict(model=label, method='O', phase='solve', elapsed_seconds=time.perf_counter()-tick))
                exact_row = dict(model=label, oracle_objective=oracle_value, oracle_quote_edges=sorted(edges), methods={})
                model_tapes = [make_primitive_tape(model, seed) for seed in validation_seeds]
                tapes_archive[label] = model_tapes
                for method, policy in policies.items():
                    tick = time.perf_counter()
                    objective = evaluate_policy(model, policy)
                    raw_pnl = evaluate_policy(raw_model, policy)
                    if objective > oracle_value+1e-10:
                        raise AssertionError('Reference oracle action set failed to dominate comparator')
                    exact_row['methods'][method] = dict(objective=objective, raw_pnl=raw_pnl,
                                                       parameters=asdict(bands.get(method, bands['B3'])))
                    samples = simulate_policy(model, policy, validation_seeds, tapes=model_tapes)
                    records.extend(dict(model=label, regime=regime, risk_weight=risk, method=method,
                                        study='matching_known_kernel', crashed=False, **row) for row in samples)
                    timings.append(dict(model=label, method=method, phase='evaluation_and_replay',
                                        elapsed_seconds=time.perf_counter()-tick, sessions=len(samples)))
                exact.append(exact_row)
                r2 = policies['R2_fixed_quotes']
                for t, q, b in product(range(model.config.horizon), range(-model.config.limit_lots, model.config.limit_lots+1), range(len(model.batches))):
                    actions = r2.candidates(t, q, b)
                    values = [r2.evaluation.action_value(t, q, b, action) for action in actions]
                    decisions.append(dict(model=label, time=t, inventory_lots=q, batch=b,
                                          candidates=[asdict(a) for a in actions], baseline_Q=values[0],
                                          baseline_continuation_Q=values, selected=asdict(r2(t, q, b))))
                print(label, {m: round(v['raw_pnl'], 6) for m, v in exact_row['methods'].items()}, flush=True)
        # Frozen balanced policies encounter untrained flow/competition kernels.
        for risk in cfg['risk_weights']:
            _, _, frozen = trained[risk, 'balanced']
            for regime in ('informed', 'persistent'):
                actual, _, _ = trained[risk, regime]
                label = f'frozen_balanced_to_{regime}_risk{risk:g}'
                tapes = [make_primitive_tape(actual, seed) for seed in stress_seeds]
                tapes_archive[label] = tapes
                for method in ('B3', 'R2_fixed_quotes', 'R2_joint'):
                    policy = frozen[method]
                    samples = simulate_policy(actual, policy, stress_seeds, tapes=tapes)
                    records.extend(dict(model=label, regime=regime, risk_weight=risk, method=method,
                                        study='frozen_model_stress', crashed=False, **row) for row in samples)
        for label in dict.fromkeys(row['model'] for row in records):
            by_method = {method: sorted([r for r in records if r['model']==label and r['method']==method], key=lambda r:r['seed'])
                         for method in dict.fromkeys(r['method'] for r in records if r['model']==label)}
            baseline = by_method['B3']
            for method, rows in by_method.items():
                assert [r['seed'] for r in rows] == [r['seed'] for r in baseline]
                mean, lo, hi = paired_bootstrap_interval([r['pnl'] for r in rows], [r['pnl'] for r in baseline], replicates=4000)
                summaries.append(dict(model=label, method=method, sessions=len(rows),
                    mean_pnl=float(np.mean([r['pnl'] for r in rows])), paired_difference=mean,
                    descriptive_ci_low=lo, descriptive_ci_high=hi,
                    lower_10pct_mean=lower_tail_mean([r['pnl'] for r in rows]),
                    loss_fraction=float(np.mean([r['pnl']<0 for r in rows])),
                    **{'mean_'+k: float(np.mean([r[k] for r in rows])) for k in
                       ('objective', 'risk_penalty', 'customer_orders', 'customer_notional', 'fees',
                        'penalties', 'exchange_notional', 'inventory_sq_time', 'max_drawdown')}))
        with (out/'sessions.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        for filename, value in (('tuning.json', tuning), ('exact_values.json', exact),
                                ('summary.json', summaries), ('decisions.json', decisions),
                                ('timings.json', timings), ('primitive_tapes.json', tapes_archive)):
            write_json(out/filename, value)
        write_json(out/'status.json', dict(status='EXPLORATORY_KNOWN_SYNTHETIC_MODEL_ONLY',
                   sessions=len(records), crashes=0, tuning_evaluations=len(tuning),
                   final_test_opened=False, R3_evaluated=False, elapsed_seconds=time.perf_counter()-start))
    except Exception:
        write_json(out/'failure.json', dict(traceback=traceback.format_exc(), completed_records=records,
                                            elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == '__main__':
    main()
