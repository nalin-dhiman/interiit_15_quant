"""Predeclared synthetic final-RFQ calibration; no deployment integration."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

from resetmm.accounting import round_price
from resetmm.calibration import (
    bernoulli_losses, fit_fill_models, fit_markout_models, in_training_support,
    in_occupied_bin_support,
    predict_fill, predict_markout, validate_final_records,
)
from resetmm.risk import reserve_batch
from resetmm.toy_exchange import make_scenario, run


class PublicExplorationCollector:
    """Features come solely from policy callbacks; outcome joins occur offline."""
    def __init__(self, seed: int, config: dict):
        sequence = np.random.SeedSequence([seed, config['exploration_stream_identifier']])
        self.rng = np.random.default_rng(sequence)
        self.edges = config['exploration_edges']
        self.chosen_edges: dict[str, float] = {}
        self.final: dict[str, dict] = {}

    def hedge(self, prices, positions, products, *, dt):
        return [(k, -q) for k, q in positions.items() if q != 0]

    def quote(self, orders, prices, positions, products, *, dt):
        accepted = reserve_batch(orders, positions, products)
        output = {}
        for order in orders:
            oid, key, side = order['order_id'], order['product_id'], order['side']
            if oid not in self.chosen_edges:
                self.chosen_edges[oid] = float(self.rng.choice(self.edges))
            reference = prices[key]
            d = 1 if side == 'BUY' else -1
            quote = (round_price(reference+d*self.chosen_edges[oid], products[key].tick, side)
                     if oid in accepted else None)
            count = self.final.get(oid, {}).get('callback_count', 0)+1
            self.final[oid] = dict(
                order_id=oid, product_id=key, side=side, quantity=order['quantity'],
                reference=reference, inventory=positions[key], final_quote=quote,
                final_edge=None if quote is None else d*(quote-reference),
                callback_count=count,
            )
            output[oid] = quote
        return output


def generate_session_records(seed: int, regime: str, config: dict) -> tuple[list[dict], dict]:
    scenario = make_scenario(seed, intervals=config['intervals'], products=config['products'],
                             rfqs=config['rfqs'], updates=config['updates'], dt=config['dt'],
                             regime=regime)
    policy = PublicExplorationCollector(seed, config)
    result = run(scenario, policy, keep_trace=True)
    records = []
    product_index = {k: i for i, k in enumerate(scenario.product_ids)}
    # No private RFQ fields are accessed. Only final own outcomes and future
    # public prices (labels only, after complete execution) join the features.
    for trace in result['trace']:
        interval = trace['interval']
        next_index = (interval+1)*config['updates']
        for outcome in trace['outcomes']:
            feature = dict(policy.final[outcome['order_id']])
            d = 1 if feature['side'] == 'BUY' else -1
            future_public_price = float(scenario.prices[next_index, product_index[feature['product_id']]])
            won = bool(outcome['won'])
            records.append(dict(
                session_id=f'{regime}:{seed}', **feature, won=won,
                markout=d*(future_public_price-feature['reference']) if won else None,
            ))
    validate_final_records(records)
    session_metrics = {k: v for k, v in result.items() if k not in ('trace', 'ledger')}
    return records, session_metrics


def evaluate_records(records: list[dict], fill_models: dict, markout_model: dict) -> dict:
    validate_final_records(records)
    quoted = [r for r in records if r['final_quote'] is not None and r['won'] is not None]
    x = np.array([r['final_edge'] for r in quoted])
    y = np.array([r['won'] for r in quoted], float)
    filled = [r for r in quoted if r['won'] and r['markout'] is not None]
    fx = np.array([r['final_edge'] for r in filled])
    fy = np.array([r['markout'] for r in filled])
    summary = {'rfq_records': len(records), 'quoted': len(quoted), 'filled': len(filled),
               'withdrawn': sum(r['final_quote'] is None for r in records),
               'censored_outcomes': sum(r['won'] is None for r in records),
               'callbacks_recorded': sum(r['callback_count'] for r in records),
               'training_edge_range_coverage': float(in_training_support(fill_models['pooled'], x).mean()),
               'training_occupied_bin_coverage': float(in_occupied_bin_support(fill_models['pooled'], x).mean()),
               'training_bin_counts': fill_models['monotone']['counts'],
               'fill_models': {}, 'markout_models': {}, 'bin_calibration': []}
    for name, model in fill_models.items():
        predicted = predict_fill(model, x)
        summary['fill_models'][name] = dict(bernoulli_losses(y, predicted),
                                          observed_rate=float(y.mean()), predicted_rate=float(predicted.mean()))
    for conditional in (False, True):
        prediction = predict_markout(markout_model, fx, conditional=conditional)
        summary['markout_models']['conditional_bins' if conditional else 'pooled'] = {
            'mse': float(np.mean((fy-prediction)**2)),
            'prediction_bias': float(np.mean(prediction-fy)),
            'observed_mean': float(fy.mean()), 'predicted_mean': float(prediction.mean()),
        }
    edges = fill_models['monotone']['bin_edges']
    for low, high in zip(edges, edges[1:]):
        selected = (x >= low) & (x < high)
        fill_selected = (fx >= low) & (fx < high)
        row = {'edge_interval': [low, high], 'quoted': int(selected.sum()),
               'filled_with_markout': int(fill_selected.sum())}
        if selected.any():
            row['observed_fill'] = float(y[selected].mean())
            row['predicted_fill'] = {name: float(predict_fill(model, x[selected]).mean())
                                     for name, model in fill_models.items()}
        if fill_selected.any():
            row['observed_markout'] = float(fy[fill_selected].mean())
            row['predicted_markout'] = float(predict_markout(markout_model, fx[fill_selected], conditional=True).mean())
        summary['bin_calibration'].append(row)
    grouped = defaultdict(list)
    for r in quoted:
        grouped[r['session_id']].append(r)
    session_losses = []
    for key, rows in sorted(grouped.items()):
        sx = [r['final_edge'] for r in rows]
        sy = [r['won'] for r in rows]
        row = {'session_id': key, 'quoted': len(rows),
               'fill': {name: bernoulli_losses(sy, predict_fill(model, sx))
                        for name, model in fill_models.items()}}
        fills = [r for r in rows if r['won'] and r['markout'] is not None]
        row['markout'] = {}
        for conditional in (False, True):
            prediction = predict_markout(markout_model, [r['final_edge'] for r in fills], conditional=conditional)
            observed = np.array([r['markout'] for r in fills])
            row['markout']['conditional_bins' if conditional else 'pooled'] = {
                'mse': float(np.mean((observed-prediction)**2)),
                'prediction_bias': float(np.mean(prediction-observed)),
            }
        session_losses.append(row)
    summary['session_losses'] = session_losses
    summary['session_paired_differences_vs_pooled'] = {}
    for name in ('monotone', 'logistic'):
        diffs = np.array([r['fill'][name]['brier']-r['fill']['pooled']['brier'] for r in session_losses])
        summary['session_paired_differences_vs_pooled'][name] = {
            'brier_difference_mean': float(diffs.mean()),
            'descriptive_standard_error': float(diffs.std(ddof=1)/np.sqrt(len(diffs))),
            'sessions': len(diffs),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not (out/'PROTOCOL.md').is_file() or not (out/'config.json').is_file():
        raise RuntimeError('Write the protocol and complete config before execution')
    if (out/'metrics.json').exists():
        raise RuntimeError('Refuse to overwrite completed calibration evidence')
    config = json.loads((out/'config.json').read_text())
    for split in ('train', 'validation', 'heldout_family'):
        first, last = config[split]['seeds']
        if first >= 3000 or last >= 3000 or last-first+1 != config['sessions_per_split']:
            raise ValueError('Predeclared development blocks only; no final seeds')
    source_paths = ['resetmm/calibration.py', 'experiments/fit_synthetic_response.py',
                    'resetmm/toy_exchange.py', 'resetmm/accounting.py', 'resetmm/risk.py',
                    'tests/test_calibration.py']
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    hashes.update({p: hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ('PROTOCOL.md', 'config.json')})
    (out/'source_hashes_before_run.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps(dict(
        python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform()), indent=2)+'\n')
    started = time.perf_counter()
    data, ledger_metrics = {}, {}
    for split in ('train', 'validation', 'heldout_family'):
        block = config[split]
        rows, ledger_metrics[split] = [], []
        first, last = block['seeds']
        for seed in range(first, last+1):
            session_rows, metrics = generate_session_records(seed, block['regime'], config)
            rows.extend(session_rows)
            ledger_metrics[split].append(dict(seed=seed, regime=block['regime'], **metrics))
        data[split] = rows
        with (out/f'{split}_final_rfq_records.jsonl.gz').open('wb') as file:
            with gzip.GzipFile(filename='', mode='wb', fileobj=file, mtime=0) as gz:
                for row in rows:
                    gz.write((json.dumps(row, sort_keys=True)+'\n').encode())
    fill_models = fit_fill_models(data['train'], config['bin_edges'],
                                  ridge=config['logistic_ridge'],
                                  max_iterations=config['logistic_max_iterations'],
                                  tolerance=config['logistic_tolerance'])
    markout_model = fit_markout_models(data['train'], config['bin_edges'],
                                       pseudocount=config['markout_pseudocount'])
    coefficients = {'fill_models': fill_models, 'markout_model': markout_model,
                    'trained_on': config['train'], 'official_or_deployable': False,
                    'markout_definition': 'positive adverse a = customer_direction * (next_update_price - final_reference)',
                    'limitations': ['synthetic fixture', 'unknown-market uncertainty uncertified',
                                    'conditional on quoted support', 'one-update markout label']}
    (out/'coefficients.json').write_text(json.dumps(coefficients, indent=2)+'\n')
    (out/'synthetic_parameters.py').write_text('# Synthetic offline coefficients; not a certified policy.\nPARAMETERS = '+repr(coefficients)+'\n')
    metrics = {split: evaluate_records(rows, fill_models, markout_model) for split, rows in data.items()}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (out/'collection_ledger_metrics.json').write_text(json.dumps(ledger_metrics, indent=2)+'\n')
    elapsed = time.perf_counter()-started
    print(json.dumps({'sessions': sum(len(x) for x in ledger_metrics.values()),
                      'rfq_records': sum(len(x) for x in data.values()),
                      'fit_split': 'train', 'elapsed_seconds': elapsed,
                      'logistic_converged': fill_models['logistic']['converged']}))


if __name__ == '__main__':
    main()
