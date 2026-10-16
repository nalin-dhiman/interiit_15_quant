"""Process-only orchestration of the already sealed research experiment.

No policy selection or seed changes. Each process gets its own logical clock,
SDK Engine and in-memory store. Parent analysis calls the sealed implementation.
An interrupted sequential prefix is retained and checked for exact replay.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import research_study as study


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def worker(task):
    scenario, seal, out = task
    out = Path(out)
    study.FAILURE_DIR = out / 'failed_cases'
    rows, tapes = [], {}
    with gzip.open(out / f'shard_{scenario}.jsonl.gz', 'wt') as stream:
        for path in range(seal['paths_per_scenario']):
            tapes[f'{scenario}/{path}'] = study.generate_prices('final', scenario, path)
            for family in study.FINAL_FAMILIES:
                for method in study.METHODS + study.EXTRAS:
                    row = study.run_case(method, {} if method == 'B0' else seal['selected'][method],
                                         'final', scenario, path, family, detail=True)
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    row.pop('own_trades')
                    row.pop('final_rfq_observations')
                    rows.append(row)
            if (path + 1) % 16 == 0:
                dump(out / f'progress_{scenario}.json', dict(scenario=scenario, completed_paths=path + 1))
    with gzip.open(out / f'tapes_{scenario}.json.gz', 'wt') as stream:
        json.dump(tapes, stream, sort_keys=True)
    return scenario, rows


def key(row):
    return row['scenario'], row['path'], row['family'], row['method']


def stable(row):
    # Whole-callback elapsed timing changes under host load; economics must not.
    return {k: v for k, v in row.items() if k != 'callback_seconds'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seal', type=Path, required=True)
    p.add_argument('--sequential-prefix', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    seal = json.loads(a.seal.read_text())
    if seal['source_hashes'] != study.sources():
        raise RuntimeError('economic source guard failed before parallel replay')
    if not 1 <= a.workers <= 8:
        p.error('workers must be between 1 and 8')
    started = time.perf_counter()
    study.init(a.out, 'same sealed final experiment; independent process orchestration')
    study.write_json(a.out / 'POLICY_SEAL.json', seal)
    prefix = {}
    with gzip.open(a.sequential_prefix / 'sessions.jsonl.gz', 'rt') as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise RuntimeError('incomplete sequential JSON record')
            row = json.loads(line)
            prefix[key(row)] = stable(row)
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(a.out / 'ORCHESTRATION_SEAL.json', dict(
        driver_sha256=driver_hash, workers=a.workers, sequential_prefix=str(a.sequential_prefix),
        prefix_records=len(prefix), same_economic_policy_seal=True, no_parameter_or_seed_changes=True,
        no_new_independent_N_from_replay=True, reason='parallel CPU orchestration; no outcome-based exclusion'))
    completed = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(worker, (s, seal, str(a.out))) for s in range(8)]
        for future in as_completed(futures):
            scenario, rows = future.result()
            completed[scenario] = rows
            dump(a.out / 'progress.json', dict(completed_scenarios=sorted(completed),
                                              completed_sessions=sum(map(len, completed.values()))))
    rows, tapes, checked = [], {}, 0
    with gzip.open(a.out / 'sessions.jsonl.gz', 'wt') as target:
        for scenario in range(8):
            with gzip.open(a.out / f'shard_{scenario}.jsonl.gz', 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    identity = key(row)
                    if identity in prefix:
                        if stable(row) != prefix[identity]:
                            dump(a.out / 'REPLAY_FAILURE.json', dict(key=identity,
                                sequential=prefix[identity], parallel=stable(row)))
                            raise AssertionError('parallel replay changed economic record')
                        checked += 1
                    target.write(line)
            rows.extend(completed[scenario])
            with gzip.open(a.out / f'tapes_{scenario}.json.gz', 'rt') as stream:
                tapes.update(json.load(stream))
    if checked != len(prefix):
        raise AssertionError('not every sequential prefix record reproduced')
    if seal['source_hashes'] != study.sources() or driver_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError('source changed during execution')
    with gzip.open(a.out / 'price_tapes.json.gz', 'wt') as stream:
        json.dump(tapes, stream, sort_keys=True)
    summary = study.summaries(rows)
    primary = study.inference(study.pairs(rows, 'R2B', seal['fallback']))
    dump(a.out / 'summary.json', summary)
    dump(a.out / 'primary_inference.json', primary)
    comparisons = (('R2B_minus_B4', 'R2B', 'B4'), ('R2B_matched_minus_B4', 'R2B_matched', 'B4'),
        ('B4_minus_reduce', 'B4', 'B4_reduce'), ('B4_minus_zeroquote', 'B4', 'B4_zeroquote'),
        ('B4_minus_B3', 'B4', 'B3'), ('BR2_minus_B4', 'BR2', 'B4'),
        ('R2B_fixedprice_minus_B4_fixedprice', 'R2B_fixedprice', 'B4_fixedprice'))
    dump(a.out / 'descriptive_ablations.json', {name: study.inference(study.pairs(rows, left, right))
                                             for name, left, right in comparisons})
    dump(a.out / 'execution.json', dict(elapsed_seconds=time.perf_counter() - started,
        scheduled_sessions=len(rows), independent_paths=seal['final_paths'], source_unchanged=True,
        sequential_prefix_reproduced=checked, economic_replay_exact=True,
        timings_excluded_from_economic_equality=True, no_new_N_from_prefix=True,
        branch_decision='SCOPED_SYNTHETIC_PRIMARY_PASS' if primary['superiority_gate'] else 'INCREMENTAL_SUPERIORITY_NOT_SUPPORTED',
        R3_enabled=False, competition_scope=False))
    study.manifest(a.out)
    print(json.dumps(dict(primary=primary, replayed_prefix=checked,
                         method_means={m: v['mean_pnl'] for m, v in summary.items()}), indent=2))


if __name__ == '__main__':
    main()
