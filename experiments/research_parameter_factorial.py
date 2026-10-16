"""Post-primary exploratory controller-by-selected-package diagnostic.

The main final results were already opened when this diagnostic was proposed.
It cannot revise the sealed primary or become new confirmatory evidence.
"""
from __future__ import annotations
from experiments.sdk_environment import KIT, source_key, source_path
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from experiments import research_study as study

ROOT = Path(__file__).resolve().parents[1]
PHASE = 'post_primary_factorial'
CELLS = tuple((controller, package) for controller in ('B4', 'R2B') for package in ('B4', 'R2B'))
DEFAULT_PACKAGES = ROOT/'data/sdk-study-parameters.json'


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    result = study.sources()
    for name in ('experiments/research_parameter_factorial.py', 'docs/RESEARCH_PARAMETER_DIAGNOSTIC.md'):
        result[name] = sha(ROOT/name)
    return result


def label(controller, package):
    return controller+'__'+package+'_package'


def declare(out, policy_seal=None):
    out.mkdir(parents=True, exist_ok=False)
    pilot = Path(policy_seal) if policy_seal else DEFAULT_PACKAGES
    packages = {method: json.loads(pilot.read_text())['selected'][method] for method in ('B4', 'R2B')}
    config = dict(stage='post-primary exploratory 2x2 parameter-package diagnostic',
        declared_utc=datetime.now(timezone.utc).isoformat(), phase=PHASE,
        main_results_already_opened=True, confirmatory=False, retuning=False,
        controllers=['B4', 'R2B'], packages=packages, scenarios=list(range(8)),
        paths_per_scenario=32, opponents=list(study.FINAL_FAMILIES),
        scheduled_sessions=2048, independent_paths=256,
        frozen_package_sha256=sha(pilot), package_input=str(pilot),
        opened_historical_aggregate_sha256=sha(ROOT/'data/research-summary.json'), source_hashes=sources(),
        uncertainty='unadjusted descriptive stratified-path bootstrap; no confirmatory gate',
        claim='effects of swapping whole frozen quote/risk/hedge packages in this synthetic domain, not distinctive reset alpha',
        runtime_R3_enabled=False, sealed_primary_revised=False)
    write(out/'config.json', config)
    write(out/'environment.json', dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),executable=sys.executable))
    for name in config['source_hashes']:
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source_path(name).read_bytes())
    return config


def contrast(rows, terms):
    grouped={}
    for row in rows:
        grouped.setdefault((row['scenario'],row['path']),{}).setdefault(row['family'],{})[row['method']]=row
    values=[]
    for (scenario,path),families in sorted(grouped.items()):
        differences=[]
        for family,cells in sorted(families.items()):
            selected=[cells[cell] for cell in terms]
            for key in ('price_tape_sha256','public_customer_tape_sha256'):
                if len({r[key] for r in selected})!=1:raise AssertionError('factorial primitive tapes differ')
            differences.append(sum(coefficient*cells[cell]['account']['pnl'] for cell,coefficient in terms.items()))
        values.append(dict(scenario=scenario,path=path,difference=float(np.mean(differences))))
    result=study.inference(values)
    result.pop('superiority_gate');result.pop('materiality')
    result.update(exploratory=True,multiplicity_adjusted=False,sealed_primary_revised=False,
                  scenario_mean_contrasts={str(s):float(np.mean([r['difference'] for r in values if r['scenario']==s])) for s in range(8)})
    return result


def summaries(rows):
    result={}
    for controller,package in CELLS:
        cell=label(controller,package);own=[r for r in rows if r['method']==cell]
        pnl=np.array([r['account']['pnl'] for r in own])
        result[cell]=dict(controller=controller,package=package,sessions=len(own),mean_pnl=float(pnl.mean()),
            worst_pnl=float(pnl.min()),pnl_quantile_05=float(np.quantile(pnl,.05)),negative_sessions=int((pnl<0).sum()),
            customer_orders=sum(r['account']['customer_trades'] for r in own),
            mean_customer_fills=float(np.mean([r['account']['customer_trades'] for r in own])),
            customer_notional=sum(r['account']['customer_notional'] for r in own),
            exchange_notional=sum(r['account']['exchange_notional'] for r in own),
            fees=sum(r['account']['fees_paid'] for r in own),penalties=sum(r['account']['penalties_paid'] for r in own),
            crashes=sum(r['crash'] for r in own),callback_errors=sum(r['callback_errors'] for r in own),
            own_rejections=sum(len(r['own_rejections']) for r in own),
            max_decomposition_error=max(r['decomposition_max_error'] for r in own),
            mean_q2_at_innovation=np.mean([r['innovation_exposure']['mean_q_squared_at_price_innovation'] for r in own],axis=0).tolist(),
            mean_monetary_components={k:float(np.mean([r['monetary_components'][k] for r in own])) for k in own[0]['monetary_components']},
            scenario_means={str(s):float(np.mean([r['account']['pnl'] for r in own if r['scenario']==s])) for s in range(8)})
    return result


def execute(out):
    config=json.loads((out/'config.json').read_text())
    if config['source_hashes']!=sources():raise RuntimeError('source/protocol changed after declaration')
    if (out/'sessions.jsonl.gz').exists():raise FileExistsError('Never overwrite executed diagnostic')
    study.FAILURE_DIR=out/'failed_cases'
    began=time.perf_counter();rows=[];tapes={}
    with gzip.open(out/'sessions.jsonl.gz','wt') as stream:
        for scenario in config['scenarios']:
            for path in range(config['paths_per_scenario']):
                tapes[f'{scenario}/{path}']=study.generate_prices(PHASE,scenario,path)
                for family in config['opponents']:
                    for controller,package in CELLS:
                        row=study.run_case(controller,copy.deepcopy(config['packages'][package]),PHASE,scenario,path,family,detail=True)
                        row.update(native_method=row['method'],method=label(controller,package),controller=controller,package=package,
                                   post_primary_exploratory=True)
                        stream.write(json.dumps(row,sort_keys=True)+'\n')
                        row.pop('own_trades');row.pop('final_rfq_observations');rows.append(row)
            write(out/'progress.json',dict(completed_scenarios=scenario+1,completed_sessions=len(rows)))
    if config['source_hashes']!=sources():
        write(out/'SOURCE_CHANGE_FAILURE.json',dict(before=config['source_hashes'],after=sources()))
        raise RuntimeError('source/protocol changed during execution; inference prohibited')
    with gzip.open(out/'price_tapes.json.gz','wt') as stream:json.dump(tapes,stream,sort_keys=True)
    a,b,c,d=[label(controller,package) for controller,package in CELLS]
    terms=dict(controller_at_B4_package={c:1,a:-1},controller_at_R2B_package={d:1,b:-1},
               package_with_B4_controller={b:1,a:-1},package_with_R2B_controller={d:1,c:-1},
               diagonal_selected_comparison={d:1,a:-1},interaction={d:1,b:-1,c:-1,a:1})
    comparisons={name:contrast(rows,coefficients) for name,coefficients in terms.items()}
    summary=summaries(rows)
    write(out/'summary.json',summary);write(out/'contrasts.json',comparisons)
    write(out/'execution.json',dict(elapsed_seconds=time.perf_counter()-began,scheduled_sessions=len(rows),independent_paths=256,
        source_unchanged=True,post_primary_exploratory=True,confirmatory=False,retuning=False,
        sealed_primary_revised=False,R3_enabled=False))
    print(json.dumps(dict(means={k:v['mean_pnl'] for k,v in summary.items()},contrasts=comparisons),indent=2))


def manifest(out):
    write(out/'SHA256SUMS.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('declare','execute'));parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--policy-seal',type=Path,help='Optional selected-parameter artifact; default is the compact historical package input.')
    args=parser.parse_args()
    if args.stage=='declare':declare(args.out,args.policy_seal);print('Frozen post-primary exploratory factorial protocol')
    else:execute(args.out)
    manifest(args.out)


if __name__=='__main__':main()
