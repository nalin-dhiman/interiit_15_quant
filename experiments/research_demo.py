"""Run a local reproducible demonstration of the frozen research policies.

This generates a synthetic path and uses the supplied SDK Engine. Its two
sessions are a demonstration, not additional held-out performance evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.research_study import ROOT, generate_prices, run_case, write_json
from experiments import research_study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', type=int, choices=range(8), default=0)
    p.add_argument('--path', type=int, default=0)
    p.add_argument('--family', choices=('standard', 'wide', 'volatility', 'sticky'), default='standard')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    artifact = json.loads((ROOT / 'configs/research_policy.json').read_text())
    out = args.out or ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_interactive')
    out.mkdir(parents=True, exist_ok=False)
    research_study.FAILURE_DIR = out / 'failed_cases'
    config = dict(scope='local synthetic demonstration; not independent performance confirmation',
        scenario=args.scenario, path=args.path, family=args.family, phase='interactive',
        default=artifact['default'], R3_enabled=False, external_trading=False)
    write_json(out / 'config.json', config)
    rows = [run_case(artifact['default'], artifact['baseline_parameters'], 'interactive',
                     args.scenario, args.path, args.family, detail=True),
            run_case('R2B', artifact['reset_parameters'], 'interactive',
                     args.scenario, args.path, args.family, detail=True)]
    write_json(out / 'raw_sessions.json', rows)
    write_json(out / 'price_tape.json', generate_prices('interactive', args.scenario, args.path))
    write_json(out / 'SHA256SUMS.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(output=str(out), default=artifact['default'],
        ledger_pnl={r['method']: r['account']['pnl'] for r in rows},
        interpretation='One reproducible demonstration; use the sealed research reports for inference.'), indent=2))


if __name__ == '__main__':
    main()
