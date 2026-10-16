"""Profile full boundary callbacks through actual Engine/Runner contract logic.

Worker and network are controlled adapters. Timings include the whole Python
Dealer callback plus adapter argument/output serialization, but exclude real
multiprocessing and network overhead, which separate localhost sessions test.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.boundary_async_replay import BoundaryReplay, make_dealer, source_hashes


async def profile(products, mode):
    r = await BoundaryReplay(make_dealer(mode, observation_limit=0, diagnostic_limit=0),
                             products=products).start()
    try:
        await r.bootstrap()
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        admitted = bool(r.positive_quote_messages())
        await r.deliver(r.engine.advance())
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        await r.deliver(r.engine.advance(final=True))
        await r.drain_actions()
        await r.tick()
        await r.drain_actions()
        await r.deliver(r.engine.finish())
        reconciliation = r.reconcile()
        by_method = {}
        for call in r.runner.worker.calls:
            by_method.setdefault(call["method"], []).append(call["elapsed_ns"] / 1e9)
        timing = {method: dict(count=len(values), max_seconds=max(values),
                               mean_seconds=sum(values)/len(values))
                  for method, values in by_method.items()}
        return dict(products=products, rfqs_per_interval=3 * products, mode=mode,
                    full_callback_timings=timing,
                    maximum_callback_seconds=max(v["max_seconds"] for v in timing.values()),
                    initial_admission_after_confirmation=admitted,
                    reconciliation=reconciliation,
                    callback_errors=r.dealer.callback_errors,
                    boundary_state=r.dealer.boundary_state,
                    boundary_failures=list(r.dealer.boundary_failures),
                    boundary_model_fallbacks=r.dealer.boundary_model_fallbacks,
                    boundary_deferred_items=getattr(r.dealer, "boundary_deferred_items", None),
                    source_controlled_transport=True)
    finally:
        await r.stop()


async def run_profiles():
    return [await profile(products, mode) for products in (2, 50, 250, 251, 1000)
            for mode in ("B3", "R2")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    before, log, begin = source_hashes(), io.StringIO(), time.perf_counter()
    with contextlib.redirect_stdout(log):
        rows = asyncio.run(run_profiles())
    result = dict(scope="whole callback plus adapter serialization; excludes real process/network overhead",
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter() - begin,
                  source_hashes=before, source_unchanged=before == source_hashes(), rows=rows)
    result["all_callbacks_under_one_second"] = all(row["maximum_callback_seconds"] < 1 for row in rows)
    result["all_accounts_reconcile"] = all(row["reconciliation"]["reconciled"] for row in rows)
    result["unsupported_profiles"] = [dict(products=row["products"], mode=row["mode"],
                                            reasons=row["boundary_failures"])
                                      for row in rows if not row["initial_admission_after_confirmation"]]
    result["all_pass"] = result["source_unchanged"] and all(
        row["maximum_callback_seconds"] < 1 and row["initial_admission_after_confirmation"]
        and row["reconciliation"]["reconciled"] and row["callback_errors"] == 0 for row in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".runner.log").write_text(log.getvalue())
    print(json.dumps({"all_pass": result["all_pass"], "rows": [{k: r[k] for k in
        ("products", "rfqs_per_interval", "mode", "maximum_callback_seconds",
         "initial_admission_after_confirmation", "callback_errors")} for r in rows]}, indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
