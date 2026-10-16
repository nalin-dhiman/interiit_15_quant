# Optional SDK research

The browser application and the ordinary NumPy tests run without the organiser
SDK. This optional source export contains the Python study harnesses,
calibration experiments, quote-freezing diagnostic, boundary controller
comparisons, parameter factorial, independent ledger audits and callback
profiles used during the research. The repository also preserves the earlier
controller as a regression fixture for the fractional-lot repair.

The SDK, supplied price CSVs and raw historical experiment archives are not
redistributed. No script downloads them or connects to an external exchange.

## Set up the local dependency

Use Python 3.11 and an SDK copy you are entitled to use. `RFQ_SDK_ROOT` points
to the supplied `IMC` directory containing `rfq/`, `examples/` and `demo/`:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sdk.txt
export RFQ_SDK_ROOT="/absolute/path/to/IMC"
.venv/bin/python -m experiments.sdk_environment
```

The preflight compares the SDK Python sources and reference participants with
the hashes in `data/sdk-contract-hashes.json`. A different SDK is rejected until
its contract has been audited. These experiments were checked with Python
3.11.15, NumPy 2.3.5 and websockets 16.0. A successful preflight verifies this
local source snapshot; it does not certify a live exchange or a submission.
The gate also rejects already imported `rfq` modules from another installation
and puts the configured SDK first on Python's import path. This prevents
accidental notebook or environment mixing; it is a reproducibility check, not
an adversarial security boundary.

The SDK can remain outside the repository. Source manifests use stable relative
keys and resolve them through `experiments.sdk_environment`; no symlink or
vendor-code commit is required. Experiment output directories may contain local
SDK source snapshots for reproducibility. Keep those ignored outputs private.

## Run the regressions

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests_sdk -v
```

With no SDK configured, the second command reports explicit module skips. It
does not pretend those contract checks passed. With the audited SDK configured,
it runs the hand-calculated contract, acknowledgement, withdrawal, capacity,
batch, fractional-lot, controller, response and asynchronous replay fixtures.
The ordinary `tests/` suite remains independent of SDK availability; it includes
the standalone decomposition, factor-diagnostic and raw-audit fixtures.

## Start with a small demonstration

Choose unused output directories; the study commands refuse to overwrite
executed results where their protocol requires a new run.

```bash
.venv/bin/python -m experiments.research_demo --out runs/sdk-demo
.venv/bin/python -m experiments.research_study smoke --out runs/sdk-smoke
.venv/bin/python -m experiments.sdk_async_replay --out runs/sdk-ack-replay
.venv/bin/python -m experiments.boundary_async_replay --out runs/sdk-boundary-replay
```

The demonstration generates its own synthetic prices and evaluates the frozen
B4 and R2B parameter packages. The smoke command exercises all main controller
families in two declared scenarios. The replay commands use controlled local
delivery schedules and the actual SDK Engine/Runner logic. They do not certify
real transport latency, worker isolation or an external market.

## Comparative studies and response estimation

```bash
.venv/bin/python -m experiments.research_study pilot --out runs/sdk-pilot
.venv/bin/python -m experiments.research_study final \
  --seal runs/sdk-pilot/POLICY_SEAL.json --out runs/sdk-final

.venv/bin/python -m experiments.research_response declare --out runs/sdk-response
.venv/bin/python -m experiments.research_response execute --out runs/sdk-response

.venv/bin/python -m experiments.research_parameter_factorial declare \
  --out runs/sdk-factorial
.venv/bin/python -m experiments.research_parameter_factorial execute \
  --out runs/sdk-factorial
```

These are substantial experiments. Review their protocol and schedule before
running them. Every new seal includes current source and SDK-gate hashes.
The factorial defaults to compact historical parameter packages in
`data/sdk-study-parameters.json`; `--policy-seal` on its `declare` command can
select a different frozen package input. Its contrasts remain exploratory.

The exported seed namespaces intentionally preserve the historical algorithms.
Rerunning those namespaces is a reproduction, not fresh held-out confirmation.
For a new confirmatory study, define a new namespace and protocol **before**
selection or observation, generate a new source seal, and retain all outcomes.
An old historical seal cannot be reused against this portable export because
file paths and dependency-gate code have changed.

Response estimation uses one final record per RFQ and separate development and
validation streams. Runtime response adaptation remains disabled. Passing
calibration inside its synthetic assignment law does not establish transport
to another opponent family, identify hidden customer limits, or prove an
incremental trading benefit.

## Audit and analysis tools

`scripts/audit_research_study.py` independently reconstructs all cash, inventory,
fees, settlement and declared statistical units from a completed study. The
other audit and decomposition utilities preserve earlier study formats.
Use `--help` for their required inputs. For example:

```bash
.venv/bin/python scripts/audit_research_study.py \
  --study runs/sdk-final --pilot runs/sdk-pilot --out runs/sdk-final-audit
.venv/bin/python scripts/profile_boundary_dealer.py --out runs/sdk-profile
```

CSV-based factor and original development-data studies need the separately
supplied price and settlement CSVs. Their defaults use the configured SDK
directory; explicit file arguments can identify other authorised local copies.
Exact historical replay, post-hoc attribution and parallel prefix verification
also need the corresponding original raw logs and seals. Aggregate results
alone cannot recreate those inputs. `replay_ack_repair` and
`run_research_final_parallel.py` therefore require explicit reference paths.

All generated outputs belong under ignored `runs/` or another private location.
The public export includes scientific source, protocols, configurations,
compact parameter inputs and tests. Manuscript builders, task ledgers, vendor
SDK files, supplied datasets and raw research archives are excluded.
