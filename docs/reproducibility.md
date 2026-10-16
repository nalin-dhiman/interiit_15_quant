# Reproducibility

## Application

The browser app uses native ES modules and a Web Worker. Its code has no runtime
dependencies. Node 18+ runs the local server, tests and build; optional Actions
templates use Node 22. Every session is determined by its applied configuration and seed.
Changing a policy's draws must not change the exogenous price tape.

`npm test` runs independent ledger, sign, rounding, capacity and random-stream
regressions. `npm run build` copies only application assets and writes an asset
SHA-256 manifest. The production server can be checked both at `/` and at the
repository prefix `/interiit_15_quant/` used by Pages.

Comparison runs use actual independent seed/path units. Within-path policy
differences are paired; many RFQs from one path are not many independent paths.
Browser intervals are empirical descriptions of that simulation. Exports
include the settings, raw session outcomes and computed summary.

## NumPy library

The scientific dependency is pinned to NumPy 2.3.5; use Python 3.11.
Tests independently reconstruct principal and fees, enumerate tiny finite
policies and full paths, test response derivatives, and preserve counterexamples
outside theorem assumptions. Run `python -m unittest discover -s tests -v`.

`experiments.run_foundation` produces 180 synthetic runs with five fixed policies
at the default 12-session setting. It preserves reset strategies that lose
after costs. These illustrative seeds are not a newly sealed tail study.

`experiments.run_continuation` records configurations, source hashes, exact
reference values, equal candidate evaluations and replay outcomes. The known
response examples reuse explicit frozen conventional parameters. No numerical
equality with an oracle is advertised as trading alpha.

## Historical SDK results

`data/research-summary.json` includes the whole-policy primary, negative
ablations, fresh exploratory parameter factorial, calibration results and
planning limitations. The main unit is a price/customer path; two opponent
repeats were averaged inside it. The 2,048-path cap bound was reached against
a 34,838-path planning requirement, so planned power is not claimed.

The browser model and aggregate tables cannot recreate historical private
SDK executions by themselves. Optional adapters require the separately
installed SDK and its exact contract. The original raw-study hash is included
to identify the evidence; no unavailable dataset or external SDK is silently
substituted by browser output.
