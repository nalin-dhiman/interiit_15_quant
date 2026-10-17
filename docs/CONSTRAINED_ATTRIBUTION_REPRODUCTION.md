# Reproduce the constrained attribution benchmark

This experiment uses the separately supplied local SDK. It is independent of
the browser simulator. Set up Python3.11/NumPy2.3.5 and the SDK gate as described
in sdk-research.md. No third-party SDK or price CSV is redistributed here.

The protocol and executable source were committed before pilot execution
(871ff40), its source-hashed declaration was published before pilot (30da4c2),
and the chosen packages/sample size were published before final paths (cd01475).
The published primary is wide private rectangle A±20/B±10, fixed quote prices,
and the common package chosen by B4 development results. Secondary views do not
replace this primary. The complete source snapshot hash map is in the public
declaration and policy-seal JSON files.

Use NEW, unused output directories. These seed namespaces reproduce this
study; re-running them does not create fresh held-out confirmation.

```bash
export RFQ_SDK_ROOT="/absolute/path/to/IMC"
.venv/bin/python -m experiments.sdk_environment
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests_sdk -v
.venv/bin/python -m experiments.risk_matched_study declare --out runs/constrained-declaration
.venv/bin/python -m experiments.risk_matched_study pilot \
  --declaration runs/constrained-declaration/DECLARATION.json --out runs/constrained-pilot
.venv/bin/python -m experiments.risk_matched_study final \
  --seal runs/constrained-pilot/POLICY_SEAL.json --out runs/constrained-final --workers 12
.venv/bin/python scripts/audit_risk_matched.py \
  --study runs/constrained-final --pilot runs/constrained-pilot \
  --declaration runs/constrained-declaration/DECLARATION.json --out runs/constrained-audit
```

The final process checks a 32-session sequential/parallel prefix and archives
that duplicate separately; it is not counted as another independent sample.
Each final independent path has 32 policy sessions, with two opponent repeats,
two quote modes, two private rectangles, two selected-package labels and two
controllers. Some package labels can select identical parameters. They remain
explicit duplicate diagnostics, never additional independent observations.

The ordinary tests run without the SDK. The SDK suite skips explicitly when
its input is absent, and fails if a different snapshot or incompatible cached
SDK module is loaded. Run these studies only with an SDK copy you can use.
Raw output snapshots can include SDK source for local reproducibility; retain
those ignored outputs privately.

Independent audit verifies raw money, the actual price-innovation exposure
clock, all reservation/pending subsets, actual trade caps, identical matched
packages/primitive tapes, fixed-price formulas, tuning budgets, validation
selection, sample planning, sequential prefix and the path-level bootstrap.
Passing it verifies this finite synthetic execution study, not market realism,
optimality, production trading compatibility or publication novelty.
