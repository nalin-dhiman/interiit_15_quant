# Prospective bounded quote-freezing mechanism diagnostic

This protocol is frozen before opening new diagnostic paths. It isolates the
`freeze_current_quotes` intervention in the unchanged scalar B3 implementation,
not the R2 continuation approximation and not the previous lifetime-admission
ablation. The previous R2 superiority claim remains NOT_SUPPORTED.

## Policy and information

Ordinary scalar B3 and identical scalar B3 with `freeze_current_quotes=True`
use the previously sealed B3 parameters in
`runs/20261016T073900Z_fresh_policy_pilot/POLICY_SEAL.json`: mode B3, edge .04,
inventory skew .005, band .6, target 0, factor hedging false, no price forecast,
12 intervals, and supplied demo settlement B=2*A. There is no retuning. All
other arguments/defaults and normal all-subset capacity reservation remain
identical. Hedge **rules** are identical; realised hedge actions, inventory,
admission and prices can differ endogenously after the intervention. No fixed
price subclass or lifetime admission budget is used. Scalar B3's freeze flag
must actually activate; a silently ineffective factor-hedging ablation is not
permitted. All hypothetical current/future information remains inaccessible.

The actual unmodified supplied Engine, matching, customer generator, products,
fees, limits and settlement are used with the existing logical-time callback
harness. Acknowledgements are immediate; no transport/compute latency is
simulated. This is a synthetic mechanism experiment, not production execution
or market evidence. The harness delivers the next market/quotes before prior
fills, matching the observed SDK ordering; this remains part of both policies.

## Primitive histories and budget

There are exactly 200 independent price/customer paths: four declared price
laws, 50 paths each, with both `tight` and `inventory` opponent families for
each path, and both policies: 800 sessions. These opponent families have already
been studied and are development families here, not held-out families. Prices
are synthetic mean-one multiplicative A and AR(1) basis B-2*A, using the four
existing scenario tuples (.08,.03,.75), (.20,.06,.90), (.35,.10,.95),
(.12,.18,.98); 124 observations per tape. No original consumed holdout is read.

Full 256-bit SHA-256 seeds use domain
`freeze-mechanism-v1/20261016T103800Z/{phase}/{scenario}/{path}/{stream}`.
Factor, basis, customers and each opponent family have separate primitive
streams. `diagnosis` and `test` phases are separate; paired policies share
prices/customers/opponents. The declared numeric seeds are audited for
collisions before diagnosis. Independent PRNG streams and generated path
independence are modelling assumptions, not properties of actual markets.

## Outcomes and accounting

Archive one final observation per RFQ, executed trades, complete price tapes,
parameters and source hashes. For actual accepted quote replacements, record
whether the live price changed. An identical accepted price does not restart
its quote age; an accepted withdrawal removes its age. At matching, age is
logical matching time minus the time its current live price was installed.
Measure age for all positive final quotes and separately customer fills;
missing quotes have no invented age. This records original Engine state after
acceptance and is offline instrumentation, not a strategy feature.

Report money PnL, signed customer spread at matching, fills/notional, fees,
penalties, exchange turnover, price-move and settlement contributions,
innovation inventory exposure, rejection/error counts, freeze activation,
omitted replacements, actual accepted price changes, quote age and all failed
checks. Independently replay principal/cash/inventory/fees and decompose PnL;
any reconciliation failure stops comparison and is preserved. No observations
or failed seeds may be excluded. Reproduce the first path in each scenario,
both opponents and policies (16 repeats); these are not new statistical units.

## Predeclared decision and interpretation

The primary difference is ordinary B3 minus frozen B3, averaging opponent
repeats within each independent path and weighting the four scenarios equally.
Use 6,000 stratified path bootstrap replicates and report a two-sided 95%
percentile interval. This assumes independent paths within scenarios and a
fixed equal scenario mixture; finite synthetic samples give empirical
uncertainty, not a uniform certificate. Opponent repeats are not independent
replicates. This is a bounded diagnostic, not a powered tail study.

Mechanism GO requires simultaneously: (i) lower 95% bound of the PnL difference
strictly exceeds 1 currency unit/session, (ii) ordinary B3 mean PnL is positive,
(iii) ordinary B3 mean customer spread exceeds frozen B3's, and (iv) freeze
activation and accounting checks pass. If any fail, report NO-GO and do not
redesign absent a precise newly evidenced diagnosis. No post-hoc alternative
gate may replace this one. Quote-age/fill changes are supporting diagnostics,
not additional confirmatory tests. A GO justifies one boundary-reset redesign
and fresh equal-budget comparison; it does not establish partial-reset value.
R3 stays disabled because the prior fill calibration gate failed.
