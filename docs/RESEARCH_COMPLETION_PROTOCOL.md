# Prospective research completion protocol

This protocol specifies a reproducible local research study of cost-aware
inventory interventions, including negative results. Experiments use synthetic
price and customer laws with the separately supplied development SDK. Competition
administration, live deployment and external trading are outside its scope.

## Primary question and equal information

Does scalar continuation-aware R2B add at least 1 currency unit/session over the
strongest conventional policy selected before opening the final paths? B4 is a
conventional, fee-aware public-settlement-basis target band. It may open exposure
from zero. R2B includes that same target, no action, and partial inventory
reductions. Both use the same public information, acknowledgement barrier,
settlement map, lot/tick rules and actual exchange fees. Its continuation is an
approximation, not a policy-improvement certificate. Factor B4 is also retained
as a strong comparator. No hidden customer limits or future prices reach any
dealer. Risk preferences are not monetary PnL. R3 remains disabled in the SDK.

Tune B2 immediate hedging, B3 scalar and factor bands, B4 scalar and factor,
legacy boundary BR2, and matched B4-continuation R2B. Each gets 24 prospective
joint quote/hedge packages and 12 tuning sessions/package (4 regular laws times
3 independent paths, alternating standard/wide opponents). B2 gets six distinct
edge/skew pairs crossed with four forecast weights; the other families get the
six declared packages crossed with those weights. Before pilot, independent
review preserved the previous selected conventional and BR2 anchors: the final
two packages in every forecast block are (.08,.002,.4,.5,1) and
(.08,.005,.6,.5,1), ordered edge/skew/band/target/risk multiplier. They replace
two exploratory packages in every family, keeping 24 evaluations each. The
other four packages retain risk multipliers 0/.5/2/0. Not every knob affects every
family. This is equal evaluation budget, not equal-dimensional exhaustive tuning.
Select each candidate by mean raw PnL, fixed enumeration breaking ties.

Validation uses 12 new paths per regular law and both development opponent
families. Select the conventional fallback among B2/B3/B3_factor/B4/B4_factor
by validation mean PnL, fixed family order breaking ties. Use R2B-minus-fallback
paired pilot variance to size the final study: ceil(((1.960+.842)*sd)^2), assuming
materiality +1 and a planning alternative +2, round to eight, floor 512 and cap
2048 independent paths. These are approximate planning assumptions. If the cap
binds, report the precision shortfall; do not claim the planned power. The pilot
variance need not transfer to the new dynamics. Seal all sources and policies
before generating any final price tape.

## Declared environments and random units

Four development laws have factor/basis innovation standard deviations and AR
coefficients (.08,.03,.75), (.20,.06,.90), (.35,.10,.95), (.12,.18,.98).
The factor follows a positive martingale geometric walk. Final equal-weight
scenarios 0--3 repeat these laws on fresh paths. Scenarios 4--7 are structural
stresses: random-walk basis rho=1 with B=2A final settlement; alternating basis
rho=-.35; rho .95/.2 regimes switching each 30 ticks with sixfold innovation
bursts every 20 ticks; and a martingale-basis LAST_PRICE settlement null. The
latter removes both mean reversion and the deterministic settlement-basis edge.
Scenarios 4--7 use factor sigma .20 and basis sigma .10. All price laws are
research inventions; the SDK execution and settlement engine is unchanged, while
the publicly configured settlement map varies as explicitly stated.

New SHA-256 namespace research-completion-v1/20261016T134827Z separates phase,
scenario, path and factor/basis/customer/opponent streams. Engine customer and
tie RNGs use its documented seed offsets, audited for collisions. Opponent RNGs
are separate. Invalid generated prices abort without clipping/reroll. The two
final opponent families are volatility and sticky; their structures were used
in an earlier study and are not novel held-out families. Their final primitive
paths are fresh. Policies cannot affect the exogenous price/customer tapes.
Conditional tie draws may differ; matched primitives do not imply equal trades.

An independent unit is a generated price/customer path. Average the two opponent
repeats within it. Bootstrap paths within each of eight equally weighted scenario
strata (5000 replicates). The sole confirmatory gate is the primary paired 95%
interval lower endpoint greater than +1. Secondary intervals are descriptive,
without simultaneous coverage or a tail-power claim. If the gate fails, retire
the tested incremental superiority claim and retain the conventional fallback.
This does not prove resetting can never help in another model.

## Ablations, records and stop rules

Final policies also include B0 supplied demo, B4_reduce (B4-selected parameters,
exchange actions reducing existing exposure only), B4_zeroquote (B4-selected
hedging but zero quote forecast), and R2B_matched (all scalar B4-selected quote
and risk parameters, changing only intervention ranking). The latter is a
matched-parameter whole-policy ablation: inventory feedback can change quotes.
B4_fixedprice and R2B_fixedprice instead use B4-selected parameters with the
inventory term removed from the common quote-price rule. At each public market
both then calculate identical prices for the same RFQ. Admission/capacity and
fills may still differ as a consequence of hedging; this isolates hedging with
fixed quoted prices, not fixed executions or guaranteed offsetting fills.
All these ablations are descriptive alongside the jointly tuned primary.
All seven tuned families are retained regardless of their pilot/final result.

Archive raw accounts, every own trade, one final RFQ observation, complete price
tapes, configurations, source hashes, quote ages, admission delay, customer
orders/notional, exchange turnover, fees, penalties, downside quantiles,
inventory/exposure and whole-callback timings. Independently replay cash and
inventory. Abort comparison on any unexplained accounting mismatch, callback
error, rejected own action or unsupported boundary history. Preserve complete
failure fixtures; never exclude paths. The synchronous schedule assumes
immediate acknowledgements; earlier real Runner/asynchronous regressions remain
separate engineering evidence, not a production certificate.

No further tuning on these final paths. Passing fill calibration alone cannot
enable response adaptation; response and known finite-model mathematical
certificates are reported independently. RL and DMFT are closed as unnecessary
extensions unless this fixed programme diagnoses an additional gap.
