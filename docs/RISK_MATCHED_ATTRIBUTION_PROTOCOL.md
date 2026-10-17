# Prospective constrained RFQ attribution benchmark

This protocol is frozen before tuning or final paths. The starting code is
4f4333c66033d1894011e58c1fc455e3c2f8cb34. Earlier positive and negative
studies are opened prior information, not observations in this study.

## Question and single primary estimand

Does R2B improve mean raw monetary PnL by more than 1 currency unit per session
relative to B4, when both use a common development-selected package, identical
causal quote-price rules and the same ex-ante admissible exposure rectangle?
The primary uses A +/-20 and B +/-10 (gross A-equivalent exposure <=40),
quotes independent of own inventory, and the package chosen by B4 validation.
The estimand averages eight equally weighted scenarios and two opponent repeats
within each independent price/customer path. This is one primary contrast.

A tighter A +/-10, B +/-5 rectangle (gross <=20), inventory-aware quote rules,
and R2B-selected packages are secondary descriptive comparisons. Same envelope
is NOT equal realised variance, drawdown or utility. No session is excluded,
rescaled or selected based on its realised exposure. Realised exposure,
participation and downside are outcomes. The rectangle is a fixed allocation,
a strict subset of a gross-exposure ball; optimal allocation is not claimed.

## Economics, information and implementation

Use the actual supplied offline SDK Engine's matching, fees, lot/tick rounding,
penalties and settlement. The SDK files and actual contract limits A50/B25 are
unchanged. Private policy limits intersect those limits BEFORE quote admission,
exchange target rounding, value evaluation and future admissibility. Every live
quote fill subset and pending hedge subset must fit the policy rectangle.
Customer fills are always booked in full. A private-cap breach never creates
an official cash penalty. Controllers retain the reconciled boundary protocol;
no capacity is obtained from an unconfirmed hedge.

The supplied pricing/fee mechanism is exogenous and simple. No real-market,
production, impact/depth or arbitrary-latency claim follows. The simulator
alone observes customer limits and all competitor quotes. Policies receive
only their original permitted public callback fields and own outcomes.
The scalar continuation is approximate and its future RFQ law is not fitted
or validated by this comparison. No mathematical improvement certificate or
response adaptation is enabled.

## Frozen tuning and selection budgets

Each controller uses the SAME existing 24 distributed joint packages from
research_study.parameters('B4'). There are two controllers, two rectangles
and two quote modes. Each candidate receives 12 cases: four regular scenarios,
three new paths, alternating the two development opponent families. Thus each
controller gets 1,152 tuning executions in total (24*12*2*2); total 2,304.
The fixed-price mode sets inventory to zero only in the quote-price formula;
it never edits holdings, capacity or cash. The other mode uses inventory skew.

For each rectangle/mode, each controller nominates its highest training-mean
package with lowest candidate index breaking ties. Evaluate both controllers
on BOTH nominations on eight scenarios, 12 new paths per scenario and both
development opponents. This gives 3,072 validation executions. Duplicate
nominations remain explicit duplicate labels, not independent samples.
Choose the common primary package by B4's validation mean over the nominations;
choose an R2B validation package by the same rule for descriptive transfer.
Both have the same candidate/evaluation budget; equal CPU is not claimed.
No final data select packages, model settings or the primary comparison.

## Generators and primitive coupling

Namespace: constrained-attribution-v1/20261017T051731Z. All price, basis,
customer, competitor and tie streams use new SHA-256 keys. Existing eight price
laws are reused with NEW paths: four autoregressive bases, random-walk basis,
alternating basis, regime/burst basis, and LAST_PRICE settlement/martingale
root control. These are known synthetic families, not new real-market data.
The LAST_PRICE scenario still has customer edge/capacity opportunities; it is
NOT a universal no-reset-value null. Existing no-customer-edge accounting
fixtures are the economic no-free-profit controls.

Development competitors are standard/wide. Final competitors are two new
predeclared structures held out from this selection: size-dependent quote edges
and public-basis responsive quote midpoints. They use only their permitted
public fields and event-keyed randomness. The SDK tie choice remains uniform,
but a research RNG adapter keys each choice by RFQ identity, rather than draw
count. It observes eligibility only inside the simulator; no hidden limit is
passed to a policy. All eligible-order queue/choice counts are asserted.
Matched shocks do not imply identical fills or outcomes. Customer/public price
hashes must match across ALL controllers/packages/budgets/modes on a path.

## Planning and final execution

Use the validation paired R2B-minus-B4 values for the primary package/mode/
rectangle. Average the two opponent repeats before computing variance.
Effective per-path variance is the mean of the eight WITHIN-scenario sample
variances, not the variance of pooled scenario means.
N_required=ceil((1.960+.842)^2 * variance / (2-1)^2).
Round up to a multiple of eight, floor1,024 and cap4,096 independent paths.
This targets approximate 80% normal planning power at true gain+2 versus
materiality+1. Pilot variance may not transfer to final opponents or paths;
if the cap binds, do not claim planned power was attained.

Seal selected packages, sample size, all executable/protocol/test source hashes
and SDK source hashes BEFORE generating any final tape. Final: eight scenarios,
N/8 paths each, two opponents, two rectangles, two quote modes, two package
labels and two controllers =32N policy executions. Every exception stops
inference and preserves its seeds/tape/trades/trace. No rerolls or exclusions.
A fixed first-path prefix is rerun to verify sequential/parallel economic
identity, ignoring only wall-clock callback durations. Generation processes
receive disjoint scenario/path tasks; no policy exploration changes another's
primitive history.

## Inference and decision gates

The independent unit is a price/customer path. Opponent repetitions are averaged
inside it. Use a 5,000-replicate equal-stratum path bootstrap, seeded separately
from market streams, and one 95% two-sided interval for the primary mean gap.
Lower>+1 supports a material benefit in this declared constrained mixture.
Upper<+1 excludes the predeclared +1 material gain at this interval level.
An interval wholly inside(-1,+1) supports practical equivalence for this ONE
mean estimand under approximate coverage; failure to reject does not do so.
Upper<0 supports a mean loss in this scope. Other intervals/scenarios/downside
quantiles are descriptive, without simultaneous coverage or tail-power claims.
No universal absence/presence of reset value or market profitability follows.

## Permanent audit and scientific reporting

Independently replay every principal, customer sign, fee, penalty and settlement
from own trades and actual price tapes. Recompute exposure at the actual
price-innovation clock, actual per-trade positions, quote capacity envelopes,
paired units, shared packages and matched fixed quote prices when both live.
Save all raw RFQs/trades/traces and every failure. Final results retain prior
counterevidence and distinguish policy intervention contrasts from accounting
PnL components; the latter do not alone identify causal mediation.
Publication novelty remains an independent question with substantial existing
inventory-control, dealer-hedging and response literature. No acceptance or
original-theorem claim is implied by this benchmark.
