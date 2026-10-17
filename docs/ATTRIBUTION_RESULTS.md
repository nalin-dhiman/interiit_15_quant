# Constrained intervention attribution: results and limits

This study asks whether the existing continuation-based partial hedge (R2B)
improves mean monetary PnL over a conventional partial-target controller (B4)
when both receive the same public information, quote-price formula,
development-selected parameter package and admissible inventory rectangle.
It is an offline synthetic experiment using the supplied exchange SDK.

The source, primary comparison and selection procedure were published before
the pilot; the chosen policies and sample size were published before the final
paths. See [protocol](RISK_MATCHED_ATTRIBUTION_PROTOCOL.md),
[reproduction](CONSTRAINED_ATTRIBUTION_REPRODUCTION.md) and
[exposure constraint](COMMON_RISK_ENVELOPE.md). Older studies remain separate.

The separately implemented raw auditor passed all final sessions, 489,100
own trades, 2,617,344 final RFQ records and 33,374,978 capacity checks; it also
reconstructed pilot selection and the primary bootstrap. Maximum cash
discrepancy was 8.74e-11. It reuses the previously verified ledger routine and
checks archived participant reservation/pending traces, rather than a complete
exchange action log. This is an implementation audit within the project,
not independent external peer review. See
[machine-readable evidence](../data/constrained-attribution-results.json).

## Primary result

The final study contains **72,704 policy sessions on 2,272 independent paths**.
Each path repeats several controller/configuration/opponent combinations;
sessions are not independent statistical units. Opponent repeats are averaged
within a path, and resampling preserves the eight scenario strata.

| Primary outcome | Conventional B4 | Continuation R2B |
|---|---:|---:|
| Mean monetary PnL | 8.3337 | 6.8864 |
| Mean customer fills | 1.1142 | 1.7388 |
| Mean fees | 1.2088 | 0.9334 |
| Mean squared factor exposure at price innovations | 325.2487 | 209.1128 |
| Descriptive fifth percentile of session PnL | −40.8498 | −31.2993 |

The predeclared primary difference, **R2B minus B4**, is **−1.4474**, with a
95% stratified path-bootstrap interval **[−2.2133, −0.6791]**. Under independent
generated paths, fixed scenario weights and approximate bootstrap coverage,
this supports a mean loss for this implementation and excludes the declared
material gain of +1. It does not establish practical equivalence.

R2B also carried 35.7% less measured squared factor exposure, paid 22.8% less in
fees and had a better observed fifth percentile. The private rectangle is a
common ex ante constraint, **not equal realised variance or downside risk**.
These results do not establish an overall risk-return ordering, stochastic
dominance or population tail improvement. Both primary policies selected a
zero explicit risk multiplier; different risk preferences were not introduced
to explain the exposure difference.
Squared factor exposure is (q_A + 2q_B)^2 in the declared price law; it excludes
basis/idiosyncratic risk and does not equal realised PnL variance.

## What the accounting explains

| Contribution to mean R2B-minus-B4 PnL | Difference |
|---|---:|
| Exogenous price-move PnL | −1.6494 |
| Customer spread | −0.0747 |
| Fee savings | +0.2754 |
| Terminal settlement residual | +0.0014 |

This is a self-financing accounting identity, not causal mediation. More fills
did not generate more customer spread. Most of the positive average PnL comes
from the synthetic price/basis dynamics and settlement exposure, rather than
customer spread. Such exposure earns no guaranteed profit in another market.

## Selection, secondary comparisons and limitations

Each controller received the same 24-package tuning budget, a bounded search
rather than exhaustive optimization. Development
selection used 2,304 tuning and 3,072 validation sessions before opening final
results. Pilot within-scenario variance determined the sealed 2,272-path
allocation; the sample-size cap did not bind. The planning formula is an
approximation and does not guarantee achieved power after opponent changes.

Both selector labels chose identical packages in every context. The selected
wide-mode inventory skew is zero, so its fixed/inventory modes also coincide.
Eight reported contrast labels therefore represent **three distinct economic
controller comparisons**, all sharing the same final paths. Slight interval
differences for identical outcomes are bootstrap simulation variation. Tight
fixed/inventory modes selected different forecast weights, so their comparison
does not isolate inventory skew. Secondary intervals have no simultaneous
coverage claim and cannot replace the primary result.

New final seeds and two held-out opponent quote families were used. Price laws
remain previously studied synthetic laws. Opponents use fixed public quoting
heuristics and submit no hedges; they are not a model of strategic equilibrium
or actual dealer competition. There is no real-market or arbitrary latency,
impact, depth or production-certification claim.

R2B uses an approximate continuation model. No matching exact oracle for the
executed RFQ/competition kernel was available in this study. Inaccurate fill
probabilities, continuation truncation, state approximation and limited
economic opportunity remain possible explanations for its performance. The
experiment cannot distinguish them. Earlier exact finite-state checks do not
make this SDK controller globally optimal.

## Decision and scientific contribution

Retain B4 for the declared mean-PnL objective and retire this R2B superiority
claim. Keep R3 disabled: an additional response-adaptation advantage has not
been established. Do not repeatedly retune opened final paths as confirmation.

The useful contribution is an auditable attribution benchmark: explicit
deadlines and acknowledgements, conservative pending-action reservation,
self-financing cash, shared information/packages/admissible exposure and
path-level inference. [Nearest work](ATTRIBUTION_PRIOR_ART.md) already studies
dealer hedging and continuation approximation. New hedging theory, publication
priority and journal acceptance remain unestablished. A negative case study
can be informative without establishing that all partial resetting is useless.
