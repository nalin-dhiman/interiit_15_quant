# Controlled SDK intervention protocol, frozen before execution

This experiment tests an explicitly synthetic observation and intervention model
through the supplied, unmodified organiser Engine and Store. It does not infer
the organiser's round law and does not reuse either supplied demo price CSV.

The frozen configuration is written to `response/config.json` before any run.
There are 100 independent training, 100 in-family validation, and 100 shifted
opponent-family validation sessions, each with 120 intervals and one RFQ per
interval. Separate seeded primitive streams generate exogenous prices, customer
attributes/hidden limits, opponent family/quote choices, quote exploration, and
Engine tie breaking. Each session has a fresh generated price history. Price
increments have an observable 0.4 lag coefficient and independent bounded
innovations; B has twice the A increment plus an independent idiosyncratic
increment. This is a research model, not an asserted market or organiser law.

At each matching deadline the actor submits one final randomly explored edge
from a fixed tick grid, independently of customer limits and future prices. The
two opponent families have fixed positive-support discrete quote laws; a
Bernoulli mixture parameter chooses a family independently for each RFQ. Public
features are own final quote edge, product, customer side, quantity, and the
already observed previous price increment. The response target is own winning
AND eligibility; losing quotes do not reveal their customer's hidden limit.
One observation is recorded per RFQ. Public equal-price outcomes flag possible
ties, without pretending they identify all eligible competitors.

Constant, pooled edge, monotone edge and logistic fill predictions are compared
on the same final observations. Markout regression uses only past increments,
product and side; its labels are post-match price changes and are never current
features. Independent sessions are the uncertainty unit. Export is disabled
unless both in-family and shifted-family validation have full edge support,
logistic Brier improvement over constant, absolute mean calibration error <=
0.03, ten-bin expected calibration error <= 0.05, and no supported-bin absolute
error above 0.10. Markout export additionally requires lower MSE than zero on
both sets. These are empirical gates, not uniform error certificates.

For the known experimental intervention theta, the opponent-mixture probability
is theta and per-unit exchange fee is 0.01 + 0.01 theta. Theta stays in [0.2,0.8].
The primitive family support remains fixed; the derivative of reward includes
the explicit fee term. Each actor fill is flattened by an actual exchange
transaction at the post-match price. No cash-state path-density score is used.
The forward primitive-history likelihood score is Z/theta-(1-Z)/(1-theta).
The additive, locally independent reward allows the lower-variance local score;
the full path score is also reported. A finite-support tangent/reference sums
the exact discrete competitor and customer-limit law with uniform tie weights.
This reference is checked against Engine outcomes, rather than substituted for
the Engine in data generation.

All estimated response methods receive the same baseline/plus/minus observation
bundle and the same three-Engine-run budget per primitive session. The local
likelihood estimator importance weights all three arms to baseline; the finite
tangent and direct conditional refit pool conditional family outcomes from all
arms. Paired finite differences use the two intervention arms. Each method may
ignore available observations, but none receives extra observations. Paired finite
differences at epsilon=0.05 are a finite intervention contrast, not an exact
derivative. Actual estimator fit and evaluation times and Engine execution
counts are recorded; the full-path score is a baseline-only diagnostic, not an
equal-information competitor. Use of a known law is a stronger-information benchmark,
explicitly distinguished from estimated response. Original Engine tie RNG is
unchanged, so policy-dependent eligible sets can consume a different count of
tie draws; matched price/customer/opponent primitives do not imply identical
endogenous outcomes.

No policy is selected from these validation outcomes. No R3 or runtime file is
enabled by this script. A failed transport gate is retained as counterevidence.

## Numerical audit erratum, 2026-10-16

The frozen generative protocol and raw outcomes remain unchanged. The original
finite-law reference treated hidden-offset eligibility as ideal decimal
arithmetic. The supplied Engine actually adds/subtracts offset limits in binary
float arithmetic, although submitted quotes are validated as exact decimal ticks.
At reference 99.96, customer BUY and limit offset 0.08, the valid tick quote
100.04 fails because the computed customer limit is 100.03999999999999. This
counterexample is retained as a regression fixture, without patching SDK source.

The revised reference conditions on matching reference and side and enumerates
the Engine's actual numerical comparisons. The ideal-decimal reference and its
earlier results remain archived with narrower status. This repairs an analytical
reference assumption; it does not alter any policy, random tape, outcome, export
gate or seed. The exact source used for each analysis is archived separately.
