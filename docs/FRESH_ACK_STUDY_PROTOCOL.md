# Prospective fresh synthetic comparison after quote-state repair

Written 16 October 2026 before evaluating the new policy comparison. The old
demo holdout is consumed and is neither read nor reused by this study. This is
an explicitly constructed research environment using the supplied unchanged
Engine and public extension interfaces, not an organiser round model.

Independent source review corrected the initial protocol before any tuning
or final-path evaluation: the original is preserved as
`FRESH_ACK_STUDY_PROTOCOL_v1.md`. Corrections align power with the materiality
gate, use checked 256-bit primitive seed digests, archive failing cases, and
make the fixed-quote ablation share actual prices and public admissions.

The primary diagnostic is repaired R2 minus a conventional policy selected
from B1/B2/B3/factor B3 using development validation only. The programme's
R3-versus-B3 primary objective is retained separately; R3 enters this study
only if its identification and runtime-validation gates pass before sealing.
Otherwise record it as unexecuted, not as R2 renamed R3.

Each session has 12 intervals, three RFQs per interval, the verified old-price
matching and post-match price jump, and three final-window updates: 124 price
rows. Products, fees, limits, customer-size/limit law and final B=2*A settlement
are the supplied demo specifications. Synthetic prices use a mean-one lognormal
factor increment, plus an independently driven AR(1) intraday B basis. The four
predeclared (factor currency volatility, basis innovation volatility, basis rho)
scenarios are (0.08,0.03,0.75), (0.20,0.06,0.90), (0.35,0.10,0.95), and
(0.12,0.18,0.98). Positive-price validation rejects a bad generated tape;
there is no clipping or reroll. These chosen laws are hypotheses, not claims
about actual market diffusion or stationarity of inventory.

Separate seeded primitive streams drive factor shocks, basis shocks, customer
attributes, opponent edges and ties. Numerical seed namespaces are unique to
this study and each phase/scenario/path; generated histories are hashed and
archived. Opponent draw consumption does not change customer or price tapes.
Original conditional tie consumption is retained and disclosed. Policies see
only public market/RFQ/own-account callbacks.

Development opponents are the original example dealer with edge ranges
[0.03,0.20] and [0.08,0.28]. Held-out opponent families are tighter original
example dealers [0.015,0.09] and a research inventory-skewed example-dealer
extension with [0.03,0.16], skew 0.003 and a half-limit hedge band. No losing
opponent quotes/private inventory are supplied to our policy. Each independent
price/customer tape is replayed against both opponent families; those two
repetitions are averaged before inference.

B1/B2/B3/factor B3/R2 each receive the same 12 prospectively recorded joint
quote/hedge candidates: edge [0.04,0.08,0.12,0.16] times skew [0,0.002,0.005],
with cyclic bands [0.2,0.4,0.6] and targets [0,0.5]. Each candidate is evaluated
on four scenarios times three development paths, alternating the two training
opponent families: 144 sessions per method, 720 tuning sessions. Selection is
mean raw monetary PnL, ties by candidate index. Validation uses 20 fresh paths
per scenario, both training opponent families, equally for every method.
No empirical directional forecast is used.

Freeze selected parameters/source hashes after validation. B0 and a genuinely
fixed-price hedge ablation are untuned diagnostics. The ablation applies the
same B3-selected edge, zero inventory skew and the first public reference of
each RFQ interval to both R2 and B3; these prices remain fixed throughout that
interval. Both ablation policies also share a causal public admission budget:
the cumulative absolute RFQ quantity admitted per product over the entire
session cannot exceed the smaller individual long/short limit. This budget is
spent on offering, regardless of fills, and never replenished. Since both
hedgers only reduce absolute own positions, every admitted subset stays
feasible. This deliberately restrictive hedge-isolation experiment is separate
from the full joint comparison. Offered prices AND withdrawal/admission masks
are checked for exact agreement. Hypothetical future quotes use the model's
current-price anchor, not the current interval's cached historic quote anchor.
Identical quote parameters alone would not fix realised prices when inventory differs. The continuously
repricing B3 remains the strong conventional comparator. Frozen-book and
continuously repricing policies have different economics; conditional
action-value ordering is not advertised as online policy dominance.

Materiality is +1 price-currency unit/session. The superiority claim requires
the lower endpoint of the two-sided 95% interval to exceed +1, approximately
a one-sided alpha .025 test of that materiality boundary. Plan for a design
alternative of +2, hence an effect gap of 1, using the 80 validation path-mean
paired differences. The final number is the next multiple of four above
max(128, ceil(((1.960+0.842)*sd/(2-1))^2)), capped at 4096 paths.
Report insufficient planned precision if the cap binds. This is a pilot rule,
not guaranteed power. Equal scenario weights and two opponent-family repeats
are maintained. Policies and final sample size are sealed BEFORE any final
namespace tape is generated. No post-hoc extension, exclusion or retuning.

Primary inference resamples independent path units within scenario strata,
with both opponent repetitions kept together, and reports a paired bootstrap
95% interval and a one-sided paired mean test under stated finite-variance
assumptions. A superiority gate requires the lower interval above +1. Pilot
power is approximate, not a guarantee for the selected policy or changed
opponent family. All other policy/ablation comparisons are descriptive, with no
unadjusted confirmatory claims. Session losses, fees, penalties, customer
orders/notional, exchange turnover, price-innovation exposure, crashes and
operational fallbacks are retained. No powered tail or live-market claim.

R2 evaluates uncertain current fills only under its explicit common frozen
quote-book and exogenous unconstrained-customer assumptions. Approximation,
transition-estimation and response errors remain separate from quote-state
robustness. The offline schedule has synchronous acknowledgements and logical
time; delayed/rejected/reconnect behaviour is tested independently using the
original Runner with controlled transport and actual WebSocket sessions.
