# Prospective boundary-reset study and stop decision

Written on 16 October 2026 before any boundary-policy tuning or new final tape.
The earlier 42,624-session test is consumed. Its negative primary result is
preserved; this is one declared development redesign, not a repeated first claim.

## Gates and research question

The separate predeclared freezing diagnostic changes only the freeze flag in
the prior scalar-B3 bank (edge 0.04, skew 0.005, band 0.6, target 0). Its mechanism
gate passed: normal-minus-freeze PnL 31.84758, 95% path interval [27.99504,35.72923],
normal mean positive and customer spread improved. The bank does not isolate
the previous R2-selected wide edge 0.16 or explain all its participation loss.
It justifies testing one controller that resumes fresh quotes after a boundary
intervention. It does not establish economic reset value or market realism.

The next gates are explicit: fail accounting/causal sequencing and stop policy
comparisons; fail the final materiality gate and retire this boundary R2 version
for the tested environment. A favorable secondary does not rescue a failed
primary. No post-hoc extension, exclusions, retuning or R3 rename is allowed.

Primary: boundary R2 (BR2) minus the strongest conventional control selected on
new validation data, measured by raw cash PnL including fees, penalties and
verified demo retained settlement. Risk penalties and participation are separate.
The programme primary R3-versus-B3 remains closed; no calibrated response model,
RL or DMFT is introduced. Combining resetting and response is not a novelty claim.

## Controller and conventional controls

BR2 suppresses initial new RFQ admission, waits for all previous RFQ results,
own fills and a reconciled account, makes one feasible lot-rounded boundary
intervention, and resumes normal inventory-aware quote replacement after
confirmed execution. The current SDK's new quote callback precedes old fill
delivery; simply calling hedge after a boundary is insufficient. Generic ACKs
are not correlated by invented IDs. Unknown/rejected/reconnect histories fail
conservatively and are counted; capacity never uses an unconfirmed hedge.

The continuation must include the upcoming public RFQ batch with
candidate-dependent quotes, eligible capacity and assumed fill probabilities.
Current cash does not cancel across these future inventory-dependent books.
Future flow/price/markout parameters remain assumed; no unknown-market error
certificate or exact continuous-state theorem is claimed.

Tunable methods: B1 (no hedge), B2 (immediate), B3 (scalar partial band), factor B3,
BB3 (boundary B3) and BR2. BB3 uses BR2's identical sequencing/admission gate with
the conventional band target. The ordinary B3 remains unrestricted by that
boundary delay. These are whole-policy comparisons: their realised inventories,
quotes, admissions and endogenous rival fills may differ. BB3 versus B3 measures
sequence/delay effects; BR2 versus BB3 includes reset/quote-feedback effects.

Provided B0 is an untuned diagnostic. A BR2 variant using BB3-selected quote
parameters is an untuned diagnostic, with independently tuned BR2 hedge
parameters retained. It is not described as identical realised prices or a
fixed-book hedge experiment. Conservative all-subset reservation remains normal;
there is no lifetime admission budget. Unsupported later-round impact/depth
is disabled. Original organiser Engine and SDK source are unchanged.

## Primitive histories and new opponents

Price laws remain the four previously declared synthetic factor/basis scenarios:
(0.08,0.03,0.75),(0.20,0.06,0.90),(0.35,0.10,0.95),(0.12,0.18,0.98).
The positive factor follows mean-one lognormal increments; independently driven
B basis follows the specified AR(1). Invalid tapes raise, without reroll/clipping.
Each session uses the supplied 12 intervals, 3 RFQs/interval, quantities 2/4/10,
fees/individual limits, 124 rows and B=2*A retained settlement. These are research
laws using demo rules, not actual organiser round price dynamics.

Independent full 256-bit SHA primitive seeds use namespace
`boundary-policy-v1/20261016T103800Z`, phase,scenario,path and stream names.
Factor,basis,customers,opponents and policy streams are independent PRNG streams;
all intended seeds including derived customer/tie streams are collision-audited.
Original conditional tie draw consumption is retained; no identical tie outcomes
or inaccessible opponent state are promised. No policy receives future prices,
hidden limits, latent attributes or losing competitor quotes. Price/customer
tape hashes must match paired policies. One final record per RFQ is retained.

Development families are the previous standard/wide original example dealers
with edges [0.03,0.20]/[0.08,0.28]. Final families are two explicitly researched
structural extensions, specified here before their statistical tapes exist:

- Volatility-responsive: two original example dealers, edge range[0.025,0.16],
  with an additive edge min(0.12,0.5*absolute latest public price change) for the
  RFQ product. They immediately hedge using their original public-account rule.
- Sticky update: original example edges [0.03,0.20], with prices refreshed only
  at public price versions whose (version-1) mod3 equals0; first sight of each
  RFQ always quotes. Intermediate calls omit the RFQ, preserving its accepted
  price. They hedge with the original rule. This is a declared cadence family,
  not an inferred competition tactic.

These differ from consumed tight/inventory opponent families. Parameters are
not tuned to favor BR2. New final tapes are unopened until all sources, policies,
parameters, opponent rules, analysis and sample size are sealed.

## Equal nominal budgets and final seal

Each tunable method receives12 distinct joint quote/hedge candidates and 12
development cases (4 scenarios×3 paths, alternating standard/wide family), 144
sessions per method: 864 tuning sessions. Quote grid edge [0.04,0.08,0.12,0.16]
times skew [0,0.002,0.005]; cyclic band [0.2,0.4,0.6], targets [0,0.5] as the prior
grid. B1/B2 still have 12 distinct quote policies. The prior strongest factor-B3
setting (edge 0.08,skew 0.005,band 0.6,target 0.5) must be present and verified before
tuning. All methods use zero empirical directional forecast. Known terminal
settlement basis remains available equally; this is not zero price-value
information. No fitted research coefficients are promoted.

New validation has 80 independent paths (4×20), two development families and six
methods: 960 sessions. Select the conventional fallback among B1/B2/B3/factorB3/
BB3 by mean validation raw PnL, fixed-order tie breaks. Candidate selection uses
mean tuning PnL with index tie breaks. Record real elapsed time: equal trial
counts are not equal CPU, exhaustive tuning or equal effective search difficulty.

Materiality null +1 currency/session; design alternative +2. Use the pooled
sample standard deviation of the 80 validation path-mean paired differences
(opponent repeats averaged). Between-scenario means remain in this pooled SD;
this can conservatively overstate the stratified variance. Compute
ceil(((1.960+0.842)*sd/(2-1))^2), minimum 128, round upward to a multiple of 4,
cap 4096 independent final paths. If the cap binds, report missing planned
precision; pilot power and transfer to new opponents are approximate.

The final comparison has eight methods: six tuned controls plus B0 and the
BB3-quote-parameter BR2 diagnostic, two final opponent repeats per independent
path. Independent inference unit is the price/customer path, stratified equally
across four scenarios. Bootstrap paths within strata, retaining opponent repeats
together; report the 95% interval and approximate one-sided materiality test.
Only the predeclared BR2-minus-fallback comparison receives the primary decision.
Superiority requires its two-sided 95% lower endpoint>+1. Additional differences,
scenario means, quote age, activity, raw downside and exposure are descriptive.
No finite-sample uniform/bootstrap/tail safety claim is made.

Stop on crashes or monetary inconsistency; archive exact failed case and all
source/data/config hashes. Never drop a seed or continue silently after failure.
Unsupported operational states and all abstention/delay counters remain results.
After the fixed final sample, failed primary means NO-GO for this implementation.
A pass supplies conditional synthetic empirical support only, retaining actual
round/production/scoring/later-liquidity certification blockers.

The boundary intervention batch is capped at 500 items, matching one Runner
chunk. Both BB3 and BR2 prioritize normalized individual exposure with stable
product-ID ties; remaining products retain physical inventory, with deferred
items/candidate diagnostics explicitly recorded. This is a computational action
constraint, not an optimal multi-product scheduling result. Quotes resume only
after the actual bounded batch is reconciled.
