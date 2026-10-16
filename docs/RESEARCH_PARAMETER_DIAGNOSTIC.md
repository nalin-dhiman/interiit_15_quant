# Post-primary exploratory parameter-package diagnostic

This protocol is written **after the sealed main comparison was opened**. The
main diagonal comparison supports a synthetic full-policy difference, while its
matched-parameter and fixed-price contrasts caution against attributing that
difference to resetting alone. This diagnostic investigates that ambiguity; it
does not revise the primary comparison or supply confirmatory evidence.

Freeze a 2x2 factorial crossing unchanged B4 and R2B controllers with the whole
B4-selected and R2B-selected parameter packages copied exactly from the sealed
pilot. The packages differ jointly in quote edge/skew, forecast weight, band and
risk preference. No new parameter is optimized or selected. In particular the
R2B package has forecast_weight=0.5/risk_multiplier=0, while the B4 package has
forecast_weight=0/risk_multiplier=2. Both controllers see the same public inputs.

Use the unchanged research_study.run_case with a new seed phase
post_primary_factorial: eight declared price/settlement scenarios, 32 independent
paths/scenario, two declared held-out opponent laws and four factorial cells.
There are 2,048 sessions on 256 independent path units. Opponent repetitions
share exogenous price/customer histories and are averaged before inference.
All accounting, live-quote, primitive-stream and source guards remain unchanged.
Archive every raw trade/final RFQ, downside outcome, fee, turnover, participation,
penalty, rejection, crash and price-innovation exposure. Failed cases abort and
remain archived; they are never omitted or replaced.

Report both within-package controller contrasts, both within-controller package
contrasts, the selected-policy diagonal and their interaction. Use stratified
path bootstrap 95% intervals as unadjusted exploratory descriptions. There is
no confirmatory p-value, performance gate, main-policy promotion or sealed-primary
revision. Fresh seeds remove sample reuse; they do not undo the post-primary
choice of question or establish general reset alpha. Capacity/quoting feedback
can differ across controllers even within a shared parameter package.

The diagnostic asks whether the R2B-selected package also strengthens a simple
conventional B4. A positive diagonal alone cannot answer that question. Preserve
all outcomes, including a result that a crossed conventional configuration
eliminates or reverses the apparent reset-specific interpretation.

Run experiments.research_parameter_factorial declare and execute against a new
output directory. Configuration, package/source/input hashes and their original
snapshot are sealed before the new data are generated. No SDK source, policy,
existing main-study source, consumed final outcome or response gate is changed.
