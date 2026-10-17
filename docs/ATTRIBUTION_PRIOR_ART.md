# Prior art and the constrained attribution question

This is a targeted comparison, not an exhaustive literature review or an
independent audit of every cited proof. A reference to an author manuscript
identifies that work/version; it does not establish novelty of this project.

| Work | Existing contribution relevant to this project | Boundary of our claim |
|---|---|---|
| Evans and Majumdar (2011), [Diffusion with stochastic resetting](https://arxiv.org/abs/1102.2704) | Diffusion interrupted by returns; stationary and first-passage consequences. | The inventory-walker analogy is motivation; it does not imply dealer profits. |
| Dai and Yao (2013), [Brownian inventory control](https://arxiv.org/abs/1110.2831) | Costly upward/downward inventory adjustment and optimal control bands. | Partial targets and transaction costs are established control concepts. |
| Guilbaud and Pham (2013), [Limit and market orders](https://arxiv.org/abs/1106.5040) | Mixed regular/impulse market-making control. | A Bellman/QVI formulation or discrete market intervention is not by itself new. |
| Barzykin, Bergault and Guéant (2022 version), [Dealer markets with hedging and impact](https://arxiv.org/abs/2106.06974v6) | Dealer internalisation, external hedging and inventory-dependent boundaries. | Comparing anticipated offsetting customer flow with costly hedging overlaps directly. |
| Boyce and Neuman (2026), [Competition with internalisation and externalisation](https://arxiv.org/abs/2606.06413) | Competing quote skew and external hedging. | The economic trade-off is not a new resetting mechanism. |
| Barzykin (2026), [Hawkes-driven OTC market making](https://arxiv.org/abs/2608.02002) | RFQ-conditioned continuation approximations and exact lifted-model numerical reference. | Continuation approximation, a strong comparator and paired simulation alone do not establish priority. |
| Barzykin and coauthors, [Optimal quoting under adverse selection and price reading](https://arxiv.org/abs/2508.20225) | Information, adverse-selection effects and quote adjustments. | Local perturbative quote changes already have direct dealer-market precedents. |
| Marín Martínez, Ardanza-Trevijano and Sabio (2026), [Causal interventions in bond platforms](https://doi.org/10.1371/journal.pone.0341369) | RFQ pricing interventions and distinction between predictive and causal effects. | Predictive fills and accounting decompositions do not identify causal mechanisms by themselves. |
| Lalor and Swishchuk (2024; revised 2026), [Market Simulation under Adverse Selection](https://arxiv.org/abs/2409.12721v3) | Simulation sensitivity to fill and adverse-selection assumptions. | Profit in a synthetic exchange does not establish market transport. |

The proposed contribution is narrower: a reproducible, deadline-batch
intervention-attribution benchmark that shares information, parameter packages
and explicit admissible exposure rectangles; preserves all failures; separates
whole-policy selection from controller swaps; and independently reconstructs
cash, quotes, pending actions and the correct price-innovation exposure clock.
It does not claim new hedging theory, a general verification theorem, universal
reset superiority, extra response information, or publication acceptance.
