# Model and accounting

The controlled walker is dealer inventory. Market prices are an exogenous
process; an exchange intervention changes exposure through a real transaction.

## Signed trades

Let `d` be the dealer's signed purchase quantity, `p` the execution price,
`m` the instrument multiplier and `C` the execution cost:

```
inventory_after = inventory_before + d
cash_after = cash_before - m*p*d - C
```

A customer BUY has `d=-quantity`; a customer SELL has `d=+quantity`.
Customer executions themselves are not clipped to a limit. An intervention is
feasible only if its actual rounded quantity satisfies the exchange rule.
At public mark `s`, wealth is `cash + sum(m*q*s)`; at settlement use the
settlement prices. A risk preference enters policy evaluation separately and
never changes cash PnL. No compulsory terminal liquidation is imposed.

## Live RFQ capacity

Quotes remain live until matching or withdrawal. Replacing a quote changes its
price; omission preserves it. The browser model acknowledges synchronously.
Opposite requests can fill independently: reserve the sum of all possible
dealer purchases and the sum of all possible dealer sales. Every fill subset
must fit the proposed capacity envelope. Do not count an unconfirmed hedge.
The browser controller intervenes between resolved RFQ batches, then quotes
using confirmed inventory.

## Conventional and continuation controls

For current inventory `q` and candidate target `z`, a conventional basis band
balances terminal basis, a quadratic holding preference and execution cost:

```
J(z;q) = m*b*z - lambda*T*z² - C(z-q).
```

Public basis `b` is settlement value implied by current root prices minus the
instrument's current price. It predicts expected terminal gains only under
the declared root-price assumptions. In the browser, B settles to twice A.
The conventional controller may open a basis position from zero.

Continuation control instead compares actual intervention costs plus expected
future customer opportunities and a baseline continuation:

```
Q_baseline(t,q,a) = immediate_reward(t,q,a)
                   + sum_next P(next|t,q,a)*V_baseline(next).
```

No action and the conventional target must be available. Recompute cost and
value after tick and lot rounding. A true baseline-Q improvement implies
policy improvement for an unchanged known sufficient finite model. It does
not imply improvement if the transition or continuation approximation is wrong.
The browser uses its explicitly declared small synthetic response model;
the Python library supplies exact finite models and exhaustive references.

## Conditional response bounds

For a fixed baseline and a twice differentiable finite family, let the
uniform kernel derivative row norms be `L1,L2`, reward bounds `R0,R1,R2`
and terminal bounds `G0,G1,G2`. With remaining horizon `h`, set

```
A0=G0; B0=G1; C0=G2
Ah=R0+A(h-1)
Bh=R1+L1*A(h-1)+B(h-1)
Ch=R2+L2*A(h-1)+2*L1*B(h-1)+C(h-1)
```

Differentiating the finite recursion and using stochastic-row contraction
gives a simultaneous action-value Taylor remainder `delta²*Ch/2`.
A linear-response action gap strictly above twice a valid error bound yields
conditional baseline improvement. These constants must genuinely bound the
whole domain; the API cannot prove caller-supplied premises or domain membership.
The result is in real arithmetic, not a floating-point certificate.

## Brownian diagnostics

For inventory diffusion `dX=sqrt(2D)dW`, independent resets to zero at rate `r`
give a Laplace stationary law and `E[X²]=2D/r`. This `D` is inventory diffusion,
not market-price volatility. Restricted full-reset Poisson and threshold cost
families have a `1/sqrt(6)` optimal-cost ratio only for positive fixed cost and
zero proportional cost. It is not a trading-profit multiplier or a universal
optimality result over partial and singular controls.

## Randomness and observations

Primitive price, customer, opponent, tie and exploration streams are separate.
Only public current features and observed outcomes reach the policy. One final
observation per RFQ is a statistical record; quote-update callbacks are not
additional samples. Matched primitive shocks can produce different fills and
different wealth paths under different policies.

Full-path likelihoods need support and differentiability premises. Changing a
deterministic target or cash coordinate can make them incompatible; common
primitive histories with a parameter-dependent reward are one valid alternative.
Explicit payoff derivatives and continuous-time escape-rate derivatives cannot
be discarded. Cash fees have no automatic thermodynamic entropy interpretation.
