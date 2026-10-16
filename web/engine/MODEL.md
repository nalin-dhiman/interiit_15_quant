# Browser model and controller

The browser runs a small, reproducible research simulator. It does not connect to a broker, place external trades, or reproduce the archived competition-SDK experiments. Every displayed result belongs to the synthetic model below.

## Prices and settlement

There are two products with unit multipliers, unit lots and a tick of 0.01. Write their public prices as \(P_A\) and \(P_B\), and their public basis as

\[
b=2P_A-P_B.
\]

Product A follows an arithmetic Gaussian random walk. The basis follows an autoregression with persistence \(\phi\) and innovation standard deviation \(\sigma_b\) per complete batch. A batch has three market observations. Its microstep persistence is \(\rho=\phi^{1/3}\); the basis innovation variance is

\[
\sigma_{b,\mathrm{micro}}^2
=\sigma_b^2\frac{1-\rho^2}{1-\phi^2},
\]

with its continuous limit \(\sigma_b^2/3\) at \(\phi=1\). This construction gives the stated full-batch autoregression before conditioning on trading decisions. Product B is always defined by \(P_B=2P_A-b\). Prices are exogenous: a policy cannot move them. Nonpositive arithmetic price paths fail visibly; they are not clipped or silently excluded.

At the last observation, contractual values are \(S_A=P_A\) and \(S_B=2P_A\). Terminal inventory is retained in the display. It is valued at settlement without inserting a forced exchange liquidation.

## Customer orders, quotes and matching

Each RFQ independently chooses a product and a customer side with equal probability. Quantity is uniform on the distinct values in \(\{1,\min(2,Q),Q\}\), where \(Q\) is the RFQ-size control.

At arrival, a private customer limit offset is drawn uniformly from \([0.04,0.30]\). A customer BUY has limit \(P+u\); a customer SELL has limit \(P-u\), using the arrival reference price. The limit remains fixed while the public market changes. The policy receives product, customer side, quantity, arrival reference and deadline. It never receives the private limit or future price shocks.

The dealer refreshes its quotes at all three market observations. A quote is acknowledged immediately. Only the latest acknowledged quote at the deadline participates. Explicit `null` withdraws a quote; an omitted identifier preserves its previous acknowledged quote. This model has no asynchronous acknowledgement uncertainty.

An opponent quotes an outward-rounded edge sampled uniformly from \([0.03,0.28]\) around the current public price. A dealer fill requires both customer eligibility and a winning competitive quote. Exact ties receive one-half allocation through a separate primitive random variable. A losing quote does not disclose the customer limit.

The quote rule is

\[
\pi_{\pm}=\operatorname{round}_{\pm}
\left(P+w b_i\pm e-\kappa q_i\right),
\]

where \(+\) denotes customer BUY, \(-\) customer SELL, \(e\) is the edge, \(\kappa\) the inventory skew, \(w\) the forecast weight, \(b_A=0\), and \(b_B=b\). B3 uses \(w b_i=0\). B4 and R2B use the same public basis and quote rule.

## Accounting and capacity

A customer BUY of size \(v\) gives the dealer inventory change \(\Delta q=-v\); a customer SELL gives \(\Delta q=+v\). Every transaction records

\[
\Delta C=-m\,\Delta q\,\pi-f,
\]

where \(m\) is the multiplier and \(f\) the cash fee. The browser products have \(m=1\), although the accounting primitive is multiplier-aware. Customer fills are never clipped or rejected after matching because of a position limit.

Before admitting quotes, the engine reserves every possible same-product fill subset. If customer BUY quotes have total size \(V_+\) and customer SELL quotes total size \(V_-\), admission requires

\[
q-V_+\geq L,\qquad q+V_-\leq U.
\]

Opposite-side quotes do not create guaranteed offsetting capacity. Quotes resolve together at their deadline. Exchange interventions occur only after the previous batch has completely resolved and while no quotes are live. The exchange transaction confirms before new quote capacity is admitted.

Exchange quantities are lot-rounded. Execution prices include directional slippage and outward tick rounding. A proportional cash fee and optional fixed intervention fee are charged. An exchange action cannot create a new breach or increase an existing breach. A crossing trade is tested by the same excess rule: crossing into an opposite-side breach is allowed when its excess does not exceed the original excess.

The ledger is independently reconstructed from all transaction records after each completed batch. Displayed PnL is

\[
\mathrm{PnL}_T=C_T+q_{A,T}S_A+q_{B,T}S_B-W_0.
\]

Quadratic inventory preferences are reported separately. Fees, negative sessions and retained losses remain in PnL. Drawdown is sampled at every observed market-price, trade-booking and terminal-settlement event, including events inside a batch. It is not a continuous-path worst-case measure between unobserved events.

## Conventional B3 and B4

B4 enumerates feasible lot-grid targets \(z\) and maximizes the frozen-position approximation

\[
J(z)=z b_i-\lambda H z^2-K(q,z),
\]

where \(H\) is the remaining number of batches and \(K\) is the actual rounded exchange cost relative to the public mark. This produces a conventional cost-aware no-trade band. B3 uses the same objective with \(b_i=0\).

The target-fraction control moves a chosen fraction toward the full conventional target. The resulting transaction is lot-rounded and its value is recomputed. A move is retained only when its modeled gain exceeds the displayed comparison margin. This margin is a preference against tiny changes; it is not a confidence bound.

## Continuation-aware R2B

R2B compares the B4 action, no intervention, zero inventory, partial inventory reductions, and partial moves toward the B4 target. Each candidate pays its actual rounded exchange cost. It then evaluates customer flow and baseline continuation.

The finite future-batch kernel integrates product, side, size, joint eligibility and competition, and conservative reservations. Every admitted quote reserves capacity until the modeled deadline even in branches where it does not fill. States retain both probability and probability-weighted customer spread. Merging branches therefore does not discard realized cash rewards.

For a candidate \(z\), a schematic recursion is

\[
Q_H(q,z;b)=-K(q,z)-\lambda z^2
+\sum_j\left[
R_j+p_j\left(q_j(b-b')+V^{\mathrm{B4}}_{H-1}(q_j,b')\right)
\right],
\]

Risk is charged on the held target \(z\), before deadline customer fills. Here \(b'=\phi b\) for B and zero for A. At the last batch, the continuation is the exact retained-inventory settlement residual \(q_j b\). The finite lookahead has depth one to three. Beyond that depth, the tail is the conventional frozen-position objective after a conventional intervention.

This approximation includes future offsetting customer flow, quote capacity, time remaining, fees, partial targets and terminal settlement. It freezes public prices within each hypothetical future batch and replaces later basis uncertainty by its conditional mean. The actual browser session continues to update prices within batches. Thus R2B values are approximate and can select inferior realized trades. A positive modeled advantage is not a mathematical improvement certificate. Response adaptation remains disabled.

## Reproducibility and comparisons

Prices, customer attributes, opponent quotes and tie allocation use distinct deterministic streams, with separate keys for each logical event. Policy draw counts cannot alter another policy's exogenous price path. Changing the number of RFQs cannot alter the primitive price tape. Policies receive only causally available public observations.

Batch comparisons reuse the same primitive seed for each policy. Their independent unit is the seed, not the number of policy rows. The interface reports paired seed-level differences against B4. Its intervals are descriptive normal approximations; small samples, repeated experimentation and tail behavior require additional analysis. Browser controls are exploratory, not a sealed confirmatory experiment.

The worker yields within sessions and supports cancellation. Explicit limits are 120 batches, six RFQs per batch, and 200 comparison seeds. Unsupported controls raise visible errors rather than silently changing the requested experiment.
