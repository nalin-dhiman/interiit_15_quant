# A common admissible exposure envelope

The new benchmark constrains both controllers to the same fixed private
inventory rectangle. Public exchange limits, settlement and fees stay intact.
This is an ex ante policy constraint, not a variance estimate or cash penalty.

For product i the private interval is the intersection of the public limit
and [-c_i,c_i]. If independently possible signed customer/pending exchange
changes are d_ij, all realised fill subsets lie in the coordinate box

    lower_i = q_i + sum_j min(0, d_ij)
    upper_i = q_i + sum_j max(0, d_ij).

Customer BUY gives negative dealer quantity; exchange BUY gives positive
quantity. Opposing requests cannot be cancelled as guaranteed fills. Because
each interval is convex, partial execution quantities between zero and an
order quantity also fit these extrema. This does not claim the supplied
exchange has a depth/partial-quantity model.

For positive multipliers m_i and fixed weights w_i, the worst gross exposure
of this box is sum_i w_i*m_i*max(abs(lower_i),abs(upper_i)). In this independent
subset model each coordinate extreme can be attained; the weighted absolute
value reaches its maximum at a box corner. Requiring every coordinate box
inside its private interval therefore bounds every fill subset and every
non-execution/execution subset of the proposed hedge batch.

In the unit-multiplier two-product study, w_A=1,w_B=2. Caps10/5 imply gross<=20;
caps20/10 imply gross<=40. This fixed allocation is stricter than allowing any
allocation inside a gross-exposure ball. It is not an optimal portfolio-risk
allocation or a complete covariance model.

Private bounds feed existing quote admission, lot-rounded candidate targets
and modeled future admissibility before costs/values are evaluated. The
adapter preserves the original inherited hedge function and its virtual SDK
clock. It never assigns holdings directly or edits customer fills.

During pending exchange acknowledgements the audit uses the original account
and every proposed leg's execution/no-execution subset. It does not add the
entire pending request to an aggregate account that may already include those
same executions. After partial confirmation this can remain conservative.
No fresh customer capacity is credited before reconciliation and confirmation.

An unexpected private-cap breach is recorded, retains actual inventory/cash,
and disables new controller decisions. It does not alter the Engine's official
penalty rule. The final benchmark halts inference on such a failure rather
than deleting that path. All callbacks, reservations and pending items are
archived for a separate raw-record audit.

Two admissible policies can carry zero and nonzero positions under the same
cap; their price-risk variance then differs. Thus common admissible rectangles
do not establish equal realised variance, drawdown, trading activity or utility.
The benchmark records actual positions at each price innovation and reports
these quantities as outcomes, without post-treatment matching or exclusions.
