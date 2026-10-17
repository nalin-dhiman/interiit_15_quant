"""Shared private inventory-envelope adapter for scalar B4 and R2B research.

Public SDK specifications, customer executions, cash and fee rules are untouched.
The rectangle is enforced before quotation and candidate valuation. The inherited
hedge method is intentionally preserved so its original logical-clock binding
continues to govern the continuation model. Unexpected envelope breaches are
recorded and disable new decisions; they are never clipped or charged as money.
"""
from __future__ import annotations

from collections import deque

from resetmm.basis_baseline import BasisBandDealer, BasisResetDealer
from resetmm.risk_envelope import InventoryEnvelope, possible_inventory_box


class CommonInventoryEnvelopeMixin:
    def __init__(self, inventory_caps, risk_weights=None, risk_trace_limit=10000, **kwargs):
        if kwargs.get('factor_risk', False):
            raise ValueError("The matched envelope adapter supports scalar B4/R2B only")
        weights = risk_weights if risk_weights is not None else {
            pid: 2. if pid == 'B' else 1. for pid in inventory_caps}
        self.inventory_envelope = InventoryEnvelope(inventory_caps, weights)
        if not isinstance(risk_trace_limit, int) or not 0 <= risk_trace_limit <= 100000:
            raise ValueError("Trace limit must be an integer between zero and 100000")
        self._risk_trace = deque(maxlen=risk_trace_limit)
        self._risk_stats = dict(checks=0, actual_breaches=0, possible_breaches=0,
                                max_actual_gross=0., max_possible_gross=0., trace_dropped=0)
        self._risk_pending_origin = None
        self._risk_pending_items = ()
        super().__init__(**kwargs)

    @property
    def risk_budget_counters(self):
        return dict(self._risk_stats)

    @property
    def risk_budget_trace(self):
        """JSON-serializable records; the bounded internal deque stays private."""
        return list(self._risk_trace)

    def _risk_product_id(self, product):
        for pid, public in self._market.get('products', {}).items():
            if public is product:
                return pid
        pid = product.get('product_id')
        if pid in self.inventory_envelope.caps:
            return pid
        raise ValueError("A public product has no declared private inventory cap")

    def _bounds(self, product):
        lower, upper = super()._bounds(product)
        return self.inventory_envelope.bounds(self._risk_product_id(product), lower, upper)

    def _risk_audit(self, event, portfolio=None, fail=True):
        account = portfolio if isinstance(portfolio, dict) else self._boundary_account
        positions = dict((account or {}).get('positions', self._positions))
        products = self._market.get('products', {})
        multipliers = {pid: float(product.get('multiplier', 1.)) for pid, product in products.items()}
        bounds = {pid: self._bounds(product) for pid, product in products.items()}
        if self._risk_pending_items:
            # The initial account plus every execution/no-execution subset is
            # valid even when callbacks carry an aggregate partially accepted
            # account. Adding pending quantities to that aggregate would count
            # the same executions twice.
            origin = self._risk_pending_origin
            possible = possible_inventory_box(origin, exchange_items=self._risk_pending_items)
        else:
            possible = possible_inventory_box(positions, self._reserved.values())
        within = self.inventory_envelope.within(positions, bounds)
        possible_ok = all(pid in bounds and lo >= bounds[pid][0] - 1e-9
                          and hi <= bounds[pid][1] + 1e-9
                          for pid, (lo, hi) in possible.items())
        actual_gross = self.inventory_envelope.gross(positions, multipliers)
        possible_gross = self.inventory_envelope.worst_gross(possible, multipliers)
        limit = self.inventory_envelope.gross_limit(multipliers, bounds)
        self._risk_stats['checks'] += 1
        self._risk_stats['actual_breaches'] += int(not within)
        self._risk_stats['possible_breaches'] += int(not possible_ok)
        self._risk_stats['max_actual_gross'] = max(self._risk_stats['max_actual_gross'], actual_gross)
        self._risk_stats['max_possible_gross'] = max(self._risk_stats['max_possible_gross'], possible_gross)
        if len(self._risk_trace) == self._risk_trace.maxlen:
            self._risk_stats['trace_dropped'] += 1
        self._risk_trace.append(dict(event=event,
            interval_id=self._market.get('interval_id'), price_version=self._market.get('price_version'),
            positions=positions, possible_box=possible, policy_bounds=bounds,
            gross=actual_gross, possible_gross=possible_gross, gross_limit=limit,
            within_envelope=within, all_subsets_within_envelope=possible_ok,
            pending_exchange_items=len(self._risk_pending_items),
            reserved_customer_orders=len(self._reserved),
            reserved_orders=[dict(record, interval_id=key[0], order_id=key[1])
                             for key, record in self._reserved.items()],
            pending_origin=dict(self._risk_pending_origin) if self._risk_pending_items else None,
            pending_items=list(self._risk_pending_items), multipliers=multipliers,
            risk_weights=dict(self.inventory_envelope.weights),
            risk_is_variance_matched=False, cash_penalty_imposed=False))
        if fail and (not within or not possible_ok):
            self._boundary_fail('private common inventory envelope exceeded; observed inventory retained')
        return within and possible_ok

    def on_market(self, market, portfolio, exchange=None):
        result = super().on_market(market, portfolio, exchange)
        self._risk_audit('market_before_fill_delivery', portfolio)
        return result

    def quote(self, orders, market, portfolio):
        self._market = market
        if not self._risk_audit('before_quote', portfolio):
            return {order['order_id']: None for order in orders or ()}
        quotes = super().quote(orders, market, portfolio)
        self._risk_audit('after_quote', portfolio)
        return quotes

    def _boundary_ready(self, market, portfolio):
        self._market = market
        if not self._risk_audit('before_boundary', portfolio):
            return False
        return super()._boundary_ready(market, portfolio)

    def _boundary_choose(self, market, portfolio):
        self._market = market
        if not self._risk_audit('before_candidates', portfolio):
            return []
        self._risk_pending_origin = dict(portfolio['positions'])
        trades = super()._boundary_choose(market, portfolio)
        self._risk_pending_items = tuple(trades)
        if not self._risk_audit('proposed_exchange_subsets', portfolio):
            raise ArithmeticError('Valued exchange candidates violate the shared inventory envelope')
        if not trades:
            self._risk_pending_origin = None
        return trades

    def on_fill(self, fill, portfolio, exchange=None):
        result = super().on_fill(fill, portfolio, exchange)
        self._risk_audit('observed_' + str(fill.get('source')) + '_fill', portfolio)
        return result

    def on_customer_result(self, result):
        value = super().on_customer_result(result)
        self._risk_audit('customer_result')
        return value

    def on_ack(self, ack):
        result = super().on_ack(ack)
        if self._boundary_pending is None:
            self._risk_pending_origin, self._risk_pending_items = None, ()
        self._risk_audit('acknowledgement')
        return result


class CommonRiskBandDealer(CommonInventoryEnvelopeMixin, BasisBandDealer):
    """Conventional scalar B4 inside the predeclared private rectangle."""


class CommonRiskResetDealer(CommonInventoryEnvelopeMixin, BasisResetDealer):
    """Scalar R2B with the same current and modeled-future private rectangle."""
