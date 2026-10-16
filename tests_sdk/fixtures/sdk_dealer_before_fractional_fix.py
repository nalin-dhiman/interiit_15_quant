"""Single-file dealer for the supplied October 2026 development SDK.

Default B3 is a conventional inventory band. R2 is a model-dependent finite
scenario policy improvement, not a certified competition-performance claim.
No runtime I/O; fit parameters and settlement coefficients offline.
"""
import math
import time
from collections import deque
from decimal import Decimal

from rfq.sdk import round_lot, round_quote


class Dealer:
    def __init__(self, mode="B3", base_edge=0.08, inventory_skew=0.002,
                 band_fraction=0.35, target_fraction=0.5, risk_weight=0.0002,
                 horizon=3, fill_probability=0.25, future_size=4.0,
                 future_side_buy=0.5, future_orders_per_product=1.5,
                 total_intervals=12, price_ticks_per_interval=10,
                 settlement_coefficients=None, settlement_constants=None,
                 price_drift=None, min_improvement=0.002, forecast_weight=0.0,
                 adverse_selection=None, fill_edge_decay=12.0,
                 observation_limit=2000, diagnostic_limit=100,
                 scenario_count=8, factor_hedge=False, max_r2_products=250,
                 max_current_fill_states=128, freeze_current_quotes=None):
        self.mode = mode if mode in ("B0", "B1", "B2", "B3", "R2") else "B3"
        self.requested_mode = mode
        self.adaptation_enabled = False
        self.base_edge = max(0.0, float(base_edge))
        self.inventory_skew = max(0.0, float(inventory_skew))
        self.band_fraction = min(1.0, max(0.01, float(band_fraction)))
        self.target_fraction = min(0.99, max(0.0, float(target_fraction)))
        self.risk_weight = max(0.0, float(risk_weight))
        self.horizon = min(4, max(1, int(horizon)))
        self.fill_probability = min(1.0, max(0.0, float(fill_probability)))
        self.future_size = max(0.0, float(future_size))
        self.future_side_buy = min(1.0, max(0.0, float(future_side_buy)))
        self.future_orders_per_product = max(0.0, float(future_orders_per_product))
        self.total_intervals = None if total_intervals is None else max(1, int(total_intervals))
        self.price_ticks_per_interval = max(1, int(price_ticks_per_interval))
        self.settlement_coefficients = settlement_coefficients or {}
        self.settlement_constants = settlement_constants or {}
        self.price_drift = price_drift or {}
        self.adverse_selection = adverse_selection or {}
        self.forecast_weight = min(1.0, max(0.0, float(forecast_weight)))
        self.min_improvement = max(0.0, float(min_improvement))
        self.fill_edge_decay = max(0.0, float(fill_edge_decay))
        self.scenario_count = min(16, max(2, int(scenario_count)))
        self.factor_hedge = bool(factor_hedge)
        self.max_r2_products = max(1, min(250, int(max_r2_products)))
        self.max_current_fill_states = max(1, min(512, int(max_current_fill_states)))
        self.freeze_current_quotes = (self.mode == "R2" if freeze_current_quotes is None
                                      else bool(freeze_current_quotes))
        self.observations = deque(maxlen=max(0, min(10000, int(observation_limit))))
        self.last_decisions = deque(maxlen=max(0, min(1000, int(diagnostic_limit))))
        self.callback_errors = 0
        self.ack_rejections = 0
        self.wins = self.losses = self.expired = 0
        # Reservations are pessimistic: any once-offered quote may still be live
        # after a rejected replacement. Withdrawals do not release capacity.
        self._reserved = {}
        self._reserved_by_product = {}
        self._reservation_totals = {}
        self._records = {}
        self._pending_hedges = {}
        self._quote_state_uncertain = False
        # The SDK does not expose the request corresponding to an ACK. No raw
        # ACK, including a duplicate after reconnect, certifies a live quote.
        self._frozen_products = set()
        self.quote_ambiguity_decisions = 0
        self.r2_decisions = 0
        self.r2_terminal_decisions = 0
        self.robust_r2_decisions = 0
        self.robust_r2_improvements = 0
        self.quote_fallback_decisions = 0
        self.fill_state_cap_fallbacks = 0
        self.load_fallback_decisions = 0
        self.repeated_market_callbacks = 0
        self._last_market_interval = None
        self._seen_acks = deque(maxlen=10000)
        self._seen_ack_set = set()
        self._seen_results = deque(maxlen=10000)
        self._seen_result_set = set()
        self._market = {}
        self._positions = {}

    @staticmethod
    def _number(value, default=0.0):
        if isinstance(value, bool):
            return default
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (ValueError, TypeError, OverflowError):
            return default

    def _spec(self, product_id, market):
        products = market.get("products", {})
        prices = market.get("prices", {})
        if product_id not in products or product_id not in prices:
            return None
        product = products[product_id]
        if not isinstance(product, dict):
            return None
        price = self._number(prices[product_id], -1.0)
        tick = self._number(product.get("tick_size"), -1.0)
        lot = self._number(product.get("lot_size"), -1.0)
        multiplier = self._number(product.get("multiplier", 1), -1.0)
        if min(price, tick, lot, multiplier) <= 0:
            return None
        return product, price, tick, lot, multiplier

    @staticmethod
    def _limit(product, key):
        value = product.get(key)
        return math.inf if value is None else max(0.0, float(value))

    def _bounds(self, product):
        return -self._limit(product, "max_short"), self._limit(product, "max_long")

    def _reserved_range(self, product_id, excluding=None):
        negative, positive = self._reservation_totals.get(product_id, (0.0, 0.0))
        record = self._reserved.get(excluding)
        if record is not None and record["product_id"] == product_id:
            if record["side"] == "BUY":
                negative -= record["quantity"]
            else:
                positive -= record["quantity"]
        pending = self._pending_hedges.get(product_id, 0.0)
        return -negative + min(0.0, pending), positive + max(0.0, pending)

    def _reserve(self, key, record):
        self._unreserve(key)
        self._reserved[key] = record
        pid = record["product_id"]
        self._reserved_by_product.setdefault(pid, {})[key] = record
        neg, pos = self._reservation_totals.get(pid, (0.0, 0.0))
        if record["side"] == "BUY":
            neg += record["quantity"]
        else:
            pos += record["quantity"]
        self._reservation_totals[pid] = (neg, pos)

    def _unreserve(self, key):
        record = self._reserved.pop(key, None)
        if record is None:
            return
        pid = record["product_id"]
        bucket = self._reserved_by_product.get(pid, {})
        bucket.pop(key, None)
        neg, pos = self._reservation_totals.get(pid, (0.0, 0.0))
        if record["side"] == "BUY":
            neg -= record["quantity"]
        else:
            pos -= record["quantity"]
        if bucket:
            self._reservation_totals[pid] = (max(0.0, neg), max(0.0, pos))
        else:
            self._reservation_totals.pop(pid, None)
            self._reserved_by_product.pop(pid, None)

    def _basis(self, product_id, market):
        prices = market.get("prices", {})
        if product_id not in self.settlement_coefficients:
            return 0.0
        value = self._number(self.settlement_constants.get(product_id, 0))
        for root, coefficient in self.settlement_coefficients[product_id].items():
            if root not in prices:
                return 0.0
            value += self._number(coefficient) * self._number(prices[root])
        return value - self._number(prices.get(product_id))

    def _edge(self, side, q, price, tick, product_id, market):
        direction = 1.0 if side == "BUY" else -1.0
        skew = 0.0 if self.mode == "B0" else self.inventory_skew
        fair = price + self.forecast_weight * self._basis(product_id, market)
        return round_quote(fair + direction * self.base_edge - skew * q, tick, side)

    def quote(self, orders, market, portfolio):
        quotes = {}
        try:
            if not isinstance(market, dict) or not isinstance(portfolio, dict):
                return {}
            if orders is not None and not isinstance(orders, (list, tuple)):
                return {}
            self._market = market
            self._positions = dict(portfolio.get("positions", {}))
            interval = market.get("interval_id", 0)
            for order in orders or ():
                if not isinstance(order, dict) or not isinstance(order.get("order_id"), str):
                    continue
                oid = order["order_id"]
                quotes[oid] = None
                pid, side = order.get("product_id"), order.get("side")
                spec = self._spec(pid, market)
                qty = self._number(order.get("quantity"), -1)
                if spec is None or side not in ("BUY", "SELL") or qty <= 0 or market.get("closing"):
                    continue
                product, price, tick, lot, multiplier = spec
                q = self._number(self._positions.get(pid, 0))
                key = (interval, oid)
                saved = self._records.get(key)
                if (self.freeze_current_quotes and (interval, pid) in self._frozen_products
                        and saved is not None):
                    # Omission preserves the live quote and avoids sending a
                    # replacement whose inventory-dependent price would break
                    # the fixed-current-book policy comparison. On reconnect
                    # the server may have withdrawn it; absence is still in the
                    # robust fill set. Capacity is never released here.
                    quotes.pop(oid, None)
                    continue
                lower, upper = self._bounds(product)
                neg, pos = self._reserved_range(pid, excluding=key)
                lo = q + neg - (qty if side == "BUY" else 0.0)
                hi = q + pos + (qty if side == "SELL" else 0.0)
                # The no-fill subset is included. An existing breach does not
                # grant permission to quote more exposure in either direction.
                allowed = lo >= lower - 1e-10 and hi <= upper + 1e-10
                # B0 retains the organiser template's absence of reservation;
                # its potential breaches are measured, not suppressed by a toy.
                if self.mode == "B0":
                    allowed = True
                if allowed and len(self._reserved) < 10000:
                    rounded = self._edge(side, q, price, tick, pid, market)
                    if rounded > 0 and math.isfinite(rounded):
                        quotes[oid] = float(rounded)
                        self._reserve(key, {"product_id": pid, "side": side, "quantity": qty})
                record = self._records.setdefault(key, {"order_id": oid, "interval_id": interval,
                                                        "product_id": pid, "side": side, "quantity": qty})
                if record.get("final_requested_quote") != quotes[oid]:
                    record["previous_requested_quote"] = record.get("final_requested_quote")
                possible = record.setdefault("possible_live_quotes", [None])
                if quotes[oid] not in possible:
                    if len(possible) < 33:
                        possible.append(quotes[oid])
                    else:
                        record["quote_history_truncated"] = True
                record.update(final_requested_quote=quotes[oid], reference_price=price,
                              confirmed_inventory=q, price_version=market.get("price_version"),
                              request_state=("withdrawal_unconfirmed" if quotes[oid] is None
                                             else "positive_quote_unconfirmed"),
                              quote_ack_verified=False)
            return quotes
        except Exception:
            self.callback_errors += 1
            # Withdraw every identifiable active order on malformed input.
            return {order["order_id"]: None for order in orders or ()
                    if isinstance(order, dict) and isinstance(order.get("order_id"), str)}

    def _fee(self, delta, spec):
        product, price, tick, lot, multiplier = spec
        return abs(delta) * (self._number(product.get("fee_per_unit"))
                             + price * multiplier * self._number(product.get("fee_rate")))

    def _band_target(self, q, spec):
        product, price, tick, lot, multiplier = spec
        lower, upper = self._bounds(product)
        limit = upper if q >= 0 else -lower
        if not math.isfinite(limit):
            # No documented finite limit: an explicit physical-unit fallback.
            limit = max(lot, self.future_size * 10)
        threshold = max(lot, limit * self.band_fraction)
        if abs(q) < threshold and lower <= q <= upper:
            return q
        target = math.copysign(round_lot(threshold * self.target_fraction, lot), q)
        return q + round_lot(target - q, lot)

    def _feasible_target(self, q, target, pid, spec):
        product, price, tick, lot, multiplier = spec
        lower, upper = self._bounds(product)
        # Revalue after rounding the actual quantity, never a fractional target.
        z = q + round_lot(target - q, lot)
        excess0 = max(0.0, q - upper, lower - q)
        excess1 = max(0.0, z - upper, lower - z)
        if excess1 > excess0 + 1e-10:
            return None
        if self.mode == "B0":
            return z
        neg, pos = self._reserved_range(pid)
        # For a breached account allow a recovery even if the no-fill subset
        # remains breached, provided no possible subset is made worse.
        if (z + neg < min(lower, q + neg) - 1e-10
                or z + pos > max(upper, q + pos) + 1e-10):
            return None
        return z

    def _probability(self, edge):
        # Assumed kernel; train-only exported fits can replace its parameters.
        exponent = min(40.0, max(-40.0, -self.fill_edge_decay * (edge - self.base_edge)))
        return min(1.0, max(0.0, self.fill_probability * math.exp(exponent)))

    def _current_outcomes(self, pid, market, spec):
        # Legacy requested-price quadrature for isolated model diagnostics only.
        # Hedge() never uses this to certify economics of an unacknowledged
        # live book; the robust comparison below supplies deterministic deltas.
        interval = market.get("interval_id", 0)
        price, multiplier = spec[1], spec[4]
        records = [(key, rec) for key, rec in self._reserved_by_product.get(pid, {}).items()
                   if key[0] == interval]
        outcomes = []
        for i in range(self.scenario_count):
            delta = spread = 0.0
            for j, (key, rec) in enumerate(records):
                saved = self._records.get(key, {})
                quote = saved.get("final_requested_quote")
                # A withdrawn/unacknowledged request has unknown live price;
                # reserve its capacity, but don't credit speculative profits.
                if quote is None:
                    continue
                direction = 1 if rec["side"] == "BUY" else -1
                margin = direction * (quote - price)
                uniform = ((i + 0.5) / self.scenario_count + j * 0.6180339887498949) % 1.0
                if uniform < self._probability(margin):
                    qty = rec["quantity"]
                    delta -= direction * qty
                    spread += qty * multiplier * (margin - self._number(self.adverse_selection.get(pid)))
            outcomes.append((1.0 / self.scenario_count, delta, spread))
        return outcomes

    def _current_fill_deltas(self, pid, market):
        """All reachable net customer deltas, without offsetting guarantees.

        Each once-offered unresolved RFQ may fill or fail. A requested
        withdrawal and a later positive replacement retain that same branch.
        Current and unresolved previous-interval records are both included: the
        runner quotes a new interval before delivering the old match's fills.
        Equal net deltas can be merged because the same current quote cashflow
        cancels in candidate-minus-baseline value for a fixed current book.
        """
        records = self._reserved_by_product.get(pid, {})
        if not records:
            return [0.0]
        spec = self._spec(pid, market)
        if spec is None:
            return None
        lot = Decimal(str(spec[3]))
        lot_num, lot_den = lot.as_integer_ratio()
        deltas = {0}
        for rec in records.values():
            qty_num, qty_den = Decimal(str(rec["quantity"])).as_integer_ratio()
            signed, remainder = divmod(qty_num * lot_den, qty_den * lot_num)
            if remainder:
                return None  # Unsupported fractional-lot input is not rounded.
            if rec["side"] == "BUY":
                signed = -signed
            deltas |= {delta + signed for delta in deltas}
            if len(deltas) > self.max_current_fill_states:
                return None
        result = [float(Decimal(delta) * lot) for delta in sorted(deltas)]
        return result if all(math.isfinite(delta) for delta in result) else None

    def _remaining(self, market):
        if market.get("closing"):
            return 0
        if self.total_intervals is None:
            return self.horizon
        return max(0, self.total_intervals - int(market.get("interval_id", 1)) + 1)

    def _continuation_values(self, q, candidates, pid, spec, market,
                             current_outcomes=None, return_outcomes=False):
        """Evaluate an unchanged B3 continuation under an explicit small model.

        One future customer event has no fill / customer BUY / customer SELL.
        Quote margins and capacity depend on inventory; offsetting flow can save
        future hedge fees. Terminal value is settlement less current mark, not
        forced liquidation. Truncation replaces omitted flow by an exposed tail.
        """
        product, price, tick, lot, multiplier = spec
        lower, upper = self._bounds(product)
        remaining = self._remaining(market)
        depth = min(self.horizon, remaining)
        dt = self._number(market.get("interval_seconds"), 5.0)
        tick_dt = dt / self.price_ticks_per_interval
        deadline = self._number(market.get("closes_at"), 0.0)
        current_dt = min(dt, max(0.0, deadline - time.time())) if deadline > 0 else dt
        current_tick_dt = min(tick_dt, current_dt)
        current_fraction = current_dt / dt
        basis = self._basis(pid, market)
        drift = self._number(self.price_drift.get(pid))
        if pid in self.settlement_coefficients:
            settlement_drift = sum(self._number(coefficient) * self._number(self.price_drift.get(root))
                                   for root, coefficient in self.settlement_coefficients[pid].items())
            basis += (max(0, remaining - 1) + current_fraction) * (settlement_drift - drift)
        adverse = self._number(self.adverse_selection.get(pid))
        size = max(lot, round_lot(self.future_size, lot))
        cache = {}

        def breach(z):
            return max(0.0, z - upper, lower - z) * self._number(product.get("breach_penalty_per_unit"))

        def terminal(z):
            # Unmodelled tail has no speculative RFQ edge. This is an explicit
            # approximation and its error is not a simultaneous certificate.
            tail = max(0, remaining - depth)
            return (z * multiplier * (basis + tail * drift) - breach(z)
                    - self.risk_weight * z * z * tail * dt)

        def future(h, z):
            key = (h, round(z / lot, 9))
            if key in cache:
                return cache[key]
            if h == 0:
                return terminal(z)
            target = self._band_target(z, spec)
            cost = self._fee(target - z, spec)
            # Future quotes reserve their three possible outcomes exactly.
            branches = []
            total = 0.0
            for side, weight in (("BUY", self.future_side_buy), ("SELL", 1 - self.future_side_buy)):
                direction = 1 if side == "BUY" else -1
                nxt = target - direction * size
                if lower <= nxt <= upper:
                    quote = self._edge(side, target, price, tick, pid, market)
                    margin = direction * (quote - price)
                    probability = min(0.95, self.future_orders_per_product * self._probability(margin)) * weight
                    reward = size * multiplier * (margin - adverse)
                    branches.append((probability, nxt, reward))
                    total += probability
            branches.append((1 - total, target, 0.0))
            value = -cost - breach(z)
            for probability, nxt, reward in branches:
                value += probability * (reward + multiplier * drift *
                                         (target * (1 - 1 / self.price_ticks_per_interval)
                                          + nxt / self.price_ticks_per_interval)
                                         - self.risk_weight * (target * target * (dt - tick_dt)
                                                               + nxt * nxt * tick_dt)
                                         + future(h - 1, nxt))
            cache[key] = value
            return value

        values = {}
        if remaining == 0:
            # No RFQs in closing window. No forced liquidation at LAST_PRICE.
            # Remaining price risk preference is explicit and separate to PnL.
            for z in candidates:
                values[z] = (-self._fee(z - q, spec) + z * multiplier * basis - breach(z)
                             - self.risk_weight * z * z * dt / self.price_ticks_per_interval)
            return values
        outcomes = (self._current_outcomes(pid, market, spec) if current_outcomes is None
                    else current_outcomes)
        rows = [{} for _ in outcomes]
        for z in candidates:
            value = -self._fee(z - q, spec) - breach(z)
            for index, (probability, delta, spread) in enumerate(outcomes):
                nxt = z + delta
                outcome_value = (spread + multiplier * drift *
                                         (z * current_fraction + delta / self.price_ticks_per_interval)
                                         - self.risk_weight * (z * z * (current_dt - current_tick_dt)
                                                               + nxt * nxt * current_tick_dt)
                                         + future(depth - 1, nxt))
                rows[index][z] = -self._fee(z - q, spec) - breach(z) + outcome_value
                value += probability * outcome_value
            values[z] = value
        return rows if return_outcomes else values

    def hedge(self, market, portfolio):
        trades = []
        try:
            if not isinstance(market, dict) or not isinstance(portfolio, dict):
                return []
            self._market = market
            self._positions = dict(portfolio.get("positions", {}))
            self._pending_hedges.clear()  # SDK never invokes hedge with a batch in flight
            if self.mode == "B1":
                return []
            load_fallback = len(market.get("products", {})) > self.max_r2_products
            for pid, raw in self._positions.items():
                q = self._number(raw)
                spec = self._spec(pid, market)
                if spec is None or q == 0 or not spec[0].get("tradable_against_exchange", False):
                    continue
                baseline = 0.0 if self.mode in ("B0", "B2") else self._band_target(q, spec)
                if market.get("closing") and self.mode not in ("B0", "B2"):
                    baseline = q
                feasible = self._feasible_target(q, baseline, pid, spec)
                baseline = q if feasible is None else feasible
                selected, gap = baseline, 0.0
                values = {}
                live_records = self._reserved_by_product.get(pid, {})
                uncertain_quotes = bool(live_records)
                fill_deltas = self._current_fill_deltas(pid, market)
                cap_fallback = fill_deltas is None
                feedback_fallback = uncertain_quotes and not self.freeze_current_quotes
                closing_book_fallback = bool(market.get("closing")) and uncertain_quotes
                quote_fallback = (self._quote_state_uncertain or cap_fallback
                                  or feedback_fallback or closing_book_fallback)
                advantages = {}
                robust = False
                if uncertain_quotes and self.freeze_current_quotes:
                    self._frozen_products.add((market.get("interval_id", 0), pid))
                if self.mode == "R2" and uncertain_quotes:
                    self.quote_ambiguity_decisions += 1
                if self.mode == "R2":
                    self.r2_decisions += 1
                    self.load_fallback_decisions += int(load_fallback and not market.get("closing"))
                if self.mode == "R2" and quote_fallback:
                    self.quote_fallback_decisions += 1
                    self.fill_state_cap_fallbacks += int(cap_fallback)
                if ((self.mode == "R2" and not quote_fallback and not load_fallback)
                        or (market.get("closing") and not closing_book_fallback
                            and self.mode in ("B3", "R2") and not self.factor_hedge)):
                    raw_targets = [q, baseline, 0.0, q * 0.25, q * 0.5, q * 0.75]
                    candidates = []
                    for target in raw_targets:
                        z = self._feasible_target(q, target, pid, spec)
                        if z is not None and z not in candidates:
                            candidates.append(z)
                    if not candidates:
                        continue
                    if self.mode == "R2" and not market.get("closing"):
                        # Quote spread cancels pathwise between candidate and
                        # B3 under a fixed current book and the supplied demo's
                        # exogenous, inventory-independent customer matching.
                        # Every feasible live/absent/rejected-quote history is
                        # covered by its net fill delta. Future continuation is
                        # still an assumed finite model, not a market bound.
                        outcomes = [(1.0, delta, 0.0) for delta in fill_deltas]
                        rows = self._continuation_values(q, candidates, pid, spec, market,
                                                         current_outcomes=outcomes, return_outcomes=True)
                        advantages = {z: min(row[z] - row[baseline] for row in rows)
                                      for z in candidates}
                        values = {z: min(row[z] for row in rows) for z in candidates}
                        selected = baseline
                        best = max(advantages, key=advantages.get)
                        gap = advantages[best]
                        robust = True
                        self.robust_r2_decisions += 1
                    else:
                        values = self._continuation_values(q, candidates, pid, spec, market)
                        self.r2_terminal_decisions += int(self.mode == "R2" and bool(market.get("closing")))
                        # Baseline wins ties, then hysteresis suppresses flips.
                        selected = baseline if baseline in values else q
                        if selected not in values:
                            selected = candidates[0]
                        best = max(values, key=values.get)
                        gap = values[best] - values[selected]
                    if gap > self.min_improvement:
                        selected = best
                        self.robust_r2_improvements += int(robust)
                delta = round_lot(selected - q, spec[3])
                self.last_decisions.append({"product_id": pid, "inventory": q, "target": q + delta,
                                            "baseline_target": baseline, "value_gap": gap,
                                            "candidate_values": values, "mode": self.mode,
                                            "interval_id": market.get("interval_id"),
                                            "closing": bool(market.get("closing")),
                                            "quote_state_fallback": quote_fallback,
                                            "quote_state_ambiguous": uncertain_quotes,
                                            "current_book_frozen": (market.get("interval_id", 0), pid)
                                                in self._frozen_products,
                                            "robust_current_fill_comparison": robust,
                                            "current_fill_states": None if cap_fallback else len(fill_deltas),
                                            "candidate_advantage_lower_bounds": advantages,
                                            "fill_state_cap_fallback": cap_fallback,
                                            "quote_feedback_fallback": feedback_fallback,
                                            "closing_book_fallback": closing_book_fallback,
                                            "load_fallback": self.mode == "R2" and load_fallback
                                                and not bool(market.get("closing")),
                                            "error_bound_certified": False})
                if delta:
                    trades.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
                    self._pending_hedges[pid] = delta
            if (self.factor_hedge and self.mode == "B3"
                    and (not market.get("closing") or not self._reserved)):
                # A separate stronger conventional comparator: published exact
                # settlement factors, individual limits, and band hedges in
                # tradable roots. Intraday basis risk remains unmodelled here.
                trades = self._factor_band(trades, market)
            return trades
        except Exception:
            self.callback_errors += 1
            self._pending_hedges.clear()
            return []

    def _factor_band(self, trades, market):
        if market.get("closing"):
            return self._factor_closing(market)
        changes = {pid: (qty if side == "BUY" else -qty) for pid, side, qty in trades}
        positions = {pid: self._number(q) + changes.get(pid, 0.0) for pid, q in self._positions.items()}
        factors = {}
        for pid, q in positions.items():
            spec = self._spec(pid, market)
            if spec is None:
                continue
            for root, coefficient in self.settlement_coefficients.get(pid, {pid: 1.0}).items():
                factors[root] = factors.get(root, 0.0) + q * spec[4] * self._number(coefficient)
        # Pending deltas in _feasible_target are our own candidate trades here,
        # not an earlier in-flight batch. Test the final combined root quantity.
        self._pending_hedges.clear()
        for root, exposure in factors.items():
            spec = self._spec(root, market)
            if spec is None or not spec[0].get("tradable_against_exchange", False):
                continue
            equivalent = exposure / spec[4]
            desired = self._band_target(equivalent, spec)
            delta = round_lot(desired - equivalent, spec[3])
            q = self._number(self._positions.get(root))
            target = self._feasible_target(q, positions.get(root, q) + delta, root, spec)
            if target is None:
                continue
            changes[root] = round_lot(target - q, spec[3])
        result = []
        for pid, delta in changes.items():
            if delta:
                result.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
                self._pending_hedges[pid] = delta
        return result

    def _factor_closing(self, market):
        # Greedy finite candidates for a conventional factor comparator, not a
        # proof of global optimality. No-action is the incumbent at settlement.
        positions = {pid: self._number(q) for pid, q in self._positions.items()}
        factors = {}
        for pid, q in positions.items():
            spec = self._spec(pid, market)
            if spec:
                for root, coefficient in self.settlement_coefficients.get(pid, {pid: 1.0}).items():
                    factors[root] = factors.get(root, 0.0) + q * spec[4] * self._number(coefficient)
        self._pending_hedges.clear()
        dt = self._number(market.get("interval_seconds"), 5.0) / self.price_ticks_per_interval
        for pid in sorted(set(positions) | set(factors)):
            spec = self._spec(pid, market)
            if spec is None or not spec[0].get("tradable_against_exchange", False):
                continue
            q = positions.get(pid, 0.0)
            coefficients = self.settlement_coefficients.get(pid, {pid: 1.0})
            candidates = [q, 0.0, 0.25 * q, 0.5 * q, 0.75 * q]
            if pid not in self.settlement_coefficients:
                candidates.append(q - factors.get(pid, 0.0) / spec[4])
            best, best_gain = q, self.min_improvement
            lower, upper = self._bounds(spec[0])
            penalty_rate = self._number(spec[0].get("breach_penalty_per_unit"))
            for target in candidates:
                z = self._feasible_target(q, target, pid, spec)
                if z is None:
                    continue
                delta = z - q
                gain = delta * spec[4] * self._basis(pid, market) - self._fee(delta, spec)
                gain -= (max(0, z - upper, lower - z) - max(0, q - upper, lower - q)) * penalty_rate
                for root, coefficient in coefficients.items():
                    f = factors.get(root, 0.0)
                    change = delta * spec[4] * self._number(coefficient)
                    gain -= self.risk_weight * dt * ((f + change) ** 2 - f ** 2)
                if gain > best_gain:
                    best, best_gain = z, gain
            if best != q:
                for root, coefficient in coefficients.items():
                    factors[root] = factors.get(root, 0.0) + (best - q) * spec[4] * self._number(coefficient)
                positions[pid] = best
        result = []
        for pid, z in positions.items():
            delta = z - self._number(self._positions.get(pid))
            if delta:
                result.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
                self._pending_hedges[pid] = delta
        return result

    def on_market(self, market, portfolio, exchange=None):
        if isinstance(market, dict):
            self._market = market
            interval = market.get("interval_id")
            if interval == self._last_market_interval:
                # This can be a normal startup market, a snapshot, or reconnect.
                # There is no public reconnect flag. Server withdrawal means
                # absence remains possible; no ACK can resurrect certainty.
                self.repeated_market_callbacks += 1
                for record in self._records.values():
                    record["transport_reset_possible"] = True
            self._last_market_interval = interval
            retained = {(key[0], rec["product_id"]) for key, rec in self._reserved.items()}
            self._frozen_products.intersection_update(retained)
        if isinstance(portfolio, dict) and isinstance(portfolio.get("positions"), dict):
            self._positions = dict(portfolio.get("positions", {}))

    def on_fill(self, fill, portfolio, exchange=None):
        if not isinstance(fill, dict):
            return
        if isinstance(portfolio, dict) and isinstance(portfolio.get("positions"), dict):
            self._positions = dict(portfolio.get("positions", {}))
        pid = fill.get("product_id")
        if fill.get("source") == "customer":
            key = (fill.get("interval_id"), fill.get("order_id"))
            self._unreserve(key)
            if key in self._records:
                self._records[key]["actual_fill_price"] = fill.get("price")
        elif fill.get("source") == "exchange":
            self._pending_hedges.pop(pid, None)

    def on_customer_result(self, result):
        if not isinstance(result, dict):
            return
        if not isinstance(result.get("order_id"), str) or not isinstance(result.get("interval_id"), int):
            return
        key = (result.get("interval_id"), result.get("order_id"))
        if key in self._seen_result_set:
            return
        if len(self._seen_results) == self._seen_results.maxlen:
            self._seen_result_set.discard(self._seen_results[0])
        self._seen_results.append(key)
        self._seen_result_set.add(key)
        self._unreserve(key)
        if not any(old_key[0] == key[0] for old_key in
                   self._reserved_by_product.get(result.get("product_id"), {})):
            # product_id is optional on malformed/manual result fixtures; the
            # bounded market-time cleanup still handles that case.
            self._frozen_products.discard((key[0], result.get("product_id")))
        if not self._reserved:
            self._quote_state_uncertain = False
        observation = self._records.pop(key, {"interval_id": key[0], "order_id": key[1],
                                               "quote_ack_verified": False})
        observation.update(status=result.get("status"), winner=result.get("winner"),
                           winning_price=result.get("price"), traded=bool(result.get("traded")))
        self.observations.append(observation)
        if result.get("status") == "expired":
            self.expired += 1
        elif result.get("traded"):
            self.wins += 1
        else:
            self.losses += 1

    def on_ack(self, ack):
        if isinstance(ack, dict) and isinstance(ack.get("action_id"), str):
            action_id = ack["action_id"]
            if action_id in self._seen_ack_set:
                return
            if len(self._seen_acks) == self._seen_acks.maxlen:
                self._seen_ack_set.discard(self._seen_acks[0])
            self._seen_acks.append(action_id)
            self._seen_ack_set.add(action_id)
        if isinstance(ack, dict) and (ack.get("status") == "rejected" or ack.get("errors")):
            self.ack_rejections += 1
            if self._reserved:
                self._quote_state_uncertain = True
        # ACK contains no original batched quote mapping. Do not infer acceptance
        # of a particular replacement or release its all-subset reservation.
        if isinstance(ack, dict) and ack.get("status") == "rejected" and ack.get("code") in (
                "STALE_PRICE", "POSITION_LIMIT", "LIMIT", "NOT_TRADABLE"):
            self._pending_hedges.clear()
