"""Single-file opt-in boundary experiment for the October development SDK.

Default B3 is a conventional inventory band. Boundary R2 uses a declared finite
continuation model, not a certified competition-performance improvement.
No runtime I/O; fit parameters and settlement coefficients offline.
"""
import math
import time
from collections import deque
from decimal import Decimal

from rfq.sdk import round_lot, round_quote


class _CoreDealer:
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

    @staticmethod
    def _quantity_for_target(q, target, lot):
        """Recover the actual lot quantity of an already constructed target.

        Float subtraction can move q + one valid lot just below that same lot.
        Recover a lot quantity only when its actual float inventory update
        reconstructs the valued target exactly. Fresh non-grid targets retain
        the organiser SDK's toward-zero rule. The integer-lot domain retains
        its original arithmetic path.
        """
        raw = target - q
        if (float(lot).is_integer() and float(q).is_integer()
                and float(target).is_integer()):
            return round_lot(raw, lot)
        lots = int(round(raw / lot))
        quantity = float(Decimal(lots) * Decimal(str(lot)))
        if q + quantity == target:
            return quantity
        return round_lot(raw, lot)

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
        return q + self._quantity_for_target(q, target, lot)

    def _feasible_target(self, q, target, pid, spec):
        product, price, tick, lot, multiplier = spec
        lower, upper = self._bounds(product)
        # Revalue after rounding the actual quantity, never a fractional target.
        z = q + self._quantity_for_target(q, target, lot)
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
            cost = self._fee(self._quantity_for_target(z, target, lot), spec)
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
                values[z] = (-self._fee(self._quantity_for_target(q, z, lot), spec) + z * multiplier * basis - breach(z)
                             - self.risk_weight * z * z * dt / self.price_ticks_per_interval)
            return values
        outcomes = (self._current_outcomes(pid, market, spec) if current_outcomes is None
                    else current_outcomes)
        rows = [{} for _ in outcomes]
        for z in candidates:
            value = -self._fee(self._quantity_for_target(q, z, lot), spec) - breach(z)
            for index, (probability, delta, spread) in enumerate(outcomes):
                nxt = z + delta
                outcome_value = (spread + multiplier * drift *
                                         (z * current_fraction + delta / self.price_ticks_per_interval)
                                         - self.risk_weight * (z * z * (current_dt - current_tick_dt)
                                                               + nxt * nxt * current_tick_dt)
                                         + future(depth - 1, nxt))
                rows[index][z] = -self._fee(self._quantity_for_target(q, z, lot), spec) - breach(z) + outcome_value
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
                delta = self._quantity_for_target(q, selected, spec[3])
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
            delta = self._quantity_for_target(equivalent, desired, spec[3])
            q = self._number(self._positions.get(root))
            target = self._feasible_target(q, positions.get(root, q) + delta, root, spec)
            if target is None:
                continue
            changes[root] = self._quantity_for_target(q, target, spec[3])
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
                delta = self._quantity_for_target(q, z, spec[3])
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
                    factors[root] = factors.get(root, 0.0) + self._quantity_for_target(q, best, spec[3]) * spec[4] * self._number(coefficient)
                positions[pid] = best
        result = []
        for pid, z in positions.items():
            old_q = self._number(self._positions.get(pid))
            spec = self._spec(pid, market)
            delta = (self._quantity_for_target(old_q, z, spec[3]) if spec is not None else 0.0)
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


class Dealer(_CoreDealer):
    """Opt-in, sequential boundary experiment for the supplied development SDK.

    Default construction retains conventional B3. This class never learns a
    response model. Unsupported histories stop new admission rather than infer
    quote acknowledgements, lost results, or exchange capacity.
    """
    def __init__(self, boundary_reset=False, max_boundary_orders=12,
                 max_boundary_states=128, **kwargs):
        if boundary_reset:
            kwargs["freeze_current_quotes"] = False
        super().__init__(**kwargs)
        self.boundary_reset = bool(boundary_reset)
        self.max_boundary_orders = max(1, min(12, int(max_boundary_orders)))
        self.max_boundary_states = max(1, min(128, int(max_boundary_states)))
        self.boundary_state = "uninitialised"
        self.boundary_failures = deque(maxlen=100)
        self.boundary_epochs = self.boundary_actions = 0
        self.boundary_suppressed_quotes = self.boundary_model_fallbacks = 0
        self.boundary_model_decisions = self.boundary_nonbaseline = 0
        self.boundary_execution_items = self.boundary_partial_batches = 0
        self.boundary_duplicate_events = 0
        self.boundary_deferred_items = self.boundary_unsupported_histories = 0
        self._boundary_epoch = None
        self._boundary_version = None
        self._boundary_orders = {}
        self._boundary_old = {}
        self._boundary_results = {}
        self._boundary_fills = {}
        self._boundary_seen_results = set()
        self._boundary_seen_fills = set()
        self._boundary_seen_acks = set()
        self._boundary_account = None
        self._boundary_pending = None
        self._boundary_offered = False
        self._boundary_once = False

    def _boundary_fail(self, reason):
        self.boundary_unsupported_histories += int(self.boundary_state != "unsupported")
        self.boundary_state = "unsupported"
        self.boundary_failures.append(str(reason))

    @property
    def boundary_counters(self):
        return {"epochs": self.boundary_epochs, "actions": self.boundary_actions,
                "suppressed_quotes": self.boundary_suppressed_quotes,
                "model_fallbacks": self.boundary_model_fallbacks,
                "model_decisions": self.boundary_model_decisions,
                "nonbaseline_decisions": self.boundary_nonbaseline,
                "execution_items": self.boundary_execution_items,
                "partial_batches": self.boundary_partial_batches,
                "duplicate_events": self.boundary_duplicate_events,
                "unsupported_histories": self.boundary_unsupported_histories,
                "deferred_items": self.boundary_deferred_items}

    @staticmethod
    def _close_number(a, b):
        return (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(a) and math.isfinite(b)
                and abs(a - b) <= 1e-8 + 1e-12 * max(abs(a), abs(b)))

    def _account_copy(self, portfolio):
        fields = ("cash", "customer_trades", "exchange_trades", "fees_paid", "penalties_paid")
        if not isinstance(portfolio, dict) or not isinstance(portfolio.get("positions"), dict):
            return None
        result = {key: portfolio.get(key) for key in fields}
        result["positions"] = dict(portfolio["positions"])
        if any(not self._close_number(value, value) for key, value in result.items()
               if key != "positions"):
            return None
        if any(not self._close_number(value, value) for value in result["positions"].values()):
            return None
        if any(type(result[key]) is not int or result[key] < 0
               for key in ("customer_trades", "exchange_trades")):
            return None
        return result

    def _account_matches(self, actual, expected=None, allow_penalties=False):
        expected = self._boundary_account if expected is None else expected
        observed = self._account_copy(actual)
        if expected is None or observed is None:
            return False
        if set(observed["positions"]) != set(expected["positions"]):
            return False
        if any(not self._close_number(observed["positions"][pid], q)
               for pid, q in expected["positions"].items()):
            return False
        penalty = observed["penalties_paid"] - expected["penalties_paid"]
        if penalty < -1e-8 or (not allow_penalties and abs(penalty) > 1e-8):
            return False
        if not self._close_number(observed["cash"], expected["cash"] - penalty):
            return False
        return all(self._close_number(observed[key], expected[key])
                   for key in ("customer_trades", "exchange_trades", "fees_paid"))

    def _apply_account_fill(self, account, fill):
        pid = fill["product_id"]
        spec = self._spec(pid, self._market)
        if spec is None or pid not in account["positions"]:
            return False
        qty, price, fee = fill.get("quantity"), fill.get("price"), fill.get("fee")
        if (not self._close_number(qty, qty) or qty <= 0
                or not self._close_number(price, price) or price <= 0
                or not self._close_number(fee, fee) or fee < 0
                or fill.get("side") not in ("BUY", "SELL")):
            return False
        signed = qty if fill["side"] == "BUY" else -qty
        account["positions"][pid] += signed
        account["cash"] -= signed * price * spec[4] + fee
        account["fees_paid"] += fee
        account["customer_trades" if fill["source"] == "customer" else "exchange_trades"] += 1
        return True

    def _load_boundary_orders(self, market):
        orders = market.get("active_orders")
        if not isinstance(orders, list) or len(orders) > 10000:
            self._boundary_fail("unsupported public RFQ batch")
            return {}
        result = {}
        for order in orders:
            if (not isinstance(order, dict) or not isinstance(order.get("order_id"), str)
                    or order["order_id"] in result or order.get("side") not in ("BUY", "SELL")
                    or self._spec(order.get("product_id"), market) is None
                    or self._number(order.get("quantity"), -1) <= 0):
                self._boundary_fail("malformed public RFQ")
                return {}
            result[order["order_id"]] = dict(order)
        return result

    def on_market(self, market, portfolio, exchange=None):
        if not self.boundary_reset:
            return super().on_market(market, portfolio, exchange)
        if not isinstance(market, dict) or type(market.get("interval_id")) is not int:
            self._boundary_fail("malformed market")
            return
        self._market = market
        if self.boundary_state == "unsupported":
            return
        epoch = (market["interval_id"], bool(market.get("closing")))
        if type(market.get("price_version")) is not int:
            self._boundary_fail("missing market version")
            return
        if epoch == self._boundary_epoch:
            # No participant reconnect marker exists. A repeated startup with
            # no offered book/action is harmless; later snapshots are refused.
            if self._boundary_offered or self._boundary_pending is not None:
                self._boundary_fail("same-epoch market may be reconnect")
            elif market["price_version"] != self._boundary_version:
                self._boundary_fail("unexpected same-epoch market version")
            return
        if self._boundary_epoch is not None:
            old_interval, old_closing = self._boundary_epoch
            valid_next = ((not old_closing and epoch == (old_interval + 1, False))
                          or (not old_closing and epoch == (old_interval, True)))
            if (not valid_next or self._boundary_pending is not None or self._boundary_old
                    or market["price_version"] <= self._boundary_version):
                self._boundary_fail("unresolved or reordered boundary")
                return
            self._boundary_old = {(old_interval, oid): order
                                  for oid, order in self._boundary_orders.items()}
        else:
            self._boundary_account = self._account_copy(portfolio)
            if (self._boundary_account is None
                    or portfolio.get("interval_id") != market["interval_id"]
                    or portfolio.get("price_version") != market["price_version"]):
                self._boundary_fail("initial snapshot account unavailable")
                return
        self._boundary_epoch, self._boundary_version = epoch, market["price_version"]
        self._boundary_orders = self._load_boundary_orders(market)
        self._boundary_results, self._boundary_fills = {}, {}
        self._boundary_once = self._boundary_offered = False
        self.boundary_epochs += 1
        if self.boundary_state != "unsupported":
            self.boundary_state = "reconciling"

    def quote(self, orders, market, portfolio):
        if not self.boundary_reset:
            return super().quote(orders, market, portfolio)
        self._market = market if isinstance(market, dict) else self._market
        suppress = {o["order_id"]: None for o in orders or ()
                    if isinstance(o, dict) and isinstance(o.get("order_id"), str)}
        if self.boundary_state != "open" or self._boundary_pending is not None:
            self.boundary_suppressed_quotes += len(suppress)
            return suppress
        if (not self._account_matches(portfolio)
                or any(o.get("order_id") not in self._boundary_orders for o in orders or ())):
            self._boundary_fail("unreconciled quotation account or RFQ")
            return suppress
        quotes = super().quote(orders, market, portfolio)
        self._boundary_offered |= any(price is not None for price in quotes.values())
        return quotes

    def on_fill(self, fill, portfolio, exchange=None):
        if not self.boundary_reset:
            return super().on_fill(fill, portfolio, exchange)
        if self.boundary_state == "unsupported":
            return
        if not isinstance(fill, dict) or fill.get("source") not in ("customer", "exchange"):
            self._boundary_fail("unsupported fill")
            return
        if fill["source"] == "customer":
            key = (fill.get("interval_id"), fill.get("order_id"))
            if key in self._boundary_seen_fills:
                self.boundary_duplicate_events += 1
                return
            order = self._boundary_old.get(key)
            if (order is None or self.boundary_state != "reconciling"
                    or fill.get("product_id") != order["product_id"]
                    or fill.get("side") != ("SELL" if order["side"] == "BUY" else "BUY")
                    or not self._close_number(fill.get("quantity"), order["quantity"])
                    or not self._close_number(fill.get("fee"), 0.0)
                    or not self._apply_account_fill(self._boundary_account, fill)):
                self._boundary_fail("customer fill does not match prior public RFQ")
                return
            self._boundary_fills[key] = dict(fill)
            self._boundary_seen_fills.add(key)
            super().on_fill(fill, portfolio, exchange)
            # The Runner's intermediate account can omit a just-charged
            # penalty and retains old version tags. Only final hedge confirms.
            return
        pending = self._boundary_pending
        action = fill.get("action_id")
        if action in self._boundary_seen_fills:
            self.boundary_duplicate_events += 1
            return
        if pending is None or not isinstance(action, str) or ":" not in action:
            self._boundary_fail("unexpected exchange execution")
            return
        prefix, index = action.rsplit(":", 1)
        if (not index.isdigit() or str(int(index)) != index or int(index) >= len(pending["items"])
                or not prefix or pending.get("prefix", prefix) != prefix):
            self._boundary_fail("execution identity does not match single pending batch")
            return
        index = int(index)
        item = pending["items"][index]
        spec = self._spec(item[0], self._market)
        expected_fee = (item[2] * (self._number(spec[0].get("fee_per_unit"))
                        + self._number(fill.get("price"), -1) * spec[4]
                        * self._number(spec[0].get("fee_rate")))) if spec else -1
        if (index in pending["fills"] or fill.get("product_id") != item[0]
                or fill.get("side") != item[1]
                or fill.get("interval_id") != self._boundary_epoch[0]
                or type(fill.get("price_version")) is not int
                or fill["price_version"] < self._boundary_version
                or not self._close_number(fill.get("quantity"), item[2])
                or not self._close_number(fill.get("fee"), expected_fee)
                or not self._apply_account_fill(pending["expected"], fill)):
            self._boundary_fail("exchange item or fee mismatch")
            return
        pending["prefix"] = prefix
        pending["fills"][index] = dict(fill)
        pending["portfolio"] = self._account_copy(portfolio)
        self._boundary_seen_fills.add(action)
        self.boundary_execution_items += 1
        # Aggregate account accompanies every item. It is checked after ACK
        # coverage is complete, never against an intermediate partial sum.

    def on_customer_result(self, result):
        if not self.boundary_reset:
            return super().on_customer_result(result)
        if self.boundary_state == "unsupported":
            return
        if not isinstance(result, dict):
            self._boundary_fail("malformed customer result")
            return
        key = (result.get("interval_id"), result.get("order_id"))
        if key in self._boundary_seen_results:
            self.boundary_duplicate_events += 1
            return
        order = self._boundary_old.get(key)
        if (order is None or type(result.get("traded")) is not bool
                or result.get("status") not in ("filled", "expired")
                or result["traded"] != (key in self._boundary_fills)
                or (result["traded"] and result["status"] != "filled")):
            self._boundary_fail("result order/winner/fill mismatch or reordered result")
            return
        if result.get("quantity") is not None and not self._close_number(result["quantity"], order["quantity"]):
            self._boundary_fail("result quantity mismatch")
            return
        if (result.get("product_id", order["product_id"]) != order["product_id"]
                or (result["traded"] and result.get("price") is not None
                    and not self._close_number(result["price"], self._boundary_fills[key]["price"]))):
            self._boundary_fail("result product or winning price mismatch")
            return
        self._boundary_results[key] = dict(result)
        self._boundary_seen_results.add(key)
        super().on_customer_result(result)

    def _boundary_ready(self, market, portfolio):
        if (self._boundary_epoch != (market.get("interval_id"), bool(market.get("closing")))
                or portfolio.get("interval_id") != self._boundary_epoch[0]
                or type(portfolio.get("price_version")) is not int
                or portfolio["price_version"] < self._boundary_version
                or set(self._boundary_results) != set(self._boundary_old)):
            return False
        if not self._account_matches(portfolio, allow_penalties=True):
            self._boundary_fail("boundary account does not reconcile")
            return False
        self._boundary_account = self._account_copy(portfolio)
        self._boundary_old = {}
        if self._reserved:
            self._boundary_fail("unresolved prior quote reservation")
            return False
        return True

    def _boundary_batch_outcomes(self, z, pid, spec, market):
        """Exact Bernoulli enumeration of candidate-dependent public batch.

        Admission follows the actual sequential all-subset capacity rule.
        Probabilities are assumed, not estimated eligibility/competition laws.
        Merging net deltas retains probability-weighted customer spread.
        """
        rows = [order for order in self._boundary_orders.values() if order["product_id"] == pid]
        if len(rows) > self.max_boundary_orders:
            return None
        lower, upper = self._bounds(spec[0])
        negative = positive = 0.0
        states = {0: (1.0, 0.0)}
        lot = Decimal(str(spec[3]))
        for order in rows:
            qty = self._number(order["quantity"], -1)
            unit_qty = Decimal(str(qty)) / lot
            if unit_qty != unit_qty.to_integral_value():
                return None
            direction = 1 if order["side"] == "BUY" else -1
            lo = z - negative - (qty if direction == 1 else 0.0)
            hi = z + positive + (qty if direction == -1 else 0.0)
            if lo < lower - 1e-10 or hi > upper + 1e-10:
                continue
            quote = self._edge(order["side"], z, spec[1], spec[2], pid, market)
            if quote <= 0 or not math.isfinite(quote):
                continue
            if direction == 1:
                negative += qty
            else:
                positive += qty
            margin = direction * (quote - spec[1])
            probability = self._probability(margin)
            signed_units = -direction * int(unit_qty)
            reward = qty * spec[4] * (margin - self._number(self.adverse_selection.get(pid)))
            next_states = {}
            for delta, (weight, spread_mass) in states.items():
                for branch, nxt, cash in ((1 - probability, delta, 0.0),
                                          (probability, delta + signed_units, reward)):
                    mass, spread = next_states.get(nxt, (0.0, 0.0))
                    next_states[nxt] = (mass + weight * branch,
                                        spread + branch * (spread_mass + weight * cash))
            states = {delta: row for delta, row in next_states.items() if row[0] > 0}
            if len(states) > self.max_boundary_states:
                return None
        return [(weight, float(Decimal(delta) * lot), spread / weight)
                for delta, (weight, spread) in sorted(states.items())]

    def _boundary_choose(self, market, portfolio):
        if self.mode != "R2":
            return super().hedge(market, portfolio)
        trades = []
        self._positions = dict(portfolio["positions"])
        self._pending_hedges.clear()
        too_many = len(market.get("products", {})) > self.max_r2_products
        for pid, q in self._positions.items():
            q = self._number(q)
            spec = self._spec(pid, market)
            if spec is None or not spec[0].get("tradable_against_exchange", False):
                continue
            baseline = q if market.get("closing") else self._band_target(q, spec)
            baseline = self._feasible_target(q, baseline, pid, spec)
            baseline = q if baseline is None else baseline
            candidates = []
            for target in (baseline, q, 0.0, .25 * q, .5 * q, .75 * q):
                z = self._feasible_target(q, target, pid, spec)
                if z is not None and z not in candidates:
                    candidates.append(z)
            values, fallback = {}, too_many
            if not fallback:
                for z in candidates:
                    outcomes = self._boundary_batch_outcomes(z, pid, spec, market)
                    if outcomes is None:
                        fallback = True
                        break
                    values[z] = self._continuation_values(q, [z], pid, spec, market,
                                                          current_outcomes=outcomes)[z]
            selected, gap = baseline, 0.0
            if fallback:
                self.boundary_model_fallbacks += 1
                values = {}
            elif values:
                self.boundary_model_decisions += 1
                best = max(values, key=values.get)
                gap = values[best] - values[baseline]
                if gap > self.min_improvement:
                    selected = best
                self.boundary_nonbaseline += int(selected != baseline)
            delta = self._quantity_for_target(q, selected, spec[3])
            self.last_decisions.append({"product_id": pid, "inventory": q, "target": q + delta,
                "baseline_target": baseline, "value_gap": gap, "candidate_values": values,
                "mode": "boundary_R2", "interval_id": market.get("interval_id"),
                "closing": bool(market.get("closing")), "model_fallback": fallback,
                "candidate_dependent_upcoming_quotes": True, "error_bound_certified": False})
            if delta:
                trades.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
                self._pending_hedges[pid] = delta
        return trades

    def hedge(self, market, portfolio):
        if not self.boundary_reset:
            return super().hedge(market, portfolio)
        self._market = market if isinstance(market, dict) else self._market
        if (self.boundary_state != "reconciling" or self._boundary_once
                or self._boundary_pending is not None):
            return []
        if not self._boundary_ready(market, portfolio):
            return []
        self._boundary_once = True
        trades = self._boundary_choose(market, portfolio)
        if len(trades) > 500:
            # One explicit portfolio action constraint: prioritise recovery of
            # individual breaches, then normalized exposure, stable ID ties.
            # Other products retain no action rather than imaginary targets.
            def priority(item):
                pid = item[0]
                q = self._number(portfolio["positions"].get(pid))
                spec = self._spec(pid, market)
                lower, upper = self._bounds(spec[0])
                scale = upper if q >= 0 else -lower
                scale = scale if math.isfinite(scale) and scale > 0 else max(spec[3], abs(q), 1.)
                return (-max(0., q - upper, lower - q) / scale, -abs(q) / scale, str(pid))
            chosen = sorted(trades, key=priority)[:500]
            active = {item[0] for item in chosen}
            deferred = {item[0] for item in trades if item[0] not in active}
            self.boundary_deferred_items += len(deferred)
            for pid in deferred:
                self._pending_hedges.pop(pid, None)
            for decision in self.last_decisions:
                if (decision.get("product_id") in deferred
                        and (decision.get("interval_id"), bool(decision.get("closing")))
                            == self._boundary_epoch):
                    if self.mode == "R2" and decision["target"] != decision["baseline_target"]:
                        self.boundary_nonbaseline -= 1
                    decision["proposed_target"] = decision["target"]
                    decision["proposed_value_gap"] = decision["value_gap"]
                    decision["target"] = self._number(portfolio["positions"].get(decision["product_id"]))
                    decision["value_gap"] = 0.
                    decision["scheduling_deferred"] = True
            trades = chosen
        if trades:
            self._boundary_pending = {"items": list(trades), "fills": {},
                "expected": self._account_copy(portfolio), "portfolio": None}
            self.boundary_state = "pending_reset"
            self.boundary_actions += 1
        else:
            self.boundary_state = "open"
        return trades

    def on_ack(self, ack):
        if not self.boundary_reset:
            return super().on_ack(ack)
        if self.boundary_state == "unsupported":
            return
        if not isinstance(ack, dict) or not isinstance(ack.get("action_id"), str):
            return
        action_id = ack["action_id"]
        if action_id in self._boundary_seen_acks or ack.get("duplicate"):
            self.boundary_duplicate_events += 1
            return
        self._boundary_seen_acks.add(action_id)
        pending = self._boundary_pending
        if pending is None:
            # Quote failures retain reservations but do not identify a reset.
            return
        if pending.get("prefix") != action_id:
            if ack.get("status") == "rejected" and not ack.get("errors"):
                self._boundary_fail("uncorrelated rejected action while reset pending")
            return
        errors, executions = ack.get("errors", {}), ack.get("executions")
        if ack.get("status") != "accepted" or not isinstance(errors, dict) or not isinstance(executions, list):
            self._boundary_fail("malformed correlated reset ACK")
            return
        indices = set()
        for key in errors:
            if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
                self._boundary_fail("unsupported reset error index")
                return
            indices.add(int(key))
        fills = pending["fills"]
        execution_map = {}
        for execution in executions:
            if not isinstance(execution, dict) or not isinstance(execution.get("action_id"), str):
                self._boundary_fail("malformed execution in reset ACK")
                return
            action = execution["action_id"]
            if ":" not in action:
                self._boundary_fail("malformed execution identity in reset ACK")
                return
            prefix, index = action.rsplit(":", 1)
            if (prefix != action_id or not index.isdigit() or str(int(index)) != index
                    or int(index) in execution_map):
                self._boundary_fail("duplicate/foreign execution in reset ACK")
                return
            execution_map[int(index)] = execution
        if (indices & set(fills) or indices | set(fills) != set(range(len(pending["items"])))
                or len(executions) != len(fills)
                or execution_map != fills
                or not self._account_matches(pending["portfolio"], pending["expected"])):
            self._boundary_fail("reset ACK coverage/account mismatch")
            return
        self.boundary_partial_batches += int(bool(indices))
        self._boundary_account = pending["expected"]
        self._boundary_pending = None
        self._pending_hedges.clear()
        self.boundary_state = "open"
