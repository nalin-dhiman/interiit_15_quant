"""Conventional public-settlement target band; research, not a deployment file.

The holding objective is deliberately simpler than the RFQ continuation model.
It is a preference under a martingale-root assumption, not an identified market
value. Actual exchange transactions and the unchanged boundary accounting gate
are inherited from the previously frozen SDK controller.
"""
from __future__ import annotations

import math

from submission.boundary_dealer import Dealer as BoundaryDealer


def fee_kink_target(center, original, curvature, fee_per_unit, lower, upper):
    """Exact box solution of -.5*h*(z-center)^2-c*abs(z-original)."""
    if curvature <= 0:
        raise ValueError("positive curvature required")
    width = fee_per_unit / curvature
    if center > original + width:
        target = center - width
    elif center < original - width:
        target = center + width
    else:
        target = original
    return min(upper, max(lower, target))


class BasisBandDealer(BoundaryDealer):
    """B4: inventory target from public settlement basis and a no-trade band.

    ``factor_risk`` uses exact published terminal root loadings plus a strictly
    labelled idiosyncratic *preference*. It is not an estimated covariance.
    Setting ``risk_multiplier=0`` gives the linear fee-aware settlement control.
    Both signs of new exposure are allowed unless ``reduce_only`` is requested.
    """

    def __init__(self, factor_risk=False, idiosyncratic_risk=.25,
                 risk_multiplier=1., max_coordinate_sweeps=12,
                 reduce_only=False, **kwargs):
        if (not math.isfinite(float(risk_multiplier)) or risk_multiplier < 0
                or not math.isfinite(float(idiosyncratic_risk))
                or idiosyncratic_risk < 0):
            raise ValueError("risk preferences must be finite and nonnegative")
        kwargs.update(mode="B3", boundary_reset=True, factor_hedge=False)
        super().__init__(**kwargs)
        self.factor_risk = bool(factor_risk)
        self.idiosyncratic_risk = float(idiosyncratic_risk) if self.factor_risk else 0.
        self.risk_multiplier = float(risk_multiplier)
        self.max_coordinate_sweeps = max(1, min(50, int(max_coordinate_sweeps)))
        self.reduce_only = bool(reduce_only)
        self.basis_coordinate_updates = 0
        self.basis_opening_actions = 0
        self.basis_last_objective_gain = 0.

    def _holding_seconds(self, market):
        """Explicit frozen-position holding approximation in physical seconds."""
        interval = max(0., self._number(market.get("interval_seconds"), 5.))
        if market.get("closing"):
            return interval / self.price_ticks_per_interval
        return interval * self._remaining(market)

    def _loadings(self, pid, spec):
        multiplier = spec[4]
        if not self.factor_risk:
            # Match the existing scalar continuation's physical q^2 penalty.
            return {pid: 1.}
        coefficients = self.settlement_coefficients.get(pid, {pid: 1.})
        return {root: multiplier * self._number(weight)
                for root, weight in coefficients.items() if self._number(weight)}

    @staticmethod
    def _exposures(positions, loadings):
        result = {}
        for pid, q in positions.items():
            for root, loading in loadings.get(pid, {}).items():
                result[root] = result.get(root, 0.) + loading * q
        return result

    def _objective(self, positions, original, specs, loadings, market):
        exposures = self._exposures(positions, loadings)
        square = sum(value * value for value in exposures.values())
        square += self.idiosyncratic_risk * sum(
            (specs[pid][4] * q) ** 2 for pid, q in positions.items())
        result = -self.risk_weight * self.risk_multiplier * self._holding_seconds(market) * square
        for pid, q in positions.items():
            spec = specs[pid]
            lower, upper = self._bounds(spec[0])
            result += spec[4] * self._basis(pid, market) * q
            result -= self._fee(self._quantity_for_target(original[pid], q, spec[3]), spec)
            result -= max(0., q - upper, lower - q) * self._number(
                spec[0].get("breach_penalty_per_unit"))
        return result

    def _boundary_choose(self, market, portfolio):
        self._market = market
        self._positions = dict(portfolio["positions"])
        self._pending_hedges.clear()
        specs = {pid: self._spec(pid, market) for pid in self._positions}
        specs = {pid: spec for pid, spec in specs.items() if spec is not None}
        original = {pid: self._number(self._positions[pid]) for pid in specs}
        positions = dict(original)
        loadings = {pid: self._loadings(pid, spec) for pid, spec in specs.items()}
        exposures = self._exposures(positions, loadings)
        coefficient = self.risk_weight * self.risk_multiplier * self._holding_seconds(market)

        def priority(pid):
            q, spec = original[pid], specs[pid]
            lower, upper = self._bounds(spec[0])
            scale = upper if q >= 0 else -lower
            scale = scale if math.isfinite(scale) and scale > 0 else max(1., abs(q), spec[3])
            return (-max(0., q - upper, lower - q) / scale, -abs(q) / scale, str(pid))

        eligible = [pid for pid, spec in specs.items()
                    if spec[0].get("tradable_against_exchange", False)]
        # Fix the supported action set BEFORE solving the coupled objective;
        # dropping one hedge leg afterwards would invalidate its cross terms.
        mutable = sorted(eligible, key=priority)[:500]
        self.boundary_deferred_items += max(0, len(eligible) - len(mutable))
        initial_value = self._objective(original, original, specs, loadings, market)
        diagnostics = {}
        for _ in range(self.max_coordinate_sweeps if self.factor_risk else 1):
            changed = False
            for pid in mutable:
                spec, q, z = specs[pid], original[pid], positions[pid]
                lower, upper = self._bounds(spec[0])
                # This only bounds our proposed exchange position. Customer
                # fills and the exchange's actual public limits are untouched.
                if not math.isfinite(lower):
                    lower = min(q, -max(spec[3], 10 * self.future_size))
                if not math.isfinite(upper):
                    upper = max(q, max(spec[3], 10 * self.future_size))
                if self.reduce_only:
                    lower, upper = max(lower, min(q, 0.)), min(upper, max(q, 0.))
                unit_fee = self._fee(1., spec)
                gain = spec[4] * self._basis(pid, market)
                own_loading = loadings[pid]
                diagonal = sum(v * v for v in own_loading.values())
                diagonal += self.idiosyncratic_risk * spec[4] ** 2
                curvature = 2 * coefficient * diagonal
                cross = sum(v * (exposures.get(root, 0.) - v * z)
                            for root, v in own_loading.items())
                if curvature > 0:
                    center = (gain - 2 * coefficient * cross) / curvature
                    cost_target = fee_kink_target(center, q, curvature, unit_fee, lower, upper)
                    finite_limit = max(abs(lower), abs(upper), spec[3])
                    band = max(unit_fee / curvature, self.band_fraction * finite_limit)
                    trigger = abs(z - center) > band
                    # A conventional displacement band, returning part-way to
                    # the fee-aware target rather than invoking RFQ Bellman.
                    proposed = cost_target + self.target_fraction * (z - cost_target)
                else:
                    # Exact linear objective; ties retain the actual account.
                    trigger = abs(gain) > unit_fee
                    proposed = upper if gain > unit_fee else lower if gain < -unit_fee else z
                    center, band = proposed, 0.
                if not trigger and lower <= z <= upper:
                    continue
                desired = min(upper, max(lower, proposed))
                actual = self._feasible_target(q, desired, pid, spec)
                if actual is None or actual == z:
                    continue
                delta = actual - z
                linear_gain = gain * delta
                risk_change = coefficient * (2 * delta * (
                    sum(v * exposures.get(root, 0.) for root, v in own_loading.items())
                    + self.idiosyncratic_risk * spec[4] ** 2 * z) + diagonal * delta ** 2)
                fee_change = (self._fee(self._quantity_for_target(q, actual, spec[3]), spec)
                              - self._fee(self._quantity_for_target(q, z, spec[3]), spec))
                penalty_rate = self._number(spec[0].get("breach_penalty_per_unit"))
                penalty_gain = (max(0., z - upper, lower - z)
                                - max(0., actual - upper, lower - actual)) * penalty_rate
                improvement = linear_gain - risk_change - fee_change + penalty_gain
                if improvement <= self.min_improvement:
                    continue
                positions[pid] = actual
                for root, loading in own_loading.items():
                    exposures[root] = exposures.get(root, 0.) + loading * delta
                changed = True
                self.basis_coordinate_updates += 1
                diagnostics[pid] = dict(center=center, no_trade_halfwidth=band,
                                        last_coordinate_gain=improvement)
            if not changed:
                break
        self.basis_last_objective_gain = (
            self._objective(positions, original, specs, loadings, market) - initial_value)
        if self.basis_last_objective_gain < -1e-8:
            raise ArithmeticError("rounded conventional holding objective decreased")
        trades = []
        for pid in mutable:
            q, spec = original[pid], specs[pid]
            delta = self._quantity_for_target(q, positions[pid], spec[3])
            if delta:
                self._pending_hedges[pid] = delta
                self.basis_opening_actions += int(q == 0.)
                trades.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
            self.last_decisions.append(dict(product_id=pid, inventory=q, target=q + delta,
                baseline_target=q + delta, value_gap=0., mode="B4_basis_band",
                interval_id=market.get("interval_id"), closing=bool(market.get("closing")),
                factor_risk=self.factor_risk, risk_adjusted_value_is_pnl=False,
                error_bound_certified=False, **diagnostics.get(pid, {})))
        return trades


    def _band_target(self, q, spec):
        """Scalar B4 target, also usable by the matching R2 continuation.

        The core signature supplies no product ID. The public market owns its
        product dictionaries, so identify this exact spec without price-tape or
        hidden customer information. Missing context retains inherited B3.
        """
        pid = next((pid for pid, product in self._market.get("products", {}).items()
                    if product is spec[0]), None)
        if pid is None or self.factor_risk:
            return super()._band_target(q, spec)
        lower, upper = self._bounds(spec[0])
        if not math.isfinite(lower):
            lower = min(q, -max(spec[3], 10 * self.future_size))
        if not math.isfinite(upper):
            upper = max(q, max(spec[3], 10 * self.future_size))
        if self.reduce_only:
            lower, upper = max(lower, min(q, 0.)), min(upper, max(q, 0.))
        gain = spec[4] * self._basis(pid, self._market)
        curvature = 2 * self.risk_weight * self.risk_multiplier * self._holding_seconds(self._market)
        fee = self._fee(1., spec)
        if curvature > 0:
            center = gain / curvature
            band = max(fee / curvature, self.band_fraction * max(abs(lower), abs(upper), spec[3]))
            if abs(q - center) <= band and lower <= q <= upper:
                return q
            target = fee_kink_target(center, q, curvature, fee, lower, upper)
            target += self.target_fraction * (q - target)
        else:
            target = upper if gain > fee else lower if gain < -fee else q
        target = min(upper, max(lower, target))
        actual = q + self._quantity_for_target(q, target, spec[3])
        improvement = (gain * (actual - q) - .5 * curvature * (actual * actual - q * q)
                       - self._fee(self._quantity_for_target(q, actual, spec[3]), spec))
        improvement += (max(0., q - upper, lower - q)
                        - max(0., actual - upper, lower - actual)) * self._number(
                            spec[0].get("breach_penalty_per_unit"))
        return actual if improvement > self.min_improvement else q


class BasisResetDealer(BasisBandDealer):
    """Scalar B4 continuation-aware improvement with the same public basis.

    Its action set includes the B4 target, including opening from zero. Factor
    B4 is intentionally excluded because the inherited scalar q^2 continuation
    does not represent a joint factor objective. Actual policy improvement is
    not guaranteed by this approximate finite-model ranking.
    """
    def __init__(self, factor_risk=False, risk_multiplier=1., **kwargs):
        if factor_risk:
            raise ValueError("R2B supports the scalar B4 continuation only")
        if not math.isfinite(float(risk_multiplier)) or risk_multiplier < 0:
            raise ValueError("risk multiplier must be finite and nonnegative")
        effective_risk = float(kwargs.pop("risk_weight", .0002)) * float(risk_multiplier)
        super().__init__(factor_risk=False, risk_multiplier=1.,
                         risk_weight=effective_risk, **kwargs)
        self.mode = "R2"

    def _boundary_choose(self, market, portfolio):
        self._market = market
        if not market.get("closing"):
            return BoundaryDealer._boundary_choose(self, market, portfolio)
        # The frozen boundary policy deliberately kept no action as its closing
        # baseline. B4 can open a fee-profitable terminal-basis position, so the
        # matched comparator must admit that SAME action at the closing epoch.
        self._positions = dict(portfolio["positions"])
        self._pending_hedges.clear()
        trades = []
        for pid, raw in self._positions.items():
            q = self._number(raw)
            spec = self._spec(pid, market)
            if spec is None or not spec[0].get("tradable_against_exchange", False):
                continue
            baseline = self._feasible_target(q, self._band_target(q, spec), pid, spec)
            baseline = q if baseline is None else baseline
            candidates = []
            for target in (baseline, q, 0., .25 * q, .5 * q, .75 * q):
                actual = self._feasible_target(q, target, pid, spec)
                if actual is not None and actual not in candidates:
                    candidates.append(actual)
            values = self._continuation_values(q, candidates, pid, spec, market)
            best = max(values, key=values.get)
            gap = values[best] - values[baseline]
            selected = best if gap > self.min_improvement else baseline
            self.boundary_model_decisions += 1
            self.boundary_nonbaseline += int(selected != baseline)
            delta = self._quantity_for_target(q, selected, spec[3])
            self.last_decisions.append(dict(product_id=pid, inventory=q, target=q + delta,
                baseline_target=baseline, value_gap=gap, candidate_values=values,
                mode="basis_boundary_R2", interval_id=market.get("interval_id"),
                closing=True, model_fallback=False,
                candidate_dependent_upcoming_quotes=True, error_bound_certified=False))
            if delta:
                trades.append((str(pid), "BUY" if delta > 0 else "SELL", float(abs(delta))))
                self._pending_hedges[pid] = delta
        return trades


def basis_candidate_grid(factor_risk=False):
    """Prospective 24-candidate joint quote/hedge grid for EACH risk family.

    Six distributed risk/quote packages crossed with four forecast weights.
    This is not a full Cartesian search over all constructor parameters.
    Scalar B4, factor B4 and scalar R2B get the same 24 quote/risk packages.
    """
    packages = (
        (.04, 0., .10, 0., 0.),
        (.08, .002, .35, .5, .5),
        (.12, .005, .60, 0., 2.),
        (.16, 0., .10, .5, 0.),
        (.04, .005, .35, 0., .5),
        (.08, .002, .60, .5, 2.),
    )
    return [dict(base_edge=edge, inventory_skew=skew, band_fraction=band,
                 target_fraction=target, risk_multiplier=risk,
                 factor_risk=bool(factor_risk), forecast_weight=forecast)
            for forecast in (0., .25, .5, 1.)
            for edge, skew, band, target, risk in packages]
