"""Meaningful accounting/causality/model-domain checks for the B4 comparator."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import hashlib
import itertools
from pathlib import Path
import unittest

from resetmm.basis_baseline import (BasisBandDealer, BasisResetDealer,
                                    basis_candidate_grid, fee_kink_target)
from submission.boundary_dealer import Dealer as BoundaryDealer
from tests_sdk.test_boundary_dealer import account, apply
from tests_sdk.test_sdk_controller import market, order


SETTLEMENT = {"B": {"A": 2.}}


class BasisBaselineTests(unittest.TestCase):
    def dealer(self, **kwargs):
        parameters = dict(settlement_coefficients=SETTLEMENT, band_fraction=.01,
                          target_fraction=0., min_improvement=0.)
        parameters.update(kwargs)
        return BasisBandDealer(**parameters)

    def choose(self, dealer, m=None, **positions):
        m = market() if m is None else m
        p = account(m, **positions)
        return dealer._boundary_choose(m, p), p

    def test_fee_kink_formula_against_dense_independent_objective(self):
        for q, center, h, fee in itertools.product((-2., 0., 2.), (-3., -.2, .2, 3.),
                                                  (.2, 2.), (0., .1, 1.)):
            target = fee_kink_target(center, q, h, fee, -4., 4.)
            objective = lambda z: -.5 * h * (z - center) ** 2 - fee * abs(z - q)
            self.assertGreaterEqual(objective(target) + 1e-10,
                                    max(objective(-4. + k / 100) for k in range(801)))

    def test_fee_kink_requires_positive_curvature(self):
        with self.assertRaises(ValueError):
            fee_kink_target(0, 0, 0, .1, -1, 1)

    def test_negative_basis_opens_short_and_does_not_change_ledger(self):
        d = self.dealer(risk_multiplier=0)
        trades, p = self.choose(d)
        self.assertEqual(trades, [("B", "SELL", 20.)])
        self.assertEqual(p["positions"], {"A": 0., "B": 0.})
        self.assertEqual(p["cash"], 100000.)
        self.assertAlmostEqual(d.basis_last_objective_gain, 19.8)

    def test_positive_basis_opens_long(self):
        m = market(); m["prices"]["B"] = 199.
        self.assertEqual(self.choose(self.dealer(risk_multiplier=0), m)[0], [("B", "BUY", 20.)])

    def test_costs_can_eliminate_positive_basis_trade(self):
        m = market(); m["prices"]["B"] = 199.995
        self.assertEqual(self.choose(self.dealer(risk_multiplier=0), m)[0], [])

    def test_closing_is_not_forced_liquidation(self):
        d = self.dealer(risk_multiplier=0)
        trades, _ = self.choose(d, market(closing=True), A=12.)
        self.assertFalse(any(pid == "A" for pid, _, _ in trades))
        self.assertEqual(trades, [("B", "SELL", 20.)])

    def test_closing_basis_target_is_available_to_matched_r2(self):
        kwargs = dict(settlement_coefficients=SETTLEMENT, risk_multiplier=0,
                      band_fraction=.01, target_fraction=0., min_improvement=0.)
        a, b = BasisBandDealer(**kwargs), BasisResetDealer(**kwargs)
        m = market(closing=True); p = account(m)
        self.assertEqual(a._boundary_choose(m, p), b._boundary_choose(m, p))
        decision = [r for r in b.last_decisions if r["product_id"] == "B"][0]
        self.assertIn(-20., decision["candidate_values"])
        self.assertEqual(decision["baseline_target"], -20.)

    def test_multiplier_changes_principal_fee_and_basis_target(self):
        m = market(); m["products"]["B"]["multiplier"] = 2.
        m["products"]["B"]["fee_rate"] = .0001
        d = self.dealer(risk_weight=.001)
        trades, p = self.choose(d, m)
        self.assertEqual(trades, [("B", "SELL", 16.)])
        qty = trades[0][2]
        fee = qty * (.01 + 201. * 2 * .0001)
        self.assertAlmostEqual(d._fee(qty, d._spec("B", m)), fee)
        f = dict(product_id="B", side="SELL", quantity=qty, price=201., fee=fee,
                 source="exchange")
        final = apply(p, f, m)
        self.assertAlmostEqual(final["cash"], 100000. + qty * 201 * 2 - fee)
        self.assertEqual(final["positions"]["B"], -qty)

    def test_fractional_lot_uses_actual_quantity_and_revalued_gain(self):
        m = market(); m["products"]["B"].update(lot_size=.1, max_long=.7, max_short=.7)
        d = self.dealer(risk_multiplier=0)
        trades, _ = self.choose(d, m, B=.2)
        qty = trades[0][2]
        self.assertLessEqual(abs(.2 - qty), .7 + 1e-12)
        self.assertAlmostEqual(qty / .1, round(qty / .1))
        self.assertAlmostEqual(d.basis_last_objective_gain, qty * .99)
        decision = [row for row in d.last_decisions if row["product_id"] == "B"][0]
        self.assertEqual(decision["target"], .2 - qty)

    def test_nontradable_position_is_retained_in_factor_risk(self):
        m = market(); m["products"]["B"]["tradable_against_exchange"] = False
        d = self.dealer(factor_risk=True, idiosyncratic_risk=.25, risk_weight=.01)
        trades, _ = self.choose(d, m, B=5.)
        self.assertFalse(any(pid == "B" for pid, _, _ in trades))
        self.assertTrue(any(pid == "A" and side == "SELL" for pid, side, _ in trades))
        self.assertGreater(d.basis_last_objective_gain, 0.)

    def test_reduce_only_control_cannot_initiate_or_reverse(self):
        d = self.dealer(risk_multiplier=0, reduce_only=True)
        self.assertEqual(self.choose(d)[0], [])
        trades, _ = self.choose(d, B=7.)
        self.assertEqual(trades, [("B", "SELL", 7.)])

    def test_scalar_target_is_identical_in_direct_and_future_model(self):
        for q, risk, fraction, closing in itertools.product((-10., 0., 10.), (0., .5, 2.),
                                                          (0., .5), (False, True)):
            d = self.dealer(risk_multiplier=risk, target_fraction=fraction)
            m = market(closing=closing)
            trades, _ = self.choose(d, m, B=q)
            target = q + sum(qty * (1 if side == "BUY" else -1)
                             for pid, side, qty in trades if pid == "B")
            self.assertEqual(target, d._band_target(q, d._spec("B", m)))

    def test_matching_r2_risk_scale_and_basis_target(self):
        for risk in (0., .5, 2.):
            a = self.dealer(risk_multiplier=risk)
            b = BasisResetDealer(settlement_coefficients=SETTLEMENT, risk_multiplier=risk,
                                 band_fraction=.01, target_fraction=0., min_improvement=0.)
            m = market(); p = account(m)
            a._market = b._market = m
            self.assertEqual(a._band_target(0., a._spec("B", m)),
                             b._band_target(0., b._spec("B", m)))
            b._boundary_choose(m, p)
            r = [r for r in b.last_decisions if r["product_id"] == "B"][0]
            self.assertIn(a._band_target(0., a._spec("B", m)), r["candidate_values"])
            self.assertAlmostEqual(b.risk_weight, a.risk_weight * risk)

    def test_factor_continuation_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "scalar"):
            BasisResetDealer(factor_risk=True)

    def test_invalid_risk_preferences_are_not_silently_clipped(self):
        for cls in (BasisBandDealer, BasisResetDealer):
            for value in (-1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    cls(risk_multiplier=value)

    def test_only_public_current_state_affects_actions(self):
        m = market()
        altered = copy.deepcopy(m)
        altered["hidden_customer_limit"] = -100000.
        altered["latent_informed"] = True
        altered["future_prices"] = {"B": 100000.}
        a, b = self.dealer(), self.dealer()
        self.assertEqual(self.choose(a, m)[0], self.choose(b, altered)[0])
        self.assertEqual(a.last_decisions, b.last_decisions)

    def test_basis_quotes_are_identical_to_existing_public_basis_rule(self):
        orders = [order("b", "BUY", product="B"), order("s", "SELL", product="B")]
        m = market(orders)
        a = self.dealer(forecast_weight=.5, base_edge=.08, inventory_skew=.002)
        b = BoundaryDealer(settlement_coefficients=SETTLEMENT, forecast_weight=.5,
                           base_edge=.08, inventory_skew=.002)
        for side in ("BUY", "SELL"):
            self.assertEqual(a._edge(side, 3, 201, .01, "B", m),
                             b._edge(side, 3, 201, .01, "B", m))

    def test_boundary_execution_waits_for_real_cash_and_confirmed_fills(self):
        d = self.dealer(risk_multiplier=0)
        o = order("future", "BUY", quantity=4., product="B")
        m = market([o]); p = account(m)
        d.on_market(m, p)
        self.assertEqual(d.quote([o], m, p), {"future": None})
        trades = d.hedge(m, p)
        self.assertEqual(trades, [("B", "SELL", 20.)])
        self.assertEqual(d.boundary_state, "pending_reset")
        self.assertEqual(d.quote([o], m, p), {"future": None})
        f = dict(product_id="B", side="SELL", quantity=20., price=201., fee=.2,
                 source="exchange", action_id="actual:0", interval_id=1, price_version=10)
        final = apply(p, f, m)
        d.on_fill(f, final)
        d.on_ack(dict(action_id="actual", status="accepted", executions=[f]))
        self.assertEqual(d.boundary_state, "open")
        self.assertAlmostEqual(final["cash"], 104019.8)
        self.assertEqual(final["positions"]["B"], -20.)
        # A customer BUY would worsen the short limit. No imaginary hedge
        # capacity or opposite-side guaranteed offset admits it.
        self.assertIsNone(d.quote([o], m, final)["future"])

    def test_equal_fixed_grid_budgets_and_zero_forecast_control(self):
        for factor in (False, True):
            grid = basis_candidate_grid(factor)
            self.assertEqual(len(grid), 24)
            self.assertEqual(len({tuple(sorted(p.items())) for p in grid}), 24)
            self.assertEqual({p["forecast_weight"] for p in grid}, {0., .25, .5, 1.})
            self.assertEqual({p["risk_multiplier"] for p in grid}, {0., .5, 2.})
            self.assertTrue(all(p["factor_risk"] == factor for p in grid))

    def test_frozen_sdk_policy_sources_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        expected = {"sdk_dealer.py": "ff5a54ad9057521620903ab94f0b7017f3c61a905d7ce91563d6a3b207c33fbd",
                    "boundary_dealer.py": "d17b056ee7a3371cd89e22581f9ed57f3009a6eea8a048546e50dee82beb031e"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / "submission" / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
