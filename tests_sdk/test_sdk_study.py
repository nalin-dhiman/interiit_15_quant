"""Regression tests for research-only direct execution of supplied SDK Engine."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import unittest
import copy
import numpy as np

from experiments.sdk_study import (KIT, SynchronousSession, block_differences,
                                   build_config, candidate_grid, one_session)
from rfq.models import CustomerOrder
from experiments.sdk_calibration import features, monotone_decreasing


class RecordDealer:
    def __init__(self):
        self.quote_portfolios = []
        self.hedge_portfolios = []
        self.fill_portfolios = []

    def quote(self, orders, market, portfolio):
        self.quote_portfolios.append((market["interval_id"], portfolio["positions"].copy()))
        return {o["order_id"]: market["prices"][o["product_id"]] for o in orders}

    def hedge(self, market, portfolio):
        self.hedge_portfolios.append((market["price_version"], portfolio["positions"].copy()))
        return []

    def on_fill(self, fill, portfolio):
        self.fill_portfolios.append(portfolio["positions"].copy())


class KnownBuy:
    def generate(self, interval_id, market, state, rng):
        return None, [CustomerOrder(f"r{interval_id}", "c", "A", "BUY", 2, interval_id)]


class SDKStudyTests(unittest.TestCase):
    def test_exact_boundary_quote_before_prior_customer_fill_portfolio(self):
        config = build_config(None, 0, 5, rows=[dict(A=100., B=200.), dict(A=101., B=202.)])
        config.customer_model = KnownBuy()
        dealer = RecordDealer()
        session = SynchronousSession(config, dealer, opponents=0)
        try:
            session.accept_market("dealer_1", session.engine.market())
            session.advance(final=False)
            self.assertEqual(dealer.quote_portfolios, [(1, dict(A=0., B=0.)), (2, dict(A=0., B=0.))])
            self.assertEqual(dealer.fill_portfolios, [dict(A=-2., B=0.)])
            self.assertEqual(dealer.hedge_portfolios[-1], (2, dict(A=-2., B=0.)))
            self.assertEqual(session.engine.accounts["dealer_1"]["cash"], 100200.)
            self.assertEqual(session.engine.portfolio("dealer_1")["pnl"], -2.)
            self.assertEqual(session.observations[0]["dealer_signed_one_tick_markout"], -1.)
            self.assertTrue(session.observations[0]["own_quote_acknowledged"])
        finally:
            session.store.close()

    def test_official_124_row_schedule_one_final_observation_per_rfq(self):
        row, obs, _, rejects = one_session("B0", {}, KIT / "demo/demo_data.csv", 0, 42, collect=True)
        self.assertFalse(row["crash"], row.get("traceback"))
        self.assertEqual(row["price_rows_consumed"], 124)
        self.assertEqual(len(obs), 36)
        self.assertEqual(len({o["order_id"] for o in obs}), 36)
        self.assertEqual(rejects, [])
        # Independently matched original WebSocket run (parent's evidence).
        self.assertAlmostEqual(row["account"]["pnl"], -7.0132, places=7)
        self.assertEqual(row["account"]["customer_trades"], 8)
        self.assertEqual(row["account"]["exchange_trades"], 7)
        self.assertLess(row["reconciliation"]["max_absolute_difference"], 1e-7)
        self.assertGreater(max(row["max_abs_inventory"]), 0.)

    def test_logical_clock_preserves_decreasing_deadline_remaining(self):
        config = build_config(None, 0, 5, rows=[dict(A=100., B=200.)]*124)
        session = SynchronousSession(config, RecordDealer(), opponents=0)
        try:
            session.accept_market("dealer_1", session.engine.market())
            self.assertEqual(session.engine.closes_at-session.virtual_time, 5.)
            for _ in range(9):
                session.tick()
            self.assertEqual(session.engine.closes_at-session.virtual_time, .5)
            session.advance(final=True)
            self.assertEqual(session.engine.closes_at-session.virtual_time, 2.)
            session.tick()
            self.assertEqual(session.engine.closes_at-session.virtual_time, 1.5)
        finally:
            session.store.close()

    def test_exogenous_tapes_unchanged_by_candidate_quote_action(self):
        first, _, _, _ = one_session("B0", {}, KIT / "demo/demo_data.csv", 0, 42)
        second, _, _, _ = one_session("B0", {}, KIT / "demo/demo_data.csv", 0, 42)
        self.assertEqual(first["price_tape_sha256"], second["price_tape_sha256"])
        self.assertEqual(first["public_customer_tape_sha256"], second["public_customer_tape_sha256"])
        self.assertEqual(first["account"], second["account"])

    def test_equal_unique_quote_candidates_and_common_joint_grid(self):
        for method in ("B1", "B2", "B3", "B3_factor", "R2"):
            grid = candidate_grid(method)
            self.assertEqual(len(grid), 12)
            self.assertEqual(len({(g["base_edge"], g["inventory_skew"]) for g in grid}), 12)
        a, b = candidate_grid("B3"), candidate_grid("R2")
        self.assertEqual([{k: v for k, v in g.items() if k != "mode"} for g in a],
                         [{k: v for k, v in g.items() if k != "mode"} for g in b])

    def test_pairing_rejects_wrong_exogenous_history(self):
        rows = [dict(method=m, block=0, seed=1, crash=False, account=dict(pnl=0.),
                     price_tape_sha256=m, public_customer_tape_sha256="same")
                for m in ("R2", "B3")]
        with self.assertRaises(AssertionError):
            block_differences(rows, "R2", "B3")

    def test_calibration_features_exclude_future_price_and_hidden_limit(self):
        record = dict(edge=.1, product_id="B", side="BUY", quantity=4,
                      next_reference=200., dealer_signed_one_tick_markout=.4)
        changed = copy.deepcopy(record)
        changed.update(next_reference=9999., dealer_signed_one_tick_markout=-8000.,
                       hidden_limit=999., won=True)
        np.testing.assert_array_equal(features([record]), features([changed]))

    def test_monotone_fill_fit_pools_order_violation(self):
        result = monotone_decreasing([.2, .8, .1], [1., 1., 1.])
        np.testing.assert_allclose(result, [.5, .5, .1])


if __name__ == "__main__":
    unittest.main()
