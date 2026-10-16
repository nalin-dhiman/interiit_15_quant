"""Unknown transport state is not an acknowledged live quote book."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import itertools
import unittest
from unittest.mock import patch

from submission.sdk_dealer import Dealer
from tests_sdk.test_sdk_controller import market, order, portfolio


class SDKAcknowledgementStateTests(unittest.TestCase):
    def test_delayed_positive_replacement_counterexample(self):
        # Hand fixture: at match ref=99.93, hidden customer BUY limit=100.03.
        # Old live ask100.08 loses, pending ask100.01 would win. Treating the
        # requested replacement as accepted wrongly credits an offsetting fill.
        d = Dealer(mode="R2", band_fraction=.4, target_fraction=0,
                   risk_weight=.01, total_intervals=1, fill_probability=1,
                   fill_edge_decay=100, inventory_skew=0)
        orders = [order("pending", "BUY", 10)]
        old = market(orders, closes_at=1000.5)
        self.assertEqual(d.quote(orders, old, portfolio(A=10))["pending"], 100.08)
        new = market(orders, closes_at=1000.5)
        new["prices"]["A"] = 99.93
        self.assertEqual(d.quote(orders, new, portfolio(A=10))["pending"], 100.01)
        self.assertGreater(100.08, 99.93 + .10)
        self.assertLess(100.01, 99.93 + .10)
        with patch("submission.sdk_dealer.time.time", return_value=1000):
            legacy = d._continuation_values(10, [10, 0], "A", d._spec("A", new), new)
            self.assertAlmostEqual(legacy[10] - legacy[0], .6)
            self.assertEqual(d.hedge(new, portfolio(A=10)), [("A", "SELL", 10.)])
        diagnostic = d.last_decisions[-1]
        self.assertTrue(diagnostic["robust_current_fill_comparison"])
        self.assertEqual(diagnostic["current_fill_states"], 2)
        self.assertAlmostEqual(diagnostic["candidate_advantage_lower_bounds"][10.], -.4)
        self.assertEqual(d._records[(1, "pending")]["possible_live_quotes"],
                         [None, 100.08, 100.01])

    def test_successful_raw_ack_does_not_certify_request(self):
        d = Dealer(mode="R2")
        orders = [order("x")]
        d.quote(orders, market(orders), portfolio(A=10))
        d.on_ack({"action_id": "opaque-id", "status": "accepted"})
        self.assertFalse(d._records[(1, "x")]["quote_ack_verified"])
        self.assertEqual(d._current_fill_deltas("A", market(orders)), [0., 4.])

    def test_partial_rejection_preserves_old_and_pending_capacity(self):
        d = Dealer(mode="R2")
        orders = [order("old", quantity=15)]
        m = market(orders)
        d.quote(orders, m, portfolio())
        d.quote(orders, m, portfolio(A=15))  # None requests withdrawal.
        d.on_ack({"action_id": "partial", "status": "accepted",
                  "errors": {"old": "UNKNOWN_ORDER"}})
        self.assertIn((1, "old"), d._reserved)
        self.assertEqual(d._records[(1, "old")]["possible_live_quotes"], [None, 99.92])
        self.assertIsNone(d.quote([order("new", quantity=10)], m, portfolio())["new"])
        d.hedge(m, portfolio(A=15))
        self.assertTrue(d.last_decisions[-1]["quote_state_fallback"])
        self.assertEqual(d.last_decisions[-1]["candidate_values"], {})

    def test_duplicate_rejection_does_not_count_twice_or_release(self):
        d = Dealer()
        orders = [order("x")]
        d.quote(orders, market(orders), portfolio())
        ack = {"action_id": "same", "status": "rejected", "code": "STALE_QUOTE"}
        d.on_ack(ack)
        d.on_ack(dict(ack, duplicate=True))
        self.assertEqual(d.ack_rejections, 1)
        self.assertIn((1, "x"), d._reserved)

    def test_reconnect_indistinguishable_market_keeps_absence_and_capacity(self):
        d = Dealer(mode="R2")
        orders = [order("x", quantity=15)]
        m = market(orders)
        d.on_market(m, portfolio())
        d.quote(orders, m, portfolio())
        d.on_ack({"action_id": "old", "status": "accepted"})
        d.on_market(m, portfolio())  # Startup duplicate OR reconnect snapshot.
        d.on_ack({"action_id": "old", "status": "accepted", "duplicate": True})
        self.assertEqual(d.repeated_market_callbacks, 1)
        self.assertTrue(d._records[(1, "x")]["transport_reset_possible"])
        self.assertEqual(d._current_fill_deltas("A", m), [0., 15.])
        self.assertIsNone(d.quote([order("new", quantity=10)], m, portfolio())["new"])

    def test_boundary_stale_quote_ack_cannot_remove_old_fill_branch(self):
        d = Dealer(mode="R2")
        d.quote([order("old", "BUY", 15)], market(interval=1), portfolio())
        d.on_market(market(interval=2), portfolio())
        d.on_ack({"action_id": "late", "status": "rejected", "code": "STALE_QUOTE"})
        self.assertEqual(d._current_fill_deltas("A", market(interval=2)), [-15., 0.])
        d.on_customer_result({"interval_id": 1, "order_id": "old", "product_id": "A",
                              "status": "expired", "traded": False})
        self.assertEqual(d._current_fill_deltas("A", market(interval=2)), [0.])
        self.assertFalse(d._quote_state_uncertain)

    def test_all_fill_subsets_preserved_in_exact_lot_units(self):
        d = Dealer(mode="R2")
        orders = [order("a", "BUY", .1), order("b", "SELL", .2), order("c", "BUY", .4)]
        m = market(orders)
        m["products"]["A"]["lot_size"] = .1
        d.quote(orders, m, portfolio())
        expected = sorted({float(sum(flag * signed for flag, signed in zip(flags, (-1, 2, -4))) / 10)
                           for flags in itertools.product((0, 1), repeat=3)})
        self.assertEqual(d._current_fill_deltas("A", m), expected)

    def test_fill_enumeration_cap_falls_back_without_truncating(self):
        d = Dealer(mode="R2", max_current_fill_states=2, band_fraction=.5, target_fraction=0)
        orders = [order("a", "BUY", 2), order("b", "SELL", 3)]
        m = market(orders)
        d.quote(orders, m, portfolio(A=15))
        self.assertIsNone(d._current_fill_deltas("A", m))
        d.hedge(m, portfolio(A=15))
        self.assertTrue(d.last_decisions[-1]["fill_state_cap_fallback"])
        self.assertEqual(d.last_decisions[-1]["target"], d.last_decisions[-1]["baseline_target"])
        self.assertEqual(d.fill_state_cap_fallbacks, 1)

    def test_r2_fixed_book_omits_repricing_and_preserves_withdrawal(self):
        d = Dealer(mode="R2", risk_weight=0)
        orders = [order("old", "BUY", 4)]
        m = market(orders)
        d.quote(orders, m, portfolio(A=10))
        d.hedge(m, portfolio(A=10))
        changed = market(orders)
        changed["prices"]["A"] = 110.
        self.assertEqual(d.quote(orders, changed, portfolio(A=5)), {})
        self.assertEqual(d._records[(1, "old")]["final_requested_quote"], 100.06)
        self.assertIn((1, "old"), d._reserved)

    def test_b3_frozen_ablation_is_explicit_and_default_stays_repricing(self):
        orders = [order("old", "BUY", 4)]
        normal, frozen = Dealer(mode="B3"), Dealer(mode="B3", freeze_current_quotes=True)
        m = market(orders)
        for d in (normal, frozen):
            d.quote(orders, m, portfolio(A=10))
            d.hedge(m, portfolio(A=10))
        changed = market(orders)
        changed["prices"]["A"] = 110.
        self.assertIn("old", normal.quote(orders, changed, portfolio(A=5)))
        self.assertEqual(frozen.quote(orders, changed, portfolio(A=5)), {})

    def test_unfrozen_r2_uses_b3_when_quote_feedback_unknown(self):
        d = Dealer(mode="R2", freeze_current_quotes=False)
        orders = [order("x")]
        m = market(orders)
        d.quote(orders, m, portfolio(A=15))
        d.hedge(m, portfolio(A=15))
        diag = d.last_decisions[-1]
        self.assertTrue(diag["quote_feedback_fallback"])
        self.assertEqual(diag["candidate_values"], {})
        self.assertEqual(diag["target"], diag["baseline_target"])

    def test_robust_model_action_can_differ_from_b3_with_unknown_book(self):
        d = Dealer(mode="R2", risk_weight=0, band_fraction=.4, target_fraction=0,
                   total_intervals=1)
        orders = [order("x", "BUY", 10)]
        m = market(orders)
        d.quote(orders, m, portfolio(A=10))
        self.assertEqual(d.hedge(m, portfolio(A=10)), [])
        diag = d.last_decisions[-1]
        self.assertTrue(diag["quote_state_ambiguous"])
        self.assertFalse(diag["quote_state_fallback"])
        self.assertAlmostEqual(diag["candidate_advantage_lower_bounds"][10.], .1)
        self.assertEqual(d.robust_r2_improvements, 1)

    def test_explicit_withdrawal_is_never_changed_to_old_offer(self):
        d = Dealer(mode="R2")
        orders = [order("x", quantity=15)]
        m = market(orders)
        d.quote(orders, m, portfolio())
        self.assertEqual(d.quote(orders, m, portfolio(A=15)), {"x": None})
        d.hedge(m, portfolio(A=15))
        self.assertEqual(d.quote(orders, m, portfolio()), {})
        self.assertIsNone(d._records[(1, "x")]["final_requested_quote"])
        self.assertIn((1, "x"), d._reserved)

    def test_closing_does_not_value_unreceived_customer_fill_as_absent(self):
        d = Dealer(mode="R2", risk_weight=0, settlement_coefficients={"B": {"A": 2}})
        orders = [order("last", "BUY", 4, "B")]
        d.quote(orders, market(orders), portfolio(B=10))
        self.assertEqual(d.hedge(market(closing=True), portfolio(B=10)), [])
        self.assertTrue(d.last_decisions[-1]["closing_book_fallback"])
        self.assertEqual(d.last_decisions[-1]["candidate_values"], {})
        d.on_customer_result({"interval_id": 1, "order_id": "last", "product_id": "B",
                              "status": "expired", "traded": False})
        self.assertEqual(d.hedge(market(closing=True), portfolio(B=10)), [("B", "SELL", 10.)])

    def test_fractional_lot_rfqs_trigger_fallback_without_support_rounding(self):
        d = Dealer(mode="R2")
        orders = [order("invalid", "BUY", .15)]
        m = market(orders)
        m["products"]["A"]["lot_size"] = .1
        d.quote(orders, m, portfolio(A=10))
        self.assertIsNone(d._current_fill_deltas("A", m))
        d.hedge(m, portfolio(A=10))
        self.assertTrue(d.last_decisions[-1]["quote_state_fallback"])


if __name__ == "__main__":
    unittest.main()
