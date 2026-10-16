"""SDK controller regressions; model identities do not establish market alpha."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import itertools
import math
import unittest

from submission.sdk_dealer import Dealer


def market(orders=(), closing=False, interval=1, **changes):
    result = {"interval_id": interval, "price_version": interval * 10, "interval_seconds": 5.,
              "closing": closing, "active_orders": list(orders), "prices": {"A": 100., "B": 201.},
              "products": {pid: {"tick_size": .01, "lot_size": 1., "multiplier": 1.,
                                    "max_long": 20., "max_short": 20., "fee_per_unit": .01,
                                    "fee_rate": 0., "breach_penalty_per_unit": 1.,
                                    "tradable_against_exchange": True} for pid in ("A", "B")}}
    result.update(changes)
    return result


def portfolio(**positions):
    return {"positions": {"A": positions.get("A", 0.), "B": positions.get("B", 0.)}, "cash": 100000.}


def order(oid, side="SELL", quantity=4., product="A"):
    return {"order_id": oid, "product_id": product, "side": side, "quantity": quantity}


class SDKControllerTests(unittest.TestCase):
    def test_customer_direction_and_rounding(self):
        d = Dealer(mode="B1", base_edge=.081, inventory_skew=0)
        orders = [order("buy", "BUY"), order("sell", "SELL")]
        self.assertEqual(d.quote(orders, market(orders), portfolio()), {"buy": 100.09, "sell": 99.91})

    def test_all_subsets_no_opposite_netting(self):
        orders = [order("s1", quantity=15), order("b1", "BUY", 15), order("s2", quantity=10)]
        d = Dealer()
        quotes = d.quote(orders, market(orders), portfolio())
        self.assertIsNotNone(quotes["s1"])
        self.assertIsNotNone(quotes["b1"])
        self.assertIsNone(quotes["s2"])
        live = [o for o in orders if quotes[o["order_id"]] is not None]
        for flags in itertools.product((0, 1), repeat=len(live)):
            q = sum(flag * o["quantity"] * (-1 if o["side"] == "BUY" else 1)
                    for flag, o in zip(flags, live))
            self.assertLessEqual(abs(q), 20)

    def test_old_interval_unknown_fills_reserved(self):
        d = Dealer()
        old = order("old", quantity=15)
        d.quote([old], market([old]), portfolio())
        new = order("new", quantity=10)
        self.assertIsNone(d.quote([new], market([new], interval=2), portfolio())["new"])
        d.on_fill({"source": "customer", "interval_id": 1, "order_id": "old", "product_id": "A",
                   "side": "BUY", "quantity": 15, "price": 99.9}, portfolio(A=15))
        d.on_customer_result({"interval_id": 1, "order_id": "old", "status": "filled", "traded": True})
        self.assertIsNone(d.quote([new], market([new], interval=2), portfolio(A=15))["new"])

    def test_ack_does_not_release_failed_withdrawal(self):
        d = Dealer()
        old = order("old", quantity=15)
        m = market([old])
        d.quote([old], m, portfolio())
        self.assertIsNone(d.quote([old], m, portfolio(A=15))["old"])
        d.on_ack({"status": "accepted", "errors": {"old": "STALE_QUOTE"}})
        new = order("new", quantity=10)
        self.assertIsNone(d.quote([new], market([new]), portfolio())["new"])

    def test_pending_hedge_not_capacity(self):
        d = Dealer(mode="B2")
        self.assertEqual(d.hedge(market(), portfolio(A=15)), [("A", "SELL", 15.)])
        buy = order("buy", "BUY", 10)
        # Stale +15 portfolio, but a possible unconfirmed sale takes it to zero.
        # 15 of sale + a 10 customer BUY would exceed the short limit if stale
        # position were already zero; never allocate from a pending sale.
        quotes = d.quote([buy], market([buy]), portfolio(A=0))
        self.assertIsNone(quotes["buy"])

    def test_r2_unknown_live_quote_uses_conventional_fallback(self):
        d = Dealer(mode="R2", band_fraction=.5, target_fraction=0, risk_weight=0)
        orders = [order("x", "BUY", 5)]
        d.quote(orders, market(orders), portfolio(A=15))
        d.on_ack({"status": "accepted", "errors": {"x": "BAD_TICK"}})
        d.hedge(market(orders), portfolio(A=15))
        diagnostic = d.last_decisions[-1]
        self.assertTrue(diagnostic["quote_state_fallback"])
        self.assertEqual(diagnostic["candidate_values"], {})
        self.assertEqual(diagnostic["target"], diagnostic["baseline_target"])

    def test_hedge_preserves_live_subset_capacity(self):
        d = Dealer(mode="B2")
        buy = order("buy", "BUY", 30)
        d.quote([buy], market([buy]), portfolio(A=15))
        self.assertEqual(d.hedge(market([buy]), portfolio(A=15)), [])

    def test_breached_recovery_is_real_lot_transaction(self):
        d = Dealer(mode="B2")
        self.assertEqual(d.hedge(market(), portfolio(A=31)), [("A", "SELL", 31.)])
        self.assertEqual(d.quote([order("s")], market(), portfolio(A=31)), {"s": None})

    def test_nontradable_no_exchange_action(self):
        m = market()
        m["products"]["A"]["tradable_against_exchange"] = False
        self.assertEqual(Dealer(mode="B2").hedge(m, portfolio(A=20)), [])

    def test_fee_multiplier_and_rate(self):
        d, m = Dealer(), market()
        m["products"]["A"].update(multiplier=5., fee_per_unit=.02, fee_rate=.001)
        self.assertAlmostEqual(d._fee(3, d._spec("A", m)), 3 * (.02 + 100 * 5 * .001))

    def test_lot_targets_recomputed(self):
        m = market()
        m["products"]["A"]["lot_size"] = 3
        d = Dealer(mode="R2", risk_weight=0, fill_probability=0, total_intervals=1)
        values = d._continuation_values(10, [10, 1, 4], "A", d._spec("A", m), m)
        self.assertAlmostEqual(values[1], -.09)
        self.assertAlmostEqual(values[4], -.06)
        for trade in d.hedge(m, portfolio(A=10)):
            self.assertEqual(trade[2] % 3, 0)

    def test_terminal_residual_exact_published_graph(self):
        d = Dealer(mode="R2", settlement_coefficients={"B": {"A": 2}}, risk_weight=0)
        m = market(closing=True)
        values = d._continuation_values(10, [10, 0], "B", d._spec("B", m), m)
        self.assertEqual(values[10], -10.)
        self.assertAlmostEqual(values[0], -.1)
        self.assertEqual(d.hedge(m, portfolio(B=10)), [("B", "SELL", 10.)])

    def test_no_forced_liquidation_last_price(self):
        m = market(closing=True)
        for mode in ("B1", "B3", "R2"):
            self.assertEqual(Dealer(mode=mode, risk_weight=0).hedge(m, portfolio(A=20)), [])

    def test_b0_matches_provided_template_actions(self):
        d = Dealer(mode="B0", base_edge=.10)
        orders = [order("b", "BUY", 40), order("s", "SELL", 40)]
        self.assertEqual(d.quote(orders, market(orders), portfolio(A=20)), {"b": 100.1, "s": 99.9})
        self.assertEqual(d.hedge(market(orders), portfolio(A=20)), [("A", "SELL", 20.)])

    def test_one_final_observation_not_callbacks(self):
        d = Dealer()
        orders = [order("x")]
        for _ in range(20):
            d.quote(orders, market(orders), portfolio())
        result = {"interval_id": 1, "order_id": "x", "status": "filled", "winner": "other",
                  "traded": False, "price": 99.95}
        d.on_customer_result(result)
        d.on_customer_result(result)
        self.assertEqual(len(d.observations), 1)
        self.assertFalse(d.observations[0]["quote_ack_verified"])

    def test_unknown_adaptation_falls_back(self):
        d = Dealer(mode="R3")
        self.assertEqual(d.mode, "B3")
        self.assertFalse(d.adaptation_enabled)

    def test_no_future_or_latent_features_used(self):
        d, d2 = Dealer(), Dealer()
        orders = [order("x")]
        tainted = dict(orders[0], reservation_price=1e9, limit_offset=1e9, latent_informed=True)
        self.assertEqual(d.quote(orders, market(orders), portfolio()),
                         d2.quote([tainted], market([tainted]), portfolio()))

    def test_malformed_empty_callbacks(self):
        d = Dealer()
        for m, p in (({}, {}), (None, None), ({"prices": {}}, {"positions": {"x": math.nan}})):
            self.assertEqual(d.quote([], m, p), {})
            self.assertEqual(d.hedge(m, p), [])
        self.assertEqual(d.quote([{}, None, {"order_id": "x", "quantity": math.inf}], {}, {}), {"x": None})

    def test_partial_band_and_candidate_inclusion(self):
        d = Dealer(mode="R2", band_fraction=.5, target_fraction=.5)
        d.hedge(market(), portfolio(A=15))
        diagnostic = d.last_decisions[-1]
        values = diagnostic["candidate_values"]
        self.assertIn(15, values)
        self.assertIn(diagnostic["baseline_target"], values)
        self.assertGreaterEqual(values[diagnostic["target"]] + d.min_improvement,
                                values[diagnostic["baseline_target"]] - 1e-10)
        self.assertFalse(diagnostic["error_bound_certified"])

    def test_observed_offsetting_rfq_can_save_reset_fee(self):
        d = Dealer(mode="R2", band_fraction=.4, target_fraction=0, risk_weight=0,
                   total_intervals=1, fill_probability=1, fill_edge_decay=0)
        orders = [order("offset", "BUY", 10)]
        m = market(orders)
        d.quote(orders, m, portfolio(A=10))
        self.assertEqual(d.hedge(m, portfolio(A=10)), [])
        values = d.last_decisions[-1]["candidate_values"]
        self.assertGreater(values[10], values[0])

    def test_exact_tiny_future_enumeration(self):
        # Independent depth-two enumeration of the explicitly assumed kernel.
        d = Dealer(mode="R2", risk_weight=.003, total_intervals=3, horizon=3,
                   band_fraction=1, target_fraction=0, future_size=2,
                   fill_probability=.2, future_orders_per_product=1, inventory_skew=0,
                   fill_edge_decay=0)
        m = market()
        spec = d._spec("A", m)
        actual = d._continuation_values(0, [0], "A", spec, m)[0]
        expected = 0.
        # Current observed batch is empty; then two future RFQ events.
        for path in itertools.product((0, -2, 2), repeat=2):
            probability, q, payoff = 1., 0., 0.
            for delta in path:
                probability *= .8 if delta == 0 else .1
                old_q = q
                q += delta
                payoff += (abs(delta) * .08) - .003 * (old_q * old_q * 4.5 + q * q * .5)
            expected += probability * payoff
        self.assertAlmostEqual(actual, expected, places=10)

    def test_hysteresis_retains_noaction_near_boundary(self):
        d = Dealer(mode="R2", risk_weight=0, min_improvement=1,
                   settlement_coefficients={"B": {"A": 2}})
        m = market(closing=True)
        m["prices"]["B"] = 200.011
        for _ in range(20):
            self.assertEqual(d.hedge(m, portfolio(B=10)), [])

    def test_factor_band_uses_published_root_and_individual_limits(self):
        d = Dealer(mode="B3", factor_hedge=True, band_fraction=.5, target_fraction=0,
                   settlement_coefficients={"B": {"A": 2}})
        m = market()
        # B alone is below its own band; its A-equivalent reaches A's band.
        self.assertEqual(d.hedge(m, portfolio(B=8)), [("A", "SELL", 16.)])
        d2 = Dealer(mode="B3", factor_hedge=True, risk_weight=0,
                    settlement_coefficients={"B": {"A": 2}})
        m["prices"]["B"] = 200
        self.assertEqual(d2.hedge(dict(m, closing=True), portfolio(A=-16, B=8)), [])

    def test_1000_products_and_quotes_plain_types(self):
        m = market()
        spec = m["products"]["A"]
        m["products"] = {str(i): dict(spec) for i in range(1000)}
        m["prices"] = {str(i): 100. for i in range(1000)}
        p = {"positions": {str(i): 15. for i in range(1000)}}
        orders = [order(str(i), "BUY", 4, str(i)) for i in range(1000)]
        d = Dealer(mode="R2", observation_limit=10, diagnostic_limit=5)
        quotes = d.quote(orders, m, p)
        trades = d.hedge(m, p)
        self.assertEqual(len(quotes), 1000)
        self.assertEqual(d.callback_errors, 0)
        self.assertLessEqual(len(d.last_decisions), 5)
        self.assertTrue(all(x["load_fallback"] for x in d.last_decisions))
        self.assertTrue(all(x["candidate_values"] == {} for x in d.last_decisions))
        self.assertTrue(all(type(value) in (float, type(None)) for value in quotes.values()))
        self.assertTrue(all(type(trade[2]) is float for trade in trades))


if __name__ == "__main__":
    unittest.main()
