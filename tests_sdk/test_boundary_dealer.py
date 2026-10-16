"""Sequential boundary regression cases; none establishes market improvement."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import itertools
import unittest
from copy import deepcopy
from unittest.mock import patch

from submission.boundary_dealer import Dealer
from submission.sdk_dealer import Dealer as Original
from tests_sdk.test_sdk_controller import market, order


def account(m, **positions):
    return {"positions": {pid: float(positions.get(pid, 0)) for pid in m["products"]},
            "cash": 100000., "customer_trades": 0, "exchange_trades": 0,
            "fees_paid": 0., "penalties_paid": 0.,
            "interval_id": m["interval_id"], "price_version": m["price_version"]}


def apply(p, fill, m):
    p = deepcopy(p)
    signed = fill["quantity"] * (1 if fill["side"] == "BUY" else -1)
    p["positions"][fill["product_id"]] += signed
    p["cash"] -= signed * fill["price"] * m["products"][fill["product_id"]]["multiplier"] + fill["fee"]
    p["fees_paid"] += fill["fee"]
    p["customer_trades" if fill["source"] == "customer" else "exchange_trades"] += 1
    return p


def customer(o, interval=1):
    return {"source": "customer", "interval_id": interval, "order_id": o["order_id"],
            "product_id": o["product_id"], "side": "SELL" if o["side"] == "BUY" else "BUY",
            "quantity": o["quantity"], "price": 100., "fee": 0.}


def result(o, won=False, interval=1):
    return {"interval_id": interval, "order_id": o["order_id"], "product_id": o["product_id"],
            "quantity": o["quantity"], "status": "filled" if won else "expired", "traded": won}


class BoundaryDealerTests(unittest.TestCase):
    def start(self, orders=(), **kwargs):
        d = Dealer(boundary_reset=True, band_fraction=.6, target_fraction=0,
                   risk_weight=0, **kwargs)
        m = market(orders)
        p = account(m)
        d.on_market(m, p)
        self.assertEqual(d.quote(orders, m, p), {o["order_id"]: None for o in orders})
        self.assertEqual(d.hedge(m, p), [])
        self.assertEqual(d.boundary_state, "open")
        d.quote(orders, m, p)
        return d, m, p

    def boundary(self, d, old, p, new=(), fills=(), closing=False):
        m = market(new, interval=1 if closing else 2, closing=closing)
        if closing:
            m["price_version"] += 1
        d.on_market(m, p)
        self.assertEqual(d.quote(new, m, p), {o["order_id"]: None for o in new})
        for o in fills:
            f = customer(o)
            p = apply(p, f, m)
            d.on_fill(f, p)
        for o in old:
            d.on_customer_result(result(o, o in fills))
        p.update(interval_id=m["interval_id"], price_version=m["price_version"])
        return m, p

    def complete(self, d, trades, p, m, rejected=()):
        executions = []
        for i, (pid, side, qty) in enumerate(trades):
            if i in rejected:
                continue
            spec = d._spec(pid, m)
            f = {"source": "exchange", "interval_id": m["interval_id"],
                 "price_version": m["price_version"], "product_id": pid,
                 "side": side, "quantity": qty, "price": m["prices"][pid],
                 "fee": d._fee(qty, spec), "action_id": "reset:" + str(i)}
            p = apply(p, f, m)
            executions.append(f)
        for f in executions:
            d.on_fill(f, p)
        ack = {"action_id": "reset", "status": "accepted", "executions": executions}
        if rejected:
            ack["errors"] = {str(i): "BAD_LOT" for i in rejected}
        d.on_ack(ack)
        return p, ack

    def test_default_is_original_conventional_dealer(self):
        m = market([order("x")])
        p = account(m, A=10)
        a, b = Dealer(), Original()
        self.assertEqual(a.quote(m["active_orders"], m, p), b.quote(m["active_orders"], m, p))
        self.assertEqual(a.hedge(m, p), b.hedge(m, p))
        self.assertFalse(a.adaptation_enabled)

    def test_new_quotes_wait_for_old_outcomes_and_final_account(self):
        old, new = [order("old", quantity=15)], [order("new", quantity=10)]
        d, _, p = self.start(old)
        m = market(new, interval=2)
        d.on_market(m, p)
        self.assertEqual(d.quote(new, m, p), {"new": None})
        f = customer(old[0]); p = apply(p, f, m); d.on_fill(f, p)
        self.assertFalse(d._reserved)  # This alone is NOT the barrier.
        final = dict(p, interval_id=2, price_version=m["price_version"])
        self.assertEqual(d.hedge(m, final), [])
        d.on_customer_result(result(old[0], True))
        self.assertEqual(d.hedge(m, p), [])  # Runner synthetic fill has old tags.
        trades = d.hedge(m, final)
        self.assertEqual(trades, [("A", "SELL", 15.)])
        self.assertEqual(d.quote(new, m, final), {"new": None})
        self.assertEqual(d.hedge(m, final), [])
        p, _ = self.complete(d, trades, final, m)
        self.assertEqual(d.boundary_state, "open")
        self.assertIsNotNone(d.quote(new, m, p)["new"])

    def test_every_public_rfq_result_required_even_if_never_offered(self):
        old = [order("first", quantity=15), order("not-offered", quantity=15)]
        d, _, p = self.start(old)
        self.assertNotIn((1, "not-offered"), d._reserved)
        m = market(interval=2); d.on_market(m, p)
        d.on_customer_result(result(old[0]))
        p.update(interval_id=2, price_version=m["price_version"])
        self.assertEqual(d.hedge(m, p), [])
        self.assertEqual(d.boundary_state, "reconciling")
        d.on_customer_result(result(old[1]))
        self.assertEqual(d.hedge(m, p), [])
        self.assertEqual(d.boundary_state, "open")

    def test_generic_ack_cannot_release_unconfirmed_capacity(self):
        old, new = [order("old", quantity=15)], [order("new", quantity=10)]
        d, _, p = self.start(old)
        m, p = self.boundary(d, old, p, new, old)
        trades = d.hedge(m, p)
        d.on_ack({"action_id": "quote-ack", "status": "accepted"})
        self.assertEqual(d.boundary_state, "pending_reset")
        self.assertEqual(d.quote(new, m, p), {"new": None})
        self.assertEqual(d._boundary_account["positions"]["A"], 15.)
        p, _ = self.complete(d, trades, p, m)
        self.assertEqual(p["positions"]["A"], 0.)

    def test_partial_batch_opens_using_actual_positions_not_requested_targets(self):
        old = [order("a", quantity=15), order("b", quantity=15, product="B")]
        d, _, p = self.start(old)
        m, p = self.boundary(d, old, p, fills=old)
        trades = d.hedge(m, p)
        self.assertEqual(len(trades), 2)
        p, ack = self.complete(d, trades, p, m, rejected=(1,))
        self.assertEqual(d.boundary_state, "open")
        self.assertEqual(d._boundary_account["positions"], {"A": 0., "B": 15.})
        self.assertEqual(d.boundary_partial_batches, 1)
        cash = d._boundary_account["cash"]
        d.on_ack(dict(ack, duplicate=True))
        d.on_fill(ack["executions"][0], p)
        self.assertEqual(d._boundary_account["cash"], cash)
        self.assertEqual(d.boundary_duplicate_events, 2)

    def test_full_reject_cannot_be_correlated_and_fails_closed(self):
        old = [order("old", quantity=15)]
        d, _, p = self.start(old)
        m, p = self.boundary(d, old, p, fills=old)
        d.hedge(m, p)
        d.on_ack({"action_id": "opaque", "status": "rejected", "code": "STALE_PRICE"})
        self.assertEqual(d.boundary_state, "unsupported")
        self.assertIsNotNone(d._boundary_pending)

    def test_reordered_result_without_own_fill_fails_closed(self):
        old = [order("old")]; d, _, p = self.start(old)
        m = market(interval=2); d.on_market(m, p)
        d.on_customer_result(result(old[0], True))
        self.assertEqual(d.boundary_state, "unsupported")

    def test_cash_and_counter_mismatch_fails_closed(self):
        for field, change in (("cash", 1.), ("customer_trades", 1), ("exchange_trades", 1)):
            with self.subTest(field=field):
                d, _, p = self.start()
                m = market(interval=2); d.on_market(m, p)
                p.update(interval_id=2, price_version=m["price_version"])
                p[field] += change
                self.assertEqual(d.hedge(m, p), [])
                self.assertEqual(d.boundary_state, "unsupported")

    def test_explicit_penalty_counter_reconciles_without_hiding_charge(self):
        d, _, p = self.start()
        m = market(interval=2); d.on_market(m, p)
        p.update(interval_id=2, price_version=m["price_version"])
        p["cash"] -= 3.; p["penalties_paid"] += 3.
        d.hedge(m, p)
        self.assertEqual(d.boundary_state, "open")
        self.assertEqual(d._boundary_account["penalties_paid"], 3.)

    def test_closing_same_interval_reconciles_and_does_not_force_liquidation(self):
        old = [order("last", quantity=4)]
        d, _, p = self.start(old)
        m, p = self.boundary(d, old, p, fills=old, closing=True)
        self.assertEqual(d._boundary_epoch, (1, True))
        self.assertEqual(d.hedge(m, p), [])
        self.assertEqual(p["positions"]["A"], 4.)
        self.assertEqual(d.boundary_epochs, 2)

    def test_resume_reprices_on_every_callback(self):
        orders = [order("x")]; d, m, p = self.start(orders)
        before = d.quote(orders, m, p)["x"]
        m = deepcopy(m); m["prices"]["A"] += 2; m["price_version"] += 1
        after = d.quote(orders, m, p)["x"]
        self.assertEqual(after - before, 2.)
        self.assertFalse(d.freeze_current_quotes)

    def test_reconnect_with_live_book_does_not_forget_results_or_reservations(self):
        orders = [order("x")]; d, m, p = self.start(orders)
        d.on_market(m, p)
        self.assertEqual(d.boundary_state, "unsupported")
        self.assertIn((1, "x"), d._reserved)
        self.assertEqual(d.quote(orders, m, p), {"x": None})

    def test_missing_results_are_not_cleared_at_next_interval(self):
        orders = [order("x")]; d, _, p = self.start(orders)
        d.on_market(market(interval=2), p)
        d.on_market(market(interval=3), p)
        self.assertEqual(d.boundary_state, "unsupported")
        self.assertIn((1, "x"), d._boundary_old)

    def test_malformed_execution_and_ack_never_raise_or_open(self):
        old = [order("old", quantity=15)]
        d, _, p = self.start(old); m, p = self.boundary(d, old, p, fills=old)
        d.hedge(m, p)
        d.on_fill({"source": "exchange", "action_id": "bad"}, p)
        self.assertEqual(d.boundary_state, "unsupported")

    def test_duplicate_customer_fill_and_result_do_not_double_book(self):
        old = [order("old", quantity=4)]; d, _, p = self.start(old)
        m = market(interval=2); d.on_market(m, p)
        f = customer(old[0]); p = apply(p, f, m)
        d.on_fill(f, p); d.on_fill(f, p)
        d.on_customer_result(result(old[0], True)); d.on_customer_result(result(old[0], True))
        self.assertEqual(d._boundary_account["positions"]["A"], 4.)
        self.assertEqual(d._boundary_account["customer_trades"], 1)
        self.assertEqual(d.boundary_duplicate_events, 2)

    def test_fractional_lot_execution_uses_actual_quantity_and_fees(self):
        d = Dealer(boundary_reset=True, band_fraction=.4, target_fraction=.5)
        m = market(); m["products"]["A"]["lot_size"] = .1
        m["products"]["A"]["max_long"] = 1.
        m["products"]["A"]["max_short"] = 1.
        p = account(m, A=.7); d.on_market(m, p)
        trades = d.hedge(m, p)
        # The organiser's toward-zero helper sees .2-.7 below .5 in binary
        # arithmetic, so the core correctly values/emits the actual .4 lot
        # quantity and .3 residual, rather than claiming the requested .2.
        self.assertEqual(trades, [("A", "SELL", .4)])
        p, _ = self.complete(d, trades, p, m)
        self.assertAlmostEqual(d._boundary_account["positions"]["A"], .3)
        self.assertAlmostEqual(p["fees_paid"], .004)

    def test_upcoming_batch_kernel_matches_independent_path_enumeration(self):
        orders = [order("a", "BUY", 3), order("b", "SELL", 2), order("c", "BUY", 1)]
        d, m, _ = self.start(orders, mode="R2", inventory_skew=.003, fill_probability=.4)
        spec = d._spec("A", m)
        for z in (-10., 0., 10.):
            actual = d._boundary_batch_outcomes(z, "A", spec, m)
            hand = {}
            for flags in itertools.product((0, 1), repeat=3):
                weight, delta, spread = 1., 0., 0.
                for flag, o in zip(flags, orders):
                    direction = 1 if o["side"] == "BUY" else -1
                    quote = d._edge(o["side"], z, spec[1], spec[2], "A", m)
                    margin = direction * (quote - spec[1])
                    prob = .4 * __import__("math").exp(-12 * (margin - d.base_edge))
                    prob = min(1., max(0., prob))
                    weight *= prob if flag else 1 - prob
                    delta -= flag * direction * o["quantity"]
                    spread += flag * o["quantity"] * margin
                mass, reward = hand.get(delta, (0., 0.))
                hand[delta] = mass + weight, reward + weight * spread
            self.assertAlmostEqual(sum(w for w, _, _ in actual), 1.)
            for weight, delta, spread in actual:
                self.assertAlmostEqual(weight, hand[delta][0])
                self.assertAlmostEqual(weight * spread, hand[delta][1])

    def test_candidate_inventory_changes_both_admission_and_quote_reward(self):
        orders = [order("a", "SELL", 10)]
        d, m, _ = self.start(orders, mode="R2", inventory_skew=.01, fill_probability=.5)
        spec = d._spec("A", m)
        self.assertEqual(d._boundary_batch_outcomes(15, "A", spec, m), [(1., 0., 0.)])
        self.assertGreater(len(d._boundary_batch_outcomes(0, "A", spec, m)), 1)
        zero = d._boundary_batch_outcomes(0, "A", spec, m)
        five = d._boundary_batch_outcomes(5, "A", spec, m)
        self.assertNotEqual(zero, five)

    def test_model_cap_returns_exact_b3_target_without_truncation(self):
        orders = [order("a"), order("b", "BUY")]
        d = Dealer(boundary_reset=True, mode="R2", max_boundary_orders=1,
                   band_fraction=.5, target_fraction=0)
        m = market(orders); p = account(m, A=15); d.on_market(m, p)
        self.assertEqual(d.hedge(m, p), [("A", "SELL", 15.)])
        self.assertTrue(d.last_decisions[0]["model_fallback"])
        self.assertEqual(d.last_decisions[0]["candidate_values"], {})

    def test_candidate_comparison_baseline_wins_tie_and_recomputes_actual_values(self):
        d = Dealer(boundary_reset=True, mode="R2", risk_weight=0, band_fraction=.5,
                   target_fraction=0, fill_probability=0, total_intervals=1,
                   min_improvement=0)
        m = market(); m["products"]["A"]["fee_per_unit"] = 0
        p = account(m, A=15); d.on_market(m, p)
        with patch("submission.boundary_dealer.time.time", return_value=1000):
            trades = d.hedge(m, p)
        self.assertEqual(trades, [("A", "SELL", 15.)])
        self.assertEqual(d.last_decisions[0]["value_gap"], 0.)

    def test_tiny_one_interval_candidate_values_equal_direct_cash_paths(self):
        orders = [order("a", "BUY", 10), order("b", "SELL", 8)]
        d = Dealer(boundary_reset=True, mode="R2", risk_weight=0, band_fraction=.5,
                   target_fraction=0, fill_probability=.25, inventory_skew=.003,
                   total_intervals=1, min_improvement=0)
        m = market(orders); p = account(m, A=15); d.on_market(m, p)
        with patch("submission.boundary_dealer.time.time", return_value=1000):
            d.hedge(m, p)
        values = d.last_decisions[0]["candidate_values"]
        for z, actual in values.items():
            # Independent two-event tree: public lot/tick helpers, the declared
            # model probability formula, exact customer cash and current mark.
            neg = pos = 0.
            eligible = []
            for o in orders:
                direction = 1 if o["side"] == "BUY" else -1
                if z - neg - (o["quantity"] if direction == 1 else 0) < -20:
                    continue
                if z + pos + (o["quantity"] if direction == -1 else 0) > 20:
                    continue
                from rfq.sdk import round_quote
                quote = round_quote(100 + direction * .08 - .003 * z, .01, o["side"])
                margin = direction * (quote - 100)
                prob = min(1., .25 * __import__("math").exp(-12 * (margin - .08)))
                eligible.append((o, direction, quote, prob))
                neg += o["quantity"] if direction == 1 else 0
                pos += o["quantity"] if direction == -1 else 0
            expected = 0.
            for flags in itertools.product((0, 1), repeat=len(eligible)):
                weight, cash, q = 1., -(z - 15) * 100 - abs(z - 15) * .01, z
                for flag, (o, direction, quote, prob) in zip(flags, eligible):
                    weight *= prob if flag else 1 - prob
                    if flag:
                        delta = -direction * o["quantity"]
                        cash -= delta * quote
                        q += delta
                expected += weight * (cash + q * 100 - 15 * 100)
            self.assertAlmostEqual(actual, expected, places=10)

    def test_unsupported_is_absorbing_even_after_late_valid_ack(self):
        old = [order("old", quantity=15)]
        d, _, p = self.start(old); m, p = self.boundary(d, old, p, fills=old)
        trades = d.hedge(m, p)
        d._boundary_fail("injected unsupported history")
        self.complete(d, trades, p, m)
        self.assertEqual(d.boundary_state, "unsupported")

    def test_thousand_product_reset_caps_one_batch_and_reconciles_hand_cash(self):
        for mode in ("B3", "R2"):
            with self.subTest(mode=mode):
                d = Dealer(boundary_reset=True, mode=mode, band_fraction=.5,
                           target_fraction=0, diagnostic_limit=1000)
                m = market()
                template = dict(m["products"]["A"])
                ids = ["P" + str(i).zfill(4) for i in range(1000)]
                m["products"] = {pid: dict(template) for pid in ids}
                m["prices"] = {pid: 100. for pid in ids}
                p = account(m, **{pid: 20. if i % 2 else -20. for i, pid in enumerate(ids)})
                d.on_market(m, p)
                trades = d.hedge(m, p)
                self.assertEqual(len(trades), 500)
                self.assertEqual([item[0] for item in trades], ids[:500])
                self.assertEqual(d.boundary_counters["deferred_items"], 500)
                self.assertEqual(d.boundary_state, "pending_reset")
                for diag in d.last_decisions:
                    if diag["product_id"] in ids[500:]:
                        self.assertEqual(diag["target"], p["positions"][diag["product_id"]])
                        self.assertTrue(diag["scheduling_deferred"])
                p, ack = self.complete(d, trades, p, m)
                self.assertEqual(d.boundary_state, "open")
                self.assertAlmostEqual(p["cash"], 100000. - 500 * 20 * .01)
                self.assertAlmostEqual(p["fees_paid"], 100.)
                self.assertEqual(p["exchange_trades"], 500)
                self.assertTrue(all(p["positions"][pid] == 0 for pid in ids[:500]))
                self.assertTrue(all(abs(p["positions"][pid]) == 20 for pid in ids[500:]))
                self.assertEqual(d.hedge(m, p), [])
                # Enumerate all eight customer-fill subsets over three retained
                # products; capacity is based on actual q, never unsent target.
                orders = [order("buy", "BUY", 4, ids[500]),
                          order("sell", "SELL", 4, ids[501]),
                          order("toward", "SELL", 4, ids[502])]
                m["active_orders"] = orders
                d._boundary_orders = {o["order_id"]: o for o in orders}
                quotes = d.quote(orders, m, p)
                admitted = [o for o in orders if quotes[o["order_id"]] is not None]
                for flags in itertools.product((0, 1), repeat=len(admitted)):
                    q = dict(p["positions"])
                    for flag, o in zip(flags, admitted):
                        q[o["product_id"]] += flag * o["quantity"] * (-1 if o["side"] == "BUY" else 1)
                    self.assertTrue(all(-20 <= z <= 20 for z in q.values()))

    def test_batch_scheduler_prioritises_breach_before_absolute_exposure(self):
        d = Dealer(boundary_reset=True, band_fraction=.5, target_fraction=0)
        m = market(); template = dict(m["products"]["A"])
        ids = ["P" + str(i).zfill(4) for i in range(501)]
        m["products"] = {pid: dict(template) for pid in ids}
        m["prices"] = {pid: 100. for pid in ids}
        p = account(m, **{pid: 20. for pid in ids})
        p["positions"][ids[-1]] = 21.
        d.on_market(m, p); trades = d.hedge(m, p)
        self.assertEqual(trades[0], (ids[-1], "SELL", 21.))
        self.assertNotIn(ids[-2], {item[0] for item in trades})


if __name__ == "__main__":
    unittest.main()
