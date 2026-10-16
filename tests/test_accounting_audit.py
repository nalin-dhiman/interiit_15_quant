"""Independent numerical economics fixtures; not SDK conformance tests.

The constants below are calculated directly from cash principal, contractual
multiplier, fees and marked residual holdings. No organiser data are assumed.
"""
from __future__ import annotations

import itertools
import unittest
import numpy as np

from resetmm.accounting import Ledger, Product, round_price
from resetmm.inventory_mdp import build_inventory_mdp
from resetmm.risk import reserve_batch
from resetmm.settlement import resolve
from resetmm.toy_exchange import PrivateRFQ, QuoteBook, Scenario, match, run


class IndependentAccountingAudit(unittest.TestCase):
    def test_customer_sell_principal_and_multiplier(self):
        ledger = Ledger({'A': Product('A', lot=2, multiplier=10)}, starting_cash=1000)
        ledger.customer('A', 'SELL', 6, 100)
        self.assertEqual(ledger.cash, -5000)
        self.assertEqual(ledger.positions, {'A': 6})
        self.assertEqual(ledger.customer_notional, 6000)
        self.assertEqual(ledger.pnl({'A': 97}), -180)

    def test_partial_reset_preserves_loss_and_pays_execution_slippage(self):
        product = Product('A', lot=2, multiplier=10,
                          fee_per_unit=.25, fee_notional=.001)
        ledger = Ledger({'A': product}, starting_cash=1000)
        ledger.customer('A', 'SELL', 6, 100)
        self.assertTrue(ledger.exchange('A', -4, 96.5))
        # Principal received = 3860; fees = 1 + 3.86. Two units remain.
        self.assertAlmostEqual(ledger.cash, -1144.86)
        self.assertAlmostEqual(ledger.fees, 4.86)
        self.assertEqual(ledger.positions['A'], 2)
        self.assertAlmostEqual(ledger.pnl({'A': 97}), -204.86)
        ledger.reconcile()

    def test_customer_buy_then_adverse_full_reset(self):
        ledger = Ledger({'A': Product('A', multiplier=5, fee_per_unit=.2)},
                        starting_cash=1000)
        ledger.customer('A', 'BUY', 3, 102)
        self.assertTrue(ledger.exchange('A', 3, 104))
        self.assertEqual(ledger.positions['A'], 0)
        self.assertAlmostEqual(ledger.cash, 969.4)
        self.assertAlmostEqual(ledger.pnl({'A': 104}), -30.6)

    def test_exchange_round_trip_without_customer_edge_loses_both_fees(self):
        ledger = Ledger({'A': Product('A', multiplier=4,
                                     fee_per_unit=.3, fee_notional=.002)})
        self.assertTrue(ledger.exchange('A', 2, 50))
        self.assertTrue(ledger.exchange('A', -2, 50))
        self.assertEqual(ledger.positions['A'], 0)
        self.assertAlmostEqual(ledger.pnl({'A': 50}), -2.8)
        self.assertEqual(ledger.customer_notional, 0)
        self.assertEqual(ledger.customer_orders, 0)
        self.assertEqual(ledger.exchange_notional, 800)

    def test_rejected_worsening_hedge_is_atomic(self):
        ledger = Ledger({'A': Product('A', long_limit=3, short_limit=2)})
        ledger.customer('A', 'SELL', 5, 100)
        before = (ledger.cash, dict(ledger.positions), ledger.fees,
                  ledger.exchange_notional, list(ledger.log))
        self.assertFalse(ledger.exchange('A', 1, 110))
        self.assertEqual(before, (ledger.cash, ledger.positions, ledger.fees,
                                  ledger.exchange_notional, ledger.log))

    def test_breach_recovery_can_cross_to_equal_excess_opposite_breach(self):
        ledger = Ledger({'A': Product('A', long_limit=3, short_limit=2,
                                     fee_per_unit=0)})
        ledger.customer('A', 'SELL', 5, 100)
        self.assertTrue(ledger.exchange('A', -1, 100))
        self.assertEqual(ledger.positions['A'], 4)
        self.assertFalse(ledger.exchange('A', -8, 100))
        self.assertTrue(ledger.exchange('A', -7, 100))
        self.assertEqual(ledger.positions['A'], -3)
        self.assertTrue(ledger.exchange('A', 1, 100))
        self.assertEqual(ledger.positions['A'], -2)

    def test_boundary_and_terminal_penalties_leave_residual_holdings(self):
        ledger = Ledger({'A': Product('A', multiplier=3, long_limit=2,
                                     breach_per_unit=.7)})
        ledger.customer('A', 'SELL', 5, 20)
        self.assertAlmostEqual(ledger.charge_breaches(), 2.1)
        self.assertAlmostEqual(ledger.charge_breaches(), 2.1)
        self.assertEqual(ledger.positions['A'], 5)
        self.assertAlmostEqual(ledger.pnl({'A': 21}), 10.8)
        ledger.reconcile()

    def test_affine_settlement_residual_uses_each_product_multiplier(self):
        ledger = Ledger({'A': Product('A', multiplier=2),
                         'B': Product('B', multiplier=5)})
        ledger.customer('A', 'SELL', 3, 100)
        ledger.customer('B', 'BUY', 2, 201)
        prices = resolve({'A': 102, 'B': 199},
                         {'A': None, 'B': (1, {'A': 2})})
        # +3*2*(102-100) -2*5*(205-201) = -28.
        self.assertEqual(prices, {'A': 102, 'B': 205})
        self.assertEqual(ledger.pnl(prices), -28)
        self.assertEqual(ledger.positions, {'A': 3, 'B': -2})

    def test_formula_settlement_does_not_invent_tick_rounding(self):
        prices = resolve({'A': 100.037, 'B': 200},
                         {'A': None, 'B': (.1, {'A': 1})})
        self.assertAlmostEqual(prices['B'], 100.137)
        self.assertNotAlmostEqual(prices['B'], round(prices['B'], 2))

    def test_lot_rejection_does_not_change_cash_or_positions(self):
        ledger = Ledger({'A': Product('A', lot=2)})
        with self.assertRaises(ValueError):
            ledger.customer('A', 'SELL', 3, 100)
        self.assertEqual(ledger.cash, ledger.starting_cash)
        self.assertEqual(ledger.positions, {'A': 0})
        self.assertEqual(ledger.log, [])

    def test_tick_rounding_on_both_sides_and_exact_tick(self):
        self.assertAlmostEqual(round_price(12.031, .05, 'BUY'), 12.05)
        self.assertAlmostEqual(round_price(12.031, .05, 'SELL'), 12)
        self.assertAlmostEqual(round_price(12.05, .05, 'BUY'), 12.05)
        self.assertAlmostEqual(round_price(12.05, .05, 'SELL'), 12.05)


class IndependentEventAudit(unittest.TestCase):
    def test_latest_quote_replaces_and_omission_preserves_that_quote(self):
        book = QuoteBook()
        book.update({'a': 100.02}, {'a'})
        book.update({'a': 100.04}, {'a'})
        book.update({}, {'a'})
        rfq = PrivateRFQ('a', 'A', 'BUY', 1, .2, (.03,), .1)
        self.assertEqual(book.values, {'a': 100.04})
        self.assertEqual(match(rfq, book.values['a'], 100, .01), (False, 100.03))

    def test_explicit_withdrawal_then_omission_cannot_revive_quote(self):
        book = QuoteBook()
        book.update({'a': 100}, {'a'})
        book.update({'a': None}, {'a'})
        book.update({}, {'a'})
        self.assertEqual(book.values, {})

    def test_matching_reference_changes_eligibility_not_execution_quote(self):
        rfq = PrivateRFQ('a', 'A', 'BUY', 2, .04, (.10,), .1)
        self.assertEqual(match(rfq, 100.03, 100, .01), (True, 100.03))
        self.assertEqual(match(rfq, 100.03, 99.98, .01), (False, None))

    def test_three_way_tie_threshold(self):
        for side, quote in [('BUY', 100.1), ('SELL', 99.9)]:
            for uniform, won in [(0.32, True), (0.34, False)]:
                rfq = PrivateRFQ('a', 'A', side, 1, .2, (.1, .1), uniform)
                self.assertEqual(match(rfq, quote, 100, .01), (won, quote))

    def test_simultaneous_customer_overshoot_is_executed_and_penalised(self):
        class UnreservedQuotes:
            def quote(self, orders, prices, positions, products, *, dt):
                return {o['order_id']: 100.0 for o in orders}

            def hedge(self, prices, positions, products, *, dt):
                return []

        rfqs = tuple(PrivateRFQ(str(i), 'A', 'SELL', 2, .2, (.1,), .1)
                     for i in range(2))
        scenario = Scenario(np.array([[100.0], [100.0]]), (rfqs,), ('A',), 1, .5)
        result = run(scenario, UnreservedQuotes(), limit=3, keep_trace=True)
        self.assertEqual(result['customer_orders'], 2)
        self.assertEqual(result['terminal_positions'], {'A': 4})
        self.assertAlmostEqual(result['penalties'], .2)
        self.assertAlmostEqual(result['pnl'], -.2)

    def test_every_reserved_subset_fits_without_assuming_opposite_fills(self):
        products = {'A': Product('A', long_limit=5, short_limit=3)}
        orders = [dict(order_id='a', product_id='A', side='SELL', quantity=4),
                  dict(order_id='b', product_id='A', side='BUY', quantity=5),
                  dict(order_id='c', product_id='A', side='SELL', quantity=1)]
        accepted = reserve_batch(orders, {'A': 1}, products)
        self.assertEqual(accepted, {'a'})
        selected = [o for o in orders if o['order_id'] in accepted]
        for bits in itertools.product((False, True), repeat=len(selected)):
            q = 1 + sum((o['quantity'] if o['side'] == 'SELL' else -o['quantity'])
                        for o, filled in zip(selected, bits) if filled)
            self.assertLessEqual(q, 5)
            self.assertGreaterEqual(q, -3)

    def test_unconfirmed_hedge_cannot_release_reservation_capacity(self):
        product = Product('A', long_limit=4, short_limit=4)
        order = dict(order_id='a', product_id='A', side='SELL', quantity=2)
        # The actual position is still 3, even if a sale of 3 was requested.
        self.assertEqual(reserve_batch([order], {'A': 3}, {'A': product}), set())

    def test_old_finite_model_masks_breach_quote_but_ledger_does_not_clip(self):
        model, states, actions = build_inventory_mdp(horizon=1, limit=1,
                                                      allow_reset=False)
        state = states.index((-1, 1))  # Customer BUY reduces inventory.
        quote = actions.index((None, .02))
        self.assertFalse(model.feasible[0, state, quote])
        ledger = Ledger({'A': Product('A', long_limit=1, short_limit=1)})
        ledger.customer('A', 'BUY', 1, 100)
        ledger.customer('A', 'BUY', 1, 100)
        self.assertEqual(ledger.positions['A'], -2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
