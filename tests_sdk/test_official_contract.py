"""Hand-calculated fixtures against the supplied organiser engine/runner.

These exercise the exact local kit bytes, not an unseen competition server or
later-round impact model. Source hashes and old rejected assumptions are archived
in runs/20261015T135015Z_sdk/contract/.
"""
from __future__ import annotations

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SDK_ROOT = KIT / 'rfq'
if SDK_ROOT.is_dir():
    sys.path.insert(0, str(SDK_ROOT))

from rfq.engine import Engine, Reject, increment_ok
from rfq.models import (CONSTANT, LAST_PRICE, CustomerOrder, HostConfig,
                        Product as SDKProduct, ReplayPrices, settle_with)
from rfq.sdk import round_lot, round_quote
from rfq.settlement import parse_settlement
from rfq.store import Store
from resetmm.accounting import Ledger, Product, round_price
from resetmm.settlement import resolve


class ConstantOrders:
    def __init__(self, orders=()):
        self.orders = orders

    def generate(self, interval_id, market, state, rng):
        return state, [CustomerOrder(f'r{interval_id}-{i}', 'c', *order,
                                    expires_interval=interval_id)
                       for i, order in enumerate(self.orders)]


class OfficialEngineFixtures(unittest.TestCase):
    def make(self, *, product=None, orders=(), rows=None, rules=None, **kwargs):
        product = product or SDKProduct('A', 100, max_long=3, max_short=3,
                                       breach_penalty_per_unit=.7,
                                       fee_per_unit=.2, fee_rate=.001)
        rows = rows or [{'A': 100}, {'A': 101}, {'A': 102}]
        store = Store(':memory:')
        self.addCleanup(store.close)
        engine = Engine(HostConfig([product], ReplayPrices(rows, rules),
                                   ConstantOrders(orders), **kwargs), store)
        engine.register('d')
        return engine

    def action(self, engine, *, who='d', **action):
        msg = dict(action_id=f'a{len(engine.trades)}-{self.seq}', **action)
        self.seq += 1
        if msg['type'] in ('quote', 'quotes'):
            msg.setdefault('interval_id', engine.interval_id)
        else:
            msg.setdefault('price_version', engine.price_version)
        return engine.action(who, msg)[-1][1]

    def setUp(self):
        self.seq = 0

    def assert_ledger(self, engine, ledger):
        a = engine.accounts['d']
        self.assertAlmostEqual(a['cash'], ledger.cash)
        self.assertEqual(a['positions'], ledger.positions)
        self.assertAlmostEqual(a['fees_paid'], ledger.fees)
        self.assertAlmostEqual(a['penalties_paid'], ledger.penalties)
        self.assertAlmostEqual(a['customer_notional'], ledger.customer_notional)
        self.assertAlmostEqual(a['exchange_notional'], ledger.exchange_notional)
        self.assertEqual(a['customer_trades'], ledger.customer_orders)
        ledger.reconcile()

    def test_customer_sell_multiplier_fee_and_partial_reset_reconcile(self):
        p = SDKProduct('A', 100, lot_size=2, multiplier=10, max_long=10,
                       max_short=10, fee_per_unit=.25, fee_rate=.001)
        e = self.make(product=p, orders=[('A', 'SELL', 6)], rows=[{'A': 100}, {'A': 96.5}])
        self.action(e, type='quotes', quotes={'r1-0': 100})
        e.advance(final=True)
        ledger = Ledger({'A': Product('A', lot=2, multiplier=10,
                                     long_limit=10, short_limit=10,
                                     fee_per_unit=.25, fee_notional=.001)}, starting_cash=1000)
        # Align initial capital, preserving all transactions.
        e.accounts['d']['cash'] -= 99000
        e.config.starting_cash = 1000
        ledger.customer('A', 'SELL', 6, 100)
        self.assertEqual(e.accounts['d']['positions']['A'], 6)
        ack = self.action(e, type='trades', trades=[dict(product_id='A', side='SELL', quantity=4)])
        self.assertEqual(ack['status'], 'accepted')
        self.assertTrue(ledger.exchange('A', -4, 96.5))
        self.assertAlmostEqual(ledger.cash, -1144.86)
        self.assertAlmostEqual(ledger.fees, 4.86)
        self.assert_ledger(e, ledger)
        self.assertAlmostEqual(e.portfolio('d')['pnl'], -214.86)

    def test_customer_buy_means_dealer_sell_no_customer_fee(self):
        e = self.make(product=SDKProduct('A', 100, multiplier=5), orders=[('A', 'BUY', 3)])
        self.action(e, type='quotes', quotes={'r1-0': 102})
        events = e.advance(final=True)
        fill = next(x for _, x in events if x['type'] == 'fill')
        self.assertEqual(fill['side'], 'SELL')
        self.assertEqual(fill['fee'], 0)
        self.assertEqual(e.accounts['d']['positions'], {'A': -3})
        self.assertEqual(e.accounts['d']['cash'], 101530)
        self.assertEqual(e.accounts['d']['customer_notional'], 1530)

    def test_exchange_roundtrip_loses_both_fees_no_customer_volume(self):
        p = SDKProduct('A', 50, multiplier=4, fee_per_unit=.3, fee_rate=.002)
        e = self.make(product=p, rows=[{'A': 50}])
        ack = self.action(e, type='trades', trades=[dict(product_id='A', side=s, quantity=2)
                                                  for s in ('BUY', 'SELL')])
        self.assertEqual(len(ack['executions']), 2)
        self.assertAlmostEqual(e.portfolio('d')['pnl'], -2.8)
        self.assertEqual(e.accounts['d']['customer_trades'], 0)
        self.assertEqual(e.accounts['d']['customer_notional'], 0)
        self.assertEqual(e.accounts['d']['exchange_notional'], 800)

    def test_equal_excess_opposite_crossing_is_allowed_and_reconciles(self):
        e = self.make(product=SDKProduct('A', 100, max_long=25, max_short=25, fee_per_unit=.01))
        e.fill('d', 'A', 'BUY', 30, 100, 'customer')
        ledger = Ledger({'A': Product('A', long_limit=25, short_limit=25)})
        ledger.customer('A', 'SELL', 30, 100)
        ack = self.action(e, type='trades', trades=[dict(product_id='A', side='SELL', quantity=60)])
        self.assertNotIn('errors', ack)
        self.assertTrue(ledger.exchange('A', -60, 100))
        self.assertEqual(e.accounts['d']['positions'], {'A': -30})
        self.assert_ledger(e, ledger)

    def test_more_excess_opposite_crossing_rejects_atomically(self):
        e = self.make(product=SDKProduct('A', 100, max_long=25, max_short=25))
        e.fill('d', 'A', 'BUY', 30, 100, 'customer')
        before = dict(e.portfolio('d'))
        ack = self.action(e, type='trades', trades=[dict(product_id='A', side='SELL', quantity=62)])
        self.assertEqual(ack['errors'], {'0': 'POSITION_LIMIT'})
        self.assertEqual(before, e.portfolio('d'))

    def test_asymmetric_limits_compare_excess_not_absolute_position(self):
        e = self.make(product=SDKProduct('A', 100, max_long=3, max_short=2))
        e.fill('d', 'A', 'BUY', 5, 100, 'customer')
        self.assertFalse(e.worsens_breach('d', 'A', -9))  # +5 excess2 -> -4 excess2
        self.assertTrue(e.worsens_breach('d', 'A', -10))
        self.assertFalse(e.worsens_breach('d', 'A', -8))

    def test_batch_trades_sequential_errors_do_not_rollback_other_items(self):
        e = self.make()
        trades = [dict(product_id='A', side='BUY', quantity=4),
                  dict(product_id='A', side='BUY', quantity=3),
                  dict(product_id='A', side='SELL', quantity=6),
                  dict(product_id='A', side='SELL', quantity=1)]
        ack = self.action(e, type='trades', trades=trades)
        self.assertEqual(ack['errors'], {'0': 'POSITION_LIMIT', '3': 'POSITION_LIMIT'})
        self.assertEqual([x['quantity'] for x in ack['executions']], [3, 6])
        self.assertEqual(e.accounts['d']['positions'], {'A': -3})

    def test_penalty_before_matching_not_on_new_fill_until_next_check(self):
        e = self.make(orders=[('A', 'BUY', 2)])
        e.fill('d', 'A', 'BUY', 5, 100, 'customer')
        self.action(e, type='quotes', quotes={'r1-0': 100})
        events = e.advance(final=True)
        self.assertAlmostEqual(e.accounts['d']['penalties_paid'], 1.4)
        self.assertEqual(e.accounts['d']['positions']['A'], 3)
        kinds = [x['type'] for _, x in events]
        self.assertEqual(kinds[0], 'market')
        self.assertLess(kinds.index('penalty'), kinds.index('fill'))
        self.assertEqual(next(x for _, x in events if x['type'] == 'penalty')['checkpoint'], 'interval')
        e.finish()
        self.assertAlmostEqual(e.accounts['d']['penalties_paid'], 1.4)

    def test_customer_batch_overshoot_executes_then_settlement_penalty(self):
        e = self.make(orders=[('A', 'SELL', 2), ('A', 'SELL', 2)])
        self.action(e, type='quotes', quotes={'r1-0': 100, 'r1-1': 100})
        e.advance(final=True)
        self.assertEqual(e.accounts['d']['positions']['A'], 4)
        self.assertEqual(e.accounts['d']['penalties_paid'], 0)
        e.finish()
        self.assertAlmostEqual(e.accounts['d']['penalties_paid'], .7)
        self.assertEqual(e.accounts['d']['positions']['A'], 4)
        self.assertAlmostEqual(e.portfolio('d')['pnl'], 3.3)

    def test_optional_customer_limit_mode_is_sequential(self):
        e = self.make(orders=[('A', 'SELL', 2), ('A', 'SELL', 2)], customer_fills_respect_limits=True)
        self.action(e, type='quotes', quotes={'r1-0': 100, 'r1-1': 100})
        events = e.advance()
        results = next(x for _, x in events if x['type'] == 'customer_results')['results']
        self.assertEqual([x['status'] for x in results], ['filled', 'expired'])
        self.assertEqual(e.accounts['d']['positions']['A'], 2)

    def test_settlement_only_penalty_mode(self):
        e = self.make(penalty_checkpoints='settlement')
        e.fill('d', 'A', 'BUY', 5, 100, 'customer')
        e.advance(final=True)
        self.assertEqual(e.accounts['d']['penalties_paid'], 0)
        e.finish()
        self.assertAlmostEqual(e.accounts['d']['penalties_paid'], 1.4)

    def test_matching_uses_pre_boundary_price_fill_precedes_new_marking(self):
        e = self.make(orders=[('A', 'BUY', 2)], rows=[{'A': 100}, {'A': 200}])
        e.orders['r1-0'] = CustomerOrder('r1-0', 'c', 'A', 'BUY', 2, 1, limit_offset=.05)
        self.action(e, type='quotes', quotes={'r1-0': 100.05})
        events = e.advance(final=True)
        fill = next(x for _, x in events if x['type'] == 'fill')
        self.assertEqual(fill['price'], 100.05)
        self.assertEqual(fill['price_version'], 1)
        self.assertEqual(events[0][1]['prices']['A'], 200)
        self.assertEqual(events[0][1]['price_version'], 2)
        self.assertAlmostEqual(e.portfolio('d')['pnl'], -199.9)

    def test_current_price_changes_hidden_eligibility(self):
        e = self.make(orders=[('A', 'BUY', 1)], rows=[{'A': 100}, {'A': 99.98}, {'A': 200}])
        e.orders['r1-0'] = CustomerOrder('r1-0', 'c', 'A', 'BUY', 1, 1, limit_offset=.04)
        self.action(e, type='quotes', quotes={'r1-0': 100.03})
        self.assertEqual(e.matches(e.orders['r1-0']), {'d': 100.03})
        e.tick()
        self.assertEqual(e.matches(e.orders['r1-0']), {})

    def test_reservation_and_offset_both_apply_sell_side(self):
        e = self.make(orders=[('A', 'SELL', 1)])
        order = CustomerOrder('r1-0', 'c', 'A', 'SELL', 1, 1,
                              reservation_price=99.97, limit_offset=.1)
        e.orders['r1-0'] = order
        self.action(e, type='quotes', quotes={'r1-0': 99.95})
        self.assertEqual(e.matches(order), {})
        self.action(e, type='quotes', quotes={'r1-0': 99.97})
        self.assertEqual(e.matches(order), {'d': 99.97})

    def test_tie_uses_sorted_equal_price_dealers_and_independent_rng(self):
        e = self.make(orders=[('A', 'BUY', 1)], seed=42)
        e.register('z')
        e.register('a')
        for who in ('z', 'd', 'a'):
            self.action(e, who=who, type='quotes', quotes={'r1-0': 100})
        expected = random.Random(44).choice(['a', 'd', 'z'])
        events = e.advance()
        result = next(x for _, x in events if x['type'] == 'customer_results')['results'][0]
        self.assertEqual(result['winner'], expected)

    def test_latest_replacement_omission_withdrawal_and_bad_tick_retention(self):
        e = self.make(orders=[('A', 'BUY', 1)])
        self.action(e, type='quotes', quotes={'r1-0': 100})
        self.action(e, type='quotes', quotes={'r1-0': 100.02})
        self.action(e, type='quotes', quotes={})
        self.assertEqual(e.quotes['r1-0'], {'d': 100.02})
        ack = self.action(e, type='quotes', quotes={'r1-0': 100.021})
        self.assertEqual(ack['errors'], {'r1-0': 'BAD_TICK'})
        self.assertEqual(e.quotes['r1-0'], {'d': 100.02})
        self.action(e, type='quotes', quotes={'r1-0': None})
        self.action(e, type='quotes', quotes={})
        self.assertEqual(e.quotes['r1-0'], {})

    def test_quote_batch_partially_accepts_and_unknown_none_rejects(self):
        e = self.make(orders=[('A', 'BUY', 1), ('A', 'SELL', 1)])
        ack = self.action(e, type='quotes', quotes={'r1-0': 100, 'r1-1': 100.001, 'missing': None})
        self.assertEqual(ack['status'], 'accepted')
        self.assertEqual(ack['errors'], {'r1-1': 'BAD_TICK', 'missing': 'UNKNOWN_ORDER'})
        self.assertEqual(e.quotes, {'r1-0': {'d': 100}})

    def test_stale_quote_and_closed_order_codes(self):
        e = self.make(orders=[('A', 'BUY', 1)])
        ack = self.action(e, type='quotes', quotes={'r1-0': 100}, interval_id=0)
        self.assertEqual(ack['code'], 'STALE_QUOTE')
        e.advance(final=True)
        ack = self.action(e, type='quotes', quotes={'r1-0': 100})
        self.assertEqual(ack['errors'], {'r1-0': 'UNKNOWN_ORDER'})

    def test_trade_lag_executes_latest_rejects_old_and_future_versions(self):
        e = self.make()
        e.tick()
        e.tick()
        ack = self.action(e, type='trades', price_version=1,
                          trades=[dict(product_id='A', side='BUY', quantity=1)])
        self.assertEqual(ack['executions'][0]['price'], 102)
        for version in (0, 4, 3.0, True):
            ack = self.action(e, type='trades', price_version=version,
                              trades=[dict(product_id='A', side='BUY', quantity=1)])
            self.assertEqual(ack['code'], 'STALE_PRICE')

    def test_limit_price_not_tradable_and_bad_lot(self):
        e = self.make(product=SDKProduct('A', 100, lot_size=.1))
        cases = [(dict(product_id='A', side='BUY', quantity=.1, limit_price=99), 'LIMIT'),
                 (dict(product_id='A', side='SELL', quantity=.1, limit_price=101), 'LIMIT'),
                 (dict(product_id='A', side='BUY', quantity=.30000000000000004), 'BAD_LOT')]
        for trade, code in cases:
            self.assertEqual(self.action(e, type='trades', trades=[trade])['errors'], {'0': code})
        e = self.make(product=SDKProduct('A', 100, tradable_against_exchange=False))
        self.assertEqual(self.action(e, type='trades', trades=[dict(product_id='A', side='BUY', quantity=1)])['errors'], {'0': 'NOT_TRADABLE'})

    def test_dedup_same_action_does_not_trade_twice_changed_id_content_rejects(self):
        e = self.make()
        msg = dict(action_id='same', type='trades', price_version=1,
                   trades=[dict(product_id='A', side='BUY', quantity=1)])
        e.action('d', msg)
        before = e.portfolio('d')
        ack = e.action('d', msg)[-1][1]
        self.assertTrue(ack['duplicate'])
        self.assertEqual(before, e.portfolio('d'))
        ack = e.action('d', {**msg, 'price_version': 2})[-1][1]
        self.assertEqual(ack['code'], 'ID_REUSED')
        self.assertEqual(before, e.portfolio('d'))

    def test_final_window_trading_then_residual_settlement_no_liquidation(self):
        e = self.make()
        e.advance(final=True)
        self.assertTrue(e.market()['closing'])
        self.assertEqual(e.market()['active_orders'], [])
        self.action(e, type='trades', trades=[dict(product_id='A', side='BUY', quantity=2)])
        e.tick()
        e.finish()
        self.assertEqual(e.accounts['d']['positions']['A'], 2)
        self.assertAlmostEqual(e.portfolio('d')['pnl'], 1.398)
        ack = self.action(e, type='trades', trades=[dict(product_id='A', side='SELL', quantity=2)])
        self.assertEqual(ack['code'], 'ENDED')

    def test_public_observation_omits_customer_limits_and_opponent_quotes(self):
        e = self.make(orders=[('A', 'BUY', 1)])
        e.orders['r1-0'] = CustomerOrder('r1-0', 'c', 'A', 'BUY', 1, 1,
                                       reservation_price=101, limit_offset=.1)
        e.register('opponent')
        self.action(e, who='opponent', type='quotes', quotes={'r1-0': 100.05})
        public = e.snapshot('d')
        order = public['market']['active_orders'][0]
        self.assertNotIn('reservation_price', order)
        self.assertNotIn('limit_offset', order)
        self.assertNotIn('quotes', public)
        events = e.advance()
        result = next(x for _, x in events if x['type'] == 'customer_results')['results'][0]
        self.assertEqual(result['winner'], 'opponent')
        self.assertEqual(result['price'], 100.05)
        self.assertNotIn('limit_offset', result)
        self.assertNotIn('customer_side', result)  # must retain public RFQ record

    def test_disconnect_withdraws_quotes_preserves_account(self):
        e = self.make(orders=[('A', 'BUY', 1)])
        e.fill('d', 'A', 'BUY', 2, 100, 'customer')
        self.action(e, type='quotes', quotes={'r1-0': 100})
        before = e.portfolio('d')
        e.withdraw('d')
        e.register('d')
        self.assertEqual(e.portfolio('d'), before)
        self.assertEqual(e.quotes['r1-0'], {})


class OfficialSettlementAndRounding(unittest.TestCase):
    def test_rounding_matches_sdk_and_engine_accepts_ticks(self):
        for price in (12.031, 12.05, 12.450000000000001, .3, 100.00000000000001):
            for tick in (.01, .05, .1):
                for side in ('BUY', 'SELL'):
                    expected = round_quote(price, tick, side)
                    self.assertEqual(round_price(price, tick, side), expected)
                    self.assertTrue(increment_ok(expected, tick))
        self.assertEqual(round_quote(100.00000000000001, .01, 'BUY'), 100.01)

    def test_round_lot_towards_zero_and_exact_decimal_validation(self):
        self.assertEqual(round_lot(2.7, .5), 2.5)
        self.assertEqual(round_lot(-2.7, .5), -2.5)
        l = Ledger({'A': Product('A', lot=.1)})
        with self.assertRaises(ValueError):
            l.customer('A', 'SELL', .30000000000000004, 100)
        l.customer('A', 'SELL', round_lot(.30000000000000004, .1), 100)
        self.assertEqual(l.positions['A'], .3)

    def test_affine_formulas_match_research_graph_multipliers_separate(self):
        rules = {'B': parse_settlement('2*A + 3'), 'A': LAST_PRICE,
                 'C': parse_settlement('B - .5*A + 1')}
        prices = {'A': 100.037, 'B': 201, 'C': 99}
        expected = resolve(prices, {'B': (3, {'A': 2}), 'A': None,
                                    'C': (1, {'B': 1, 'A': -.5})})
        replay = ReplayPrices([prices], rules)
        self.assertEqual(replay.settle(0), expected)
        self.assertAlmostEqual(expected['B'], 203.074)
        self.assertEqual(parse_settlement('last_price'), LAST_PRICE)
        self.assertEqual(parse_settlement('-0.5*A + A*2 + 3'), {'A': 1.5, CONSTANT: 3})

    def test_formula_grammar_rejects_unsupported_expressions(self):
        for text in ('(A)', 'A/2', 'A*B', '2A', '1e-3*A', '1_000', 'A +', '', 'A B'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_settlement(text)

    def test_order_dependencies_self_unknown_and_nonpositive_reject(self):
        rows = [{'A': 100, 'B': 200, 'C': 300}]
        cases = ({'B': {'C': 1}, 'C': {'A': 1}},
                 {'B': {'B': 1}}, {'B': {'missing': 1}},
                 {'B': {'A': -1}})
        for rules in cases:
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                ReplayPrices(rows, rules)
        # Unlike the SDK's ordered CSV grammar, research resolve allows any DAG order.
        self.assertEqual(resolve(rows[0], {'B': (0, {'C': 1}), 'C': (0, {'A': 1}), 'A': None})['B'], 100)

    def test_csv_bom_case_blank_lines_default_rule_and_duplicate_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, rules = Path(tmp)/'data.csv', Path(tmp)/'rules.csv'
            data.write_text('\ufefftime_s,round,price_version,A,B\n0,1,1,100,201\n', encoding='utf-8')
            rules.write_text('\ufeffPRODUCT,SETTLEMENT\n\nB,2*A\n', encoding='utf-8')
            self.assertEqual(ReplayPrices.from_csv(data, rules).settle(0), {'A': 100, 'B': 200})
            rules.write_text('product,settlement\nA,LAST_PRICE\nA,LAST_PRICE\n')
            with self.assertRaisesRegex(ValueError, 'line 3'):
                ReplayPrices.from_csv(data, rules)


class OfficialRunnerFixtures(unittest.IsolatedAsyncioTestCase):
    async def test_batch_portfolio_precedes_ack_and_fill_callbacks_see_full_batch(self):
        from rfq.runner import DealerRunner
        store = Store(':memory:')
        self.addCleanup(store.close)
        product = SDKProduct('A', 100, fee_per_unit=.2)
        engine = Engine(HostConfig([product], ReplayPrices([{'A': 100}]),
                                  ConstantOrders()), store)
        engine.register('d')
        runner = DealerRunner.__new__(DealerRunner)
        runner.args = SimpleNamespace(id='d')
        runner.portfolio = engine.portfolio('d')
        runner.market = engine.market()
        runner.products = {'A': asdict(product)}
        runner.last_reported_interval = None
        runner.hedge_after_fills = runner.hedge_wanted = False
        runner.pending = {}
        seen = []
        async def callback(ws, method, **data):
            seen.append((method, copy.deepcopy(data)))
        runner.callback = callback

        async def deliver(message):
            runner.pending[message['action_id']] = message
            events = engine.action('d', message)
            async def messages(ws):
                for _, event in events:
                    yield event
            runner.messages = messages
            await runner.receive(None)
            return events

        # Single trade uses a separate fill message; ACK.execution must not be
        # applied as a second fill by runner.
        events = await deliver(dict(type='trade', action_id='single', price_version=1,
                                    product_id='A', side='BUY', quantity=1))
        self.assertEqual([x['type'] for _, x in events], ['fill', 'portfolio', 'ack'])
        self.assertEqual([x[0] for x in seen], ['on_fill', 'on_ack'])
        self.assertEqual(runner.portfolio['positions']['A'], 1)
        self.assertEqual(runner.portfolio['exchange_trades'], 1)
        seen.clear()

        events = await deliver(dict(type='trades', action_id='batch', price_version=1,
                                    trades=[dict(product_id='A', side='BUY', quantity=2),
                                            dict(product_id='A', side='SELL', quantity=1)]))
        self.assertEqual([x['type'] for _, x in events], ['portfolio', 'ack'])
        self.assertEqual(len(events[-1][1]['executions']), 2)
        self.assertEqual([x[0] for x in seen], ['on_fill', 'on_fill', 'on_ack'])
        portfolios = [x[1]['portfolio'] for x in seen if x[0] == 'on_fill']
        self.assertEqual([p['positions']['A'] for p in portfolios], [2, 2])
        self.assertEqual([p['exchange_trades'] for p in portfolios], [3, 3])
        self.assertAlmostEqual(runner.portfolio['cash'], 99799.2)
        expected = engine.portfolio('d')
        self.assertEqual({key: runner.portfolio[key] for key in expected}, expected)

    async def test_sent_quote_cache_is_not_acknowledged_quote_state(self):
        from rfq.runner import DealerRunner
        runner = DealerRunner.__new__(DealerRunner)
        runner.callback_lock = asyncio.Lock()
        runner.worker = SimpleNamespace(methods={'quote'}, timeout=1,
                                        call=lambda method, data: [{'type': 'quotes', 'quotes': {'r1-0': 100.001}}])
        runner.market = {'interval_id': 1, 'price_version': 1}
        runner.sent_quotes, runner.pending, runner.slow_warnings = {}, {}, 0
        sent = []
        class Socket:
            async def send(self, value):
                sent.append(json.loads(value))
        socket = Socket()
        # We exercise the actual runner's cache/send logic. Run the already
        # mocked worker synchronously; process/thread transport is a separate
        # end-to-end gate and is not the subject of this fixture.
        async def inline_worker(fn, *args):
            return fn(*args)
        with patch('rfq.runner.asyncio.to_thread', new=inline_worker):
            await runner.callback(socket, 'quote')
        self.assertEqual(runner.sent_quotes, {'r1-0': 100.001})
        # No ACK has arrived, and the exact engine would reject BAD_TICK.
        self.assertFalse(increment_ok(sent[0]['quotes']['r1-0'], .01))
        runner.pending.clear()  # rejection ACK removes pending, but not sent cache
        with patch('rfq.runner.asyncio.to_thread', new=inline_worker):
            await runner.callback(socket, 'quote')
        self.assertEqual(len(sent), 1)

    async def test_pending_trade_defers_hedge_until_ack(self):
        from rfq.runner import DealerRunner
        runner = DealerRunner.__new__(DealerRunner)
        runner.pending = {'x': {'type': 'trades'}}
        runner.market, runner.portfolio = {}, {}
        calls = []
        async def callback(ws, method, **data):
            calls.append(method)
        runner.callback = callback
        await runner.hedge(None)
        self.assertTrue(runner.hedge_wanted)
        self.assertEqual(calls, [])
        runner.pending.clear()
        await runner.hedge(None)
        self.assertFalse(runner.hedge_wanted)
        self.assertEqual(calls, ['hedge'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
