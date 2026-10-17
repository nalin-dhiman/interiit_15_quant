"""Independent SDK, reservation, model-domain and logical-clock regressions."""
from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest('Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.')
require_sdk()

from copy import deepcopy
from dataclasses import asdict
import itertools
import unittest

from experiments.sdk_study import SynchronousSession, build_config
from resetmm.common_risk import CommonRiskBandDealer, CommonRiskResetDealer
from rfq.engine import Engine
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.store import Store
from tests_sdk import test_boundary_dealer as boundary_fixtures
from tests_sdk.test_boundary_dealer import account
from tests_sdk.test_sdk_controller import market, order


CAPS = {'A': 10., 'B': 5.}


class CommonRiskControllerTests(unittest.TestCase):
    def dealer(self, cls=CommonRiskBandDealer, **kwargs):
        parameters = dict(inventory_caps=CAPS, risk_weights={'A': 1., 'B': 2.},
                          settlement_coefficients={'B': {'A': 2.}},
                          band_fraction=.01, target_fraction=0., min_improvement=0.,
                          risk_multiplier=0.)
        parameters.update(kwargs)
        return cls(**parameters)

    def open(self, dealer, orders=(), **positions):
        m = market(orders)
        m['products']['A'].update(max_long=50., max_short=50.)
        m['products']['B'].update(max_long=25., max_short=25.)
        m['prices']['B'] = 200.  # No initial basis intervention in admission fixtures.
        p = account(m, **positions)
        dealer.on_market(m, p)
        self.assertEqual(dealer.hedge(m, p), [])
        self.assertEqual(dealer.boundary_state, 'open')
        return m, p

    def test_private_bounds_preserve_public_contract_and_match_across_policies(self):
        m = market()
        m['products']['A'].update(max_long=50., max_short=50.)
        m['products']['B'].update(max_long=25., max_short=25.)
        original = deepcopy(m)
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            dealer = self.dealer(cls)
            dealer._market = m
            self.assertEqual(dealer._bounds(m['products']['A']), (-10., 10.))
            self.assertEqual(dealer._bounds(m['products']['B']), (-5., 5.))
            self.assertEqual(m, original)

    def test_conventional_and_closing_reset_targets_obey_caps_before_valuation(self):
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            dealer = self.dealer(cls)
            m = market(closing=True)
            m['products']['B'].update(max_long=25., max_short=25.)
            p = account(m)
            actions = dealer._boundary_choose(m, p)
            self.assertEqual(actions, [('B', 'SELL', 5.)])
            self.assertEqual(p['positions'], {'A': 0., 'B': 0.})
            if cls is CommonRiskResetDealer:
                decision = list(dealer.last_decisions)[-1]
                self.assertEqual(decision['baseline_target'], -5.)
                self.assertTrue(all(abs(z) <= 5. for z in decision['candidate_values']))
                self.assertAlmostEqual(decision['candidate_values'][-5.], 4.95)

    def test_non_grid_caps_revalue_actual_lots_multipliers_and_fees(self):
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            dealer = self.dealer(cls)
            m = market(closing=True)
            m['products']['B'].update(lot_size=2., multiplier=3., fee_rate=.001,
                                      max_long=25., max_short=25.)
            p = account(m)
            actions = dealer._boundary_choose(m, p)
            self.assertEqual(actions, [('B', 'SELL', 4.)])
            self.assertEqual(m['products']['B']['max_long'], 25.)
            if cls is CommonRiskResetDealer:
                decision = list(dealer.last_decisions)[-1]
                self.assertEqual(decision['baseline_target'], -4.)
                self.assertAlmostEqual(decision['candidate_values'][-4.],
                                       4 * 3 - 4 * (.01 + 201 * 3 * .001))

    def test_all_live_quote_subsets_obey_the_shared_budget(self):
        orders = [order('as', 'SELL', 7., 'A'), order('ab', 'BUY', 7., 'A'),
                  order('as2', 'SELL', 4., 'A'), order('bs', 'SELL', 4., 'B'),
                  order('bb', 'BUY', 4., 'B'), order('bs2', 'SELL', 2., 'B')]
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            dealer = self.dealer(cls)
            m, p = self.open(dealer, orders)
            quotes = dealer.quote(orders, m, p)
            self.assertIsNone(quotes['as2'])
            self.assertIsNone(quotes['bs2'])
            admitted = [o for o in orders if quotes[o['order_id']] is not None]
            self.assertEqual(len(admitted), 4)
            for flags in itertools.product((0, 1), repeat=len(admitted)):
                q = {'A': 0., 'B': 0.}
                for flag, request in zip(flags, admitted):
                    q[request['product_id']] += flag * request['quantity'] * (-1 if request['side'] == 'BUY' else 1)
                self.assertLessEqual(abs(q['A']), 10.)
                self.assertLessEqual(abs(q['B']), 5.)
                self.assertLessEqual(abs(q['A']) + 2 * abs(q['B']), 20.)
            self.assertEqual(dealer.risk_budget_counters['possible_breaches'], 0)

    def test_an_unconfirmed_replacement_or_omission_never_releases_capacity(self):
        orders = [order('old', 'SELL', 4., 'B'), order('new', 'SELL', 2., 'B')]
        dealer = self.dealer()
        m, p = self.open(dealer, orders)
        self.assertIsNotNone(dealer.quote(orders[:1], m, p)['old'])
        dealer.on_ack({'action_id': 'quote', 'status': 'rejected', 'errors': {'old': 'fixture'}})
        self.assertIsNone(dealer.quote(orders[1:], m, p)['new'])
        self.assertIn((1, 'old'), dealer._reserved)
        self.assertTrue(dealer.risk_budget_trace[-1]['all_subsets_within_envelope'])

    def test_large_rfq_is_withdrawn_instead_of_clipped_to_private_capacity(self):
        orders = [order('large', 'SELL', 10., 'B')]
        dealer = self.dealer()
        m, p = self.open(dealer, orders)
        self.assertEqual(dealer.quote(orders, m, p), {'large': None})
        self.assertEqual(orders[0]['quantity'], 10.)
        self.assertEqual(dealer._reserved, {})

    def test_partial_hedge_ack_subsets_do_not_create_unconfirmed_capacity(self):
        dealer = self.dealer(risk_weight=.1, risk_multiplier=1.)
        requests = [order('new', 'SELL', 2., 'B')]
        m = market(requests)
        m['prices']['B'] = 200.
        p = account(m, A=9., B=4.)
        dealer.on_market(m, p)
        trades = dealer.hedge(m, p)
        self.assertEqual(len(trades), 2)
        self.assertEqual(dealer.boundary_state, 'pending_reset')
        self.assertEqual(dealer.quote(requests, m, p), {'new': None})
        final, _ = boundary_fixtures.BoundaryDealerTests().complete(dealer, trades, p, m, rejected=(1,))
        self.assertEqual(dealer.boundary_state, 'open')
        self.assertEqual(final['positions']['B'], 4.)
        self.assertEqual(dealer.quote(requests, m, final), {'new': None})
        self.assertEqual(dealer.risk_budget_counters['possible_breaches'], 0)
        self.assertTrue(all(row['possible_gross'] <= row['gross_limit'] + 1e-9 for row in dealer.risk_budget_trace))

    def test_future_current_batch_kernel_uses_the_same_private_caps(self):
        dealer = self.dealer(CommonRiskResetDealer)
        requests = [order('buy', 'BUY', 4., 'B'), order('sell', 'SELL', 4., 'B')]
        m = market(requests)
        dealer._market = m
        dealer._boundary_orders = {o['order_id']: o for o in requests}
        outcomes = dealer._boundary_batch_outcomes(-5., 'B', dealer._spec('B', m), m)
        self.assertAlmostEqual(sum(probability for probability, _, _ in outcomes), 1.)
        self.assertTrue(all(-5. <= -5. + delta <= 5. for _, delta, _ in outcomes))
        self.assertTrue(any(delta == 4. for _, delta, _ in outcomes))
        self.assertFalse(any(delta == -4. for _, delta, _ in outcomes))

    def test_sdk_customer_fill_can_exceed_the_private_cap_without_being_clipped(self):
        class Orders:
            def generate(self, interval_id, public, state, rng):
                return state, [CustomerOrder('forced', 'c', 'A', 'SELL', 15., interval_id, limit_offset=1.)]
        store = Store(':memory:')
        self.addCleanup(store.close)
        engine = Engine(HostConfig([Product('A', 100., max_long=50., max_short=50.)],
            ReplayPrices([{'A': 100.}, {'A': 100.}]), Orders()), store)
        engine.register('d')
        engine.action('d', dict(type='quotes', action_id='external-fixture', interval_id=1, quotes={'forced': 99.9}))
        engine.advance()
        observed = engine.portfolio('d')
        self.assertEqual(observed['positions']['A'], 15.)
        self.assertAlmostEqual(observed['cash'], 100000. - 15 * 99.9)
        dealer = self.dealer()
        cash_before = observed['cash']
        public = dict(engine.market(), products={pid: asdict(product) for pid, product in engine.products.items()})
        dealer.on_market(public, observed)
        self.assertEqual(dealer.boundary_state, 'unsupported')
        self.assertEqual(dealer.hedge(public, observed), [])
        self.assertEqual(observed['positions']['A'], 15.)
        self.assertEqual(observed['cash'], cash_before)
        self.assertGreater(dealer.risk_budget_counters['actual_breaches'], 0)
        self.assertTrue(all(not row['cash_penalty_imposed'] for row in dealer.risk_budget_trace))

    def test_inherited_hedge_preserves_logical_time_and_remaining_holding_value(self):
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            dealer = self.dealer(cls, total_intervals=1, risk_weight=.001, risk_multiplier=1.)
            self.assertEqual(dealer.hedge.__func__.__module__, 'submission.boundary_dealer')
            self.assertIs(dealer.hedge.__func__.__globals__, dealer._continuation_values.__func__.__globals__)
            runner = SynchronousSession(build_config(None, 0, 17, rows=[{'A': 100., 'B': 200.}] * 124), dealer)
            self.addCleanup(runner.store.close)
            m = dict(runner.engine.market(), products={pid: asdict(product) for pid, product in runner.engine.products.items()})
            dealer._market = m
            spec = dealer._spec('A', m)
            value = dealer._continuation_values(10., [10.], 'A', spec, m, current_outcomes=[(1., 0., 0.)])[10.]
            self.assertAlmostEqual(value, -.5)
            runner.virtual_time += 2.
            value = dealer._continuation_values(10., [10.], 'A', spec, m, current_outcomes=[(1., 0., 0.)])[10.]
            self.assertAlmostEqual(value, -.3)

    def test_factor_mode_is_explicitly_outside_the_scalar_matched_adapter(self):
        for cls in (CommonRiskBandDealer, CommonRiskResetDealer):
            with self.assertRaisesRegex(ValueError, 'scalar'):
                self.dealer(cls, factor_risk=True)
