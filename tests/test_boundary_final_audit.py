"""Hand-calculated fixtures for the independent post-completion audit."""
import copy
import unittest
import hashlib
import json
from scripts.audit_boundary_final import independent_ledger,recompute_inference,audit_observations

class BoundaryFinalAuditTests(unittest.TestCase):
    def fixture(self):
        trades=[dict(dealer_id='dealer_1',trade_no=1,product_id='A',price_version=1,
            quantity=2.,price=101.,fee=0.,side='SELL',counterparty='customer',reference_price=100.,
            notional=202.,position_after=-2.,cash_after=100202.),
            dict(dealer_id='dealer_1',trade_no=2,product_id='A',price_version=2,
            quantity=1.,price=102.,fee=.01,side='BUY',counterparty='exchange',reference_price=102.,
            notional=102.,position_after=-1.,cash_after=100099.99)]
        row=dict(account=dict(price_version=2,cash=100099.99,positions={'A':-1.,'B':0.},
            fees_paid=.01,customer_trades=1,exchange_trades=1,customer_notional=202.,exchange_notional=102.,
            pnl=-2.01,penalties_paid=0.),own_trades=trades,final_reference={'A':102.,'B':204.},
            final_settlement={'A':102.,'B':204.},monetary_components=dict(customer_spread=2.,
            exchange_execution_edge=0.,fees=.01,exogenous_price_move_pnl=-4.,terminal_settlement_residual=0.,
            penalties=0.,decomposed_pnl=-2.01),innovation_exposure=dict(mean_q_squared_at_price_innovation=[4.,0.],
            mean_multiplier_exposure_squared_at_price_innovation=[4.,0.],mean_terminal_factor_squared_at_price_innovation=4.,
            price_innovations=1,price_innovations_with_nonzero_inventory=1))
        return row,[{'A':100.,'B':200.},{'A':102.,'B':204.}]

    def test_customer_sell_dealer_short_and_paid_partial_exchange(self):
        row,prices=self.fixture();result=independent_ledger(row,prices)
        self.assertLess(max(result['maximum_errors'].values()),1e-8)
        self.assertEqual(result['components']['customer_spread'],2.)
        self.assertEqual(result['components']['exogenous_price_move_pnl'],-4.)

    def test_missing_fee_cash_and_wrong_sign_are_detected(self):
        original,prices=self.fixture()
        for field in ('cash','pnl'):
            row=copy.deepcopy(original);row['account'][field]+=.01
            with self.assertRaises(AssertionError):independent_ledger(row,prices)
        row=copy.deepcopy(original);row['own_trades'][0]['side']='BUY'
        with self.assertRaises(AssertionError):independent_ledger(row,prices)

    def test_innovation_exposure_cannot_be_replaced_by_posthedge_sample(self):
        row,prices=self.fixture();row['innovation_exposure']['mean_q_squared_at_price_innovation']=[1.,0.]
        with self.assertRaises(AssertionError):independent_ledger(row,prices)

    def test_known_constant_stratified_path_difference_and_materiality(self):
        pairs={(s,p):2. for s in range(4) for p in range(2)}
        result=recompute_inference(pairs)
        self.assertEqual(result['independent_paths'],8)
        self.assertEqual(result['paired_mean_difference'],2.)
        self.assertEqual(result['stratified_path_bootstrap_95'],[2.,2.])
        self.assertTrue(result['superiority_gate'])
        result=recompute_inference({key:.9 for key in pairs})
        self.assertFalse(result['superiority_gate'])

    def test_authoritative_penalty_total_is_preserved_in_cash(self):
        row,prices=self.fixture();row['account']['penalties_paid']=.25
        row['account']['cash']-=.25;row['account']['pnl']-=.25
        row['monetary_components']['penalties']=.25;row['monetary_components']['decomposed_pnl']-=.25
        self.assertLess(max(independent_ledger(row,prices)['maximum_errors'].values()),1e-8)

    def observation_fixture(self):
        observations=[]
        for i in range(36):
            interval=i//3+1
            observations.append(dict(order_id=f'order{i}',customer_id='customer',product_id='A',
                side='BUY',quantity=2.,expires_interval=interval,interval_id=interval,
                own_quote=None,accepted_quote_age_seconds=None,first_positive_virtual_time=None,
                quote_installation_virtual_time=None,matching_virtual_time=1000.+5*interval,
                first_seen_virtual_time=995.+5*interval,admission_delay_seconds=None,won=False,
                matching_reference=100.,next_reference=100.,dealer_signed_one_tick_markout=0.))
        public=[{k:o[k] for k in ('order_id','customer_id','product_id','side','quantity','expires_interval')} for o in observations]
        row=dict(final_rfq_observations=observations,own_trades=[],
            public_customer_tape_sha256=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest(),
            quote_age=dict(positive_final_quotes=0,sum_age_seconds=0.,maximum_age_seconds=0.),
            admission=dict(ever_admitted=0,never_admitted=36,sum_delay_seconds=0.,maximum_delay_seconds=0.))
        return row,[{'A':100.,'B':200.} for _ in range(124)]

    def test_losing_rfq_reference_checked_against_exact_price_tape(self):
        row,prices=self.observation_fixture();audit_observations(row,prices)
        row['final_rfq_observations'][0]['matching_reference']=101.
        with self.assertRaises(AssertionError):audit_observations(row,prices)

    def test_duplicate_or_unrepresented_customer_trade_cannot_be_overwritten(self):
        row,prices=self.observation_fixture()
        trade=dict(interval_id=1,order_id='order0',counterparty='customer')
        row['own_trades']=[trade,trade.copy()]
        with self.assertRaises(AssertionError):audit_observations(row,prices)
        row,prices=self.observation_fixture();row['own_trades']=[trade]
        with self.assertRaises(AssertionError):audit_observations(row,prices)

    def test_missing_admission_cannot_hide_a_positive_delay(self):
        row,prices=self.observation_fixture()
        row['final_rfq_observations'][0]['admission_delay_seconds']=.5
        with self.assertRaises(AssertionError):audit_observations(row,prices)
