"""Hand-calculated adversarial fixtures for the separate archived-study audit."""
import copy
import unittest

from scripts.audit_research_study import (independent_ledger, fixed_price_quote,
                                          recompute_inference, check_parameters,
                                          economic_fingerprint)


class ResearchStudyAuditTests(unittest.TestCase):
    def fixture(self, scenario):
        last=scenario==7
        prices=[{'A':100.,'B':200.},{'A':102.,'B':205.}]
        settlement={'A':102.,'B':205. if last else 204.}
        pnl=5. if last else 4.
        trade=dict(dealer_id='dealer_1',trade_no=1,product_id='B',price_version=1,
            quantity=1.,price=200.,fee=0.,side='BUY',counterparty='customer',
            reference_price=200.,notional=200.,position_after=1.,cash_after=99800.)
        row=dict(scenario=scenario,settlement_graph='LAST_PRICE' if last else 'B=2A',
            account=dict(price_version=2,cash=99800.,positions={'A':0.,'B':1.},
                fees_paid=0.,customer_trades=1,exchange_trades=0,customer_notional=200.,
                exchange_notional=0.,pnl=pnl,penalties_paid=0.),own_trades=[trade],
            final_reference=prices[-1],final_settlement=settlement,
            monetary_components=dict(customer_spread=0.,exchange_execution_edge=0.,fees=0.,
                exogenous_price_move_pnl=5.,terminal_settlement_residual=0. if last else -1.,
                penalties=0.,decomposed_pnl=pnl),
            innovation_exposure=dict(mean_q_squared_at_price_innovation=[0.,1.],
                mean_multiplier_exposure_squared_at_price_innovation=[0.,1.],
                mean_terminal_factor_squared_at_price_innovation=4.,price_innovations=1,
                price_innovations_with_nonzero_inventory=1))
        return row,prices

    def test_graph_specific_settlement_residual_and_cash(self):
        for scenario in (0,7):
            row,prices=self.fixture(scenario)
            result=independent_ledger(row,prices)
            self.assertLess(max(result['maximum_errors'].values()),1e-8)
            self.assertEqual(result['components']['terminal_settlement_residual'],0. if scenario==7 else -1.)

    def test_silently_forcing_old_settlement_graph_is_detected(self):
        row,prices=self.fixture(7);row['final_settlement']['B']=204.
        with self.assertRaises(AssertionError):independent_ledger(row,prices)
        row,prices=self.fixture(7);row['settlement_graph']='B=2A'
        with self.assertRaises(AssertionError):independent_ledger(row,prices)

    def test_missing_principal_fee_or_customer_sign_is_detected(self):
        original,prices=self.fixture(0)
        for modify in (lambda r:r['account'].update(cash=100000.),
                       lambda r:r['own_trades'][0].update(fee=.02),
                       lambda r:r['own_trades'][0].update(side='SELL')):
            row=copy.deepcopy(original);modify(row)
            with self.assertRaises(AssertionError):independent_ledger(row,prices)

    def test_fixed_quote_uses_installed_public_state_not_future_reference(self):
        prices=[{'A':100.,'B':201.},{'A':110.,'B':240.}]
        params=dict(settlement_coefficients={'B':{'A':2.}},forecast_weight=.5,base_edge=.081)
        obs=dict(product_id='B',side='BUY',accepted_quote_price_version=1)
        self.assertEqual(fixed_price_quote(obs,prices,params),200.59)
        obs['side']='SELL'
        self.assertEqual(fixed_price_quote(obs,prices,params),200.41)
        changed=copy.deepcopy(prices);changed[1]={'A':1.,'B':1.}
        self.assertEqual(fixed_price_quote(obs,prices,params),fixed_price_quote(obs,changed,params))

    def test_last_price_fixed_quote_has_zero_settlement_forecast(self):
        params=dict(settlement_coefficients={},forecast_weight=1.,base_edge=.08)
        obs=dict(product_id='B',side='BUY',accepted_quote_price_version=1)
        self.assertEqual(fixed_price_quote(obs,[{'A':100.,'B':201.}],params),201.08)

    def test_eight_strata_have_independent_path_counts_and_gate(self):
        values={(s,p):2. for s in range(8) for p in range(3)}
        result=recompute_inference(values)
        self.assertEqual(result['independent_paths'],24)
        self.assertEqual(result['stratified_path_bootstrap_95'],[2.,2.])
        self.assertTrue(result['superiority_gate'])
        self.assertFalse(recompute_inference({k:1. for k in values})['superiority_gate'])
        del values[7,2]
        with self.assertRaises(AssertionError):recompute_inference(values)

    def test_scenario_local_parameter_change_is_only_public_settlement(self):
        seal=dict(selected={'B4':dict(settlement_coefficients={'B':{'A':2.}},forecast_weight=.5)})
        row=dict(method='B4',scenario=7,parameters=dict(settlement_coefficients={},forecast_weight=.5))
        check_parameters(row,seal)
        row['parameters']['forecast_weight']=1.
        with self.assertRaises(AssertionError):check_parameters(row,seal)

    def test_parallel_prefix_equality_ignores_only_host_timing(self):
        row,_=self.fixture(0)
        row['callback_seconds']={'hedge':{'maximum':.1}}
        row['own_trades'][0]['time']=100.
        replay=copy.deepcopy(row)
        replay['callback_seconds']={'hedge':{'maximum':.3}}
        replay['own_trades'][0]['time']=200.
        self.assertEqual(economic_fingerprint(row),economic_fingerprint(replay))
        replay['account']['cash']+=.01
        self.assertNotEqual(economic_fingerprint(row),economic_fingerprint(replay))


if __name__=='__main__':unittest.main()
