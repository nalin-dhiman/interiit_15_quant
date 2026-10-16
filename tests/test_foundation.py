from __future__ import annotations
import ast
import importlib.util
import itertools
import math
from pathlib import Path
import unittest
import numpy as np
from resetmm.analytic import *
from resetmm.accounting import Product, Ledger, signed_customer_delta, round_price
from resetmm.risk import fill_envelope,reserve_batch,psd_covariance,factor_risk
from resetmm.settlement import resolve
from resetmm.mdp import *
from resetmm.inventory_mdp import build_inventory_mdp
from resetmm.response import response_fixture,bernoulli_score
from resetmm.statistics import paired_bootstrap_interval,lower_tail_mean
from resetmm.toy_exchange import PrivateRFQ,QuoteBook,match,make_scenario,run
from resetmm.policies import InventoryQuote,ImmediateReset,ThresholdReset,PoissonReset,CostAwareReset

class AnalyticTests(unittest.TestCase):
    def test_moments(self):
        self.assertEqual(poisson_moments(2,.5),(2,8))
    def test_density_normalization(self):
        D,r=1.3,.7; alpha=math.sqrt(r/D)
        x=np.linspace(-30/alpha,30/alpha,100001)
        p=alpha/2*np.exp(-alpha*abs(x))
        self.assertAlmostEqual(np.trapezoid(p,x),1,places=6)
    def test_density_second_moment(self):
        D,r=1.3,.7; alpha=math.sqrt(r/D)
        x=np.linspace(-30/alpha,30/alpha,100001)
        p=alpha/2*np.exp(-alpha*abs(x))
        self.assertAlmostEqual(np.trapezoid(p*x*x,x),2*D/r,places=6)
    def test_fixed_cost_poisson_optimum(self):
        c=ResetCosts(2,.7,.3,0)
        self.assertAlmostEqual(optimal_poisson_rate(c),math.sqrt(2*c.a*c.D/c.K),places=12)
    def test_fixed_cost_threshold_optimum(self):
        c=ResetCosts(2,.7,.3,0)
        self.assertAlmostEqual(optimal_threshold(c),(12*c.D*c.K/c.a)**.25,places=12)
    def test_fixed_cost_ratio(self):
        c=ResetCosts(2,.7,.3,0)
        ratio=threshold_cost(optimal_threshold(c),c)/poisson_cost(optimal_poisson_rate(c),c)
        self.assertAlmostEqual(ratio,1/math.sqrt(6),places=12)
    def test_proportional_only_optimum(self):
        c=ResetCosts(2,.7,0,.3)
        self.assertAlmostEqual(optimal_threshold(c),(6*c.D*c.c/c.a)**(1/3),places=12)
        self.assertAlmostEqual(optimal_poisson_rate(c),(4*c.a*math.sqrt(c.D)/c.c)**(2/3),places=12)
    def test_mixed_cost_global_grid(self):
        c=ResetCosts(.8,.03,.12,.04)
        r,b=optimal_poisson_rate(c),optimal_threshold(c)
        for x in np.logspace(-2,2,80):
            self.assertLessEqual(poisson_cost(r,c),poisson_cost(r*x,c)+1e-10)
            self.assertLessEqual(threshold_cost(b,c),threshold_cost(b*x,c)+1e-10)
    def test_free_reset_no_finite_optimum(self):
        c=ResetCosts(1,.1)
        with self.assertRaises(ValueError): optimal_threshold(c)
        with self.assertRaises(ValueError): optimal_poisson_rate(c)
    def test_exit_zero_rate(self):
        self.assertEqual(reset_exit_time(1,0,2),2)
    def test_exit_small_rate_limit(self):
        self.assertAlmostEqual(reset_exit_time(1,1e-9,2),2,places=7)
    def test_exit_increases(self):
        times=[reset_exit_time(1,r,2) for r in [0,.01,.1,1]]
        self.assertTrue(all(x<y for x,y in zip(times,times[1:])))
    def test_invalid_parameters(self):
        for args in [(-1,1,1,1),(1,0,1,1),(1,1,-1,0),(math.nan,1,1,1)]:
            with self.assertRaises(ValueError): ResetCosts(*args)

class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.p=Product('A',multiplier=2,long_limit=3,short_limit=3,fee_per_unit=.1,fee_notional=.001)
        self.l=Ledger({'A':self.p})
    def test_customer_side_sign(self):
        self.l.customer('A','BUY',2,101)
        self.assertEqual(self.l.positions['A'],-2)
        self.assertEqual(self.l.cash,100404)
    def test_customer_spread_wealth(self):
        self.l.customer('A','BUY',2,101)
        self.assertEqual(self.l.pnl({'A':100}),4)
    def test_reset_no_free_money(self):
        self.l.customer('A','SELL',2,99)
        before=self.l.wealth({'A':100})
        self.l.exchange('A',-2,100)
        self.assertAlmostEqual(self.l.wealth({'A':100})-before,-.6)
        self.assertEqual(self.l.positions['A'],0)
    def test_exchange_turnover_not_customer(self):
        self.l.exchange('A',1,100)
        self.assertEqual(self.l.customer_orders,0)
        self.assertEqual(self.l.customer_notional,0)
    def test_customer_breach_not_clipped(self):
        self.l.customer('A','SELL',5,100)
        self.assertEqual(self.l.positions['A'],5)
        self.assertFalse(self.l.exchange('A',1,100))
    def test_breach_reduction_allowed(self):
        self.l.customer('A','SELL',5,100)
        self.assertTrue(self.l.exchange('A',-1,100))
        self.assertEqual(self.l.positions['A'],4)
    def test_equal_excess_opposite_breach_allowed(self):
        self.l.customer('A','SELL',5,100)
        self.assertTrue(self.l.exchange('A',-10,100))
        self.assertEqual(self.l.positions['A'],-5)
    def test_repeated_penalties(self):
        self.l.customer('A','SELL',5,100)
        self.l.charge_breaches();self.l.charge_breaches()
        self.assertAlmostEqual(self.l.penalties,.4)
    def test_settlement_revaluation(self):
        self.l.customer('A','SELL',2,100)
        self.assertEqual(self.l.pnl({'A':110}),40)
    def test_lot_validation(self):
        with self.assertRaises(ValueError): self.l.customer('A','BUY',.3,100)
    def test_nonfinite_price(self):
        with self.assertRaises(ValueError): self.l.exchange('A',1,math.nan)
    def test_reconciliation(self):
        self.l.customer('A','SELL',4,100);self.l.charge_breaches();self.l.exchange('A',-2,101)
        self.l.reconcile()
    def test_bad_side(self):
        with self.assertRaises(ValueError): signed_customer_delta('bid',1)
    def test_rounding_directions(self):
        self.assertAlmostEqual(round_price(100.004,.01,'BUY'),100.01)
        self.assertAlmostEqual(round_price(100.004,.01,'SELL'),100.00)

class RiskTests(unittest.TestCase):
    def test_envelope_exhaustive(self):
        orders=[('BUY',2),('SELL',3),('SELL',1),('BUY',4)]
        lo,hi=fill_envelope(2,orders)
        outcomes=[2+sum(signed_customer_delta(s,n)*b for (s,n),b in zip(orders,bits))
                  for bits in itertools.product([0,1],repeat=4)]
        self.assertEqual((lo,hi),(min(outcomes),max(outcomes)))
    def test_opposite_orders_do_not_net(self):
        self.assertEqual(fill_envelope(0,[('BUY',10),('SELL',10)]),(-10,10))
    def test_batch_reservation(self):
        p={'A':Product('A',long_limit=5,short_limit=5)}
        orders=[dict(order_id=str(i),product_id='A',side='SELL',quantity=3) for i in range(3)]
        allowed=reserve_batch(orders,{'A':0},p)
        self.assertEqual(allowed,{'0'})
    def test_duplicate_order_ids(self):
        o=dict(order_id='0',product_id='A',side='BUY',quantity=1)
        with self.assertRaises(ValueError):reserve_batch([o,o],{'A':0},{'A':Product('A')})
    def test_psd_projection(self):
        C=psd_covariance(np.array([[1,2],[2,1.]]))
        self.assertGreaterEqual(np.linalg.eigvalsh(C).min(),-1e-12)
    def test_factor_risk_dense_agreement(self):
        rng=np.random.default_rng(0);q=rng.normal(size=12);B=rng.normal(size=(12,3));d=rng.random(12)
        self.assertAlmostEqual(factor_risk(q,B,d),float(q@(B@B.T+np.diag(d))@q),places=10)
    def test_negative_residual_invalid(self):
        with self.assertRaises(ValueError):factor_risk(np.ones(2),np.ones((2,1)),np.array([1,-1]))
    def test_settlement_chain(self):
        p=resolve({'A':100,'B':197,'C':140},{'A':None,'B':(0,{'A':2}),'C':(1,{'B':.5})})
        self.assertEqual(p,{'A':100,'B':200,'C':101})
    def test_settlement_cycle(self):
        with self.assertRaises(ValueError):resolve({'A':1,'B':2},{'A':(0,{'B':1}),'B':(0,{'A':1})})
    def test_settlement_missing_dependency(self):
        with self.assertRaises(ValueError):resolve({'A':1},{'A':(0,{'X':1})})

class MDPTests(unittest.TestCase):
    def test_dynamic_program_matches_all_small_policies(self):
        P=np.array([[[[.7,.3],[.2,.8]],[[.4,.6],[.9,.1]]]]*2)
        g=np.array([[[.1,.2],[.05,.3]]]*2)
        m=FiniteMDP(P,g,np.array([.2,-.1]));V,pi,_=solve(m)
        vals=[]
        for bits in itertools.product([0,1],repeat=4):
            vals.append(evaluate(m,np.array(bits).reshape(2,2))[0])
        np.testing.assert_allclose(V[0],np.max(vals,axis=0),atol=1e-12)
    def test_policy_evaluation_identity(self):
        m,_,_=build_inventory_mdp(horizon=4,limit=2);V,pi,_=solve(m)
        np.testing.assert_allclose(V,evaluate(m,pi),atol=1e-12)
    def test_more_actions_cannot_hurt_exact_model(self):
        m,_,_=build_inventory_mdp(horizon=4,limit=2)
        b,_,_=build_inventory_mdp(horizon=4,limit=2,allow_reset=False)
        self.assertTrue(np.all(solve(m)[0]>=solve(b)[0]-1e-12))
    def test_nonstochastic_matrix_rejected(self):
        with self.assertRaises(ValueError):FiniteMDP(np.ones((1,2,1,2)),np.zeros((1,2,1)),np.zeros(2))
    def test_negative_transition_rejected(self):
        with self.assertRaises(ValueError):FiniteMDP(np.array([[[[-.1,1.1]],[[.2,.8]]]]),np.zeros((1,2,1)),np.zeros(2))
    def test_no_feasible_action_rejected(self):
        with self.assertRaises(ValueError):FiniteMDP(np.ones((1,1,1,1)),np.zeros((1,1,1)),np.zeros(1),np.zeros((1,1,1),bool))
    def test_safe_switch_margin(self):
        self.assertEqual(safe_switch(np.array([0,.19]),0,.1),0)
        self.assertEqual(safe_switch(np.array([0,.21]),0,.1),1)
    def test_safe_switch_equality(self):
        self.assertEqual(safe_switch(np.array([0,.2]),0,.1),0)
    def test_error_bound_zero(self):
        self.assertEqual(model_value_error_bound(5,1,1,0,0,0),0)
    def test_error_bound_horizon(self):
        self.assertEqual(model_value_error_bound(1,1,2,.1,.2,.3),.9)
    def test_response_matches_finite_difference(self):
        m,pi,dp,dg=response_fixture();_,dv=fixed_policy_derivative(m,pi,dp,dg)
        eps=1e-5;plus,pp,_,_=response_fixture(eps);minus,mp,_,_=response_fixture(-eps)
        fd=(evaluate(plus,pp)-evaluate(minus,mp))/(2*eps)
        np.testing.assert_allclose(dv,fd,atol=1e-9)
    def test_explicit_reward_derivative_is_required(self):
        m,pi,dp,dg=response_fixture();_,full=fixed_policy_derivative(m,pi,dp,dg)
        _,missing=fixed_policy_derivative(m,pi,dp,np.zeros_like(dg))
        self.assertGreater(np.max(abs(full-missing)),.01)
    def test_support_change_rejected(self):
        m=FiniteMDP(np.array([[[[1.,0]],[[0.,1]]]]),np.zeros((1,2,1)),np.zeros(2))
        dp=np.array([[[[-1.,1]],[[1.,-1]]]])
        with self.assertRaises(ValueError):fixed_policy_derivative(m,np.zeros((1,2),int),dp,np.zeros((1,2,1)))
    def test_bernoulli_score_mean_zero(self):
        p=.3;dp=.2
        self.assertAlmostEqual(p*bernoulli_score(True,p,dp)+(1-p)*bernoulli_score(False,p,dp),0)
    def test_zero_probability_response_rejected(self):
        with self.assertRaises(ValueError):bernoulli_score(True,0,.1)

class ExchangeTests(unittest.TestCase):
    def test_omission_keeps_quote(self):
        b=QuoteBook();b.update({'a':100},{'a'});b.update({},{'a'});self.assertEqual(b.values,{'a':100})
    def test_none_withdraws(self):
        b=QuoteBook();b.update({'a':100},{'a'});b.update({'a':None},{'a'});self.assertEqual(b.values,{})
    def test_unknown_quote_rejected(self):
        with self.assertRaises(ValueError):QuoteBook().update({'a':100},{'b'})
    def test_customer_limit_at_matching(self):
        r=PrivateRFQ('a','A','BUY',1,.03,(.10,),.1)
        self.assertEqual(match(r,100.02,100,.01),(True,100.02))
        self.assertEqual(match(r,100.02,99.9,.01),(False,None))
    def test_buy_lowest_and_sell_highest(self):
        for side,quote in [('BUY',100.02),('SELL',99.98)]:
            r=PrivateRFQ('a','A',side,1,.2,(.10,),.1)
            self.assertTrue(match(r,quote,100,.01)[0])
    def test_ties_are_random(self):
        for u,expected in [(0.2,True),(.7,False)]:
            r=PrivateRFQ('a','A','BUY',1,.2,(.1,),u)
            self.assertEqual(match(r,100.1,100,.01)[0],expected)
    def test_no_hidden_fields_in_public_order(self):
        r=make_scenario(1,intervals=1,products=1,rfqs=1).orders[0][0]
        self.assertEqual(set(r.public()),{'order_id','product_id','side','quantity'})
    def test_seed_replay(self):
        s=make_scenario(3,intervals=5)
        a=run(s,ThresholdReset());b=run(s,ThresholdReset());self.assertEqual(a,b)
    def test_poisson_seed_replay(self):
        s=make_scenario(4,intervals=5)
        self.assertEqual(run(s,PoissonReset(seed=55)),run(s,PoissonReset(seed=55)))
    def test_policies_respect_batch_limit(self):
        s=make_scenario(5,intervals=10,rfqs=15)
        for policy in [InventoryQuote(),ImmediateReset(),ThresholdReset(),PoissonReset(),CostAwareReset()]:
            self.assertEqual(run(s,policy)['penalties'],0)
    def test_tape_immutable(self):
        with self.assertRaises(ValueError):make_scenario(1).prices[0,0]=2

class StatisticalTests(unittest.TestCase):
    def test_paired_constant_difference(self):
        mean,lo,hi=paired_bootstrap_interval([3,4,5],[1,2,3],replicates=100)
        self.assertEqual((mean,lo,hi),(2,2,2))
    def test_tail_mean(self):
        self.assertEqual(lower_tail_mean([-10,0,1,2],.5),-5)
    def test_unpaired_rejected(self):
        with self.assertRaises(ValueError):paired_bootstrap_interval([1,2],[1])

class ScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(__file__).resolve().parents[1]/'submission'/'dealer_scaffold.py'
        spec=importlib.util.spec_from_file_location('scaffold',cls.path)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
    def test_default_constructor(self):
        self.assertIsNotNone(self.module.Dealer())
    def test_unconfigured_withdraws(self):
        bot=self.module.Dealer()
        self.assertEqual(bot.quote([{'order_id':'a'}],{},{}),{'a':None})
        self.assertEqual(bot.hedge({},{}),[])
    def test_no_forbidden_imports(self):
        tree=ast.parse(self.path.read_text())
        imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]
        self.assertEqual(imports,['math'])

if __name__=='__main__':
    unittest.main(verbosity=2)
