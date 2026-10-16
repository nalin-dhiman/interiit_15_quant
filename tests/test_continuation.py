"""Independent accounting and finite-control checks for the stated RFQ model."""
from dataclasses import replace
from itertools import product
import math
import unittest

from resetmm.continuation import (
    RFQ, PrimitiveAtom, Batch, ModelConfig, Action, FiniteRFQModel,
    BandPolicy, PolicyEvaluation, evaluate_policy, improve_policy,
    solve_optimal, simulate_policy, tune_band, make_synthetic_model, make_primitive_tape,
)


def certain_batch(side="BUY", quantity=1, shock=0.0):
    return Batch((RFQ(side, quantity),),
                 (PrimitiveAtom(1.0, (.5,), ((.4,),), (.1,), shock),))


def model_for(batch=None, **kwargs):
    cfg = ModelConfig(horizon=2, limit_lots=2, exchange_fee=.03,
                      quote_edges=(.1,), **kwargs)
    return FiniteRFQModel(cfg, (batch or certain_batch(),))


class ContinuationAccountingTests(unittest.TestCase):
    def test_inherited_breach_pays_boundary_and_settlement_without_clipping(self):
        # Two excess lots, each two physical units; .7 currency per unit.
        # One decision boundary and settlement each charge2*2*.7=2.8.
        c = ModelConfig(horizon=1, limit_lots=3, lot=2, multiplier=5,
                        breach_penalty=.7, terminal_basis=.1)
        m = FiniteRFQModel(c, (certain_batch(),))
        withdraw = lambda t, q, b: Action(None, (None,))
        row = simulate_policy(m, withdraw, [814], q0=5)[0]
        self.assertAlmostEqual(row['penalties'], 5.6)
        self.assertAlmostEqual(row['terminal_inventory'], 10)
        self.assertAlmostEqual(row['cash'], -5005.6)
        self.assertAlmostEqual(row['pnl'], -.6)  # +5 settlement basis-5.6 fees
        self.assertAlmostEqual(evaluate_policy(m, withdraw, q0=5), -.6)
        self.assertEqual(row['exchange_quantity'], 0)

    def test_customer_buy_sign_principal_multiplier_and_terminal_residual(self):
        m = model_for(lot=2, multiplier=5, terminal_basis=.07)
        o = m.outcomes(0, 0, Action(None, (.1,)))[0]
        self.assertEqual(o.next_inventory, -1)
        self.assertAlmostEqual(o.cash_delta_at_reference, 1001)
        self.assertAlmostEqual(o.ledger_pnl, 1)
        self.assertAlmostEqual(m.terminal(-1), -.7)
        self.assertAlmostEqual(o.cash_delta_at_reference + o.next_inventory*2*5*(100+.07), .3)

    def test_customer_sell_sign_and_principal(self):
        m = model_for(certain_batch("SELL", 2), lot=3, multiplier=2)
        o = m.outcomes(0, 0, Action(None, (.1,)))[0]
        self.assertEqual(o.next_inventory, 2)
        self.assertAlmostEqual(o.cash_delta_at_reference, -1198.8)
        self.assertAlmostEqual(o.ledger_pnl, 1.2)

    def test_partial_hedge_fee_slippage_and_cash_are_real(self):
        m = model_for(lot=2, multiplier=5, fixed_cost=.2, slippage_ticks=2)
        o = m.outcomes(2, 0, Action(1, (None,)))[0]
        # Sell two physical units at 99.98, fee .2+2*.03.
        self.assertAlmostEqual(o.cash_delta_at_reference, 999.54)
        self.assertAlmostEqual(o.ledger_pnl, -.46)
        self.assertAlmostEqual(o.fees, .26)
        self.assertEqual(o.next_inventory, 1)
        self.assertAlmostEqual(o.cash_delta_at_reference + (1-2)*2*5*100, -.46)

    def test_same_target_pays_no_fixed_cost(self):
        m = model_for(fixed_cost=9)
        self.assertEqual(m.hedge_cost(1, 1), 0)
        self.assertEqual(m.outcomes(1, 0, Action(1, (None,)))[0].fees, 0)

    def test_risk_is_not_subtracted_from_cash_or_ledger_pnl(self):
        m = model_for(risk_weight=.2, dt=.5)
        o = m.outcomes(0, 0, Action(None, (.1,)))[0]
        self.assertAlmostEqual(o.ledger_pnl, .1)
        self.assertAlmostEqual(o.risk_penalty, .1)
        self.assertAlmostEqual(o.objective, 0)
        self.assertAlmostEqual(o.cash_delta_at_reference, 100.1)

    def test_customer_batch_breach_is_not_clipped_when_reservation_bypassed(self):
        m = model_for(certain_batch("SELL", 3))
        a = Action(None, (.1,))
        self.assertFalse(m.feasible(0, 0, a))
        o = m.outcomes(0, 0, a, reserve=False)[0]
        self.assertEqual(o.next_inventory, 3)
        self.assertAlmostEqual(o.ledger_pnl, .3-.1)
        self.assertAlmostEqual(o.cash_delta_at_reference, -299.8)

    def test_exchange_cannot_worsen_breach_but_can_reduce(self):
        m = model_for()
        self.assertFalse(m.exchange_feasible(4, 5))
        self.assertFalse(m.exchange_feasible(4, -3))
        self.assertTrue(m.exchange_feasible(4, 3))
        self.assertTrue(m.exchange_feasible(4, 0))
        self.assertTrue(m.feasible(4, 0, Action(None, (None,))))

    def test_adverse_selection_is_post_fill_inventory_price_term(self):
        m = model_for(certain_batch("BUY", shock=.2))
        filled = m.outcomes(0, 0, Action(None, (.1,)))[0]
        nofill = m.outcomes(0, 0, Action(None, (.6,)))[0]
        self.assertAlmostEqual(filled.ledger_pnl, -.1)
        self.assertAlmostEqual(nofill.ledger_pnl, 0)


class ContinuationSemanticsTests(unittest.TestCase):
    def test_opposite_rfq_capacity_does_not_net(self):
        b = Batch((RFQ("BUY", 2), RFQ("SELL", 2)),
                  (PrimitiveAtom(1, (.5,.5), ((.4,),(.4,)), (.1,.1)),))
        m = model_for(b)
        self.assertFalse(m.capacity_feasible(1, 0, (.1,.1)))
        self.assertTrue(m.capacity_feasible(0, 0, (.1,.1)))
        self.assertTrue(m.capacity_feasible(1, 0, (.1,None)))
        self.assertFalse(m.capacity_feasible(1, 0, (None,.1)))

    def test_all_subset_reservation_independently_enumerated(self):
        b = Batch((RFQ("BUY", 1),RFQ("SELL",2),RFQ("BUY",2)),
                  (PrimitiveAtom(1,(.5,)*3,((.4,),)*3,(.1,)*3),))
        m = model_for(b)
        for q in range(-2,3):
            for live in product((False,True), repeat=3):
                edges = tuple(.1 if x else None for x in live)
                possible = []
                for fills in product((False,True), repeat=3):
                    if all(not f or l for f,l in zip(fills, live)):
                        possible.append(q-sum(r.direction*r.quantity_lots for r,f in zip(b.rfqs,fills) if f))
                expected = all(-2<=v<=2 for v in possible)
                self.assertEqual(m.capacity_feasible(q,0,edges), expected)

    def test_tick_rounding_changes_eligibility_and_reward_before_decision(self):
        b = Batch((RFQ("BUY"),), (PrimitiveAtom(1,(.105,),((.2,),),(.1,)),))
        m = model_for(b)
        self.assertEqual(m.outcomes(0,0,Action(None,(.101,)))[0].fills,(False,))
        self.assertAlmostEqual(m.round_edge(.101), .11)
        self.assertAlmostEqual(m.outcomes(0,0,Action(None,(.099,)))[0].ledger_pnl,.1)

    def test_customer_eligibility_and_random_tie_are_both_required(self):
        atoms = (PrimitiveAtom(.5,(.1,),((.1,),),(.1,)),
                 PrimitiveAtom(.5,(.1,),((.1,),),(.9,)))
        m = model_for(Batch((RFQ("BUY"),),atoms))
        out = m.outcomes(0,0,Action(None,(.1,)))
        self.assertEqual(tuple(o.fills for o in out), ((True,),(False,)))
        self.assertAlmostEqual(sum(o.probability*o.ledger_pnl for o in out), .05)
        self.assertTrue(all(not o.fills[0] for o in m.outcomes(0,0,Action(None,(.11,)))))

    def test_partial_targets_are_lot_valid_same_sign_and_include_no_action(self):
        m = model_for()
        self.assertEqual(m.partial_targets(2),(2,0,1))
        self.assertEqual(m.partial_targets(-2),(-2,0,-1))
        self.assertEqual(m.partial_targets(0),(0,))
        self.assertTrue(all(isinstance(q,int) for q in m.partial_targets(4)))

    def test_future_offsetting_flow_avoids_unnecessary_final_reset(self):
        # An offsetting SELL tomorrow, settlement has no liquidation charge.
        b = certain_batch("SELL")
        m = FiniteRFQModel(ModelConfig(horizon=1,limit_lots=2,exchange_fee=.1,
                                       risk_weight=0,quote_edges=(.1,)),(b,))
        band = BandPolicy(.1,0,1,0)
        r2 = improve_policy(m,band)
        self.assertEqual(r2(0,-1,0).target_lots,None)
        self.assertEqual(m.outcomes(-1,0,r2(0,-1,0))[0].next_inventory,0)
        self.assertGreater(evaluate_policy(m,r2,q0=-1),evaluate_policy(m,band.bind(m),q0=-1))

    def test_final_settlement_residual_can_favor_retaining_inventory(self):
        b = Batch((),(PrimitiveAtom(1,(),(),()),))
        m = FiniteRFQModel(ModelConfig(horizon=1,limit_lots=2,terminal_basis=.5), (b,))
        r2 = improve_policy(m,BandPolicy(threshold_lots=1,target_fraction=0))
        self.assertEqual(r2(0,1,0),Action(None,()))
        self.assertAlmostEqual(evaluate_policy(m,r2,q0=1),.5)

    def test_policy_only_receives_public_state(self):
        m = make_synthetic_model(horizon=2)
        observed = []
        def spy(t,q,b):
            observed.append((t,q,b))
            return Action(None,(None,)*len(m.batches[b].rfqs))
        simulate_policy(m,spy,[4321])
        self.assertTrue(all(len(x)==3 and all(isinstance(v,int) for v in x) for x in observed))

    def test_model_rejects_invalid_probabilities_and_units(self):
        with self.assertRaises(ValueError):
            FiniteRFQModel(ModelConfig(),(Batch((RFQ("BUY"),),(PrimitiveAtom(.9,(.5,),((.4,),),(.1,)),)),))
        with self.assertRaises(ValueError):
            model_for(lot=0)
        with self.assertRaises(ValueError):
            model_for(certain_batch(shock=.005))
        with self.assertRaises(ValueError):
            BandPolicy(target_fraction=1)
        with self.assertRaises(ValueError):
            model_for(reference_price=100.005)

    def test_nonpositive_quotes_are_rejected(self):
        m = model_for(certain_batch("SELL"))
        self.assertFalse(m.feasible(0,0,Action(None,(101.,))))

    def test_primitive_tape_archives_independent_fixed_streams(self):
        m = make_synthetic_model(horizon=5,regime="informed")
        tape = make_primitive_tape(m,1987)
        self.assertEqual(tape,make_primitive_tape(m,1987))
        self.assertEqual(len(tape.batch_types),5)
        self.assertEqual(len(tape.atom_indices),5)
        self.assertEqual(len(tape.batch_uniforms),6)
        self.assertNotEqual(tape.price_uniforms,tape.customer_uniforms)
        self.assertNotEqual(tape.customer_uniforms,tape.opponent_uniforms)
        self.assertNotEqual(tape.opponent_uniforms,tape.exploration_uniforms)
        policy = BandPolicy().bind(m)
        self.assertEqual(simulate_policy(m,policy,[1987]),simulate_policy(m,policy,[],tapes=[tape]))


class ContinuationDPTests(unittest.TestCase):
    def test_exact_dp_matches_exhaustive_tiny_markov_policy_enumeration(self):
        # Two decisions, one observed batch. Enumerate every deterministic map
        # on all time-1 states reachable from any feasible initial action.
        b = Batch((RFQ("BUY"),),
                  (PrimitiveAtom(.5,(.5,),((.4,),),(.1,)),
                   PrimitiveAtom(.5,(.05,),((.4,),),(.1,))))
        m = FiniteRFQModel(ModelConfig(horizon=2,limit_lots=1,quote_edges=(.1,),
                                       exchange_fee=.03,risk_weight=.02), (b,))
        root_actions = m.actions(0,0)
        states = sorted({o.next_inventory for a in root_actions for o in m.outcomes(0,0,a)})
        best, enumerated = -math.inf, 0
        for a0 in root_actions:
            for later in product(*(m.actions(q,0) for q in states)):
                action_map = dict(zip(states,later))
                val = 0.0
                for o0 in m.outcomes(0,0,a0):
                    for o1 in m.outcomes(o0.next_inventory,0,action_map[o0.next_inventory]):
                        val += o0.probability*o1.probability*(o0.objective+o1.objective+m.terminal(o1.next_inventory))
                best = max(best,val)
                enumerated += 1
        opt = solve_optimal(m)
        self.assertGreater(enumerated, 5)
        self.assertAlmostEqual(opt.value(0,0,0),best,places=12)
        self.assertAlmostEqual(evaluate_policy(m,opt),best,places=12)

    def test_baseline_evaluation_matches_independent_primitive_cash_paths(self):
        # Independent forward cash-principal computation; no model.outcomes.
        batch = Batch((RFQ("BUY"),),
                      (PrimitiveAtom(.5,(.5,),((.4,),),(.1,),.02),
                       PrimitiveAtom(.5,(.05,),((.4,),),(.1,),-.02)))
        m = FiniteRFQModel(ModelConfig(horizon=2,limit_lots=2,quote_edges=(.1,),
                                       lot=2,multiplier=3,exchange_fee=.03,fixed_cost=.01,
                                       risk_weight=.004,terminal_basis=.03), (batch,))
        band = BandPolicy(.1,0,1,0)
        policy = band.bind(m)
        total = 0.0
        for i,j in product(range(2),repeat=2):
            q,cash,price,risk = 0,0.0,100.0,0.0
            probability = 1.0
            for t, ix in enumerate((i,j)):
                atom = batch.atoms[ix]
                probability *= atom.probability
                action = policy(t,q,0)
                z = q if action.target_lots is None else action.target_lots
                delta = (z-q)*2
                cash -= delta*3*price
                if delta:
                    cash -= .01+abs(delta)*.03
                q = z
                edge = action.quote_edges[0]
                # This fixture has one competitor above our .1; tie impossible.
                if edge is not None and edge <= atom.limit_edges[0] and edge < .4:
                    q -= 1
                    cash += 2*3*(price+edge)
                price += atom.price_change
                risk += .004*(q*2)**2
            total += probability*(cash+q*2*3*(price+.03)-risk)
        self.assertAlmostEqual(evaluate_policy(m,policy),total,places=10)

    def test_r2_baseline_improvement_on_every_small_state(self):
        m = make_synthetic_model(horizon=3,limit_lots=2)
        band = BandPolicy(.07,.01,2,.5)
        r2 = improve_policy(m,band)
        baseline = PolicyEvaluation(m,band.bind(m))
        improved = PolicyEvaluation(m,r2)
        for t in range(3):
            for q in range(-2,3):
                for b in range(2):
                    a = r2(t,q,b)
                    self.assertGreaterEqual(baseline.action_value(t,q,b,a)+1e-11,baseline.value(t,q,b))
                    self.assertGreaterEqual(improved.value(t,q,b)+1e-11,baseline.value(t,q,b))

    def test_no_customer_edge_martingale_reset_cannot_create_expected_profit(self):
        b = Batch((),(PrimitiveAtom(.5,(),(),(),.02),PrimitiveAtom(.5,(),(),(),-.02)))
        m = FiniteRFQModel(ModelConfig(horizon=3,limit_lots=2,exchange_fee=.1), (b,))
        hold = BandPolicy(threshold_lots=None).bind(m)
        reset = BandPolicy(threshold_lots=1,target_fraction=0).bind(m)
        self.assertAlmostEqual(evaluate_policy(m,hold,q0=1),0)
        self.assertAlmostEqual(evaluate_policy(m,reset,q0=1),-.1)
        self.assertAlmostEqual(solve_optimal(m).value(0,1,0),0)

    def test_baseline_retained_on_exact_zero_advantage(self):
        b = Batch((),(PrimitiveAtom(1,(),(),()),))
        m = FiniteRFQModel(ModelConfig(horizon=1,limit_lots=2,exchange_fee=0), (b,))
        band = BandPolicy(threshold_lots=1,target_fraction=0)
        self.assertEqual(improve_policy(m,band)(0,1,0),band.bind(m)(0,1,0))

    def test_joint_oracle_dominates_r2_when_quote_sets_match(self):
        # Constant .08 quote belongs to the oracle's explicit quote grid.
        m = make_synthetic_model(horizon=3,limit_lots=2)
        band = BandPolicy(.08,0,2,.5)
        baseline = evaluate_policy(m,band.bind(m))
        r2 = evaluate_policy(m,improve_policy(m,band))
        oracle = evaluate_policy(m,solve_optimal(m))
        self.assertGreaterEqual(r2+1e-11,baseline)
        self.assertGreaterEqual(oracle+1e-11,r2)

    def test_tuning_records_equal_parameter_candidate_counts(self):
        m = make_synthetic_model(horizon=2,limit_lots=2)
        grid = [BandPolicy(.04,0,None),BandPolicy(.08,.01,2,.5)]
        b3, rows3 = tune_band(m,grid)
        r2, rows2 = tune_band(m,grid,improved=True)
        self.assertEqual(len(rows3),len(rows2))
        self.assertEqual(len(rows3),2)
        self.assertIn(b3,grid)
        self.assertIn(r2,grid)

    def test_reproducible_matched_streams_and_cash_reconciliation(self):
        m = make_synthetic_model(horizon=4,regime="informed")
        band = BandPolicy(.04,.01,2,.5)
        policy = improve_policy(m,band)
        rows = simulate_policy(m,policy,[1001,1002],q0=1)
        self.assertEqual(rows,simulate_policy(m,policy,[1001,1002],q0=1))
        for row in rows:
            self.assertAlmostEqual(row["pnl"],row["cash"]+row["terminal_inventory"]*m.config.multiplier*row["settlement_price"])
            self.assertAlmostEqual(row["objective"],row["pnl"]-row["risk_penalty"])
            self.assertGreaterEqual(row["exchange_notional"],0)
            self.assertGreaterEqual(row["customer_notional"],0)

    def test_batch_dependency_changes_value(self):
        balanced = make_synthetic_model(horizon=4,regime="balanced",risk_weight=0)
        persistent = make_synthetic_model(horizon=4,regime="persistent",risk_weight=0)
        band = BandPolicy(.04,0,None)
        self.assertGreater(evaluate_policy(balanced,band.bind(balanced)),evaluate_policy(persistent,band.bind(persistent)))


if __name__ == "__main__":
    unittest.main()
