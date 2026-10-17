"""Independent primitive coupling and shared-package regressions."""
from unittest import SkipTest
from experiments.sdk_environment import sdk_available, require_sdk
if not sdk_available(): raise SkipTest('Optional supplied SDK unavailable')
require_sdk()

import copy
import unittest
from experiments.risk_matched_study import EventTies, EventOpponent, prices_for, seed_for, paired_groups
from experiments.research_study import seed_for as old_seed


class RiskMatchedStudyTests(unittest.TestCase):
    def test_one_actual_sdk_case_serializes_complete_ledger_and_risk_trace(self):
        import json
        import tempfile
        from experiments.risk_matched_study import run_case
        from experiments.research_study import parameters
        with tempfile.TemporaryDirectory() as directory:
            row = run_case('B4', parameters('B4')[0], 'fixture',0,0,'size_mix','tight','fixed',directory)
        json.dumps(row,allow_nan=False)
        self.assertEqual(row['block'],0)
        self.assertEqual(len(row['final_rfq_observations']),36)
        self.assertLess(row['decomposition_max_error'],1e-8)
        self.assertEqual(row['risk_budget_counters']['actual_breaches'],0)

    def test_new_namespace_is_disjoint_from_old_and_between_phases(self):
        self.assertNotEqual(seed_for('final', 0, 0, 'factor'), old_seed('final', 0, 0, 'factor'))
        self.assertNotEqual(seed_for('final', 0, 0, 'factor'), seed_for('tuning', 0, 0, 'factor'))
        self.assertNotEqual(seed_for('final', 0, 0, 'factor'), seed_for('final', 0, 0, 'basis'))
        self.assertEqual(prices_for('smoke', 0, 0), prices_for('smoke', 0, 0))

    def test_tie_primitive_is_by_rfq_not_preceding_eligibility(self):
        choices = ['a', 'b', 'c']
        full = EventTies([(1, 'old'), (1, 'target')], 'test', 0, 0)
        full.choice(choices)
        target = full.choice(choices)
        shortened = EventTies([(1, 'target')], 'test', 0, 0)
        self.assertEqual(target, shortened.choice(choices))
        with self.assertRaises(AssertionError): shortened.choice(choices)

    def test_opponent_event_draws_ignore_callback_order_and_private_fields(self):
        market = dict(prices={'A': 100., 'B': 201.}, products={p:dict(tick_size=.01) for p in ('A','B')})
        orders = [dict(order_id='r1',product_id='A',side='BUY',quantity=2),
                  dict(order_id='r2',product_id='B',side='SELL',quantity=10)]
        for family in ('size_mix','public_basis'):
            opponent = EventOpponent(family, 7)
            baseline = opponent.quote(orders, market, {})
            opponent.quote(list(reversed(orders)), market, {'positions':{'A':500}})
            changed = copy.deepcopy(orders)
            for order in changed: order.update(hidden_limit=-100, future_price=999, informed=True)
            self.assertEqual(baseline, opponent.quote(changed, market, {}))

    def test_opponent_repeats_and_cells_do_not_inflate_independent_units(self):
        rows = []
        for scenario in range(8):
            for path in range(2):
                for family in ('one','two'):
                    for controller,pnl in [('B4',3.),('R2B',5.)]:
                        rows.append(dict(scenario=scenario,path=path,family=family,controller=controller,
                            package='B4',budget='wide',quote_mode='fixed',parameters={'x':1},
                            account={'pnl':pnl},price_tape_sha256='p',public_customer_tape_sha256='c'))
        groups = paired_groups(rows,'B4','wide','fixed')
        self.assertEqual(groups,[[2.,2.]]*8)
        rows[0]['parameters']={'x':2}
        with self.assertRaises(AssertionError):paired_groups(rows,'B4','wide','fixed')


if __name__=='__main__':unittest.main()
