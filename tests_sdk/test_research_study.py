
from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import unittest
from experiments.research_study import (METHODS, DOMAIN, FixedPriceBand, FixedPriceReset,
    generate_prices, parameters, seed_for, inference, run_case)
from tests_sdk.test_sdk_controller import market


class ResearchStudyTests(unittest.TestCase):
    def test_equal_tuning_budgets_and_distinct_joint_packages(self):
        for method in METHODS:
            grid = parameters(method)
            self.assertEqual(len(grid), 24)
            self.assertEqual(len({repr(sorted(p.items())) for p in grid}), 24)
            self.assertEqual({p['forecast_weight'] for p in grid}, {0., .25, .5, 1.})
            if method != 'B2':
                self.assertTrue(any(p['base_edge'] == .08 and p['inventory_skew'] == .002
                    and p['band_fraction'] == .4 and p['target_fraction'] == .5
                    and p['forecast_weight'] == 0. for p in grid))

    def test_disjoint_seed_streams_and_phases(self):
        values = [seed_for(phase, scenario, path, stream)
                  for phase in ('test', 'tuning', 'validation', 'final') for scenario in range(8)
                  for path in range(3) for stream in ('factor', 'basis', 'customers', 'opponents/sticky')]
        self.assertEqual(len(values), len(set(values)))
        self.assertNotIn('boundary-policy', DOMAIN)

    def test_tape_is_deterministic_and_invalid_scenario_rejected(self):
        for scenario in range(8):
            rows = generate_prices('test', scenario, 0)
            self.assertEqual(rows, generate_prices('test', scenario, 0))
            self.assertEqual(len(rows), 124)
            self.assertTrue(all(v > 0 for r in rows for v in r.values()))
        with self.assertRaises(ValueError):
            generate_prices('test', 8, 0)

    def test_fixed_price_ablation_removes_inventory_quote_feedback(self):
        m = market()
        a, b = FixedPriceBand(inventory_skew=.5), FixedPriceReset(inventory_skew=.5)
        for side in ('BUY', 'SELL'):
            prices = {d._edge(side, q, 201., .01, 'B', m)
                      for d in (a, b) for q in (-20, 0, 20)}
            self.assertEqual(len(prices), 1)

    def test_last_price_null_uses_public_map_and_reconciles(self):
        row = run_case('B4', parameters('B4')[0], 'test', 7, 0, 'standard', detail=True)
        self.assertEqual(row['parameters']['settlement_coefficients'], {})
        self.assertEqual(row['final_settlement'], row['final_reference'])
        self.assertAlmostEqual(row['monetary_components']['terminal_settlement_residual'], 0.)
        self.assertLess(row['decomposition_max_error'], 1e-8)

    def test_path_stratification_and_materiality_boundary(self):
        values = [dict(scenario=s, path=p, difference=1.) for s in range(8) for p in range(2)]
        result = inference(values, replicates=100)
        self.assertFalse(result['superiority_gate'])
        self.assertEqual(result['independent_paths'], 16)
        self.assertEqual(result['stratified_path_bootstrap_95'], [1., 1.])
        for row in values:
            row['difference'] = 1.01
        self.assertTrue(inference(values, replicates=100)['superiority_gate'])


if __name__ == '__main__':
    unittest.main()
