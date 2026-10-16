"""Response of an optimal restricted Brownian reset boundary."""
import unittest
from dataclasses import replace
from resetmm.analytic import (ResetCosts, optimal_threshold, threshold_cost,
                             threshold_sensitivity, linearized_threshold)

class ThresholdResponseTests(unittest.TestCase):
    def setUp(self):
        self.costs = ResetCosts(D=1.0, a=.2, K=.1, c=.05)

    def test_gradient_matches_central_difference(self):
        gradient = threshold_sensitivity(self.costs)
        for name in gradient:
            h = 1e-5
            plus = replace(self.costs, **{name: getattr(self.costs, name)+h})
            minus = replace(self.costs, **{name: getattr(self.costs, name)-h})
            fd = (optimal_threshold(plus)-optimal_threshold(minus))/(2*h)
            self.assertAlmostEqual(gradient[name], fd, places=7)

    def test_sensitivity_signs(self):
        g = threshold_sensitivity(self.costs)
        self.assertLess(g['a'], 0)
        self.assertTrue(all(g[k] > 0 for k in ('D','K','c')))

    def test_first_order_boundary_error_is_second_order(self):
        errors = []
        for h in (.04, .02, .01):
            b = linearized_threshold(self.costs, {'D': h})
            exact = optimal_threshold(replace(self.costs, D=self.costs.D+h))
            errors.append(abs(b-exact))
        self.assertTrue(3.5 < errors[0]/errors[1] < 4.5)
        self.assertTrue(3.5 < errors[1]/errors[2] < 4.5)

    def test_local_cost_regret_scales_as_fourth_order(self):
        regrets = []
        for h in (.08, .04, .02):
            c = replace(self.costs, D=self.costs.D+h)
            b = linearized_threshold(self.costs, {'D': h})
            regrets.append(threshold_cost(b, c)-threshold_cost(optimal_threshold(c), c))
        self.assertTrue(all(r > 0 for r in regrets))
        self.assertTrue(13 < regrets[0]/regrets[1] < 19)
        self.assertTrue(13 < regrets[1]/regrets[2] < 19)

    def test_invalid_perturbations_rejected(self):
        for d in ({'wrong': .1}, {'D': -2}, {'a': float('nan')}):
            with self.assertRaises(ValueError):
                linearized_threshold(self.costs, d)
        with self.assertRaises(ValueError):
            threshold_sensitivity(ResetCosts(1,.1))
