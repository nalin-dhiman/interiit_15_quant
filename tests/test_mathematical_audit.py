"""Independent fixtures and counterexamples for the mathematical audit.

Counterexample tests preserve the assumptions needed by the repaired claims;
they are not evidence that the synthetic market represents an actual market.
"""
import itertools
import math
import unittest
from dataclasses import replace

import numpy as np

from resetmm.accounting import Ledger, Product
from resetmm.analytic import (ResetCosts, linearized_threshold,
                             optimal_poisson_rate, optimal_threshold,
                             poisson_cost, threshold_cost)
from resetmm.mdp import (FiniteMDP, evaluate, fixed_policy_derivative,
                         model_value_error_bound, safe_switch, solve)
from resetmm.response import bernoulli_score
from resetmm.risk import fill_envelope


class BrownianAuditTests(unittest.TestCase):
    def test_small_poisson_root_has_relative_resolution(self):
        costs = ResetCosts(1, 1, 1e200)
        expected = math.sqrt(2/costs.K)
        self.assertLess(abs(optimal_poisson_rate(costs)/expected-1), 1e-14)

    def test_small_threshold_root_has_relative_resolution(self):
        costs = ResetCosts(1, 1, 1e-200)
        expected = (12*costs.K)**.25
        self.assertLess(abs(optimal_threshold(costs)/expected-1), 1e-14)

    def test_proportional_only_ratio_is_not_fixed_cost_ratio(self):
        costs = ResetCosts(2, .3, 0, .7)
        ratio = (threshold_cost(optimal_threshold(costs), costs)
                 / poisson_cost(optimal_poisson_rate(costs), costs))
        self.assertAlmostEqual(ratio, 3**(-1/3), places=13)
        self.assertGreater(abs(ratio-1/math.sqrt(6)), .2)

    def test_explicit_local_quartic_regret_bounds(self):
        # D is dimensionless after normalisation at D0=1. For D in [1,1.2],
        # b(D)=12**(1/4)*D**(1/4), |b''|<=B. Along both boundaries b>=b0,
        # g''=1/3+12*D/b**4 <= L. This supplies actual constants, not a fit.
        costs = ResetCosts(1, 1, 1)
        b0 = 12**.25
        B, L = 3*b0/16, 1/3+1.2
        for h in (.2, .1, .05, .025):
            new = replace(costs, D=1+h)
            proposal = linearized_threshold(costs, {"D": h})
            exact = (12*(1+h))**.25
            error = abs(proposal-exact)
            regret = threshold_cost(proposal, new)-threshold_cost(exact, new)
            self.assertLessEqual(error, B*h*h/2+1e-15)
            self.assertGreaterEqual(regret, -1e-15)
            self.assertLessEqual(regret, L*B*B*h**4/8+1e-15)

    def test_fixed_lot_grid_can_leave_nonzero_regret_at_zero_change(self):
        costs = ResetCosts(1, 1, 1)
        b = optimal_threshold(costs)
        grid = .25
        rounded = round(b/grid)*grid
        regret = threshold_cost(rounded, costs)-threshold_cost(b, costs)
        self.assertGreater(regret, .005)
        # A fixed grid does not inherit a vanishing O(delta**4) error.
        self.assertLessEqual(abs(rounded-b), grid/2)

    def test_large_linear_update_is_not_uniformly_accurate(self):
        costs = ResetCosts(1, 1, 1)
        new = replace(costs, D=101)
        proposal = linearized_threshold(costs, {"D": 100})
        self.assertGreater(proposal/optimal_threshold(new), 8)
        self.assertGreater(threshold_cost(proposal, new)
                           / threshold_cost(optimal_threshold(new), new), 30)


class AccountingAuditTests(unittest.TestCase):
    def test_martingale_cash_only_normalisation_needs_zero_initial_inventory(self):
        # Constant price is a bounded martingale. A donated initial share is
        # omitted if only c0 is subtracted. This is the exact proof gap found.
        c0, q0, s0 = 0, 1, 100
        cash_normalised_pnl = c0+q0*s0-c0
        wealth_normalised_gain = c0+q0*s0-(c0+q0*s0)
        self.assertEqual(cash_normalised_pnl, 100)
        self.assertEqual(wealth_normalised_gain, 0)

    def test_real_reset_preserves_loss_and_books_fee(self):
        ledger = Ledger({"X": Product("X", multiplier=3, fee_per_unit=.2)})
        ledger.customer("X", "SELL", 2, 100)
        before = ledger.pnl({"X": 90})
        self.assertEqual(before, -60)
        ledger.exchange("X", -2, 90)
        self.assertAlmostEqual(ledger.pnl({"X": 90}), -60.4)
        self.assertEqual(ledger.positions["X"], 0)
        ledger.reconcile()

    def test_impact_identity_includes_post_trade_exposure(self):
        ledger = Ledger({"X": Product("X", multiplier=3, fee_per_unit=.2)})
        ledger.customer("X", "SELL", 2, 100)
        before = ledger.wealth({"X": 100})
        ledger.exchange("X", -1, 101)
        actual = ledger.wealth({"X": 103})-before
        predicted = -3*(-1)*(101-100)-.2+3*(2-1)*(103-100)
        self.assertAlmostEqual(actual, predicted)
        self.assertAlmostEqual(actual, 11.8)

    def test_adaptive_predictable_martingale_tree_loses_only_fees(self):
        outcomes = []
        for first, second in itertools.product((-1, 1), repeat=2):
            ledger = Ledger({"X": Product("X", fee_per_unit=.1)})
            ledger.exchange("X", 1, 100)
            price = 100+first
            # Position after observing first shock, before second shock.
            target = -1 if first > 0 else 1
            ledger.exchange("X", target-ledger.positions["X"], price)
            outcomes.append((ledger.pnl({"X": price+second}), ledger.fees))
        self.assertAlmostEqual(np.mean([p for p, f in outcomes]),
                               -np.mean([f for p, f in outcomes]))

    def test_terminal_residual_is_required_without_liquidation(self):
        ledger = Ledger({"X": Product("X", multiplier=2)})
        ledger.customer("X", "SELL", 3, 100)
        marked = ledger.pnl({"X": 100})
        final = ledger.pnl({"X": 105})
        self.assertEqual(final-marked, 2*3*(105-100))
        self.assertEqual(ledger.exchange_notional, 0)

    def test_mixed_quote_matching_markout_references_double_count_move(self):
        quote_price, quote_reference, match_reference, later = 102, 100, 101, 103
        matching_edge = quote_price-match_reference
        aligned_markout = later-match_reference
        misaligned_markout = later-quote_reference
        actual_new_short_pnl = quote_price-later
        self.assertEqual(matching_edge-aligned_markout, actual_new_short_pnl)
        self.assertEqual(actual_new_short_pnl, -1)
        self.assertEqual(matching_edge-misaligned_markout, -2)

    def test_pending_position_change_invalidates_unadjusted_reservation(self):
        lo, hi = fill_envelope(0, [("SELL", 1)])
        self.assertEqual((lo, hi), (0, 1))  # safe for [-1,1] if q stays zero
        confirmed_after_pending_buy = 1
        _, actual_hi = fill_envelope(confirmed_after_pending_buy, [("SELL", 1)])
        self.assertEqual(actual_hi, 2)  # missing premise cannot be inferred away


class FiniteControlAuditTests(unittest.TestCase):
    @staticmethod
    def independent_path_value(m, policy, initial):
        # Enumerate every state path explicitly, without Bellman evaluation.
        total = 0.
        T, S, _, _ = m.P.shape
        for successors in itertools.product(range(S), repeat=T):
            state, probability, reward = initial, 1., 0.
            for t, next_state in enumerate(successors):
                action = policy[t, state]
                reward += m.reward[t, state, action]
                probability *= m.P[t, state, action, next_state]
                state = next_state
            total += probability*(reward+m.terminal[state])
        return total

    def test_dp_matches_all_64_policies_and_explicit_path_enumeration(self):
        rng = np.random.default_rng(20261015)
        P = rng.uniform(.1, 1, (3, 2, 2, 2))
        P /= P.sum(-1, keepdims=True)
        m = FiniteMDP(P, rng.normal(size=(3, 2, 2)), np.array([.4, -.7]))
        V, pi, _ = solve(m)
        policies = (np.array(p).reshape(3, 2)
                    for p in itertools.product(range(2), repeat=6))
        enumerated = np.array([[self.independent_path_value(m, p, s)
                                for s in range(2)] for p in policies])
        np.testing.assert_allclose(V[0], enumerated.max(0), atol=1e-13, rtol=0)
        for s in range(2):
            self.assertAlmostEqual(V[0, s], self.independent_path_value(m, pi, s))

    def test_perturbation_bound_on_independent_finite_pairs(self):
        rng = np.random.default_rng(271828)
        for _ in range(20):
            P = rng.uniform(.1, 1, (3, 2, 2, 2))
            P /= P.sum(-1, keepdims=True)
            P2 = .93*P+.07*np.flip(P, axis=-1)
            r = rng.uniform(-1, 1, (3, 2, 2))
            r2 = r+rng.uniform(-.04, .04, r.shape)
            term = rng.uniform(-.5, .5, 2)
            term2 = term+rng.uniform(-.03, .03, 2)
            m, m2 = FiniteMDP(P, r, term), FiniteMDP(P2, r2, term2)
            bound = model_value_error_bound(
                3, max(abs(r).max(), abs(r2).max()),
                max(abs(term).max(), abs(term2).max()), abs(r-r2).max(),
                abs(term-term2).max(), abs(P-P2).sum(-1).max())
            self.assertLessEqual(abs(solve(m)[0][0]-solve(m2)[0][0]).max(), bound)
            policy = rng.integers(0, 2, (3, 2))
            self.assertLessEqual(abs(evaluate(m, policy)[0]
                                     -evaluate(m2, policy)[0]).max(), bound)

    def test_common_feasible_action_set_is_a_necessary_premise(self):
        P = np.ones((1, 1, 2, 1))
        reward = np.array([[[0., 100.]]])
        m = FiniteMDP(P, reward, np.zeros(1), np.array([[[True, False]]]))
        m2 = FiniteMDP(P, reward, np.zeros(1))
        invalid_bound = model_value_error_bound(1, 100, 0, 0, 0, 0)
        gap = abs(solve(m)[0][0, 0]-solve(m2)[0][0, 0])
        self.assertEqual(invalid_bound, 0)
        self.assertEqual(gap, 100)

    def test_guard_uses_strict_two_error_margin(self):
        eps = .1
        self.assertEqual(safe_switch(np.array([0., 2*eps]), 0, eps), 0)
        candidate = 2*eps+1e-4
        self.assertEqual(safe_switch(np.array([0., candidate]), 0, eps), 1)
        worst_true_baseline, worst_true_candidate = eps, candidate-eps
        self.assertGreater(worst_true_candidate, worst_true_baseline)

    def test_empirical_margin_is_not_a_certificate(self):
        estimated = np.array([0., .3])
        actual = np.array([1., 0.])
        chosen = safe_switch(estimated, 0, .1)
        self.assertEqual(chosen, 1)
        self.assertLess(actual[chosen], actual[0])
        self.assertGreater(abs(estimated-actual).max(), .1)

    def test_nonfinite_error_bounds_are_rejected(self):
        valid = [2, 1., 1., .1, .1, .1]
        for index in range(len(valid)):
            for bad in (float("nan"), float("inf"), -float("inf")):
                args = valid.copy()
                args[index] = bad
                with self.assertRaises(ValueError):
                    model_value_error_bound(*args)

    def test_overflowed_error_certificate_is_rejected(self):
        with self.assertRaises(ArithmeticError):
            model_value_error_bound(3, 1e308, 1, 0, 0, 1)


class ResponseSupportAuditTests(unittest.TestCase):
    def test_parameter_dependent_payoff_needs_explicit_derivative(self):
        p, dp, theta = .4, .2, 2.
        score_term = sum(prob*theta*y*bernoulli_score(y, p, dp)
                         for y, prob in ((False, 1-p), (True, p)))
        explicit_term = p  # derivative of theta*fill holding primitive fill fixed
        self.assertAlmostEqual(score_term, theta*dp)
        self.assertAlmostEqual(score_term+explicit_term, p+theta*dp)
        self.assertNotAlmostEqual(score_term, p+theta*dp)

    def test_escape_rate_term_is_needed_even_for_constant_observable(self):
        # Poisson event count N over T: E[N/r]=T, while the correct score
        # N/r-T has mean zero. This exact identity needs no Monte Carlo.
        rate, horizon = .7, 3.
        naive_expected_score = (rate*horizon)/rate
        correct_expected_score = naive_expected_score-horizon
        self.assertAlmostEqual(naive_expected_score, horizon)
        self.assertAlmostEqual(correct_expected_score, 0)

    def test_zero_rate_response_has_missing_mass_under_baseline_likelihood(self):
        # P_theta(Y=1)=theta, theta>=0. At zero the right derivative is one,
        # but no realised baseline path contains Y=1 to carry that derivative.
        h = 1e-6
        right_derivative = (h-0)/h
        self.assertEqual(right_derivative, 1)
        with self.assertRaises(ValueError):
            bernoulli_score(False, 0, 1)
        P = np.array([[[[1., 0.]], [[0., 1.]]]])
        dP = np.array([[[[-1., 1.]], [[0., 0.]]]])
        m = FiniteMDP(P, np.zeros((1, 2, 1)), np.array([0., 1.]))
        with self.assertRaisesRegex(ValueError, "Support-changing"):
            fixed_policy_derivative(m, np.zeros((1, 2), int), dP, np.zeros((1, 2, 1)))

    def test_optimal_policy_switch_produces_a_value_kink(self):
        def value(theta):
            m = FiniteMDP(np.ones((1, 1, 2, 1)),
                          np.array([[[theta, -theta]]]), np.zeros(1))
            return solve(m)[0][0, 0]
        h = 1e-6
        left, right = (value(0)-value(-h))/h, (value(h)-value(0))/h
        self.assertEqual((left, right), (-1, 1))
        self.assertEqual((value(h)-value(-h))/(2*h), 0)

    def test_terminal_payoff_derivative_is_not_omitted(self):
        P = np.array([[[[.25, .75]], [[.5, .5]]]])
        m = FiniteMDP(P, np.zeros((1, 2, 1)), np.zeros(2))
        _, derivative = fixed_policy_derivative(
            m, np.zeros((1, 2), int), np.zeros_like(P),
            np.zeros((1, 2, 1)), np.array([2., -1.]))
        np.testing.assert_allclose(derivative[0], [-.25, .5], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
