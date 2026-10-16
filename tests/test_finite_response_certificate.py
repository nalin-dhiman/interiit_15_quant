"""Independent path enumeration and premise counterexamples for response bounds."""
import itertools
import math
import unittest

import numpy as np

from resetmm.finite_response_certificate import (
    UniformDerivativeBounds, derivative_envelopes, guarded_linear_policy,
    softmax_affine_payoff_bounds)
from resetmm.mdp import FiniteMDP, evaluate, fixed_policy_derivative, solve


def fixture(theta=0.0, horizon=3):
    """Known regular family with action-dependent kernel and explicit payoff."""
    logits = np.array([[[.3, -.1], [.1, .2]],
                       [[-.2, .4], [.2, -.3]]])
    direction = np.array([[[.08, -.12], [-.09, .11]],
                          [[-.05, .15], [.13, -.07]]])
    weights = np.exp(logits + theta * direction)
    p = weights / weights.sum(axis=-1, keepdims=True)
    dp = p * (direction - (p * direction).sum(axis=-1, keepdims=True))
    base = np.array([[.01, -.03], [-.02, -.04]])
    slope = np.array([[.02, .60], [-.03, .55]])
    terminal = np.array([.45, -.10]) + theta * np.array([.07, -.04])
    model = FiniteMDP(np.tile(p, (horizon, 1, 1, 1)),
                      np.tile(base + theta * slope, (horizon, 1, 1)), terminal)
    return (model, np.tile(dp, (horizon, 1, 1, 1)),
            np.tile(slope, (horizon, 1, 1)), np.array([.07, -.04]))


def family_bounds():
    # Whole interval [-.2,.2]: |g| <= .04+.2*.6=.16, terminal <= .464.
    # g is an already-expected affine reward, not a softmax primitive reward.
    return UniformDerivativeBounds(.1, .02, .16, .6, 0.0, .464, .07, 0.0)


def path_value(model, policy, start=0, time=0, first_action=None):
    """Sum complete successor paths directly, independent of Bellman recursion."""
    horizon, states, _, _ = model.P.shape
    total = 0.0
    for tail in itertools.product(range(states), repeat=horizon-time):
        history = (start,) + tail
        probability, reward = 1.0, 0.0
        for step, t in enumerate(range(time, horizon)):
            state, nxt = history[step:step+2]
            action = first_action if step == 0 and first_action is not None else policy[t, state]
            probability *= model.P[t, state, action, nxt]
            reward += model.reward[t, state, action]
        total += probability * (reward + model.terminal[history[-1]])
    return float(total)


def oracle_policy(model):
    horizon, states, actions, _ = model.P.shape
    records = []
    for entries in itertools.product(range(actions), repeat=horizon*states):
        policy = np.array(entries).reshape(horizon, states)
        if not model.feasible[np.arange(horizon)[:, None], np.arange(states), policy].all():
            continue
        values = tuple(path_value(model, policy, s) for s in range(states))
        records.append((values, policy))
    return np.max([x[0] for x in records], axis=0), records


class FiniteResponseCertificateTests(unittest.TestCase):
    def test_known_bound_covers_every_first_action_at_every_state_and_time(self):
        base, dp, dr, dt = fixture()
        pi = np.zeros((3, 2), int)
        for delta in (-.2, -.1, -.03, .0, .03, .1, .2):
            actual = fixture(delta)[0]
            response = guarded_linear_policy(base, pi, dp, dr, delta, family_bounds(),
                                              terminal_derivative=dt)
            for t, state, action in itertools.product(range(3), range(2), range(2)):
                direct = path_value(actual, pi, state, t, first_action=action)
                error = abs(direct-response.linear_action_values[t, state, action])
                self.assertLessEqual(error, response.action_error_bounds[t]+2e-15)

    def test_guarded_repeated_policy_improves_all_initial_states_in_known_family(self):
        base, dp, dr, dt = fixture()
        pi = np.zeros((3, 2), int)
        switches = 0
        for delta in (-.2, -.1, .0, .05, .1, .2):
            actual = fixture(delta)[0]
            response = guarded_linear_policy(base, pi, dp, dr, delta, family_bounds(),
                                              terminal_derivative=dt)
            switches += np.count_nonzero(response.policy != pi)
            for state in range(2):
                self.assertGreaterEqual(path_value(actual, response.policy, state)+2e-15,
                                        path_value(actual, pi, state))
        self.assertGreater(switches, 0)  # Reachable, nonvacuous action changes.

    def test_independent_exhaustive_oracle_bounds_direct_and_response_selectors(self):
        base, dp, dr, dt = fixture()
        pi = np.zeros((3, 2), int)
        for delta in (-.1, .1, .2):
            actual = fixture(delta)[0]
            oracle, records = oracle_policy(actual)  # All 64 policies, full paths.
            solved = solve(actual)[0][0]
            np.testing.assert_allclose(oracle, solved, atol=2e-15)
            response = guarded_linear_policy(base, pi, dp, dr, delta, family_bounds(),
                                              terminal_derivative=dt)
            baseline_value = evaluate(actual, pi)
            q = actual.reward + np.einsum("tsaj,tj->tsa", actual.P, baseline_value[1:])
            direct_policy = np.argmax(q, axis=-1)
            for state in range(2):
                self.assertLessEqual(path_value(actual, response.policy, state), oracle[state]+2e-15)
                self.assertLessEqual(path_value(actual, direct_policy, state), oracle[state]+2e-15)
            self.assertEqual(len(records), 64)

    def test_explicit_payoff_and_terminal_derivatives_are_required(self):
        base, dp, dr, dt = fixture()
        pi = np.ones((3, 2), int)
        _, full = fixed_policy_derivative(base, pi, dp, dr, dt)
        _, incomplete = fixed_policy_derivative(base, pi, dp, np.zeros_like(dr))
        h = 1e-5
        direct = ((path_value(fixture(h)[0], pi)-path_value(fixture(-h)[0], pi))/(2*h))
        self.assertAlmostEqual(full[0, 0], direct, places=9)
        self.assertGreater(abs(incomplete[0, 0]-direct), 1.5)

    def test_softmax_primitive_bounds_include_payoff_cross_derivative(self):
        bounds = softmax_affine_payoff_bounds(direction_span=2.0,
                    primitive_reward_abs=3.0, primitive_reward_slope_abs=.7,
                    terminal_abs=1.0)
        self.assertEqual(bounds.kernel_first, 1.0)
        self.assertEqual(bounds.kernel_second, 2.0)
        self.assertAlmostEqual(bounds.reward_first, 3.7)
        self.assertAlmostEqual(bounds.reward_second, 7.4)
        # Directly differentiate E_theta[c+theta*d] for five primitive atoms.
        x = np.array([-1., -.5, .2, .6, 1.])
        c = np.array([1., -2., .7, 1.2, -.9])
        d = np.array([.7, -.3, .2, .4, -.6])
        for theta in np.linspace(-.5, .5, 21):
            p = np.exp(theta*x); p /= p.sum()
            centered = x-p@x
            dp = p*centered
            ddp = p*(centered**2-p@(centered**2))
            reward, slope = c+theta*d, d
            self.assertLessEqual(np.abs(dp).sum(), bounds.kernel_first+1e-15)
            self.assertLessEqual(np.abs(ddp).sum(), bounds.kernel_second+1e-15)
            self.assertLessEqual(abs(ddp@reward+2*dp@slope), bounds.reward_second)

    def test_exact_two_error_boundary_retains_baseline(self):
        model = FiniteMDP(np.ones((1, 1, 2, 1)), np.array([[[0., 2.]]]), np.zeros(1))
        bounds = UniformDerivativeBounds(0, 0, 2, 0, 2, 0)
        result = guarded_linear_policy(model, np.zeros((1, 1), int),
                    np.zeros_like(model.P), np.zeros_like(model.reward), 1, bounds)
        self.assertEqual(result.action_error_bounds[0], 1)
        self.assertEqual(result.policy[0, 0], 0)

    def test_additional_bound_is_uniform_and_can_force_fallback(self):
        base, dp, dr, dt = fixture()
        pi = np.zeros((3, 2), int)
        result = guarded_linear_policy(base, pi, dp, dr, .1, family_bounds(),
                    terminal_derivative=dt, additional_uniform_error=100)
        np.testing.assert_array_equal(result.policy, pi)

    def test_infeasible_action_is_never_selected(self):
        base, dp, dr, dt = fixture()
        base.feasible[:, :, 1] = False
        pi = np.zeros((3, 2), int)
        result = guarded_linear_policy(base, pi, dp, dr, .1, family_bounds(),
                                              terminal_derivative=dt)
        np.testing.assert_array_equal(result.policy, pi)

    def test_invalid_and_overflowing_premises_are_rejected(self):
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                UniformDerivativeBounds(invalid, 0, 0, 0, 0, 0)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                derivative_envelopes(invalid, family_bounds())
        with self.assertRaises(ArithmeticError):
            derivative_envelopes(3, UniformDerivativeBounds(1e308, 1e308, 1, 1, 1, 1))

    def test_policy_switch_optimum_is_nondifferentiable_despite_smooth_branches(self):
        # Both branches have zero second derivative. Their maximum does not.
        h = 1e-4
        optimal = lambda theta: max(theta, -theta)
        left = (optimal(0)-optimal(-h))/h
        right = (optimal(h)-optimal(0))/h
        self.assertEqual((left, right), (-1, 1))
        self.assertGreater(optimal(h)-optimal(0), 0)

    def test_tick_crossing_invalidates_zero_curvature_premise(self):
        # A buy quote rounded down to ticks wins when it reaches 1.00.
        win = lambda theta: float(math.floor((.999+theta)*100)/100 >= 1.0)
        self.assertEqual(win(0), 0)
        self.assertEqual(win(.002), 1)
        # Its local derivative at zero is zero, not a certificate across .002.
        self.assertEqual((win(1e-5)-win(-1e-5))/(2e-5), 0)

    def test_support_boundary_needs_separate_one_sided_tangent(self):
        # P_theta(0,1)=theta has a one-sided matrix derivative at zero, but no
        # ordinary likelihood score for the absent jump at theta=0.
        p = np.array([[[[1., 0.]], [[0., 1.]]]])
        dp = np.array([[[[-1., 1.]], [[0., 0.]]]])
        model = FiniteMDP(p, np.zeros((1, 2, 1)), np.array([0., 1.]))
        with self.assertRaisesRegex(ValueError, "Support-changing"):
            fixed_policy_derivative(model, np.zeros((1, 2), int), dp, np.zeros((1, 2, 1)))
        actual = FiniteMDP(p+.1*dp, np.zeros((1, 2, 1)), np.array([0., 1.]))
        self.assertAlmostEqual(path_value(actual, np.zeros((1, 2), int)), .1)


if __name__ == "__main__":
    unittest.main()
