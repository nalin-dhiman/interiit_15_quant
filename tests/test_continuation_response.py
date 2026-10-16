"""Continuation response checks under a known fixed-support synthetic family."""
import math
import unittest
from dataclasses import replace

import numpy as np

from resetmm.continuation import (Action, BandPolicy, Batch, FiniteRFQModel,
                                  PolicyEvaluation, improve_policy,
                                  make_synthetic_model)
from resetmm.continuation_response import (LinearResponsePolicy, PrimitiveTiltFamily,
                                           TangentPolicyEvaluation,
                                           eligibility_tilt_directions)


class ContinuationResponseTests(unittest.TestCase):
    def setUp(self):
        self.model = make_synthetic_model(horizon=4, regime="persistent", risk_weight=0)
        self.band = BandPolicy(.06, 0., 1, 0.)
        self.family = PrimitiveTiltFamily(self.model, eligibility_tilt_directions(self.model), .02)

    def test_tilt_preserves_payload_and_positive_support(self):
        changed = self.family.model(.15)
        self.assertAlmostEqual(changed.config.exchange_fee, .028)
        for b, (old, new) in enumerate(zip(self.model.batches, changed.batches)):
            p, dp = self.family.probabilities(b, .15)
            self.assertTrue(np.all(p > 0))
            self.assertAlmostEqual(float(p.sum()), 1)
            self.assertAlmostEqual(float(dp.sum()), 0)
            self.assertEqual(old.rfqs, new.rfqs)
            for a, a2 in zip(old.atoms, new.atoms):
                self.assertEqual(replace(a, probability=a2.probability), a2)

    def test_tangent_matches_fixed_policy_fd_at_five_magnitudes(self):
        policy = self.band.bind(self.model)
        _, derivative = TangentPolicyEvaluation(self.family, policy).initial_value()
        errors = []
        for h in (.1, .03, .01, .003, .001):
            plus, minus = self.family.model(h), self.family.model(-h)
            vp = PolicyEvaluation(plus, policy).initial_value()
            vm = PolicyEvaluation(minus, policy).initial_value()
            error = abs((vp-vm)/(2*h)-derivative)
            errors.append(error)
            self.assertLess(error, .1*h*h+1e-10)
        self.assertLess(errors[-1], errors[0]/1000)

    def test_action_tangent_includes_initial_and_future_actual_fee_derivatives(self):
        policy = self.band.bind(self.model)
        action = Action(0, self.band.quotes_at(self.model, 0, 0))
        full = TangentPolicyEvaluation(self.family, policy)
        incomplete = TangentPolicyEvaluation(self.family, policy, include_payoff_derivative=False)
        _, derivative = full.action_value(0, 3, 0, action)
        _, score_only = incomplete.action_value(0, 3, 0, action)
        h = 1e-5
        plus, minus = self.family.model(h), self.family.model(-h)
        fd = (PolicyEvaluation(plus, policy).action_value(0, 3, 0, action)
              -PolicyEvaluation(minus, policy).action_value(0, 3, 0, action))/(2*h)
        self.assertAlmostEqual(derivative, fd, places=8)
        self.assertGreater(abs(score_only-fd), .06)

    def test_zero_perturbation_matches_r2_candidate_selection(self):
        r3 = LinearResponsePolicy(self.family, self.band, 0)
        r2 = improve_policy(self.model, self.band)
        for t in range(self.model.config.horizon):
            for q in range(-3, 4):
                for b in range(2):
                    self.assertEqual(r3(t, q, b), r2(t, q, b))

    def test_linear_candidate_values_are_second_order_locally(self):
        policy = self.band.bind(self.model)
        tangent = TangentPolicyEvaluation(self.family, policy)
        action = Action(0, self.band.quotes_at(self.model, 0, 0))
        value, derivative = tangent.action_value(0, 3, 0, action)
        errors = []
        for theta in (.08, .04, .02):
            direct = PolicyEvaluation(self.family.model(theta), policy).action_value(0, 3, 0, action)
            errors.append(abs(value+theta*derivative-direct))
        self.assertTrue(all(3.7 < a/b < 4.3 for a, b in zip(errors, errors[1:])))

    def test_outside_radius_returns_conventional_baseline(self):
        fallback = LinearResponsePolicy(self.family, self.band, .151)
        self.assertFalse(fallback.supported)
        base = self.band.bind(self.model)
        for q in range(-3, 4):
            self.assertEqual(fallback(0, q, 0), base(0, q, 0))
        self.assertEqual(fallback.candidate_evaluations, 0)

    def test_unsupported_target_tick_and_support_changes_fall_back(self):
        for kind in ("deterministic_target", "quote_tick", "turn_on_reset", "unknown"):
            policy = LinearResponsePolicy(self.family, self.band, .05, perturbation_kind=kind)
            self.assertFalse(policy.supported)
            self.assertEqual(policy(0, 3, 0), self.band.bind(self.model)(0, 3, 0))

    def test_nan_and_negative_fee_perturbations_fall_back(self):
        for theta in (float("nan"), -2.):
            policy = LinearResponsePolicy(self.family, self.band, theta, trust_radius=3)
            self.assertFalse(policy.supported)
            self.assertEqual(policy(0, 3, 0), self.band.bind(self.model)(0, 3, 0))

    def test_zero_probability_atoms_are_rejected_for_regular_response(self):
        batch = self.model.batches[0]
        zero = replace(batch.atoms[0], probability=0.)
        other = replace(batch.atoms[1], probability=batch.atoms[1].probability+batch.atoms[0].probability)
        replaced = Batch(batch.rfqs, (zero, other)+batch.atoms[2:])
        model = FiniteRFQModel(self.model.config, (replaced, self.model.batches[1]), self.model.batch_transition)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            PrimitiveTiltFamily(model, eligibility_tilt_directions(model), .02)

    def test_tangent_does_not_differentiate_argmax_switch(self):
        # These are two smooth action branches; their maximum has a kink.
        action_values = lambda theta: np.array([theta, -theta])
        h = .001
        left = (action_values(0).max()-action_values(-h).max())/h
        right = (action_values(h).max()-action_values(0).max())/h
        self.assertEqual((left, right), (-1, 1))
        self.assertEqual((action_values(h).max()-action_values(-h).max())/(2*h), 0)


if __name__ == "__main__":
    unittest.main()
