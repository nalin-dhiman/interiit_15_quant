"""Independent forward recursion checks for the reachable-response diagnostic."""
from dataclasses import asdict
import unittest
import numpy as np

from experiments.run_reachable_response import (CandidateOracle, build_model,
    first_switch_probability, occupancy, occupancy_moments)
from experiments.run_continuation_response import moments
from resetmm.continuation import BandPolicy, ImprovedPolicy, ModelConfig
from resetmm.continuation_response import LinearResponsePolicy, PrimitiveTiltFamily, eligibility_tilt_directions


def specification(horizon=3):
    return dict(model_config=asdict(ModelConfig(horizon=horizon, limit_lots=5)),
        regimes=["balanced", "persistent", "informed"],
        rfqs=[[["BUY", 1]], [["SELL", 1]], [["BUY", 2], ["BUY", 1]],
              [["SELL", 2], ["SELL", 1]], [["BUY", 1], ["SELL", 2]]],
        atom_classes=[[.35, .18, .10], [.4, .10, .06], [.25, .04, .03]],
        tie_uniforms=[.1, .9], uninformed_price_changes=[-.02, .02])


class ReachableResponseTests(unittest.TestCase):
    def test_unit_rfqs_make_nonmultiples_of_three_reachable(self):
        model = build_model(specification(), "balanced")
        policy = BandPolicy(.02, 0., None, 0.).bind(model)
        levels = occupancy(model, policy)
        self.assertIn((1, -1, 0), levels[1])
        self.assertIn((1, 1, 0), levels[1])
        self.assertGreater(levels[1][1, -1, 0], 0.)
        for level in levels:
            self.assertAlmostEqual(sum(level.values()), 1.)

    def test_forward_moments_agree_with_independent_backward_recursion(self):
        model = build_model(specification(), "informed")
        policy = ImprovedPolicy(model, BandPolicy(.06, .01, 3, .5))
        expected = moments(model, policy)
        forward = occupancy_moments(model, policy, occupancy(model, policy))
        np.testing.assert_allclose(forward, [expected[key] for key in
            ("raw_pnl", "risk_penalty", "fees", "exchange_quantity", "customer_orders",
             "customer_quantity", "inventory_sq_time")], atol=1e-12, rtol=0)

    def test_matching_candidate_oracle_dominates_on_same_model(self):
        model = build_model(specification(), "persistent")
        band = BandPolicy(.08, .01, 3, .5)
        oracle = CandidateOracle(model, band)
        r2 = ImprovedPolicy(model, band)
        self.assertGreaterEqual(moments(model, oracle)["objective"]+1e-12,
                                moments(model, r2)["objective"])
        for q in range(-5, 6):
            for b in range(5):
                self.assertEqual(oracle.candidates(0, q, b), r2.candidates(0, q, b))

    def test_first_switch_is_zero_for_identical_policy(self):
        model = build_model(specification(), "balanced")
        band = BandPolicy(.08, .01, None, 0.)
        policy = ImprovedPolicy(model, band)
        self.assertAlmostEqual(first_switch_probability(model, policy, policy), 0.)

    def test_initial_action_difference_has_unit_first_switch_probability(self):
        model = build_model(specification(), "balanced")
        first = BandPolicy(.02, 0., None, 0.).bind(model)
        second = BandPolicy(.08, 0., None, 0.).bind(model)
        self.assertEqual(first_switch_probability(model, first, second), 1.)

    def test_zero_response_equals_frozen_r2_on_every_reachable_state(self):
        model = build_model(specification(), "balanced")
        band = BandPolicy(.08, .01, None, 0.)
        family = PrimitiveTiltFamily(model, eligibility_tilt_directions(model), .02)
        r2, r3 = ImprovedPolicy(model, band), LinearResponsePolicy(family, band, 0.)
        levels = occupancy(model, r2)
        for level in levels[:-1]:
            for state in level:
                self.assertEqual(r2(*state), r3(*state))


if __name__ == "__main__":
    unittest.main()
