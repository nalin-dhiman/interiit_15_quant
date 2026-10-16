"""Causal-feature, numerical-context and abstention regressions."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import unittest

import numpy as np

from resetmm.research_response import (DOMAIN, EDGES, fit_models, predict,
    public_features, validate_records, markout_fit, markout_predict)
from experiments.sdk_intervention_response import quote_at, sdk_fill_probability


def fixture():
    rows = []
    for theta in (.2, .8):
        for side in ("BUY", "SELL"):
            for edge in EDGES:
                for i in range(12):
                    rows.append(dict(session_seed=len(rows), order_id="r1", theta=theta,
                        product_id="A", side=side, edge=edge, matching_reference=99.96,
                        current_reference=99.96, own_quote=quote_at(99.96, edge, side),
                        quantity=2, previous_increment=.02, won=i < (8 if theta == .8 else 3),
                        signed_markout=.01, shifted_family=False))
    return rows


class TestResearchResponse(unittest.TestCase):
    def test_latent_future_and_outcome_fields_do_not_change_features(self):
        row = fixture()[0]
        before = public_features(row)
        row.update(won=not row["won"], signed_markout=9999., expected_signed_markout=999.,
                   intervention_family=999, primitive_score=999., limit_offset=999., public_winner="future")
        np.testing.assert_array_equal(before, public_features(row))

    def test_duplicate_callbacks_rejected_but_distinct_treatment_arm_allowed(self):
        row = fixture()[0]
        with self.assertRaises(ValueError):
            validate_records([row, copy.deepcopy(row)])
        second = dict(row, theta=.8)
        validate_records([row, second])

    def test_public_float_context_detects_actual_sdk_boundary(self):
        row = dict(fixture()[0], edge=.08, own_quote=100.04)
        self.assertEqual(public_features(row)[7], 0.)
        self.assertAlmostEqual(sdk_fill_probability(.08, 1, 99.96, "BUY"), 2/3)
        row.update(matching_reference=100., current_reference=100., own_quote=100.08)
        self.assertEqual(public_features(row)[7], 1.)
        self.assertAlmostEqual(sdk_fill_probability(.08, 1, 100., "BUY"), 5/6)

    def test_linear_mixture_interpolation_is_not_extra_response_information(self):
        rows = fixture()
        model = fit_models(rows, minimum_cell_n=2)
        row = rows[0]
        p = [predict(model, [dict(row, theta=t)], "conditional_finite")[0][0] for t in (.2, .5, .8)]
        self.assertAlmostEqual(p[1], (p[0]+p[2])/2)
        self.assertGreater(p[2], p[0])
        fee = lambda t: .01+.01*t
        payoff = lambda t: ((t-.2)/.6*p[2]+(.8-t)/.6*p[0])*(.04-fee(t))
        h = .05
        derivative = (p[2]-p[0])/.6*(.04-fee(.5))-.01*p[1]
        self.assertAlmostEqual(derivative, (payoff(.5+h)-payoff(.5-h))/(2*h))

    def test_domain_and_support_guards_abstain_instead_of_extrapolate(self):
        rows = fixture()
        model = fit_models(rows, minimum_cell_n=2)
        for changes in (dict(theta=.81), dict(shifted_family=True), dict(product_id="C"),
                        dict(quantity=3), dict(current_reference=100.1), dict(previous_increment=1.)):
            p, support = predict(model, [dict(rows[0], **changes)], "conditional_finite")
            self.assertFalse(support[0])
            self.assertTrue(np.isnan(p[0]))
        self.assertFalse(predict(model, [rows[0]], "constant", domain="unknown-law")[1][0])
        model["minimum_cell_n"] = 10000
        self.assertFalse(predict(model, [rows[0]], "conditional_finite")[1][0])

    def test_markout_uses_observed_own_fills_and_only_past_features(self):
        rows = fixture()
        model = markout_fit(rows)
        self.assertEqual(model["training_fills"], sum(r["won"] for r in rows))
        before = markout_predict(model, rows)
        for row in rows:
            row.update(signed_markout=99999., expected_signed_markout=-9999., public_winner="future")
        np.testing.assert_array_equal(before, markout_predict(model, rows))


if __name__ == "__main__":
    unittest.main()
