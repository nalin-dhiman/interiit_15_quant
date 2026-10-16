from __future__ import annotations

import copy
import unittest
import numpy as np

from resetmm.calibration import (
    bernoulli_losses, decreasing_pava, fit_fill_models, fit_markout_models,
    in_training_support, in_occupied_bin_support, predict_fill, predict_markout,
    validate_final_records,
)
from experiments.fit_synthetic_response import generate_session_records


def record(i, edge, won, markout=None):
    return dict(session_id='s', order_id=str(i), product_id='A', side='BUY',
                quantity=1, reference=100., inventory=0., final_quote=100.+edge,
                final_edge=edge, callback_count=3, won=won, markout=markout)


class CalibrationTests(unittest.TestCase):
    def test_duplicate_callbacks_are_rejected(self):
        row = record(0, .02, True)
        with self.assertRaises(ValueError):
            validate_final_records([row, dict(row)])

    def test_unobservable_feature_field_is_rejected(self):
        row = record(0, .02, True)
        row['hidden_limit'] = .05
        with self.assertRaises(ValueError):
            validate_final_records([row])

    def test_final_reference_must_reconcile_edge(self):
        row = record(0, .02, True)
        row['reference'] = 101.
        with self.assertRaises(ValueError):
            validate_final_records([row])

    def test_withdrawals_and_censored_outcomes_are_not_labelled_losses(self):
        rows = [record(0, .02, True), record(1, .02, None)]
        withdrawn = record(2, .02, False)
        withdrawn['final_quote'] = withdrawn['final_edge'] = None
        rows.append(withdrawn)
        model = fit_fill_models(rows, [0, .1])
        self.assertEqual(model['pooled']['probability'], 1)

    def test_markout_requires_observed_fill(self):
        with self.assertRaises(ValueError):
            validate_final_records([record(0, .02, False, .1)])

    def test_pava_pools_violation_with_correct_weights(self):
        np.testing.assert_allclose(decreasing_pava(np.array([9, 2, 4, 1]),
                                                  np.array([10, 10, 10, 10])),
                                   [.9, .3, .3, .1])
        np.testing.assert_allclose(decreasing_pava(np.array([1, 6]), np.array([2, 8])), [.7, .7])

    def test_pava_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            decreasing_pava(np.array([3]), np.array([2]))

    def test_logistic_recovers_known_edge_response(self):
        rng = np.random.default_rng(187)
        x = rng.choice([.025, .055, .085, .115, .145, .175], 5000)
        true = 1/(1+np.exp(-(1.8-.3*100*x)))
        outcomes = rng.random(len(x)) < true
        rows = [record(i, float(edge), bool(won)) for i, (edge, won) in enumerate(zip(x, outcomes))]
        models = fit_fill_models(rows, [0, .04, .07, .10, .13, .16, .2])
        self.assertTrue(models['logistic']['converged'])
        predictions = predict_fill(models['logistic'], [.025, .175])
        np.testing.assert_allclose(predictions, 1/(1+np.exp(-(1.8-.3*np.array([2.5, 17.5])))), atol=.025)

    def test_future_markout_labels_cannot_change_fill_coefficients(self):
        rows = [record(i, .02 if i < 10 else .08, i % 3 == 0,
                       .1 if i % 3 == 0 else None) for i in range(20)]
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['won']:
                row['markout'] = -999
        self.assertEqual(fit_fill_models(rows, [0, .05, .1]),
                         fit_fill_models(changed, [0, .05, .1]))

    def test_support_is_not_invented_from_bin_extent(self):
        rows = [record(0, .02, True), record(1, .08, False)]
        model = fit_fill_models(rows, [0, .05, .1])['monotone']
        np.testing.assert_array_equal(in_training_support(model, [.01, .02, .08, .09]),
                                      [False, True, True, False])

    def test_markout_shrinkage_and_censored_fill(self):
        rows = [record(0, .02, True, -.1), record(1, .02, True, .1),
                record(2, .08, True, .3), record(3, .08, True, None)]
        model = fit_markout_models(rows, [0, .05, .1], pseudocount=1)
        self.assertAlmostEqual(model['pooled_mean'], .1)
        np.testing.assert_allclose(predict_markout(model, [.02, .08], conditional=True), [1/30, .2])

    def test_empty_interior_bin_is_not_occupied_support(self):
        rows = [record(0, .02, True), record(1, .18, False)]
        models = fit_fill_models(rows, [0, .05, .1, .15, .2])
        for model in models.values():
            np.testing.assert_array_equal(in_training_support(model, [.08]), [True])
            np.testing.assert_array_equal(in_occupied_bin_support(model, [.02, .08, .18]),
                                          [True, False, True])

    def test_bernoulli_loss_hand_calculation(self):
        loss = bernoulli_losses([1, 0], [.75, .25])
        self.assertEqual(loss['brier'], .0625)
        self.assertAlmostEqual(loss['log_loss'], -np.log(.75))

    def test_public_collector_has_one_record_per_rfq_and_replays(self):
        config = dict(intervals=2, products=2, rfqs=3, updates=4, dt=.5,
                      exploration_edges=[.02, .08], exploration_stream_identifier=600613)
        records, metrics = generate_session_records(1000, 'stationary', config)
        replay, replay_metrics = generate_session_records(1000, 'stationary', config)
        self.assertEqual(records, replay)
        self.assertEqual(metrics, replay_metrics)
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(r['callback_count'] for r in records), 24)
        self.assertTrue(all(r['inventory'] == 0 for r in records))
        self.assertEqual(len({(r['session_id'], r['order_id']) for r in records}), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
