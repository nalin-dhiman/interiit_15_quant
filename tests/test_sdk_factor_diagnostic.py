import unittest
import numpy as np

from experiments.sdk_factor_diagnostic import (chronological_folds, low_rank_idiosyncratic,
                                                portfolio_risk, transient_example)


class SDKFactorDiagnosticTests(unittest.TestCase):
    def test_low_rank_idiosyncratic_symmetric_psd(self):
        matrix = np.array([[1., 1.8], [1.8, 4.]])
        fitted, common, idio = low_rank_idiosyncratic(matrix)
        np.testing.assert_allclose(fitted, fitted.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(fitted).min(), -1e-12)
        self.assertTrue(np.all(idio >= 0))
        np.testing.assert_allclose(np.diag(fitted), np.diag(matrix))
        self.assertEqual(np.linalg.matrix_rank(common, tol=1e-10), 1)

    def test_hedged_terminal_factor_keeps_intraday_basis_risk(self):
        # dA=F, dB=2F+E, independent Var(F)=1 and Var(E)=.1.
        covariance = np.array([[1, 2], [2, 4.1]])
        result = portfolio_risk([-2, 1], covariance, [1, 1], [50, 25], [50, 25])
        self.assertAlmostEqual(result["variance_per_update"], .1)
        self.assertTrue(result["within_individual_limits"])

    def test_multipliers_and_every_individual_limit(self):
        result = portfolio_risk([-4, 1], np.eye(2), [1, 4], [3, 2], [3, 2])
        self.assertAlmostEqual(result["variance_per_update"], 32)
        self.assertFalse(result["within_individual_limits"])

    def test_lowrank_idio_can_understate_hedged_direction_risk(self):
        covariance = np.array([[1., 2.], [2., 4.1]])
        fitted, _, _ = low_rank_idiosyncratic(covariance)
        exposure = np.array([-2., 1.])
        self.assertLess(float(exposure @ fitted @ exposure), float(exposure @ covariance @ exposure))
        # Matching each diagonal and PSD are insufficient risk certificates.
        np.testing.assert_allclose(np.diag(fitted), np.diag(covariance))

    def test_chronological_folds_have_no_future_training(self):
        folds = chronological_folds(1200)
        self.assertEqual(len(folds), 4)
        for begin, end, vbegin, vend in folds:
            self.assertEqual(begin, 0)
            self.assertEqual(end, vbegin)
            self.assertLess(vbegin, vend)
            self.assertLessEqual(vend, 1200)

    def test_stable_directed_map_has_finite_transient_growth(self):
        result = transient_example()
        self.assertLess(result["spectral_radius"], 1)
        self.assertGreater(result["peak"]["operator_norm"], 8)
        self.assertLess(result["rows"][-1]["operator_norm"], 1)
        # A^h e_2 = (4*h*.8^(h-1), .8^h) for h>=1.
        h = 4
        expected = np.hypot(4*h*.8**(h-1), .8**h)
        self.assertAlmostEqual(result["rows"][h]["unit_second_coordinate_norm"], expected)


if __name__ == "__main__":
    unittest.main()
