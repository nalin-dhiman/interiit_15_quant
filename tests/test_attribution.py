"""Hand examples preserve the inferential unit and planning alternatives."""
import unittest
from resetmm.attribution import path_interval, plan_sample


class AttributionTests(unittest.TestCase):
    def test_scenario_mean_differences_do_not_inflate_within_stratum_variance(self):
        plan = plan_sample([[100., 102.], [-100., -98.]], floor=16, cap=64)
        self.assertEqual(plan['within_stratum_effective_variance'], 2.)
        self.assertEqual(plan['required_independent_paths'], 16)
        self.assertEqual(plan['final_paths'], 16)

    def test_cap_and_rounding_are_reported(self):
        plan = plan_sample([[-100., 100.]] * 8, floor=16, cap=64)
        self.assertEqual(plan['final_paths'], 64)
        self.assertTrue(plan['precision_cap_binds'])

    def test_constant_material_gain_passes_only_strict_threshold(self):
        self.assertTrue(path_interval([[2.] * 3] * 8, 7)['material_benefit_supported'])
        self.assertFalse(path_interval([[1.] * 3] * 8, 7)['material_benefit_supported'])

    def test_no_significance_is_not_equivalence(self):
        result = path_interval([[-100., 100.]] * 8, 3)
        self.assertFalse(result['material_benefit_supported'])
        self.assertFalse(result['practical_equivalence_supported'])

    def test_constant_negative_and_small_effect_have_different_claims(self):
        negative = path_interval([[-2.] * 3] * 8, 7)
        self.assertTrue(negative['material_benefit_excluded'])
        self.assertFalse(negative['practical_equivalence_supported'])
        small = path_interval([[.1] * 3] * 8, 7)
        self.assertTrue(small['practical_equivalence_supported'])
        self.assertEqual(small['independent_paths'], 24)

    def test_nonfinite_or_underspecified_samples_fail(self):
        for groups in ([], [[1]], [[1., float('nan')]]):
            with self.assertRaises(ValueError): path_interval(groups, 2)
            with self.assertRaises(ValueError): plan_sample(groups)


if __name__ == '__main__': unittest.main()
