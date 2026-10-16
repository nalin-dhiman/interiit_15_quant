"""Independent regressions for the post-seal fractional-lot defect.

These fail against the sealed 1b24c46 controller, whose study used integer lots.
The fix must be applied only after its source-frozen study has completed.
"""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import importlib.util
import unittest
from pathlib import Path

from rfq.sdk import round_lot
from submission.sdk_dealer import Dealer
from tests_sdk.test_sdk_controller import market, portfolio, order


class FractionalLotAuditTests(unittest.TestCase):
    def test_selected_one_tenth_sale_is_actually_sent(self):
        m = market(closing=True)
        m["products"]["A"]["lot_size"] = .1
        d = Dealer(mode="R2", risk_weight=.02, min_improvement=0)
        # Fee .001 and risk .0025 at q=.5 beats retaining .6 (risk .0036).
        self.assertEqual(d.hedge(m, portfolio(A=.6)), [("A", "SELL", .1)])
        diagnostic = d.last_decisions[-1]
        self.assertEqual(diagnostic["target"], .5)
        self.assertAlmostEqual(diagnostic["candidate_values"][.5], -.0035)

    def test_selected_two_tenths_sale_and_fee_are_not_one_tenth(self):
        m = market(closing=True)
        m["products"]["A"]["lot_size"] = .1
        d = Dealer(mode="R2", risk_weight=1 / 90, min_improvement=0)
        self.assertEqual(d.hedge(m, portfolio(A=1.1)), [("A", "SELL", .2)])
        diagnostic = d.last_decisions[-1]
        actual_target = 1.1 - .2
        self.assertEqual(diagnostic["target"], actual_target)
        self.assertAlmostEqual(diagnostic["candidate_values"][actual_target],
                               -.2 * .01 - (1 / 90) * actual_target ** 2 * .5)

    def test_conventional_band_executes_its_constructed_target(self):
        m = market()
        m["products"]["A"].update(lot_size=.1, max_long=1, max_short=1)
        d = Dealer(mode="B3", band_fraction=.5, target_fraction=.99)
        target = d._band_target(.6, d._spec("A", m))
        trades = d.hedge(m, portfolio(A=.6))
        self.assertTrue(trades)
        actual_delta = -trades[0][2]
        self.assertEqual(.6 + actual_delta, target)

    def test_integer_quantity_rule_matches_original_sdk_grid(self):
        # Sealed experiments use integer lots and integer physical inventories.
        # Their target construction must retain the original arithmetic path.
        for lot in (1., 2., 3., 5.):
            for q in range(-30, 31):
                for quarter_target in range(-120, 121):
                    target = quarter_target / 4
                    self.assertEqual(Dealer._quantity_for_target(q, target, lot),
                                     round_lot(target - q, lot))

    def test_integer_policy_actions_match_frozen_study_controller(self):
        fixture = Path(__file__).resolve().parents[1] / "tests_sdk/fixtures/sdk_dealer_before_fractional_fix.py"
        self.assertTrue(fixture.exists(), "Preserve the sealed source before applying the post-seal fix")
        spec = importlib.util.spec_from_file_location("sealed_fractional_audit", fixture)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in ("B2", "B3", "R2"):
            for q in range(-25, 26):
                for closing in (False, True):
                    for risk in (0., .0002, .01):
                        kwargs = dict(mode=mode, risk_weight=risk,
                                      settlement_coefficients={"B": {"A": 2}})
                        old, new = module.Dealer(**kwargs), Dealer(**kwargs)
                        orders = [] if closing else [order("a", "BUY", 2), order("b", "SELL", 4)]
                        m, p = market(orders, closing=closing), portfolio(A=q)
                        self.assertEqual(old.quote(orders, m, p), new.quote(orders, m, p))
                        self.assertEqual(old.hedge(m, p), new.hedge(m, p))
                        self.assertEqual(list(old.last_decisions), list(new.last_decisions))


if __name__ == "__main__":
    unittest.main()
