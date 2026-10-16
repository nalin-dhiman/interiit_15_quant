import unittest

from experiments.sdk_pnl_decomposition import decompose_session


def account(cash, pnl, positions, fees=0.):
    return {"method": "fixture", "block": 0, "seed": 1,
            "account": {"cash": cash, "pnl": pnl, "positions": positions,
                        "fees_paid": fees, "penalties_paid": 0., "price_version": 2}}


class SDKPnlDecompositionTests(unittest.TestCase):
    def test_customer_buy_means_dealer_sell_and_postmatch_jump_risk(self):
        prices = [{"A": 100., "B": 200.}, {"A": 101., "B": 202.}]
        trades = [{"dealer_id": "dealer_1", "trade_no": 1, "product_id": "A", "price_version": 1,
                   "side": "SELL", "quantity": 2., "price": 100.1, "fee": 0., "counterparty": "customer"},
                  {"dealer_id": "dealer_1", "trade_no": 2, "product_id": "A", "price_version": 2,
                   "side": "BUY", "quantity": 2., "price": 101., "fee": .02, "counterparty": "exchange"}]
        row = account(99998.18, -1.82, {"A": 0., "B": 0.}, .02)
        row.update(final_settlement=prices[-1], mean_squared_inventory=[0., 0.],
                   mean_squared_settlement_factor_exposure=0.)
        result, steps = decompose_session(row, trades, prices, {"A": 1., "B": 1.})
        self.assertAlmostEqual(result["customer_spread"], .2)
        self.assertAlmostEqual(result["exogenous_price_move_pnl"], -2.)
        self.assertEqual(result["failed_checks"], {})
        self.assertEqual(result["exposure"]["mean_q_squared_after_callback"], [0., 0.])
        self.assertEqual(result["exposure"]["mean_q_squared_at_price_innovation"], [4., 0.])
        self.assertEqual(steps[0]["postmatch_prehedge_positions"]["A"], -2.)

    def test_retained_b_has_real_terminal_residual_and_multiplier(self):
        prices = [{"A": 100., "B": 201.}, {"A": 101., "B": 203.}]
        trades = [{"dealer_id": "dealer_1", "trade_no": 1, "product_id": "B", "price_version": 1,
                   "side": "BUY", "quantity": 2., "price": 200.9, "fee": 0., "counterparty": "customer"}]
        # Two B with multiplier three: cash -1205.4, final settlement 202*6=1212.
        row = account(98794.6, 6.6, {"A": 0., "B": 2.})
        row["final_settlement"] = {"A": 101., "B": 202.}
        result, _ = decompose_session(row, trades, prices, {"A": 1., "B": 3.})
        self.assertAlmostEqual(result["customer_spread"], .6)
        self.assertAlmostEqual(result["exogenous_price_move_pnl"], 12.)
        self.assertAlmostEqual(result["terminal_settlement_residual"], -6.)
        self.assertEqual(result["failed_checks"], {})

    def test_monetary_failure_is_preserved(self):
        prices = [{"A": 100., "B": 200.}, {"A": 101., "B": 202.}]
        row = account(100000., 1., {"A": 0., "B": 0.})
        row["final_settlement"] = prices[-1]
        result, _ = decompose_session(row, [], prices, {"A": 1., "B": 1.})
        self.assertIn("decomposition_pnl", result["failed_checks"])

    def test_actual_penalties_are_cash_not_risk_preferences(self):
        prices = [{"A": 100., "B": 200.}, {"A": 101., "B": 202.}]
        row = account(99998., -2., {"A": 0., "B": 0.})
        row["account"]["penalties_paid"] = 2.
        row["final_settlement"] = prices[-1]
        result, _ = decompose_session(row, [], prices, {"A": 1., "B": 1.})
        self.assertEqual(result["penalties"], 2.)
        self.assertEqual(result["decomposed_pnl"], -2.)
        self.assertEqual(result["failed_checks"], {})


if __name__ == "__main__":
    unittest.main()
