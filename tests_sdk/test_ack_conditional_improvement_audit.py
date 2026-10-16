"""Cash-based fixtures for the scope of current-book cancellation."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import unittest

from rfq.engine import Engine
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.store import Store


class OneCustomer:
    def generate(self, interval_id, market, state, rng):
        return state, [CustomerOrder("one", "c", "A", "BUY", 1., interval_id,
                                     reservation_price=100.)]


def execute_one(hold, quote):
    store = Store(":memory:")
    engine = Engine(HostConfig(products=[Product("A", 100., max_long=5., max_short=5.,
                                                  fee_per_unit=.1)],
                              customer_model=OneCustomer(),
                              price_model=ReplayPrices([{"A": 100.}, {"A": 100.}]),
                              final_hedge_seconds=0), store)
    engine.register("dealer_1")
    # Initial q=1 has its real cash purchase; common setup fee is retained.
    engine.action("dealer_1", {"type": "trades", "action_id": "setup", "price_version": 1,
                              "trades": [{"product_id": "A", "side": "BUY", "quantity": 1.}]})
    initial_pnl = engine.portfolio("dealer_1")["pnl"]
    if not hold:
        engine.action("dealer_1", {"type": "trades", "action_id": "band", "price_version": 1,
                                  "trades": [{"product_id": "A", "side": "SELL", "quantity": 1.}]})
    engine.action("dealer_1", {"type": "quotes", "action_id": "quote", "interval_id": 1,
                              "quotes": {"one": quote}})
    engine.advance(final=True)
    engine.finish()
    result = engine.portfolio("dealer_1")["pnl"] - initial_pnl
    store.close()
    return result


class ConditionalImprovementAuditTests(unittest.TestCase):
    def test_common_frozen_quote_cash_cancels_for_fill_and_no_fill(self):
        for quote in (99.1, 100.1):
            hold, band = execute_one(True, quote), execute_one(False, quote)
            self.assertAlmostEqual(hold - band, .1)

    def test_one_rfq_counterexample_after_inventory_dependent_repricing(self):
        # Ask=reference+.1-inventory. Holding q=1 gives99.1 and fills at a loss;
        # hedging to0 gives100.1 and loses to the customer's limit100.
        hold, band = execute_one(True, 99.1), execute_one(False, 100.1)
        self.assertAlmostEqual(hold, -.9)
        self.assertAlmostEqual(band, -.1)
        self.assertAlmostEqual(hold - band, -.8)

    def test_minimum_paired_advantages_are_not_differences_of_minima(self):
        # Two modeled outcomes. min(Fa)-min(Fb)=+1, yet one paired difference=-9.
        baseline = [0., 10.]
        candidate = [1., 1.]
        self.assertEqual(min(candidate) - min(baseline), 1.)
        self.assertEqual(min(a - b for a, b in zip(candidate, baseline)), -9.)


if __name__ == "__main__":
    unittest.main()
