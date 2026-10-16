"""Boundary-reset contract fixtures using the supplied, unchanged Engine/Runner.

These verify the observable event contract, not a proposed policy or production
server. ControlledRunner changes transport/Worker only; its receive path is the
organiser implementation. Hidden customer limits remain inside Engine.
"""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import unittest

from experiments.sdk_async_replay import Replay, ScriptedDealer
from rfq.runner import CALLBACKS


class BoundaryTraceDealer(ScriptedDealer):
    def __init__(self):
        super().__init__()
        self.trace = []

    def on_market(self, market, portfolio, exchange):
        self.trace.append(("market", copy.deepcopy(market), copy.deepcopy(portfolio)))
        return super().on_market(market, portfolio, exchange)

    def quote(self, orders, market, portfolio):
        self.trace.append(("quote", copy.deepcopy(market), copy.deepcopy(portfolio)))
        return super().quote(orders, market, portfolio)

    def hedge(self, market, portfolio):
        self.trace.append(("hedge", copy.deepcopy(market), copy.deepcopy(portfolio)))
        return super().hedge(market, portfolio)

    def on_fill(self, fill, portfolio, exchange):
        self.trace.append(("fill", copy.deepcopy(fill), copy.deepcopy(portfolio)))
        return super().on_fill(fill, portfolio, exchange)

    def on_customer_result(self, result):
        self.trace.append(("result", copy.deepcopy(result), None))
        return super().on_customer_result(result)


class BoundaryContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dealer = BoundaryTraceDealer()
        self.r = await Replay(dealer=self.dealer, orders=(("x", "BUY", 2.),)).start()

    async def asyncTearDown(self):
        await self.r.stop()

    async def install_customer_quote(self):
        action = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(action))
        self.dealer.trace.clear()

    async def test_first_new_batch_quote_precedes_old_fill_and_outcome(self):
        await self.install_customer_quote()
        await self.r.deliver(self.r.engine.advance())
        names = [row[0] for row in self.dealer.trace]
        self.assertEqual(names, ["market", "quote", "fill", "result", "hedge"])
        quote = self.dealer.trace[1]
        self.assertEqual(quote[1]["interval_id"], 2)
        self.assertEqual(quote[2]["interval_id"], 1)
        self.assertEqual(quote[2]["positions"]["A"], 0)
        self.assertEqual(self.dealer.trace[-1][2]["positions"]["A"], -2)

    async def test_customer_fill_snapshot_keeps_old_portfolio_version(self):
        await self.install_customer_quote()
        old = copy.deepcopy(self.r.runner.portfolio)
        await self.r.deliver(self.r.engine.advance())
        fill = next(row for row in self.dealer.trace if row[0] == "fill")
        hedge = next(row for row in self.dealer.trace if row[0] == "hedge")
        self.assertEqual(fill[2]["price_version"], old["price_version"])
        self.assertEqual(fill[2]["interval_id"], old["interval_id"])
        self.assertEqual(hedge[2]["price_version"], hedge[1]["price_version"])
        self.assertEqual(hedge[2]["interval_id"], hedge[1]["interval_id"])
        self.assertEqual(fill[2]["positions"], hedge[2]["positions"])
        self.assertEqual(fill[2]["cash"], hedge[2]["cash"])

    async def test_closing_transition_keeps_interval_but_increments_version(self):
        await self.install_customer_quote()
        interval = self.r.runner.market["interval_id"]
        version = self.r.runner.market["price_version"]
        await self.r.deliver(self.r.engine.advance(final=True))
        market = self.dealer.trace[0][1]
        self.assertEqual(market["interval_id"], interval)
        self.assertEqual(market["price_version"], version + 1)
        self.assertTrue(market["closing"])
        self.assertEqual(market["active_orders"], [])
        self.assertEqual([row[0] for row in self.dealer.trace],
                         ["market", "fill", "result", "hedge"])

    async def test_results_cover_even_unquoted_orders_before_authoritative_hedge(self):
        # The losing/expired public outcome exists even if no quote was offered.
        await self.r.deliver(self.r.engine.advance())
        self.assertEqual(len(self.dealer.results), 1)
        self.assertEqual(self.dealer.results[0]["status"], "expired")
        self.assertFalse(self.dealer.results[0]["traded"])
        names = [row[0] for row in self.dealer.trace]
        self.assertLess(names.index("result"), names.index("hedge"))
        self.assertEqual(self.dealer.trace[-1][2]["customer_trades"], 0)

    async def test_in_flight_reset_suppresses_second_runner_hedge(self):
        first = (await self.r.hedge([("A", "BUY", 2)]))[0]
        self.assertTrue(self.r.runner.trades_in_flight())
        before = len(self.r.socket.sent)
        await self.r.runner.hedge(self.r.socket)
        self.assertEqual(len(self.r.socket.sent), before)
        self.assertTrue(self.r.runner.hedge_wanted)
        await self.r.deliver(self.r.execute(first))
        self.assertFalse(self.r.runner.trades_in_flight())
        self.assertEqual(self.r.runner.portfolio["positions"]["A"], 2)

    async def test_partial_reset_batch_has_aggregate_portfolio_and_indexed_failure(self):
        action = (await self.r.hedge([("A", "BUY", 2), ("A", "BUY", 100)]))[0]
        events = self.r.execute(action)
        self.assertEqual([e[1]["type"] for e in events], ["portfolio", "ack"])
        ack = events[-1][1]
        self.assertEqual(ack["status"], "accepted")
        self.assertEqual(ack["errors"], {"1": "POSITION_LIMIT"})
        self.assertEqual(len(ack["executions"]), 1)
        await self.r.deliver(events)
        self.assertEqual(self.dealer.fills[-1]["portfolio"]["positions"]["A"], 2)
        self.assertTrue(self.r.reconcile()["reconciled"])

    async def test_rejected_reset_has_no_new_portfolio_or_execution(self):
        action = (await self.r.hedge([("A", "BUY", 2)]))[0]
        action["price_version"] = -100
        events = self.r.execute(action)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1]["status"], "rejected")
        self.assertEqual(events[0][1]["code"], "STALE_PRICE")
        self.assertNotIn("executions", events[0][1])
        self.assertEqual(self.r.engine.portfolio("dealer_1")["positions"]["A"], 0)

    async def test_lost_reset_ack_replay_returns_execution_again_but_cash_once(self):
        action = (await self.r.hedge([("A", "BUY", 2)]))[0]
        self.r.execute(action)  # Committed response was not delivered.
        committed_cash = self.r.engine.portfolio("dealer_1")["cash"]
        # Model the authoritative reconnect snapshot, without guessing a flag.
        await self.r.deliver([("dealer_1", self.r.engine.snapshot("dealer_1"))])
        replayed = self.r.execute(action)
        self.assertEqual(len(replayed), 1)
        self.assertTrue(replayed[0][1]["duplicate"])
        self.assertEqual(len(replayed[0][1]["executions"]), 1)
        await self.r.deliver(replayed)
        self.assertEqual(len(self.dealer.fills), 1)
        self.assertEqual(self.dealer.fills[0]["portfolio"]["cash"], committed_cash)
        self.assertEqual(self.r.engine.portfolio("dealer_1")["cash"], committed_cash)
        self.assertTrue(self.r.reconcile()["reconciled"])

    async def test_reconnect_snapshot_does_not_restore_missing_final_outcomes(self):
        await self.install_customer_quote()
        events = self.r.engine.advance()  # Outage loses boundary event delivery.
        self.assertTrue(any(e[1]["type"] == "customer_results" for e in events))
        await self.r.deliver([("dealer_1", self.r.engine.snapshot("dealer_1"))])
        self.assertEqual(self.dealer.results, [])
        self.assertEqual(self.r.runner.portfolio["positions"]["A"], -2)
        self.assertEqual(self.r.runner.portfolio["customer_trades"], 1)

    async def test_snapshot_and_market_share_on_market_payload_shape(self):
        self.assertNotIn("on_portfolio", CALLBACKS)
        await self.r.deliver([("dealer_1", self.r.engine.snapshot("dealer_1"))])
        snapshot_callback = self.dealer.trace[0]
        self.dealer.trace.clear()
        await self.r.deliver([("*", self.r.engine.market())])
        market_callback = self.dealer.trace[0]
        self.assertEqual(snapshot_callback, market_callback)
        self.assertNotIn("reconnect", snapshot_callback[1])

    async def test_unconfirmed_reset_cannot_supply_customer_capacity(self):
        # q=18, desired SELL 10 has not reached the Engine. Quoting SELL 4
        # using imagined post-reset q=8 breaches +20 when customer fills first.
        await self.r.deliver(self.r.engine.action("dealer_1", {
            "type": "trades", "action_id": "initial-q18", "price_version": 1,
            "trades": [{"product_id": "A", "side": "BUY", "quantity": 18}]}))
        pending = (await self.r.hedge([("A", "SELL", 10)]))[0]
        from rfq.models import CustomerOrder
        self.r.engine.orders = {"new": CustomerOrder("new", "fixture", "A", "SELL", 4,
                                                     self.r.engine.interval_id, limit_offset=1.)}
        self.r.engine.quotes = {"new": {"dealer_1": 99.9}}
        self.r.engine.advance()
        self.assertEqual(self.r.engine.portfolio("dealer_1")["positions"]["A"], 22)
        self.assertGreater(self.r.engine.excess("A", 22), 0)
        self.assertIn(pending["action_id"], self.r.runner.pending)

    async def test_no_action_final_settlement_keeps_inventory_and_cash_principal(self):
        action = (await self.r.hedge([("A", "BUY", 2)]))[0]
        await self.r.deliver(self.r.execute(action))
        cash = self.r.engine.portfolio("dealer_1")["cash"]
        await self.r.deliver(self.r.engine.advance(final=True))
        await self.r.deliver(self.r.engine.finish())
        account = self.r.engine.portfolio("dealer_1")
        self.assertEqual(account["positions"]["A"], 2)
        self.assertEqual(account["cash"], cash)
        self.assertEqual(account["exchange_trades"], 1)
        self.assertTrue(self.r.reconcile()["reconciled"])


if __name__ == "__main__":
    unittest.main()
