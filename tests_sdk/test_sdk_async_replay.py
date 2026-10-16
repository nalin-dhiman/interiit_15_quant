"""Permanent counterexamples for the supplied asynchronous SDK contract."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import asyncio
import copy
import itertools
import unittest

from experiments.sdk_async_replay import (CHUNK, MAX_BATCH, ControlledRunner, Replay,
                                          ScriptedDealer, run_scenarios)
from rfq.runner import DealerRunner
from submission.sdk_dealer import Dealer


class AsyncContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.r = await Replay().start()

    async def asyncTearDown(self):
        await self.r.stop()

    async def test_original_runner_methods_inherited_unchanged(self):
        for name in ("receive", "callback", "send_action", "hedge", "accept_market"):
            self.assertIs(getattr(ControlledRunner, name), getattr(DealerRunner, name))

    async def test_quote_ack_omits_original_request(self):
        action = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(action))
        ack = self.r.dealer.acks[-1]
        self.assertEqual(ack["action_id"], action["action_id"])
        self.assertNotIn("quotes", ack)
        self.assertNotIn("original", ack)
        self.assertNotIn("request", ack)
        self.assertNotIn("order_id", ack)
        self.assertNotIn("request_type", ack)

    async def test_action_id_assigned_by_runner(self):
        await self.r.runner.send_action(self.r.socket, {"type": "get_snapshot", "action_id": "participant-id"})
        self.assertNotEqual(self.r.socket.sent[-1]["action_id"], "participant-id")
        self.assertEqual(len(self.r.socket.sent[-1]["action_id"]), 32)

    async def test_rejected_replacement_retains_old_quote_and_suppresses_same_retry(self):
        first = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(first))
        bad = (await self.r.quote({"x": 100.105}))[0]
        await self.r.deliver(self.r.execute(bad))
        self.assertEqual(self.r.engine.quotes["x"]["dealer_1"], 100.1)
        self.assertEqual(await self.r.quote({"x": 100.105}), [])

    async def test_omission_preserves_live_quote(self):
        action = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(action))
        self.assertEqual(await self.r.quote({}), [])
        self.assertEqual(self.r.engine.quotes["x"]["dealer_1"], 100.1)

    async def test_withdrawal_is_pending_until_exchange_receives(self):
        action = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(action))
        withdrawal = (await self.r.quote({"x": None}))[0]
        self.assertEqual(self.r.engine.quotes["x"]["dealer_1"], 100.1)
        await self.r.deliver(self.r.execute(withdrawal))
        self.assertNotIn("dealer_1", self.r.engine.quotes["x"])

    async def test_boundary_matches_old_quote_then_rejects_late_replacement(self):
        old = (await self.r.quote({"x": 100.1}))[0]
        await self.r.deliver(self.r.execute(old))
        replacement = (await self.r.quote({"x": 100.2}))[0]
        await self.r.deliver(self.r.engine.advance())
        events = self.r.execute(replacement)
        self.assertEqual(events[-1][1]["code"], "STALE_QUOTE")
        self.assertEqual(self.r.engine.trades[0]["price"], 100.1)
        await self.r.deliver(events)

    async def test_trade_in_flight_defers_hedge(self):
        action = (await self.r.hedge([("A", "BUY", 2)]))[0]
        self.assertTrue(self.r.runner.trades_in_flight())
        self.assertEqual(await self.r.hedge([]), [])
        self.assertTrue(self.r.runner.hedge_wanted)
        await self.r.deliver(self.r.execute(action))
        self.assertFalse(self.r.runner.trades_in_flight())

    async def test_partial_batch_fill_callback_has_aggregate_portfolio(self):
        action = (await self.r.hedge([("A", "BUY", 2), ("A", "BUY", 3)]))[0]
        await self.r.deliver(self.r.execute(action))
        self.assertEqual(len(self.r.dealer.fills), 2)
        for call in self.r.dealer.fills:
            self.assertEqual(call["portfolio"]["positions"]["A"], 5.)
        self.assertTrue(self.r.reconcile()["reconciled"])

    async def test_duplicate_trade_ack_does_not_repeat_fill_or_cash(self):
        action = (await self.r.hedge([("A", "BUY", 2)]))[0]
        await self.r.deliver(self.r.execute(action))
        cash = self.r.engine.portfolio("dealer_1")["cash"]
        await self.r.deliver(self.r.execute(action))
        self.assertEqual(len(self.r.dealer.fills), 1)
        self.assertEqual(self.r.engine.portfolio("dealer_1")["cash"], cash)

    async def test_duplicate_quote_ack_does_not_restore_disconnected_quote(self):
        action = (await self.r.quote({"x": 100.1}))[0]
        self.r.execute(action)  # Server committed, but ACK is lost.
        self.r.engine.withdraw("dealer_1")
        events = self.r.execute(action)
        self.assertTrue(events[0][1]["duplicate"])
        self.assertEqual(events[0][1]["status"], "accepted")
        self.assertNotIn("dealer_1", self.r.engine.quotes["x"])
        await self.r.deliver(events)

    async def test_runner_chunks_at_500_before_json_key_sort(self):
        self.assertEqual(CHUNK, 500)
        self.assertEqual(MAX_BATCH, 1000)
        names = ["q" + str(n).zfill(4) for n in reversed(range(1001))]
        messages = await self.r.quote(dict.fromkeys(names, 100.1))
        self.assertEqual([len(a["quotes"]) for a in messages], [500, 500, 1])
        self.assertEqual(set(messages[0]["quotes"]), set(names[:500]))

    async def test_callback_lock_does_not_keep_all_chunks_contiguous(self):
        # Controlled backpressure after the first message demonstrates why
        # callback invocation order alone is not a generic ACK correlation map.
        # We do not claim a frequency for this permissible scheduling pattern.
        class TimerDealer(ScriptedDealer):
            def on_timer(self, market, portfolio, exchange, customers):
                exchange.snapshot()

        r = await Replay(dealer=TimerDealer()).start()
        original_send = r.socket.send
        first_sent, resume_first = asyncio.Event(), asyncio.Event()

        async def send(raw):
            await original_send(raw)
            if len(r.socket.sent) == 1:
                first_sent.set()
                await resume_first.wait()

        r.socket.send = send  # Only explicit transport backpressure is adapted.
        r.dealer.quote_outputs.append({"q" + str(n): 100.1 for n in range(1001)})
        quote_task = asyncio.create_task(r.runner.callback(
            r.socket, "quote", orders=r.runner.market["active_orders"],
            market=r.runner.market, portfolio=r.runner.portfolio))
        try:
            await asyncio.wait_for(first_sent.wait(), 2.)
            await r.runner.callback(r.socket, "on_timer", market=r.runner.market,
                                    portfolio=r.runner.portfolio)
            resume_first.set()
            await quote_task
            self.assertEqual([a["type"] for a in r.socket.sent],
                             ["quotes", "get_snapshot", "quotes", "quotes"])
        finally:
            resume_first.set()
            await asyncio.gather(quote_task, return_exceptions=True)
            await r.stop()

    async def test_entire_scenario_suite_reconciles(self):
        results = await run_scenarios()
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r["reconciliation"]["reconciled"] for r in results.values()))

    async def test_r2_freezes_quote_book_while_initial_quotes_are_pending(self):
        dealer = Dealer(mode="R2", risk_weight=0, total_intervals=1)
        r = await Replay(dealer=dealer).start()
        try:
            # Acquire real inventory, including its principal and fee.
            await r.deliver(r.engine.action("dealer_1", {
                "type": "trades", "action_id": "initial-inventory",
                "price_version": 1, "trades": [{"product_id": "A", "side": "BUY", "quantity": 10}]}))
            await r.runner.callback(r.socket, "quote", orders=r.runner.market["active_orders"],
                                    market=r.runner.market, portfolio=r.runner.portfolio)
            pending = copy.deepcopy(r.socket.sent[-1])
            self.assertEqual(pending["type"], "quotes")
            await r.runner.hedge(r.socket)
            decision = dealer.last_decisions[-1]
            self.assertTrue(decision["current_book_frozen"])
            self.assertTrue(decision["robust_current_fill_comparison"])
            # Pending quotes may reach the Engine after the comparison. Their
            # fill subsets were included even before this eventual acceptance.
            await r.deliver(r.execute(pending))
            before = len(r.socket.sent)
            await r.deliver(r.engine.tick())
            self.assertFalse(any(a["type"] == "quotes" for a in r.socket.sent[before:]))
            self.assertEqual(dealer.callback_errors, 0)
            self.assertTrue(r.reconcile()["reconciled"])
        finally:
            await r.stop()

    async def test_r2_continuous_requoting_falls_back_during_quote_ambiguity(self):
        dealer = Dealer(mode="R2", freeze_current_quotes=False)
        r = await Replay(dealer=dealer).start()
        try:
            await r.deliver(r.engine.action("dealer_1", {
                "type": "trades", "action_id": "initial-inventory",
                "price_version": 1, "trades": [{"product_id": "A", "side": "BUY", "quantity": 10}]}))
            await r.runner.callback(r.socket, "quote", orders=r.runner.market["active_orders"],
                                    market=r.runner.market, portfolio=r.runner.portfolio)
            await r.runner.hedge(r.socket)
            self.assertTrue(dealer.last_decisions[-1]["quote_feedback_fallback"])
            self.assertFalse(dealer.last_decisions[-1]["robust_current_fill_comparison"])
        finally:
            await r.stop()


class AsyncControllerReservationTests(unittest.TestCase):
    def test_pending_replacement_withdrawal_and_omission_reserve_every_subset(self):
        r = Replay(orders=(("x", "SELL", 12.), ("y", "BUY", 9.), ("z", "SELL", 10.)))
        try:
            d = Dealer(mode="B3", inventory_skew=0)
            market = copy.deepcopy(r.runner.market)
            portfolio = copy.deepcopy(r.runner.portfolio)
            out = d.quote(market["active_orders"], market, portfolio)
            self.assertIsNotNone(out["x"])
            self.assertIsNotNone(out["y"])
            self.assertIsNone(out["z"])
            # Positive replacement is not ACKed; omission of y must retain it.
            market["prices"]["A"] += .02
            d.quote(market["active_orders"][:1], market, portfolio)
            # Withdraw x under changed inventory; none is unconfirmed as well.
            portfolio["positions"]["A"] = 12
            d.quote(market["active_orders"][:1], market, portfolio)
            # No capacity may be created by those requests: return to the last
            # confirmed portfolio and verify a new +10 RFQ is still refused.
            portfolio["positions"]["A"] = 0
            self.assertIsNone(d.quote([market["active_orders"][2]], market, portfolio)["z"])
            for take_x, take_y in itertools.product((0, 1), repeat=2):
                self.assertLessEqual(abs(12 * take_x - 9 * take_y), 20)
        finally:
            r.store.close()


if __name__ == "__main__":
    unittest.main()
