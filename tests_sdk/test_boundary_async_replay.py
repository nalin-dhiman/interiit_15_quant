"""Independent actual-Runner boundary sequencing regression scenarios."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import unittest

from experiments.boundary_async_replay import BoundaryReplay, make_dealer, run_scenarios
from experiments.sdk_async_replay import ControlledRunner
from rfq.runner import DealerRunner


class BoundaryAsyncReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_replay_schedules_reconcile_and_preserve_gate(self):
        scenarios = await run_scenarios()
        self.assertEqual(set(scenarios), {"normal_BB3", "normal_BR2", "delayed_reset_ack",
                                         "stale_reset_rejection", "partial_reset_batch",
                                         "duplicate_ack", "reconnect_missing_results"})
        self.assertTrue(all(row["reconciliation"]["reconciled"] for row in scenarios.values()))
        for name in ("normal_BB3", "normal_BR2"):
            self.assertTrue(scenarios[name]["details"]["quote_after_confirmed_reset"])
        self.assertTrue(scenarios["delayed_reset_ack"]["details"]["delayed_reset_ack_withheld_capacity"])
        self.assertTrue(scenarios["partial_reset_batch"]["details"]["admission_uses_actual_accepted_subset"])

    async def test_unknown_accepted_ack_does_not_release_pending_reset(self):
        r = await BoundaryReplay(make_dealer()).start()
        try:
            await r.bootstrap()
            actions = r.take_actions()
            self.assertTrue(any(action["type"] == "trades" for action in actions))
            await r.deliver([("dealer_1", dict(type="ack", action_id="unknown-old",
                                               status="accepted"))])
            before = len(r.socket.sent)
            await r.tick()
            self.assertEqual(r.positive_quote_messages(before), [])
            self.assertTrue(r.runner.trades_in_flight())
        finally:
            await r.stop()

    async def test_hedge_reconciliation_cannot_invent_an_extra_quote_callback(self):
        r = await BoundaryReplay(make_dealer()).start()
        try:
            await r.bootstrap()
            await r.drain_actions()
            self.assertEqual(r.positive_quote_messages(), [])
            await r.tick()
            await r.drain_actions()
            self.assertTrue(r.positive_quote_messages())
            self.assertTrue(r.reconcile()["reconciled"])
        finally:
            await r.stop()

    async def test_same_epoch_market_after_quote_admission_fails_closed(self):
        r = await BoundaryReplay(make_dealer()).start()
        try:
            await r.bootstrap()
            await r.drain_actions()
            await r.tick()
            await r.drain_actions()
            self.assertTrue(r.positive_quote_messages())
            r.engine.withdraw("dealer_1")
            r.runner.sent_quotes = {}
            before = len(r.socket.sent)
            await r.deliver([("dealer_1", r.engine.snapshot("dealer_1"))])
            await r.tick()
            await r.drain_actions()
            self.assertEqual(r.positive_quote_messages(before), [])
        finally:
            await r.stop()

    async def test_reordered_outcome_before_own_fill_fails_closed(self):
        r = await BoundaryReplay(make_dealer()).start()
        try:
            await r.bootstrap()
            await r.drain_actions()
            await r.tick()
            await r.drain_actions()
            before = len(r.socket.sent)
            events = r.engine.advance()
            market = [item for item in events if item[1]["type"] == "market"]
            results = [item for item in events if item[1]["type"] == "customer_results"]
            fills = [item for item in events if item[1]["type"] == "fill"]
            rest = [item for item in events if item[1]["type"] not in
                    ("market", "customer_results", "fill")]
            # Explicit fault delivery; the actual supplied exchange queues
            # fills before public results. Reordering has no inferred recovery.
            await r.deliver(market + results + fills + rest)
            await r.tick()
            await r.drain_actions()
            self.assertEqual(r.positive_quote_messages(before), [])
            self.assertEqual(r.dealer.boundary_state, "unsupported")
            self.assertTrue(r.reconcile()["reconciled"])
        finally:
            await r.stop()

    async def test_lost_reset_ack_and_reconnect_do_not_guess_completion(self):
        r = await BoundaryReplay(make_dealer()).start()
        try:
            await r.bootstrap()
            actions = r.take_actions()
            trade = next(action for action in actions if action["type"] == "trades")
            for action in actions:
                if action["type"] != "trades":
                    await r.deliver(r.execute(action))
            r.execute(trade)  # Commit, but lose portfolio+ACK delivery.
            r.engine.withdraw("dealer_1")
            r.runner.sent_quotes = {}
            await r.deliver([("dealer_1", r.engine.snapshot("dealer_1"))])
            await r.deliver(r.execute(trade))  # Duplicate response, no second cash trade.
            before = len(r.socket.sent)
            await r.tick()
            await r.drain_actions()
            self.assertEqual(r.positive_quote_messages(before), [])
            self.assertEqual(r.dealer.boundary_state, "unsupported")
            self.assertTrue(r.reconcile()["reconciled"])
        finally:
            await r.stop()

    async def test_large_reset_book_uses_one_batch_and_actual_deferred_positions(self):
        r = await BoundaryReplay(make_dealer(), products=600).start()
        try:
            await r.bootstrap()
            actions = r.take_actions()
            reset_actions = [action for action in actions if action["type"] == "trades"]
            self.assertEqual(len(reset_actions), 1)
            self.assertLessEqual(len(reset_actions[0]["trades"]), 500)
            acted = {trade["product_id"] for trade in reset_actions[0]["trades"]}
            self.assertEqual(len(acted), 500)
            before_positions = dict(r.runner.portfolio["positions"])
            for action in actions:
                await r.deliver(r.execute(action))
            await r.tick()
            await r.drain_actions()
            self.assertTrue(r.positive_quote_messages())
            self.assertEqual(r.dealer.boundary_state, "open")
            for pid, position in before_positions.items():
                if pid not in acted:
                    self.assertEqual(r.runner.portfolio["positions"][pid], position)
            self.assertTrue(r.reconcile()["reconciled"])
        finally:
            await r.stop()

    def test_replay_inherits_actual_runner_methods(self):
        for name in ("receive", "callback", "send_action", "hedge", "accept_market"):
            self.assertIs(getattr(ControlledRunner, name), getattr(DealerRunner, name))


if __name__ == "__main__":
    unittest.main()
