"""Regression tests for independent network-session reconciliation."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import csv
import json
from pathlib import Path
import tempfile
import unittest

from rfq.engine import Engine
from rfq.models import CustomerOrder, HostConfig, Product, ReplayPrices
from rfq.store import Store
from scripts.audit_sdk_session import reconcile


class Orders:
    def generate(self, interval_id, market, state, rng):
        return state, [CustomerOrder(f"r{interval_id}", "c", "A", "SELL", 6,
                                    expires_interval=interval_id)]


class SDKSessionAuditTests(unittest.TestCase):
    def make_session(self, directory):
        store = Store(str(directory/"audit.sqlite"))
        product = Product("A", 100, multiplier=10, max_long=3, max_short=3,
                          breach_penalty_per_unit=0.7, fee_per_unit=0.2, fee_rate=0.001)
        engine = Engine(HostConfig([product], ReplayPrices([{"A":100}, {"A":98}, {"A":99}]), Orders()), store)
        engine.register("d")
        store.commit([("d", engine.snapshot("d"))])
        engine.action("d", dict(type="quotes", interval_id=1, action_id="quote",
                                quotes={"r1":100}))
        engine.advance()
        engine.action("d", dict(type="trade", price_version=2, action_id="single",
                                product_id="A", side="SELL", quantity=1))
        engine.action("d", dict(type="trades", price_version=2, action_id="batch",
                                trades=[dict(product_id="A", side="SELL", quantity=1)]))
        engine.advance(final=True)
        engine.finish()
        result = dict(status="completed", prices=engine.prices,
                      results={"d":engine.portfolio("d")})
        (directory/"results.json").write_text(json.dumps(result))
        with (directory/"trades.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(engine.trades[0]))
            writer.writeheader()
            writer.writerows(engine.trades)
        store.close()

    def test_single_and_embedded_batch_fills_plus_nonzero_penalties(self):
        with tempfile.TemporaryDirectory() as path:
            directory=Path(path)
            self.make_session(directory)
            result=reconcile(directory)
            self.assertTrue(result["cash_reconciles"], result["discrepancies"])
            self.assertEqual(result["embedded_batch_executions"], 1)
            self.assertEqual(result["event_counts"]["penalty"], 2)
            final=result["final_results"]["d"]
            self.assertAlmostEqual(final["penalties_paid"],1.4)
            self.assertEqual(final["exchange_trades"],2)
            self.assertAlmostEqual(final["pnl"],-83.76)

    def test_unexplained_final_cash_cannot_pass(self):
        with tempfile.TemporaryDirectory() as path:
            directory=Path(path)
            self.make_session(directory)
            target=directory/"results.json"
            result=json.loads(target.read_text())
            result["results"]["d"]["cash"]+=1
            target.write_text(json.dumps(result))
            audit=reconcile(directory)
            self.assertFalse(audit["cash_reconciles"])
            self.assertTrue(any(r["seq"]=="results.json" and r["field"]=="cash"
                                for r in audit["discrepancies"]))

    def test_settlement_prices_follow_final_portfolio_without_prices_message(self):
        class BOrders:
            def generate(self, interval_id, market, state, rng):
                return state, [CustomerOrder("b", "c", "B", "SELL", 2,
                                             expires_interval=interval_id)]
        with tempfile.TemporaryDirectory() as path:
            directory=Path(path)
            store=Store(str(directory/"audit.sqlite"))
            engine=Engine(HostConfig([Product("A",100),Product("B",200)],
                ReplayPrices([{"A":100,"B":200},{"A":101,"B":207}], {"B":{"A":2}}),
                BOrders()),store)
            engine.register("d")
            store.commit([("d",engine.snapshot("d"))])
            engine.action("d",dict(type="quotes",interval_id=1,action_id="q",quotes={"b":200}))
            engine.advance(final=True)
            engine.finish()
            final=engine.portfolio("d")
            (directory/"results.json").write_text(json.dumps(dict(status="completed",
                prices=engine.prices,results={"d":final})))
            (directory/"trades.csv").write_text("fixture hashes only; audit reconstructs SQLite events\n")
            store.close()
            result=reconcile(directory)
            self.assertTrue(result["cash_reconciles"],result["discrepancies"])
            self.assertEqual(final["positions"]["B"],2)
            self.assertEqual(final["pnl"],4)
