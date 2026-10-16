"""Independent regression fixtures for the controlled SDK response experiment."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import json
import unittest
from dataclasses import replace

import numpy as np

from experiments.sdk_intervention_response import (
    EDGES, FAMILIES, LIMITS, Engine, HostConfig, Product, ReplayPrices,
    Store, TapeCustomers, calibration, exact_fill_probability, features, make_tape,
    quote_at, response_analysis, run_tape, sdk_fill_probability, session_summary)


class TestControlledSDKResponse(unittest.TestCase):
    def test_fixed_tapes_are_reproducible_and_have_separate_streams(self):
        first, second = make_tape(8123,16), make_tape(8123,16)
        self.assertEqual(first,second)
        self.assertEqual(len(set(first["stream_seeds"].values())),4)
        self.assertNotEqual(first["price_rows"],make_tape(8124,16)["price_rows"])

    def test_price_tape_prefix_is_unaffected_by_experiment_length(self):
        short,long = make_tape(8101,4),make_tape(8101,8)
        self.assertEqual(short["price_rows"],long["price_rows"][:5])
        self.assertEqual(short["attributes"],long["attributes"][:4])

    def test_exact_tick_quotes_do_not_accumulate_binary_residuals(self):
        self.assertEqual(quote_at(100.01,.04,"BUY"),100.05)
        self.assertEqual(quote_at(100.01,.04,"SELL"),99.97)

    def test_engine_cash_fees_and_inventory_reconcile(self):
        tape=make_tape(8123,8)
        for a in tape["attributes"]:
            a["edge"]=0.
        records,total=run_tape(tape)
        self.assertTrue(all(r["won"] for r in records))
        self.assertAlmostEqual(total["pnl"],sum(r["reward"] for r in records),places=7)
        self.assertEqual(total["customer_trades"],8)
        self.assertEqual(total["exchange_trades"],8)
        self.assertEqual(total["penalties"],0.)
        self.assertTrue(total["inventory_flat"])

    def test_one_observation_per_final_rfq(self):
        records,_=run_tape(make_tape(8103,12))
        keys={(r["session_seed"],r["order_id"]) for r in records}
        self.assertEqual(len(keys),len(records))
        self.assertEqual(len(records),12)
        self.assertTrue(all("limit_offset" not in r for r in records))

    def test_feature_vector_excludes_hidden_and_future_labels(self):
        rows,_=run_tape(make_tape(8123,8))
        original=features(rows)
        mutated=copy.deepcopy(rows)
        for r in mutated:
            r.update(signed_markout=100000., expected_signed_markout=-999.,
                     public_winner="future",intervention_family=73,limit_offset=999.)
        np.testing.assert_array_equal(original,features(mutated))

    def test_response_uses_same_primitives_without_mutation(self):
        tape=make_tape(8102,12)
        serialized=json.dumps(tape,sort_keys=True)
        low,_=run_tape(tape,.45)
        high,_=run_tape(tape,.55)
        self.assertEqual(json.dumps(tape,sort_keys=True),serialized)
        self.assertEqual([r["matching_reference"] for r in low],
                         [r["matching_reference"] for r in high])
        self.assertEqual([r["signed_markout"] for r in low],
                         [r["signed_markout"] for r in high])

    def test_explicit_fee_derivative_not_dropped(self):
        tape=make_tape(8102,12)
        for a in tape["attributes"]:
            a["edge"]=0.
        baseline,_=run_tape(tape,.5)
        _,low=run_tape(tape,.45)
        _,high=run_tape(tape,.55)
        actual=(high["pnl"]-low["pnl"])/.1
        expected=sum(r["explicit_reward_derivative"] for r in baseline)
        self.assertAlmostEqual(actual,expected,places=7)
        self.assertLess(expected,0.)

    def test_exact_reference_respects_ties_and_eligibility(self):
        self.assertAlmostEqual(exact_fill_probability(.04,0),19/48)
        self.assertAlmostEqual(exact_fill_probability(.08,0),5/288)
        self.assertEqual(exact_fill_probability(.32,0),0.)
        self.assertEqual(exact_fill_probability(.32,1),0.)
        self.assertAlmostEqual(exact_fill_probability(0.,0),1.)

    def test_finite_kernel_is_monotone_without_asserting_runtime_alpha(self):
        for family in (0,1):
            p=[exact_fill_probability(e,family) for e in EDGES]
            self.assertTrue(all(a>=b-1e-12 for a,b in zip(p,p[1:])))
            self.assertTrue(all(0<=v<=1 for v in p))

    def test_intervention_rejects_outside_trust_region(self):
        with self.assertRaises(ValueError):
            run_tape(make_tape(8101,3),.0)
        with self.assertRaises(ValueError):
            run_tape(make_tape(8101,3),.81)

    def test_empty_supported_bins_are_not_a_certificate(self):
        item=calibration(np.array([0.,1.]),np.array([.5,.5]),np.array([0.,.04]))
        self.assertEqual(item["supported_bins"],0)
        self.assertEqual(item["n"],2)

    def test_statistical_unit_is_sessions(self):
        s=session_summary([1.,2.,3.])
        self.assertEqual(s["n"],3)
        self.assertAlmostEqual(s["standard_error"],1/np.sqrt(3))

    def test_common_information_tangent_matches_direct_refit(self):
        baseline,plus,minus,plus_records,minus_records=[],[],[],[],[]
        for seed in (8131,8132,8133):
            tape=make_tape(seed,12)
            b,_=run_tape(tape,.5)
            p,pt=run_tape(tape,.55)
            m,mt=run_tape(tape,.45)
            baseline.extend(b)
            plus.append(pt)
            minus.append(mt)
            plus_records.extend(p)
            minus_records.extend(m)
        result=response_analysis(baseline,plus,minus,plus_records,minus_records)
        estimates=result["estimators"]
        self.assertAlmostEqual(estimates["estimated_finite_tangent"]["mean"],
                               estimates["direct_conditional_refit"]["mean"],places=10)
        self.assertEqual(result["shared_rfqs_per_estimated_method"],108)
        self.assertEqual(result["shared_engine_runs_per_session"],3)
        self.assertTrue(result["explicit_payoff_derivative_included"])
        self.assertFalse(result["common_cash_path_density_assumed"])
        self.assertIsNone(estimates["direct_conditional_refit"]["normal_95_interval"])
        self.assertIsNone(estimates["direct_conditional_refit"]["paired_error_against_known_tangent"]["normal_95_interval"])
        self.assertEqual(len(estimates["direct_conditional_refit"]["paired_session_bootstrap_95_interval"]),2)

    def test_zero_fill_probability_does_not_break_primitive_support(self):
        tape=make_tape(8151,4)
        for a in tape["attributes"]:
            a["edge"]=.32
        rows,total=run_tape(tape,.5)
        self.assertFalse(any(r["won"] for r in rows))
        self.assertEqual(total["pnl"],0.)
        self.assertTrue(all(np.isfinite(r["primitive_score"]) for r in rows))

    def test_sdk_float_hidden_limit_can_reject_exact_boundary_quote(self):
        tape=make_tape(8151,1)
        tape["price_rows"]=[dict(A=99.96,B=199.92),dict(A=99.96,B=199.92)]
        tape["attributes"][0].update(product_id="A",side="BUY",quantity=2,
            edge=.08,limit_offset=.08,family_u=.1,opponent_indices=[0,0])
        rows,total=run_tape(tape,.5)
        self.assertEqual(rows[0]["own_quote"],100.04)
        self.assertFalse(rows[0]["won"])
        self.assertEqual(rows[0]["public_status"],"expired")
        self.assertEqual(total["pnl"],0.)
        self.assertAlmostEqual(exact_fill_probability(.08,1),5/6)
        self.assertAlmostEqual(sdk_fill_probability(.08,1,99.96,"BUY"),2/3)

    def test_ideal_and_sdk_kernel_agree_away_from_float_limit_boundary(self):
        for family in (0,1):
            for side in ("BUY","SELL"):
                self.assertAlmostEqual(exact_fill_probability(0.,family),
                                       sdk_fill_probability(0.,family,100.,side))
                self.assertAlmostEqual(exact_fill_probability(.06,family),
                                       sdk_fill_probability(.06,family,100.,side))

    def test_sdk_finite_sum_matches_exhaustive_original_matching(self):
        for reference,side in ((99.96,"BUY"),(99.9,"SELL")):
            tape=make_tape(8151,1)
            tape["attributes"][0].update(product_id="A",side=side,quantity=2)
            rows=[dict(A=reference,B=2*reference)]*2
            store=Store(":memory:")
            engine=Engine(HostConfig([Product("A",reference),Product("B",2*reference)],
                ReplayPrices(rows),TapeCustomers(tape["attributes"])),store)
            for name in ("dealer_1","dealer_2","dealer_3"):
                engine.register(name)
            order=next(iter(engine.orders.values()))
            for family,grid in enumerate(FAMILIES):
                total=0.
                for limit in LIMITS:
                    fixture_order=replace(order,limit_offset=limit)
                    engine.orders[order.order_id]=fixture_order
                    for a in grid:
                        for b in grid:
                            for name,edge in (("dealer_1",.08),("dealer_2",a),("dealer_3",b)):
                                engine.set_quote(name,order.order_id,quote_at(reference,edge,side))
                            eligible=engine.matches(fixture_order)
                            if "dealer_1" in eligible:
                                best=(min if side=="BUY" else max)(eligible.values())
                                if eligible["dealer_1"]==best:
                                    tied=sum(price==best for price in eligible.values())
                                    total+=1/(len(LIMITS)*len(grid)**2*tied)
                self.assertAlmostEqual(total,sdk_fill_probability(.08,family,reference,side))
            store.close()


if __name__ == "__main__":
    unittest.main()
