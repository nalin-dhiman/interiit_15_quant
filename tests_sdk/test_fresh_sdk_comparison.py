
from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import unittest
from experiments.fresh_sdk_comparison import (generate_prices,seed_for,run_case,
    path_pairs,inference,candidate_grid,METHODS)


class FreshComparisonTests(unittest.TestCase):
    def test_price_namespace_is_reproducible_disjoint_and_positive(self):
        a=generate_prices("test",0,17)
        self.assertEqual(a,generate_prices("test",0,17))
        self.assertNotEqual(a,generate_prices("test",1,17))
        self.assertNotEqual(a,generate_prices("test",0,18))
        self.assertEqual(len(a),124)
        self.assertTrue(all(v>0 for r in a for v in r.values()))
        self.assertEqual(len({seed_for("test",0,17,s) for s in ("factor","basis","customers","opponents")}),4)
        self.assertGreater(seed_for("test",0,17,"factor"),2**64)

    def test_independent_pnl_decomposition_and_pairing(self):
        rows=[run_case(m,candidate_grid(m)[0],"test",0,19,"standard") for m in ("B3","R2")]
        self.assertTrue(all(r["decomposition_max_error"]<1e-8 for r in rows))
        self.assertEqual(len(path_pairs(rows,"R2","B3")),1)
        bad=copy.deepcopy(rows);bad[0]["price_tape_sha256"]="wrong"
        with self.assertRaises(ValueError):path_pairs(bad,"R2","B3")

    def test_opponent_repetitions_are_one_inference_unit(self):
        rows=[]
        for family in ("tight","inventory"):
            for method,pnl in (("R2",3),("B3",1)):
                rows.append(dict(scenario=0,path=1,family=family,method=method,
                    account={"pnl":pnl},price_tape_sha256="p",public_customer_tape_sha256="c"))
        pairs=path_pairs(rows,"R2","B3")
        self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]["difference"],2)

    def test_fresh_inference_keeps_scenarios_stratified(self):
        pairs=[dict(scenario=s,path=p,difference=2.) for s in range(4) for p in range(3)]
        result=inference(pairs,replicates=200)
        self.assertEqual(result["independent_price_paths"],12)
        self.assertEqual(result["paired_mean_difference"],2)
        self.assertTrue(result["superiority_gate"])
        for row in pairs:row["difference"]=.9
        self.assertFalse(inference(pairs,replicates=200)["superiority_gate"])

    def test_equal_candidate_budget_and_no_directional_forecast(self):
        for method in METHODS:
            candidates=candidate_grid(method)
            self.assertEqual(len(candidates),12)
            self.assertTrue(all(p["forecast_weight"]==0 and p["price_drift"] is None for p in candidates))

    def test_fixed_quote_ablation_uses_same_prices_despite_hedge_inventory(self):
        rows=[]
        for method,mode in (("R2_fixed_quotes","R2"),("B3_frozen_quotes","B3")):
            params={**candidate_grid("B3")[0],"mode":mode,"inventory_skew":0.,"freeze_current_quotes":True}
            rows.append(run_case(method,params,"test",0,21,"standard",detail=True))
        a,b=({r["order_id"]:r["own_quote"] for r in row["final_rfq_observations"]} for row in rows)
        common=[oid for oid in a.keys() & b.keys() if a[oid] is not None and b[oid] is not None]
        self.assertTrue(common)
        self.assertEqual(a,b)
