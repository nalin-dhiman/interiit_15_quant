"""Independent opponent persistence, sample-unit and prospective-budget checks."""

from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import copy
import unittest

from experiments.boundary_study import (StickyOpponent,VolatilityOpponent,parameters,
                                        pairs,seed_for)
from experiments.sdk_study import KIT,load_class
from tests_sdk.test_sdk_controller import market,portfolio,order


class BoundaryStudyTests(unittest.TestCase):
    def test_sticky_omission_preserves_original_per_rfq_draw(self):
        example=load_class(KIT/'examples/dealer.py');d=StickyOpponent(example,123)
        orders=[order('x','BUY',2)];m=market(orders);m['price_version']=1
        first=d.quote(orders,m,portfolio());edge=d.base.order_edges['x'];state=d.base.rng.getstate()
        self.assertIn('x',first)
        m['price_version']=2;m['prices']['A']+=.1
        self.assertEqual(d.quote(orders,m,portfolio()),{})
        self.assertEqual(d.base.order_edges['x'],edge);self.assertEqual(d.base.rng.getstate(),state)
        m['price_version']=4
        self.assertIn('x',d.quote(orders,m,portfolio()))
        self.assertEqual(d.base.order_edges['x'],edge);self.assertEqual(d.base.rng.getstate(),state)

    def test_volatility_uses_public_distinct_versions_without_duplicate_decay(self):
        example=load_class(KIT/'examples/dealer.py');d=VolatilityOpponent(example,123)
        orders=[order('x','BUY',2)];m=market(orders);m['price_version']=1
        d.quote(orders,m,portfolio());m['price_version']=2;m['prices']['A']+=.2
        once=d.quote(orders,m,portfolio());twice=d.quote(orders,m,portfolio())
        self.assertEqual(once,twice);self.assertAlmostEqual(d.delta['A'],.2)
        m['price_version']=3
        self.assertLess(d.quote(orders,m,portfolio())['x'],once['x'])

    def test_two_opponents_are_averaged_as_one_path_and_bad_tape_rejected(self):
        rows=[]
        for family,delta in (('a',2),('b',4)):
            for method,pnl in (('BR2',delta),('B3',0)):
                rows.append(dict(scenario=0,path=0,family=family,method=method,
                    account={'pnl':pnl},price_tape_sha256='p',public_customer_tape_sha256='c'))
        self.assertEqual(pairs(rows,'BR2','B3'),[dict(scenario=0,path=0,difference=3.)])
        changed=copy.deepcopy(rows);changed[0]['price_tape_sha256']='different'
        with self.assertRaises(AssertionError):pairs(changed,'BR2','B3')

    def test_equal_distinct_quote_budgets_and_known_strong_factor_candidate(self):
        for method in ('B1','B2','B3','B3_factor','BB3','BR2'):
            grid=parameters(method)
            self.assertEqual(len(grid),12)
            self.assertEqual(len({(p['base_edge'],p['inventory_skew']) for p in grid}),12)
        self.assertTrue(any(p['base_edge']==.08 and p['inventory_skew']==.005 and
            p['band_fraction']==.6 and p['target_fraction']==.5 and p['factor_hedge']
            for p in parameters('B3_factor')))

    def test_phase_and_primitive_seed_namespaces_are_distinct_full_width(self):
        seeds=[seed_for(phase,s,p,stream) for phase in ('test','tuning','validation','final')
               for s in range(4) for p in range(3)
               for stream in ('factor','basis','customers','opponents/standard')]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertTrue(all(0<=s<2**256 for s in seeds))
        self.assertTrue(any(s>=2**255 for s in seeds))


if __name__=='__main__':unittest.main()
