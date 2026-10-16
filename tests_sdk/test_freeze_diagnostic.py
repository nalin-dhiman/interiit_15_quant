
from experiments.sdk_environment import KIT, sdk_available, require_sdk
from unittest import SkipTest
if not sdk_available():
    raise SkipTest("Optional SDK unavailable; set RFQ_SDK_ROOT and run its preflight.")
require_sdk()

import unittest
from experiments.freeze_diagnostic import (METHODS,DiagnosticSession,generate_prices,
    policy_parameters,run_case,seed_for,paired_inference)
from experiments.sdk_study import load_class,ROOT,build_config

class FreezeDiagnosticTests(unittest.TestCase):
    def test_policy_difference_is_only_freeze_flag_and_scalar_path(self):
        a,b=(policy_parameters(m) for m in METHODS)
        self.assertFalse(a.pop('freeze_current_quotes'));self.assertTrue(b.pop('freeze_current_quotes'))
        self.assertEqual(a,b);self.assertFalse(a['factor_hedge']);self.assertEqual(a['mode'],'B3')
        self.assertEqual((a['base_edge'],a['inventory_skew'],a['band_fraction'],a['target_fraction']),(.04,.005,.6,0.))

    def test_disjoint_test_and_diagnosis_histories(self):
        tape=generate_prices('test',1,997)
        self.assertEqual(tape,generate_prices('test',1,997))
        self.assertNotEqual(tape,generate_prices('diagnosis',1,997))
        self.assertEqual(len(tape),124)
        seeds=[seed_for('test',0,1,s) for s in ('factor','basis','customers','opponents/tight','opponents/inventory')]
        self.assertEqual(len(set(seeds)),5);self.assertTrue(all(s>2**64 for s in seeds))

    def test_scalar_freeze_actually_activates_and_age_changes(self):
        a,b=(run_case(m,0,997,'tight',phase='test')[0] for m in METHODS)
        self.assertEqual(a['price_tape_sha256'],b['price_tape_sha256'])
        self.assertEqual(a['public_customer_tape_sha256'],b['public_customer_tape_sha256'])
        self.assertEqual(a['freeze_instrumentation']['activated_product_intervals'],0)
        self.assertGreater(b['freeze_instrumentation']['activated_product_intervals'],0)
        self.assertGreater(b['freeze_instrumentation']['omitted_replacements'],0)
        self.assertGreater(b['quote_age']['sum_all_positive_age_seconds'],a['quote_age']['sum_all_positive_age_seconds'])
        for row in (a,b):
            self.assertLess(row['decomposition_max_error'],1e-8)
            self.assertEqual(len(row['final_rfq_observations']),36)
            self.assertTrue(all(o['accepted_quote_age_seconds'] is None or o['accepted_quote_age_seconds']>=.5 for o in row['final_rfq_observations']))

    def test_same_accepted_price_preserves_age_and_withdrawal_removes(self):
        factory=load_class(ROOT/'submission/sdk_dealer.py');dealer=factory(**policy_parameters(METHODS[0]))
        runner=DiagnosticSession(build_config(None,0,123,rows=generate_prices('test',0,994)),dealer,'tight',456)
        oid=next(iter(runner.engine.orders));interval=runner.engine.interval_id
        runner._action('dealer_1','quotes',interval_id=interval,quotes={oid:100.})
        installed=runner.live_quote_changes[(interval,oid)].copy()
        runner.virtual_time+=.5
        runner._action('dealer_1','quotes',interval_id=interval,quotes={oid:100.})
        self.assertEqual(runner.live_quote_changes[(interval,oid)],installed)
        runner._action('dealer_1','quotes',interval_id=interval,quotes={oid:None})
        self.assertNotIn((interval,oid),runner.live_quote_changes)
        runner.store.close()

    def test_rejected_off_tick_quote_does_not_restart_age(self):
        factory=load_class(ROOT/'submission/sdk_dealer.py');dealer=factory(**policy_parameters(METHODS[0]))
        runner=DiagnosticSession(build_config(None,0,123,rows=generate_prices('test',0,994)),dealer,'tight',456)
        oid=next(iter(runner.engine.orders));interval=runner.engine.interval_id
        runner._action('dealer_1','quotes',interval_id=interval,quotes={oid:100.})
        installed=runner.live_quote_changes[(interval,oid)].copy()
        runner.virtual_time+=.5
        runner._action('dealer_1','quotes',interval_id=interval,quotes={oid:100.001})
        self.assertEqual(runner.live_quote_changes[(interval,oid)],installed)
        self.assertEqual(len(runner.rejections),1)
        runner.store.close()

    def test_pairing_groups_opponents_and_rejects_mismatched_histories(self):
        rows=[]
        for s in range(4):
            for p in range(2):
                for f in ('tight','inventory'):
                    for method,value in zip(METHODS,(3.,1.)):
                        rows.append(dict(scenario=s,path=p,family=f,method=method,account={'pnl':value},price_tape_sha256='x',public_customer_tape_sha256='y'))
        result=paired_inference(rows,100)
        self.assertEqual(result['independent_price_customer_paths'],8)
        self.assertEqual(result['normal_minus_freeze'],2)
        self.assertEqual(result['stratified_path_bootstrap_95'],[2.,2.])
        rows[0]['price_tape_sha256']='different'
        with self.assertRaises(AssertionError):paired_inference(rows,100)
