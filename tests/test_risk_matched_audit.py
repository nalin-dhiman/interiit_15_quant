"""Adversarial hand fixtures for the independently reconstructed study audit."""
import copy
import itertools
import unittest

from scripts.audit_risk_matched import (
    audit_executed_envelope, audit_risk_trace, expected_parameters,
    independent_box, independent_ledger, path_cells, recompute_primary,
)
from scripts.audit_research_study import economic_fingerprint


class RiskMatchedAuditTests(unittest.TestCase):
    def trace_row(self, pending=False):
        positions = {'A': 2., 'B': -1.}
        reservations = [dict(product_id='A', side='SELL', quantity=3.),
                        dict(product_id='B', side='BUY', quantity=2.)]
        origin, items = None, []
        possible = {'A': [2., 5.], 'B': [-3., -1.]}
        gross, worst = 4., 11.
        if pending:
            positions, reservations = {'A': 0., 'B': 0.}, []
            origin, items = {'A': 9., 'B': 4.}, [['A', 'SELL', 9.], ['B', 'SELL', 4.]]
            possible, gross, worst = {'A': [0., 9.], 'B': [0., 4.]}, 0., 17.
        record = dict(event='fixture', positions=positions, reserved_orders=reservations,
            pending_origin=origin, pending_items=items, possible_box=possible,
            policy_bounds={'A': [-10., 10.], 'B': [-5., 5.]},
            risk_weights={'A': 1., 'B': 2.}, multipliers={'A': 1., 'B': 1.},
            gross=gross, possible_gross=worst, gross_limit=20.,
            within_envelope=True, all_subsets_within_envelope=True,
            pending_exchange_items=len(items), reserved_customer_orders=len(reservations),
            risk_is_variance_matched=False, cash_penalty_imposed=False)
        counters = dict(checks=1, trace_dropped=0, actual_breaches=0, possible_breaches=0,
                        max_actual_gross=gross, max_possible_gross=worst)
        return dict(budget='tight', risk_budget_trace=[record], risk_budget_counters=counters)

    def test_reservation_extrema_match_every_customer_subset(self):
        positions = {'A': 2., 'B': -1.}
        orders = self.trace_row()['risk_budget_trace'][0]['reserved_orders']
        box = independent_box(positions, orders)
        points = []
        for flags in itertools.product((0, 1), repeat=len(orders)):
            q = dict(positions)
            for flag, order in zip(flags, orders):
                if flag:
                    q[order['product_id']] += order['quantity'] * (-1 if order['side'] == 'BUY' else 1)
            points.append(q)
        for pid in positions:
            self.assertEqual(box[pid], [min(q[pid] for q in points), max(q[pid] for q in points)])
        self.assertEqual(audit_risk_trace(self.trace_row())['reservation_checks'], 1)

    def test_opposite_quotes_are_not_guaranteed_offsets(self):
        orders = [dict(product_id='A', side=side, quantity=7.) for side in ('BUY', 'SELL')]
        self.assertEqual(independent_box({'A': 0., 'B': 0.}, orders)['A'], [-7., 7.])

    def test_pending_aggregate_portfolio_does_not_double_count_fills(self):
        row = self.trace_row(pending=True)
        result = audit_risk_trace(row)
        self.assertEqual(result['pending_checks'], 1)
        self.assertEqual(result['max_possible_gross'], 17.)
        row['risk_budget_trace'][0]['pending_origin'] = {'A': 0., 'B': 0.}
        with self.assertRaises(AssertionError):
            audit_risk_trace(row)

    def test_partial_quote_capacity_cannot_be_hidden_by_a_forged_box(self):
        row = self.trace_row()
        trace = row['risk_budget_trace'][0]
        trace['reserved_orders'].append(dict(product_id='B', side='BUY', quantity=4.))
        trace['reserved_customer_orders'] += 1
        # The reported box still pretends only the earlier two-unit quote exists.
        with self.assertRaisesRegex(AssertionError, 'extremum'):
            audit_risk_trace(row)

    def test_missing_pending_items_or_truncated_trace_are_detected(self):
        for modify in (
            lambda row: row['risk_budget_trace'][0].update(pending_exchange_items=2),
            lambda row: row['risk_budget_counters'].update(trace_dropped=1),
            lambda row: row['risk_budget_counters'].update(checks=2),
            lambda row: row['risk_budget_trace'][0].update(cash_penalty_imposed=True),
            lambda row: row['risk_budget_trace'][0].update(risk_is_variance_matched=True),
        ):
            row = self.trace_row(); modify(row)
            with self.assertRaises(AssertionError):
                audit_risk_trace(row)

    def financial_fixture(self):
        prices = [{'A': 100., 'B': 200.}, {'A': 102., 'B': 205.}]
        trade = dict(dealer_id='dealer_1', trade_no=1, product_id='B', price_version=1,
            quantity=1., price=200., fee=0., side='BUY', counterparty='customer',
            reference_price=200., notional=200., position_after=1., cash_after=99800.)
        row = dict(scenario=0, budget='tight', settlement_graph='B=2A',
            account=dict(price_version=2, cash=99800., positions={'A': 0., 'B': 1.},
                fees_paid=0., customer_trades=1, exchange_trades=0, customer_notional=200.,
                exchange_notional=0., pnl=4., penalties_paid=0.),
            max_abs_inventory=[0., 1.], own_trades=[trade], final_reference=prices[-1],
            final_settlement={'A': 102., 'B': 204.},
            monetary_components=dict(customer_spread=0., exchange_execution_edge=0., fees=0.,
                exogenous_price_move_pnl=5., terminal_settlement_residual=-1., penalties=0., decomposed_pnl=4.),
            innovation_exposure=dict(mean_q_squared_at_price_innovation=[0., 1.],
                mean_multiplier_exposure_squared_at_price_innovation=[0., 1.],
                mean_terminal_factor_squared_at_price_innovation=4., price_innovations=1,
                price_innovations_with_nonzero_inventory=1))
        return row, prices

    def test_principal_fee_and_customer_sign_failures_remain_detectable(self):
        row, prices = self.financial_fixture()
        self.assertAlmostEqual(independent_ledger(row, prices)['components']['decomposed_pnl'], 4.)
        for modify in (lambda r: r['account'].update(cash=100000.),
                       lambda r: r['own_trades'][0].update(fee=.02),
                       lambda r: r['own_trades'][0].update(side='SELL')):
            bad = copy.deepcopy(row); modify(bad)
            with self.assertRaises(AssertionError):
                independent_ledger(bad, prices)

    def test_private_cap_is_not_an_official_money_penalty(self):
        row, _ = self.financial_fixture()
        self.assertEqual(audit_executed_envelope(row)['actual_max_gross'], 2.)
        bad = copy.deepcopy(row)
        bad['account']['penalties_paid'] = 3.
        with self.assertRaisesRegex(AssertionError, 'cash penalty'):
            audit_executed_envelope(bad)
        # A hypothetical full customer overshoot remains an audit failure;
        # the audit never edits the trade or invents a smaller executed quantity.
        bad = copy.deepcopy(row)
        bad['own_trades'][0].update(quantity=6., position_after=6.)
        with self.assertRaisesRegex(AssertionError, 'Actual customer'):
            audit_executed_envelope(bad)
        self.assertEqual(bad['own_trades'][0]['quantity'], 6.)

    def cells(self):
        rows = []
        for family, budget, mode, package, controller in itertools.product(
                ('size_mix', 'public_basis'), ('tight', 'wide'), ('fixed', 'inventory'), ('B4', 'R2B'), ('B4', 'R2B')):
            rows.append(dict(scenario=0, path=0, phase='final', family=family, budget=budget,
                quote_mode=mode, package=package, controller=controller,
                parameters={'forecast_weight': .5}, price_tape_sha256='same-price',
                public_customer_tape_sha256='same-customer', account={'pnl': 8. if controller == 'R2B' else 5.},
                final_rfq_observations=[dict(interval_id=1, order_id='x', own_quote=100.1)]))
        return rows

    def test_all_32_cells_make_one_path_with_two_paired_opponents(self):
        result = path_cells(self.cells())
        self.assertEqual(result['primary_difference'], 3.)
        self.assertEqual(result['fixed_common'], 8)
        with self.assertRaises(AssertionError):
            path_cells(self.cells()[:-1])

    def test_unmatched_parameters_prices_or_fixed_quotes_are_detected(self):
        for modify in (
            lambda rows: rows[0]['parameters'].update(forecast_weight=1.),
            lambda rows: rows[0].update(price_tape_sha256='different-price'),
            lambda rows: rows[0]['final_rfq_observations'][0].update(own_quote=100.11),
        ):
            rows = self.cells(); modify(rows)
            with self.assertRaises(AssertionError):
                path_cells(rows)

    def test_primary_counts_paths_and_uses_strict_materiality_boundaries(self):
        for value, supported, excluded, equivalent in ((2., True, False, False),
                (1., False, False, False), (0., False, True, True), (-2., False, True, False)):
            values = {(s, p): value for s in range(8) for p in range(3)}
            result = recompute_primary(values)
            self.assertEqual(result['independent_paths'], 24)
            self.assertEqual(result['stratified_path_bootstrap_95'], [value, value])
            self.assertEqual(result['material_benefit_supported'], supported)
            self.assertEqual(result['material_benefit_excluded'], excluded)
            self.assertEqual(result['practical_equivalence_supported'], equivalent)
        del values[7, 2]
        with self.assertRaises(AssertionError):
            recompute_primary(values)

    def test_only_the_public_settlement_map_changes_in_last_price_scenario(self):
        seal = dict(selected={'wide/fixed': {'B4': dict(forecast_weight=.5, settlement_coefficients={'B': {'A': 2.}})}})
        row = dict(budget='wide', quote_mode='fixed', package='B4', scenario=7)
        self.assertEqual(expected_parameters(seal, row), dict(forecast_weight=.5, settlement_coefficients={}))
        self.assertEqual(seal['selected']['wide/fixed']['B4']['settlement_coefficients'], {'B': {'A': 2.}})

    def test_prefix_risk_instrumentation_is_not_removed_as_host_timing(self):
        row, _ = self.financial_fixture()
        row.update(callback_seconds={'quote': {'maximum': .1}}, risk_budget_trace=[{'possible_gross': 2.}])
        replay = copy.deepcopy(row)
        replay['callback_seconds']['quote']['maximum'] = .9
        self.assertEqual(economic_fingerprint(row), economic_fingerprint(replay))
        replay['risk_budget_trace'][0]['possible_gross'] = 3.
        self.assertNotEqual(economic_fingerprint(row), economic_fingerprint(replay))


if __name__ == '__main__':
    unittest.main()
