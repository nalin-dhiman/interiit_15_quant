"""Independent combinatorial checks of the private ex ante inventory envelope."""
import itertools
import math
import unittest

from resetmm.risk_envelope import InventoryEnvelope, possible_inventory_box


class InventoryEnvelopeTests(unittest.TestCase):
    def test_declared_rectangles_imply_the_shared_gross_limits(self):
        for caps, limit in (({'A': 10., 'B': 5.}, 20.), ({'A': 20., 'B': 10.}, 40.)):
            envelope = InventoryEnvelope(caps, {'A': 1., 'B': 2.})
            self.assertEqual(envelope.gross_limit(), limit)
            for a, b in itertools.product((-caps['A'], caps['A']), (-caps['B'], caps['B'])):
                self.assertEqual(envelope.gross({'A': a, 'B': b}), limit)

    def test_rectangular_allocation_is_stricter_than_the_gross_ball(self):
        envelope = InventoryEnvelope({'A': 10., 'B': 5.}, {'A': 1., 'B': 2.})
        self.assertEqual(envelope.gross({'A': 20., 'B': 0.}), 20.)
        self.assertFalse(envelope.within({'A': 20., 'B': 0.}))

    def test_public_limits_remain_an_additional_constraint(self):
        envelope = InventoryEnvelope({'A': 10.})
        self.assertEqual(envelope.bounds('A', -3., 8.), (-3., 8.))
        self.assertEqual(envelope.bounds('A', -50., 50.), (-10., 10.))
        self.assertEqual(envelope.bounds('A'), (-10., 10.))

    def test_all_customer_fill_subsets_are_inside_the_computed_box(self):
        positions = {'A': 2., 'B': -1.}
        orders = [dict(product_id='A', side='BUY', quantity=3.),
                  dict(product_id='A', side='SELL', quantity=4.),
                  dict(product_id='B', side='BUY', quantity=2.),
                  dict(product_id='B', side='SELL', quantity=3.)]
        box = possible_inventory_box(positions, orders)
        self.assertEqual(box, {'A': (-1., 6.), 'B': (-3., 2.)})
        all_points = []
        for flags in itertools.product((False, True), repeat=len(orders)):
            q = dict(positions)
            for flag, order in zip(flags, orders):
                if flag:
                    q[order['product_id']] += order['quantity'] * (-1 if order['side'] == 'BUY' else 1)
            self.assertTrue(all(lo <= q[pid] <= hi for pid, (lo, hi) in box.items()))
            all_points.append(q)
        envelope = InventoryEnvelope({'A': 10., 'B': 5.}, {'A': 1., 'B': 2.})
        self.assertEqual(envelope.worst_gross(box), max(envelope.gross(q) for q in all_points))

    def test_opposite_customer_sides_are_not_netted_as_guaranteed_fills(self):
        orders = [dict(product_id='A', side=side, quantity=7.) for side in ('BUY', 'SELL')]
        self.assertEqual(possible_inventory_box({'A': 0.}, orders), {'A': (-7., 7.)})

    def test_pending_hedge_reduction_does_not_fund_an_unconfirmed_other_leg(self):
        positions = {'A': 10., 'B': 0.}
        items = [('A', 'SELL', 10.), ('B', 'BUY', 10.)]
        box = possible_inventory_box(positions, exchange_items=items)
        self.assertEqual(box, {'A': (0., 10.), 'B': (0., 10.)})
        envelope = InventoryEnvelope({'A': 10., 'B': 5.}, {'A': 1., 'B': 2.})
        self.assertEqual(envelope.worst_gross(box), 30.)
        self.assertFalse(envelope.within({'A': 10., 'B': 10.}))

    def test_partial_exchange_execution_subsets_preserve_a_valid_rectangle(self):
        positions = {'A': 9., 'B': -4.}
        items = [('A', 'SELL', 9.), ('B', 'BUY', 4.)]
        envelope = InventoryEnvelope({'A': 10., 'B': 5.}, {'A': 1., 'B': 2.})
        for flags in itertools.product((False, True), repeat=2):
            q = dict(positions)
            for flag, (pid, side, quantity) in zip(flags, items):
                if flag:
                    q[pid] += quantity * (1 if side == 'BUY' else -1)
            self.assertTrue(envelope.within(q))
            self.assertLessEqual(envelope.gross(q), 20.)

    def test_multiplier_weighted_exposure_is_explicit(self):
        envelope = InventoryEnvelope({'A': 10., 'B': 5.}, {'A': 1., 'B': 2.})
        self.assertEqual(envelope.gross({'A': 3., 'B': -2.}, {'A': 5., 'B': 3.}), 27.)
        self.assertEqual(envelope.gross_limit({'A': 5., 'B': 3.}), 80.)

    def test_mutating_constructor_inputs_does_not_change_a_frozen_allocation(self):
        caps, weights = {'A': 10.}, {'A': 2.}
        envelope = InventoryEnvelope(caps, weights)
        caps['A'], weights['A'] = 1000., 1000.
        self.assertEqual(envelope.gross_limit(), 20.)

    def test_invalid_caps_weights_and_reservations_are_rejected(self):
        for caps in ({}, {'A': -1.}, {'A': math.inf}, {'A': math.nan}):
            with self.assertRaises(ValueError):
                InventoryEnvelope(caps)
        for weights in ({}, {'B': 1.}, {'A': 0.}, {'A': math.nan}):
            with self.assertRaises(ValueError):
                InventoryEnvelope({'A': 1.}, weights)
        with self.assertRaises(ValueError):
            possible_inventory_box({'A': 0.}, [dict(product_id='A', side='BUY', quantity=-1.)])
