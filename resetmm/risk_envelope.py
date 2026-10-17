"""Private inventory envelopes, separate from public exchange limits and money.

A fixed rectangle is a deliberately conservative allocation of a common ex ante
exposure budget. It does not equalize realized variance, activity or drawdown.
"""
from __future__ import annotations

import math


class InventoryEnvelope:
    def __init__(self, caps, weights=None):
        if not isinstance(caps, dict) or not caps:
            raise ValueError("Nonempty per-product inventory caps are required")
        self.caps = {str(pid): float(cap) for pid, cap in caps.items()}
        if any(not math.isfinite(cap) or cap < 0 for cap in self.caps.values()):
            raise ValueError("Inventory caps must be finite and nonnegative")
        if len(self.caps) != len(caps):
            raise ValueError("Product identifiers must remain distinct")
        self.weights = ({pid: float(weight) for pid, weight in weights.items()}
                        if weights is not None else {pid: 1. for pid in self.caps})
        if set(self.weights) != set(self.caps) or any(
                not math.isfinite(weight) or weight <= 0 for weight in self.weights.values()):
            raise ValueError("Every capped product requires one finite positive exposure weight")

    def bounds(self, product_id, contract_lower=-math.inf, contract_upper=math.inf):
        cap = self.caps[product_id]
        if math.isnan(contract_lower) or math.isnan(contract_upper) or contract_lower > contract_upper:
            raise ValueError("Invalid public contract bounds")
        lower, upper = max(-cap, contract_lower), min(cap, contract_upper)
        if lower > upper:
            raise ValueError("Private and public position intervals do not intersect")
        return lower, upper

    def within(self, positions, box=None, tolerance=1e-9):
        if any(pid not in self.caps and q != 0 for pid, q in positions.items()):
            return False
        for pid, cap in self.caps.items():
            q = positions.get(pid, 0.)
            lower, upper = (box or {}).get(pid, (-cap, cap))
            if not math.isfinite(q) or q < lower - tolerance or q > upper + tolerance:
                return False
        return True

    def gross(self, positions, multipliers=None):
        return self.worst_gross({pid: (q, q) for pid, q in positions.items()}, multipliers)

    def worst_gross(self, box, multipliers=None):
        multipliers = multipliers or {}
        result = 0.
        for pid, (lower, upper) in box.items():
            if pid not in self.caps or not all(math.isfinite(q) for q in (lower, upper)) or lower > upper:
                raise ValueError("Unsupported or invalid exposure interval")
            multiplier = float(multipliers.get(pid, 1.))
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("Multipliers must be finite and positive")
            result += self.weights[pid] * multiplier * max(abs(lower), abs(upper))
        return result

    def gross_limit(self, multipliers=None, bounds=None):
        box = bounds or {pid: (-cap, cap) for pid, cap in self.caps.items()}
        return self.worst_gross(box, multipliers)


def possible_inventory_box(positions, reservations=(), exchange_items=()):
    """Exact coordinate extrema over all independent fill/execution subsets.

    Customer BUY changes dealer inventory negatively. Exchange BUY changes it
    positively. Unconfirmed opposite actions are not guaranteed offsets.
    """
    negative, positive = {}, {}
    for record in reservations:
        pid, side, quantity = record['product_id'], record['side'], float(record['quantity'])
        if side not in ('BUY', 'SELL') or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("Invalid customer reservation")
        totals = negative if side == 'BUY' else positive
        totals[pid] = totals.get(pid, 0.) + quantity
    for pid, side, quantity in exchange_items:
        quantity = float(quantity)
        if side not in ('BUY', 'SELL') or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("Invalid pending exchange item")
        totals = positive if side == 'BUY' else negative
        totals[pid] = totals.get(pid, 0.) + quantity
    products = set(positions) | set(negative) | set(positive)
    return {pid: (float(positions.get(pid, 0.)) - negative.get(pid, 0.),
                  float(positions.get(pid, 0.)) + positive.get(pid, 0.))
            for pid in sorted(products)}
