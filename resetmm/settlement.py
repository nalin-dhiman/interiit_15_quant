"""Acyclic affine settlement rules. No eval(), no unvalidated formula parser.

Rules are (intercept, {other_product: coefficient}); None means LAST_PRICE.
Tick rounding is NOT imposed on a formula unless the actual rules require it.
"""
from __future__ import annotations
import math
from typing import Mapping
Rule = tuple[float, Mapping[str, float]] | None


def resolve(last_prices: Mapping[str, float], rules: Mapping[str, Rule]) -> dict[str, float]:
    if set(last_prices) != set(rules):
        raise ValueError("Rule and product universes differ")
    result: dict[str, float] = {}
    visiting: set[str] = set()
    def visit(k: str) -> float:
        if k in result:
            return result[k]
        if k in visiting:
            raise ValueError("Cyclic settlement formula")
        if k not in rules:
            raise ValueError("Unknown settlement dependency")
        visiting.add(k)
        rule = rules[k]
        if rule is None:
            value = float(last_prices[k])
        else:
            intercept, terms = rule
            value = float(intercept) + sum(float(a)*visit(j) for j, a in terms.items())
        if not math.isfinite(value):
            raise ValueError("Nonfinite settlement")
        visiting.remove(k)
        result[k] = value
        return value
    for k in rules:
        visit(k)
    return result
