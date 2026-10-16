"""Batch reservation and low-rank portfolio-risk arithmetic."""
from __future__ import annotations
import math
from typing import Mapping, Iterable
import numpy as np
from .accounting import Product, signed_customer_delta


def fill_envelope(q: float, orders: Iterable[tuple[str, float]]) -> tuple[float, float]:
    """Min/max inventory over ALL subsets of fills; opposite quotes do not net."""
    lo = hi = float(q)
    if not math.isfinite(q):
        raise ValueError("Inventory must be finite")
    for side, qty in orders:
        d = signed_customer_delta(side, qty)
        lo += min(d, 0.0)
        hi += max(d, 0.0)
    return lo, hi


def reserve_batch(orders: list[dict], positions: Mapping[str, float],
                  products: Mapping[str, Product]) -> set[str]:
    """Conservative deterministic greedy acceptance. This is not a quote optimizer.
    Every active order in this batch must be included exactly once. Caller must
    cancel rejected existing quotes explicitly, and account for pending hedges.
    """
    if len({o["order_id"] for o in orders}) != len(orders):
        raise ValueError("Duplicate order ids")
    low = dict(positions)
    high = dict(positions)
    accepted: set[str] = set()
    for o in sorted(orders, key=lambda z: z["order_id"]):
        k = o["product_id"]
        p = products[k]
        d = signed_customer_delta(o["side"], o["quantity"])
        l, h = low[k] + min(d, 0.0), high[k] + max(d, 0.0)
        if l >= -p.short_limit-1e-9 and h <= p.long_limit+1e-9:
            accepted.add(o["order_id"])
            low[k], high[k] = l, h
    return accepted


def psd_covariance(cov: np.ndarray, floor: float = 0.0) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all() or floor < 0:
        raise ValueError("Invalid covariance matrix or floor")
    e, v = np.linalg.eigh((c+c.T)/2)
    return (v*np.maximum(e, floor)) @ v.T


def factor_risk(q: np.ndarray, loadings: np.ndarray, residual: np.ndarray) -> float:
    q, B, d = map(lambda x: np.asarray(x, dtype=float), (q, loadings, residual))
    if B.ndim != 2 or B.shape[0] != len(q) or d.shape != q.shape or np.any(d < 0):
        raise ValueError("Invalid factor-risk shapes or residual variances")
    if not all(np.isfinite(a).all() for a in (q, B, d)):
        raise ValueError("Nonfinite factor-risk input")
    f = B.T @ q
    return float(f@f + (d*q)@q)
