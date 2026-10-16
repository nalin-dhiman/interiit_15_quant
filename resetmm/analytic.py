"""Exactly derived diagnostics for a driftless Brownian inventory.

Inventory evolves as dQ = sqrt(2*D) dW. Holding cost is a*Q**2 per unit
of time; a reset to zero costs K + c*abs(Q). D is INVENTORY diffusion,
not asset-price diffusion. These functions do not model an RFQ auction.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ResetCosts:
    D: float
    a: float
    K: float = 0.0
    c: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.D, self.a, self.K, self.c)):
            raise ValueError("All parameters must be finite")
        if self.D <= 0 or self.a <= 0 or self.K < 0 or self.c < 0:
            raise ValueError("Require D,a>0 and K,c>=0")


def _positive_root(f, *, steps: int = 150) -> float:
    """Bisection for a continuous function with f(0)<0 and one positive root."""
    # A fixed number of bisections from [0, 1] supplies only absolute
    # resolution: a root of order 1e-100 used to be returned as order 1e-46.
    # Bracket on both sides at comparable positive scales first.
    hi = 1.0
    value = f(hi)
    if math.isnan(value):
        raise ArithmeticError("Invalid root function value")
    if value == 0:
        return hi
    if value < 0:
        for _ in range(1024):
            lo, hi = hi, 2*hi
            if not math.isfinite(hi):
                raise ArithmeticError("Positive root exceeds floating-point range")
            value = f(hi)
            if math.isnan(value):
                raise ArithmeticError("Invalid root function value")
            if value >= 0:
                break
        else:
            raise ArithmeticError("Could not bracket a positive root")
    else:
        for _ in range(1075):
            lo = hi/2
            if lo == 0:
                raise ArithmeticError("Positive root is below floating-point range")
            value = f(lo)
            if math.isnan(value):
                raise ArithmeticError("Invalid root function value")
            if value <= 0:
                break
            hi = lo
        else:
            raise ArithmeticError("Could not bracket a positive root")
    for _ in range(steps):
        mid = lo + (hi-lo)/2
        if mid == lo or mid == hi:
            break
        value = f(mid)
        if math.isnan(value):
            raise ArithmeticError("Invalid root function value")
        if value > 0:
            hi = mid
        else:
            lo = mid
    return lo + (hi-lo)/2


def poisson_moments(D: float, r: float) -> tuple[float, float]:
    if not math.isfinite(D) or not math.isfinite(r) or min(D, r) <= 0:
        raise ValueError("D and r must be finite and positive")
    return math.sqrt(D / r), 2 * D / r


def poisson_cost(r: float, costs: ResetCosts) -> float:
    m1, m2 = poisson_moments(costs.D, r)
    return costs.a * m2 + r * (costs.K + costs.c * m1)


def optimal_poisson_rate(costs: ResetCosts) -> float:
    if costs.K == 0 and costs.c == 0:
        raise ValueError("Free resetting has no finite optimum in this model")
    D, a, K, c = costs.D, costs.a, costs.K, costs.c
    return _positive_root(lambda r: K*r*r + .5*c*math.sqrt(D)*r**1.5 - 2*a*D)


def threshold_cost(b: float, costs: ResetCosts) -> float:
    if not math.isfinite(b) or b <= 0:
        raise ValueError("Threshold must be finite and positive")
    return costs.a*b*b/6 + 2*costs.D*(costs.K + costs.c*b)/(b*b)


def optimal_threshold(costs: ResetCosts) -> float:
    """Optimum only among symmetric exit-at-+/-b, reset-to-zero policies."""
    if costs.K == 0 and costs.c == 0:
        raise ValueError("Free resetting has no positive optimal threshold")
    D, a, K, c = costs.D, costs.a, costs.K, costs.c
    # Divide the quartic first-order condition by b. The result is strictly
    # increasing for b>0. Handle b=0 by its one-sided limit.
    def root(b: float) -> float:
        if b == 0:
            return -math.inf if K else -2*D*c
        return a*b**3/3 - 2*D*c - 4*D*K/b
    return _positive_root(root)


def reset_exit_time(D: float, r: float, boundary: float) -> float:
    """Mean time from 0 to +/-boundary, with resets to 0 at rate r.
    This is a RISK-boundary hitting time, not the PRL search-target problem.
    """
    if not all(math.isfinite(x) for x in (D, r, boundary)):
        raise ValueError("Parameters must be finite")
    if D <= 0 or boundary <= 0 or r < 0:
        raise ValueError("Require D,boundary>0 and r>=0")
    if r == 0:
        return boundary**2 / (2*D)
    z = boundary * math.sqrt(r/D)
    # cosh(z)-1 = 2 sinh(z/2)^2 avoids cancellation for small z.
    return 2 * math.sinh(z/2)**2 / r

def threshold_sensitivity(costs: ResetCosts) -> dict[str, float]:
    """Exact partial derivatives of the restricted optimal threshold.

    Boundary cost parameters K=0 or c=0 use the smooth extension locally;
    economically meaningful perturbations must still keep costs nonnegative.
    """
    b = optimal_threshold(costs)
    D, a, K, c = costs.D, costs.a, costs.K, costs.c
    Fb = 4*a*b**3/3 - 2*D*c
    if not math.isfinite(Fb) or Fb <= 0:
        raise ArithmeticError("Degenerate or numerically invalid implicit derivative")
    return {"D": (2*c*b + 4*K)/Fb, "a": -b**4/(3*Fb),
            "K": 4*D/Fb, "c": 2*D*b/Fb}


def linearized_threshold(costs: ResetCosts, delta: dict[str, float]) -> float:
    """First-order proposal, NOT a verified trading policy or trust region.

    The caller must check empirical/model error and the size of the perturbation.
    """
    if any(k not in {"D", "a", "K", "c"} for k in delta):
        raise ValueError("Unknown parameter in threshold perturbation")
    if not all(math.isfinite(v) for v in delta.values()):
        raise ValueError("Perturbations must be finite")
    ResetCosts(**{k: getattr(costs, k) + delta.get(k, 0.0)
                  for k in ("D", "a", "K", "c")})
    gradient = threshold_sensitivity(costs)
    result = optimal_threshold(costs) + sum(gradient[k]*v for k, v in delta.items())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Linearised threshold left its positive domain")
    return result
