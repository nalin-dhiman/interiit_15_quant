"""Self-financing ledger. Currency, multiplier and fees are explicit.

No trade resets cash, previous PnL, the clock, or prices. Exchange actions
that worsen a per-product breach are rejected; customer fills may breach.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math
from typing import Mapping

@dataclass(frozen=True)
class Product:
    product_id: str
    tick: float = .01
    lot: float = 1.0
    multiplier: float = 1.0
    long_limit: float = 20.0
    short_limit: float = 20.0
    fee_per_unit: float = .01
    fee_notional: float = 0.0
    breach_per_unit: float = .1

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("A product id is required")
        vals = (self.tick, self.lot, self.multiplier, self.long_limit,
                self.short_limit, self.fee_per_unit, self.fee_notional,
                self.breach_per_unit)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Product parameters must be finite")
        if min(self.tick, self.lot, self.multiplier) <= 0 or min(vals[3:]) < 0:
            raise ValueError("Invalid product parameters")


def signed_customer_delta(side: str, quantity: float) -> float:
    if side not in ("BUY", "SELL") or not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("Invalid customer side or quantity")
    return -quantity if side == "BUY" else quantity


def round_price(price: float, tick: float, customer_side: str) -> float:
    """Decimal tick rounding, reconciled with the supplied rfq.sdk helper.

    Validation is stricter than the helper: research prices/ticks must be finite
    and positive. Arithmetic for that domain matches the audited SDK snapshot.
    """
    if not math.isfinite(price) or not math.isfinite(tick) or price <= 0 or tick <= 0:
        raise ValueError("Invalid price or tick")
    if customer_side not in ("BUY", "SELL"):
        raise ValueError("Unknown side")
    quantum = Decimal(str(tick))
    rounding = ROUND_CEILING if customer_side == "BUY" else ROUND_FLOOR
    ticks = (Decimal(str(float(price))) / quantum).to_integral_value(rounding=rounding)
    return float(ticks * quantum)


@dataclass
class Ledger:
    products: Mapping[str, Product]
    starting_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(init=False)
    fees: float = 0.0
    penalties: float = 0.0
    customer_orders: int = 0
    customer_notional: float = 0.0
    exchange_notional: float = 0.0
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.starting_cash):
            raise ValueError("Initial cash must be finite")
        self.cash = self.starting_cash
        self.positions = {k: 0.0 for k in self.products}

    def _validate(self, k: str, quantity: float, price: float) -> Product:
        p = self.products[k]
        if not all(math.isfinite(x) for x in (quantity, price)) or quantity <= 0 or price <= 0:
            raise ValueError("Trade quantity and price must be finite and positive")
        if Decimal(str(quantity)) % Decimal(str(p.lot)) != 0:
            raise ValueError("Quantity is not a whole number of lots")
        return p

    def customer(self, k: str, side: str, quantity: float, price: float) -> None:
        p = self._validate(k, quantity, price)
        delta = signed_customer_delta(side, quantity)
        dcash = -delta*p.multiplier*price
        self.cash += dcash
        self.positions[k] += delta
        self.customer_orders += 1
        self.customer_notional += quantity*p.multiplier*price
        self.log.append(dict(kind="customer", product=k, delta=delta,
                             price=price, cash_delta=dcash, fee=0.0))

    def exchange(self, k: str, delta: float, price: float) -> bool:
        if delta == 0:
            return True
        p = self._validate(k, abs(delta), price)
        old, new = self.positions[k], self.positions[k] + delta
        # Supplied engine compares the scalar excess, so crossing into the
        # opposite breach is legal when its excess is no larger. In particular,
        # +30 -> -30 at limits +/-25 is allowed. Prior interval tests incorrectly
        # forbade this; their bytes are retained in this session's evidence.
        excess = lambda q: max(0.0, q-p.long_limit, -p.short_limit-q)
        if excess(new) > excess(old) + 1e-12:
            return False
        fee = abs(delta)*p.fee_per_unit + abs(delta)*price*p.multiplier*p.fee_notional
        dcash = -delta*p.multiplier*price - fee
        self.positions[k] = new
        self.cash += dcash
        self.fees += fee
        self.exchange_notional += abs(delta)*price*p.multiplier
        self.log.append(dict(kind="exchange", product=k, delta=delta,
                             price=price, cash_delta=dcash, fee=fee))
        return True

    def charge_breaches(self) -> float:
        fee = sum((max(q-p.long_limit, 0.0) + max(-q-p.short_limit, 0.0))*p.breach_per_unit
                  for k, q in self.positions.items() for p in [self.products[k]])
        self.cash -= fee
        self.penalties += fee
        self.log.append(dict(kind="penalty", cash_delta=-fee, fee=fee))
        return fee

    def wealth(self, prices: Mapping[str, float]) -> float:
        if set(self.products) - set(prices):
            raise ValueError("Missing marking or settlement prices")
        if not all(math.isfinite(prices[k]) for k in self.products):
            raise ValueError("Prices must be finite")
        return self.cash + sum(q*self.products[k].multiplier*prices[k]
                               for k, q in self.positions.items())

    def pnl(self, settlement: Mapping[str, float]) -> float:
        return self.wealth(settlement) - self.starting_cash

    def reconcile(self, *, tol: float = 1e-7) -> None:
        c = self.starting_cash + sum(e["cash_delta"] for e in self.log)
        if not math.isclose(self.cash, c, abs_tol=tol, rel_tol=0):
            raise AssertionError("Cash ledger does not reconcile")
        for k in self.positions:
            q = sum(e.get("delta", 0) for e in self.log if e.get("product") == k)
            if not math.isclose(self.positions[k], q, abs_tol=tol, rel_tol=0):
                raise AssertionError("Inventory ledger does not reconcile")
