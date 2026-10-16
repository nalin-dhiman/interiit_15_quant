"""Deadline-batched RFQ fixture; an independently built toy, NOT the IMC exchange.

It preserves public/private separation, latest-quote/withdraw semantics,
all-or-nothing RFQs, random ties, marked settlement, cash fees, and independent
random tapes across policies. Price dynamics, customer distributions, opponent
quotes and event order are explicit synthetic assumptions. NO market impact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .accounting import Product, Ledger, round_price

@dataclass(frozen=True)
class PrivateRFQ:
    order_id: str
    product_id: str
    side: str
    quantity: float
    limit_edge: float
    competitor_edges: tuple[float,...]
    tie_uniform: float

    def public(self) -> dict:
        return dict(order_id=self.order_id,product_id=self.product_id,
                    side=self.side,quantity=self.quantity)

@dataclass(frozen=True)
class Scenario:
    prices: np.ndarray
    orders: tuple[tuple[PrivateRFQ,...],...]
    product_ids: tuple[str,...]
    updates: int
    dt: float


def make_scenario(seed: int, *, intervals: int=60, products: int=3,
                  rfqs: int=5, updates: int=3, dt: float=.5,
                  regime: str="stationary") -> Scenario:
    if min(intervals,products,rfqs,updates)<1 or dt<=0:
        raise ValueError("Positive scenario dimensions required")
    if regime not in ("stationary","stress","informed"):
        raise ValueError("Unknown synthetic regime")
    ss=np.random.SeedSequence(seed)
    rp,rc,ro=[np.random.default_rng(s) for s in ss.spawn(3)]
    T=intervals*updates
    common=rp.normal(size=(T,1)); indiv=rp.normal(size=(T,products))
    vol=np.full((T,1),.045)
    if regime=="stress":
        vol[T//2:]=.18
    noise=(.5*common+np.sqrt(.75)*indiv)*vol
    prices=np.vstack([np.full(products,100.0),100+np.cumsum(noise,axis=0)])
    if (prices<=0).any():
        raise ValueError("Synthetic price path became nonpositive")
    ids=tuple(f"P{i:04d}" for i in range(products))
    informed_prob=.65 if regime=="informed" else .12
    orders=[]
    for t in range(intervals):
        batch=[]
        match_index=t*updates+updates-1
        future_shock=prices[match_index+1]-prices[match_index]
        for j in range(rfqs):
            k=int(rc.integers(products))
            if rc.random()<informed_prob:
                buy=bool(future_shock[k]>=0)
            else:
                buy=bool(rc.random()<.5)
            edges=tuple(np.maximum(.015,ro.normal(.11,.035,4)))
            limit_edge=float(rc.uniform(.04,.23))
            qty=float(rc.integers(1,4))
            batch.append(PrivateRFQ(f"r{t:04d}_{j:04d}",ids[k],"BUY" if buy else "SELL",
                                    qty,limit_edge,edges,float(ro.random())))
        orders.append(tuple(batch))
    prices.setflags(write=False)
    return Scenario(prices,tuple(orders),ids,updates,dt)


class QuoteBook:
    def __init__(self) -> None:
        self.values: dict[str,float] = {}
    def update(self, updates: dict[str,float|None], active_ids: set[str]) -> None:
        if set(updates)-active_ids:
            raise ValueError("Unknown RFQ in quote response")
        for k,p in updates.items():
            if p is None:
                self.values.pop(k,None)
            else:
                if not np.isfinite(p) or p<=0:
                    raise ValueError("Invalid quote")
                self.values[k]=float(p)
        # Omission keeps existing active quotes, as in the problem statement.
        self.values={k:v for k,v in self.values.items() if k in active_ids}


def match(r: PrivateRFQ, dealer_quote: float|None, reference: float,
          tick: float) -> tuple[bool,float|None]:
    side=1 if r.side=="BUY" else -1
    competitors=[round_price(reference+side*e,tick,r.side) for e in r.competitor_edges]
    eligible=[p for p in competitors if side*(p-reference)<=r.limit_edge+1e-9]
    our_ok=(dealer_quote is not None and side*(dealer_quote-reference)<=r.limit_edge+1e-9)
    all_quotes=eligible+([float(dealer_quote)] if our_ok else [])
    if not all_quotes:
        return False,None
    best=min(all_quotes) if side==1 else max(all_quotes)
    if not our_ok or not np.isclose(dealer_quote,best,atol=1e-9,rtol=0):
        return False,best
    tied=sum(np.isclose(p,best,atol=1e-9,rtol=0) for p in eligible)
    return bool(r.tie_uniform < 1/(1+tied)),float(best)


class Policy(Protocol):
    def quote(self,orders: list[dict],prices: dict[str,float],positions: dict[str,float],
              products: dict[str,Product], *, dt: float) -> dict[str,float|None]: ...
    def hedge(self,prices: dict[str,float],positions: dict[str,float],
              products: dict[str,Product], *, dt: float) -> list[tuple[str,float]]: ...


def run(scenario: Scenario, policy: Policy, *, fee: float=.02,
        limit: float=15.0, keep_trace: bool=False) -> dict:
    specs={k:Product(k,long_limit=limit,short_limit=limit,fee_per_unit=fee)
           for k in scenario.product_ids}
    ledger=Ledger(specs)
    peak=0.0; max_drawdown=0.0; inventory_sq_time=0.0; breach_boundaries=0
    trace=[]
    for t,private_batch in enumerate(scenario.orders):
        public=[r.public() for r in private_batch]
        book=QuoteBook(); active={o["order_id"] for o in public}
        for j in range(scenario.updates):
            idx=t*scenario.updates+j
            prices=dict(zip(scenario.product_ids,map(float,scenario.prices[idx])))
            hedges=policy.hedge(dict(prices),dict(ledger.positions),specs,dt=scenario.dt)
            for k,delta in hedges:
                ledger.exchange(k,float(delta),prices[k])
            quotes=policy.quote([dict(o) for o in public],dict(prices),dict(ledger.positions),
                                specs,dt=scenario.dt)
            book.update(quotes,active)
            if j < scenario.updates-1:
                inventory_sq_time+=sum(q*q for q in ledger.positions.values())*scenario.dt
        # All quotes use pre-fill information. Fills are then booked as a batch.
        outcomes=[]
        for r in private_batch:
            won,winning_price=match(r,book.values.get(r.order_id),prices[r.product_id],specs[r.product_id].tick)
            outcomes.append(dict(order_id=r.order_id,won=won,winning_price=winning_price))
            if won:
                ledger.customer(r.product_id,r.side,r.quantity,book.values[r.order_id])
        # Holdings AFTER the fills are exposed over the next price increment.
        inventory_sq_time+=sum(q*q for q in ledger.positions.values())*scenario.dt
        if ledger.charge_breaches()>0:
            breach_boundaries+=1
        pnl=ledger.pnl(prices); peak=max(peak,pnl); max_drawdown=max(max_drawdown,peak-pnl)
        if keep_trace:
            trace.append(dict(interval=t,cash=ledger.cash,positions=dict(ledger.positions),
                              pnl_marked=pnl,outcomes=outcomes))
    settlement=dict(zip(scenario.product_ids,map(float,scenario.prices[-1])))
    # Synthetic final hedge window: one optional policy call at the final price.
    for k,delta in policy.hedge(settlement,dict(ledger.positions),specs,dt=scenario.dt):
        ledger.exchange(k,float(delta),settlement[k])
    ledger.charge_breaches()
    pnl=ledger.pnl(settlement); peak=max(peak,pnl); max_drawdown=max(max_drawdown,peak-pnl)
    ledger.reconcile()
    result=dict(pnl=pnl,customer_orders=ledger.customer_orders,
                customer_notional=ledger.customer_notional,fees=ledger.fees,
                penalties=ledger.penalties,exchange_notional=ledger.exchange_notional,
                inventory_sq_time=inventory_sq_time,breach_boundaries=breach_boundaries,
                max_drawdown=max_drawdown,terminal_positions=dict(ledger.positions))
    if keep_trace:
        result["trace"]=trace
        result["ledger"]=ledger.log
    return result
