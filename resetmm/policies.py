"""Synthetic baseline policies. Constants are illustrative, not calibrated.

All policies use exactly the same inventory-aware quoting logic so reset
ablations change only the intervention rule. No claims of optimality are made.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .risk import reserve_batch
from .accounting import round_price

@dataclass
class InventoryQuote:
    edge: float=.075
    skew: float=.004
    def quote(self,orders,prices,positions,products,*,dt):
        allowed=reserve_batch(orders,positions,products)
        out={o["order_id"]:None for o in orders}
        for o in orders:
            if o["order_id"] not in allowed:
                continue
            k=o["product_id"]; d=1 if o["side"]=="BUY" else -1
            reference=prices[k]-self.skew*positions[k]
            proposed=reference+d*self.edge
            if proposed>0 and math.isfinite(proposed):
                out[o["order_id"]]=round_price(proposed,products[k].tick,o["side"])
        return out
    def hedge(self,prices,positions,products,*,dt):
        return []

@dataclass
class ImmediateReset(InventoryQuote):
    def hedge(self,prices,positions,products,*,dt):
        return [(k,-q) for k,q in positions.items() if abs(q)>=products[k].lot-1e-9]

@dataclass
class ThresholdReset(InventoryQuote):
    threshold: float=9.0
    target_fraction: float=0.0
    def __post_init__(self):
        if self.threshold<=0 or not 0<=self.target_fraction<1:
            raise ValueError("Invalid threshold or target fraction")
    def hedge(self,prices,positions,products,*,dt):
        ans=[]
        for k,q in positions.items():
            if abs(q)>=self.threshold:
                lot=products[k].lot
                target=math.copysign(math.floor(self.target_fraction*self.threshold/lot)*lot,q)
                ans.append((k,target-q))
        return ans

@dataclass
class PoissonReset(InventoryQuote):
    rate: float=.05
    seed: int=52
    rng: np.random.Generator=field(init=False,repr=False)
    def __post_init__(self):
        if not math.isfinite(self.rate) or self.rate<0:
            raise ValueError("Invalid reset rate")
        self.rng=np.random.default_rng(self.seed)
    def hedge(self,prices,positions,products,*,dt):
        p=-math.expm1(-self.rate*dt)
        # Sample each product every callback, even if q=0. This keeps policy
        # random draws aligned when inspecting no-op vs executed resets.
        return [(k,-q) for k,q in positions.items() if self.rng.random()<p and q!=0]

@dataclass
class CostAwareReset(InventoryQuote):
    risk_weight: float=.015
    forecast_horizon: float=1.5
    error_margin: float=.02
    def __post_init__(self):
        if min(self.risk_weight,self.forecast_horizon,self.error_margin)<0:
            raise ValueError("Nonnegative reset parameters required")
    def hedge(self,prices,positions,products,*,dt):
        ans=[]
        for k,q in positions.items():
            p=products[k]
            targets={q,0.0,math.copysign(math.floor(abs(q)*.5/p.lot)*p.lot,q)}
            best=q; best_gain=0.0
            for z in sorted(targets):
                if abs(z-q)<1e-9:
                    continue
                fee=abs(z-q)*(p.fee_per_unit+prices[k]*p.multiplier*p.fee_notional)
                # One-step frozen-position risk proxy; not a continuation-value
                # or positive-profit guarantee. Full planner is a later task.
                gain=self.risk_weight*self.forecast_horizon*(q*q-z*z)-fee
                if gain>best_gain+self.error_margin:
                    best,best_gain=z,gain
            if best!=q:
                ans.append((k,best-q))
        return ans
