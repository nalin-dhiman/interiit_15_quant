"""A small one-RFQ-per-step benchmark with known fill probabilities.

Side is observed, sizes are one lot, price is a martingale, customer fee is
zero, and risk is a quadratic DESIGN objective. Reset is before quoting.
No batching, adverse selection, price impact, or partial information here.
The exact finite-MDP solution is only a mathematical/control diagnostic.
"""
from __future__ import annotations
import math
import numpy as np
from .mdp import FiniteMDP


def build_inventory_mdp(horizon: int=12, limit: int=4, *, a: float=.015,
                        fee: float=.03, fixed_cost: float=0.0,
                        margins: tuple[float,...]=(.02,.08,.16,.3),
                        buy_probability: float=.5, allow_reset: bool=True
                       ) -> tuple[FiniteMDP,list[tuple[int,int]],list[tuple[int|None,float|None]]]:
    if horizon<1 or limit<1 or int(horizon)!=horizon or int(limit)!=limit:
        raise ValueError("Positive integer horizon and limit required")
    if min(a,fee,fixed_cost)<0 or not 0<=buy_probability<=1 or not all(math.isfinite(x) for x in margins):
        raise ValueError("Invalid model parameters")
    states=[(q,d) for q in range(-limit,limit+1) for d in (-1,1)]
    # d=+1 means CUSTOMER BUY (dealer inventory decreases).
    targets=[None]+(list(range(-limit,limit+1)) if allow_reset else [])
    actions=[(z,e) for z in targets for e in (None,)+margins]
    S,A=len(states),len(actions); index={s:i for i,s in enumerate(states)}
    P=np.zeros((horizon,S,A,S)); g=np.zeros((horizon,S,A)); mask=np.ones_like(g,bool)
    for i,(q,d) in enumerate(states):
        for j,(target,edge) in enumerate(actions):
            x=q if target is None else target
            reset_cost=0.0 if x==q else fixed_cost+fee*abs(x-q)
            p=0.0 if edge is None else 1/(1+math.exp((edge-.12)/.055))
            y=x-d
            if p>0 and not -limit<=y<=limit:
                mask[:,i,j]=False
                p=0.0
            for filled,prob in ((False,1-p),(True,p)):
                if prob==0:
                    continue
                next_q=y if filled else x
                reward=(float(edge) if filled else 0)-reset_cost-a*next_q**2
                g[:,i,j]+=prob*reward
                for next_d,pd in ((1,buy_probability),(-1,1-buy_probability)):
                    P[:,i,j,index[(next_q,next_d)]]+=prob*pd
    # No forced liquidation. Zero terminal mark residual because price is a
    # martingale and terminal value is already in mark-to-market increments.
    return FiniteMDP(P,g,np.zeros(S),mask),states,actions
