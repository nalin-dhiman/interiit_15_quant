"""Session-level statistics. Resample sessions, not correlated trades."""
from __future__ import annotations
import numpy as np


def paired_bootstrap_interval(candidate, baseline, *, seed: int=917,
                              replicates: int=4000, alpha: float=.05) -> tuple[float,float,float]:
    x,y=np.asarray(candidate,float),np.asarray(baseline,float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<2 or not np.isfinite(x-y).all():
        raise ValueError("Need at least two paired finite session values")
    if replicates<100 or not 0<alpha<1:
        raise ValueError("Invalid bootstrap settings")
    d=x-y; rng=np.random.default_rng(seed)
    means=d[rng.integers(0,len(d),size=(replicates,len(d)))].mean(1)
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2])
    return float(d.mean()),float(lo),float(hi)


def lower_tail_mean(values, fraction: float=.1) -> float:
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not 0<fraction<=1:
        raise ValueError("Invalid lower-tail input")
    n=max(1,int(np.ceil(len(x)*fraction)))
    return float(np.sort(x)[:n].mean())
