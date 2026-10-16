"""Common-support likelihood checks and a reproducible response benchmark."""
from __future__ import annotations
import numpy as np
from .mdp import FiniteMDP


def bernoulli_score(outcome: bool, p: float, dp: float) -> float:
    if not 0<p<1 or not np.isfinite(dp):
        raise ValueError("Regular Bernoulli response requires 0<p<1")
    return (float(outcome)-p)*dp/(p*(1-p))


def response_fixture(theta: float=0.0, horizon: int=8) -> tuple[FiniteMDP,np.ndarray,np.ndarray,np.ndarray]:
    """Strictly positive 3-state kernels. Differentiable parameter changes both
    transitions and reward; fixed action avoids nonsmooth argmax effects."""
    if not np.isfinite(theta) or horizon<1:
        raise ValueError("Invalid fixture parameters")
    logits=np.array([[.2,-.3,.4],[.1,.6,-.2],[-.1,.1,.3]])
    directions=np.array([[.4,-.2,.1],[-.3,.5,.2],[.1,-.4,.6]])
    log=logits+theta*directions; w=np.exp(log-log.max(axis=1,keepdims=True)); p=w/w.sum(1,keepdims=True)
    dp=p*(directions-(p*directions).sum(1,keepdims=True))
    base=np.array([.2,-.1,.1]); slope=np.array([.07,.03,-.04])
    P=np.tile(p[None,:,None,:],(horizon,1,1,1))
    dP=np.tile(dp[None,:,None,:],(horizon,1,1,1))
    g=np.tile((base+theta*slope)[None,:,None],(horizon,1,1))
    dg=np.tile(slope[None,:,None],(horizon,1,1))
    m=FiniteMDP(P,g,np.array([.4,-.2,.1]))
    return m,np.zeros((horizon,3),int),dP,dg
