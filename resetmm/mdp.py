"""Exact finite-horizon model-based dynamic programming.

Exact means exact for the supplied finite transition/reward arrays, up to
floating-point arithmetic. It is NOT an exact solution of the competition.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FiniteMDP:
    P: np.ndarray      # [time, state, action, next_state]
    reward: np.ndarray # expected single-step reward [time, state, action]
    terminal: np.ndarray
    feasible: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=float)
        self.reward = np.asarray(self.reward, dtype=float)
        self.terminal = np.asarray(self.terminal, dtype=float)
        if self.P.ndim != 4:
            raise ValueError("P must have four dimensions")
        T, S, A, S2 = self.P.shape
        if min(T, S, A) < 1 or S != S2 or self.reward.shape != (T,S,A) or self.terminal.shape != (S,):
            raise ValueError("Inconsistent MDP dimensions")
        if not all(np.isfinite(x).all() for x in (self.P, self.reward, self.terminal)):
            raise ValueError("MDP arrays must be finite")
        if np.min(self.P) < 0 or not np.allclose(self.P.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError("Transition rows must be nonnegative and sum to one")
        if self.feasible is None:
            self.feasible = np.ones((T,S,A), dtype=bool)
        else:
            self.feasible = np.asarray(self.feasible, dtype=bool)
        if self.feasible.shape != (T,S,A) or not self.feasible.any(-1).all():
            raise ValueError("Every state/time must have a feasible action")


def solve(m: FiniteMDP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, S, A, _ = m.P.shape
    V = np.empty((T+1,S)); V[T] = m.terminal
    pi = np.empty((T,S), dtype=int)
    Q = np.empty((T,S,A))
    for t in range(T-1,-1,-1):
        Q[t] = m.reward[t] + np.einsum("saj,j->sa",m.P[t],V[t+1])
        Q[t][~m.feasible[t]] = -np.inf
        pi[t] = np.argmax(Q[t],axis=-1)
        V[t] = Q[t,np.arange(S),pi[t]]
    return V, pi, Q


def validate_policy(m: FiniteMDP, pi: np.ndarray) -> np.ndarray:
    T,S,A,_ = m.P.shape
    raw = np.asarray(pi)
    if raw.shape != (T,S) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Policy must be a time-by-state integer array")
    p = raw.astype(int)
    if np.any(p<0) or np.any(p>=A) or not m.feasible[np.arange(T)[:,None],np.arange(S),p].all():
        raise ValueError("Policy includes an infeasible action")
    return p


def evaluate(m: FiniteMDP, pi: np.ndarray) -> np.ndarray:
    pi = validate_policy(m,pi)
    T,S,_,_ = m.P.shape
    V = np.empty((T+1,S)); V[-1] = m.terminal
    for t in range(T-1,-1,-1):
        P = m.P[t,np.arange(S),pi[t]]
        V[t] = m.reward[t,np.arange(S),pi[t]] + P@V[t+1]
    return V


def fixed_policy_derivative(m: FiniteMDP, pi: np.ndarray, dP: np.ndarray,
                            dr: np.ndarray, dterminal: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Tangent Bellman recursion for a FIXED policy and common-support model.
    Includes the explicit derivative of the reward. Does not differentiate an
    argmax, switching threshold, changed action support, or hidden-state filter.
    """
    pi = validate_policy(m,pi)
    dP, dr = np.asarray(dP,float), np.asarray(dr,float)
    if dP.shape != m.P.shape or dr.shape != m.reward.shape:
        raise ValueError("Derivative shape mismatch")
    if not np.isfinite(dP).all() or not np.isfinite(dr).all() or not np.allclose(dP.sum(-1),0,atol=1e-10,rtol=0):
        raise ValueError("Invalid derivative of stochastic kernel")
    if np.any((m.P==0) & (np.abs(dP)>1e-14)):
        raise ValueError("Support-changing derivative: do not use regular response")
    T,S,_,_ = m.P.shape
    dv = np.zeros((T+1,S))
    if dterminal is not None:
        arr = np.asarray(dterminal,float)
        if arr.shape != (S,) or not np.isfinite(arr).all():
            raise ValueError("Invalid terminal derivative")
        dv[-1] = arr
    V = evaluate(m,pi)
    for t in range(T-1,-1,-1):
        a=pi[t]; ix=np.arange(S)
        dv[t] = dr[t,ix,a] + dP[t,ix,a]@V[t+1] + m.P[t,ix,a]@dv[t+1]
    return V,dv


def safe_switch(action_values: np.ndarray, baseline: int, error_bound: float) -> int:
    """Conditional 2*epsilon certificate, only if the supplied bound is valid."""
    q = np.asarray(action_values,float)
    if q.ndim!=1 or not np.isfinite(q).all() or not 0<=baseline<len(q):
        raise ValueError("Invalid action values or baseline")
    if error_bound<0 or not np.isfinite(error_bound):
        raise ValueError("Invalid error bound")
    best=int(np.argmax(q))
    return best if q[best]-q[baseline]>2*error_bound else baseline


def model_value_error_bound(H: int, reward_bound: float, terminal_bound: float,
                            reward_error: float, terminal_error: float,
                            transition_l1_error: float) -> float:
    """Conditional bound for finite models; inputs are uniform finite bounds.

    This arithmetic does not establish the supplied premises from data.
    """
    bounds = (reward_bound, terminal_bound, reward_error, terminal_error,
              transition_l1_error)
    if (not np.isfinite(H) or H < 0 or int(H) != H
            or not all(np.isfinite(v) and v >= 0 for v in bounds)):
        raise ValueError("Finite nonnegative bounds and integer horizon required")
    result = (H*reward_error + terminal_error + transition_l1_error*
              (reward_bound*H*(H-1)/2+H*terminal_bound))
    if not np.isfinite(result):
        raise ArithmeticError("Value-error bound exceeds floating-point range")
    return result
