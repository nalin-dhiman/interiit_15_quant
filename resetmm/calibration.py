"""Offline, final-RFQ calibration with explicit observation/support checks.

Own win is the combined winning-and-eligibility event. A losing quote does
not identify hidden limits or losing cause. Future markout is a label only.
These fitted models supply empirical predictions, not uniform certificates.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
import numpy as np

RECORD_FIELDS = frozenset({
    'session_id', 'order_id', 'product_id', 'side', 'quantity', 'reference',
    'inventory', 'final_quote', 'final_edge', 'callback_count', 'won', 'markout',
})


def validate_final_records(records: Sequence[dict]) -> None:
    seen: set[tuple[str, str]] = set()
    for r in records:
        if set(r) != RECORD_FIELDS:
            raise ValueError('Use only the specified final public/own features and labels')
        key = (r['session_id'], r['order_id'])
        if key in seen:
            raise ValueError('One final observation per session/RFQ; callbacks are not samples')
        seen.add(key)
        if r['side'] not in ('BUY', 'SELL'):
            raise ValueError('Customer side must be BUY or SELL')
        if not all(math.isfinite(r[k]) for k in ('quantity', 'reference', 'inventory')):
            raise ValueError('Nonfinite public/own feature')
        if r['quantity'] <= 0 or r['reference'] <= 0:
            raise ValueError('Positive synthetic quantity and reference required')
        if type(r['callback_count']) is not int or r['callback_count'] < 1:
            raise ValueError('Positive callback count required')
        if r['won'] is not None and type(r['won']) is not bool:
            raise ValueError('Own final win must be boolean or censored None')
        quote, edge = r['final_quote'], r['final_edge']
        if quote is None:
            if edge is not None or r['won'] is True:
                raise ValueError('A withdrawn quote cannot win or have an edge')
        else:
            if not math.isfinite(quote) or quote <= 0 or edge is None or not math.isfinite(edge):
                raise ValueError('Invalid final quote/edge')
            d = 1 if r['side'] == 'BUY' else -1
            if not math.isclose(edge, d*(quote-r['reference']), abs_tol=1e-9, rel_tol=0):
                raise ValueError('Final edge must use the final observed reference')
        if r['markout'] is not None:
            if r['won'] is not True or not math.isfinite(r['markout']):
                raise ValueError('Post-fill markout labels require an observed own fill')


def _labelled_quotes(records: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    validate_final_records(records)
    labelled = [r for r in records if r['final_quote'] is not None and r['won'] is not None]
    if not labelled:
        raise ValueError('No observed final quoted RFQ outcomes')
    return (np.array([r['final_edge'] for r in labelled], float),
            np.array([r['won'] for r in labelled], float))


def decreasing_pava(successes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Weighted Bernoulli isotonic MLE for decreasing occupied-bin rates."""
    s, n = np.asarray(successes, float), np.asarray(counts, float)
    if s.ndim != 1 or n.shape != s.shape or not len(s):
        raise ValueError('Equal nonempty vectors required')
    if not np.isfinite(s).all() or not np.isfinite(n).all() or np.any(n <= 0):
        raise ValueError('Finite occupied-bin counts required')
    if np.any(s < 0) or np.any(s > n):
        raise ValueError('Success count outside bin count')
    blocks: list[list[float | int]] = []
    for i, (si, ni) in enumerate(zip(s, n)):
        blocks.append([i, i+1, si, ni])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] < blocks[-1][2]/blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    fitted = np.empty_like(s)
    for first, end, total, weight in blocks:
        fitted[int(first):int(end)] = total/weight
    return fitted


def _bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -z))


def fit_fill_models(records: Sequence[dict], bin_edges: Sequence[float], *,
                    ridge: float = 1e-6, max_iterations: int = 100,
                    tolerance: float = 1e-10) -> dict:
    x, y = _labelled_quotes(records)
    edges = np.asarray(bin_edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Strictly increasing finite bin edges required')
    if ridge <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError('Positive fitting controls required')
    if x.min() < edges[0] or x.max() > edges[-1]:
        raise ValueError('Training observations fall outside predeclared bins')
    b = _bins(x, edges)
    counts = np.bincount(b, minlength=len(edges)-1)
    successes = np.bincount(b, weights=y, minlength=len(edges)-1)
    occupied = counts > 0
    rates = decreasing_pava(successes[occupied], counts[occupied])
    # Empty bins interpolate neighbouring fitted bins; extrapolation is flagged
    # through support checks, not presented as empirically calibrated support.
    monotone_rates = np.interp(np.arange(len(counts)), np.flatnonzero(occupied), rates)
    X = np.column_stack((np.ones(len(x)), 100*x))
    pooled = float(y.mean())
    p0 = np.clip(pooled, 1e-8, 1-1e-8)
    beta = np.array([math.log(p0/(1-p0)), 0.0])

    def objective(c: np.ndarray) -> float:
        z = X@c
        return float(np.sum(np.logaddexp(0, z)-y*z) + .5*ridge*(c@c))

    converged = False
    for iteration in range(1, max_iterations+1):
        probabilities = _sigmoid(X@beta)
        gradient = X.T@(probabilities-y) + ridge*beta
        hessian = X.T@((probabilities*(1-probabilities))[:, None]*X) + ridge*np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) <= tolerance:
            converged = True
            break
        scale = 1.0
        old_objective = objective(beta)
        while scale > 2**-30 and objective(beta-scale*step) > old_objective:
            scale *= .5
        beta -= scale*step
    support = [float(x.min()), float(x.max())]
    bin_support = dict(training_bin_edges=edges.tolist(), training_bin_counts=counts.tolist())
    return {
        'pooled': {'kind': 'pooled', 'probability': pooled, 'support': support, **bin_support},
        'monotone': {'kind': 'monotone', 'bin_edges': edges.tolist(),
                     'probabilities': monotone_rates.tolist(),
                     'counts': counts.tolist(), 'successes': successes.tolist(),
                     'support': support, **bin_support},
        'logistic': {'kind': 'logistic', 'coefficients': beta.tolist(),
                     'feature_names': ['intercept', 'final_edge_in_cents'],
                     'ridge': ridge, 'iterations': iteration, 'converged': converged,
                     'support': support, **bin_support},
    }


def predict_fill(model: dict, x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Finite edge vector required')
    if model['kind'] == 'pooled':
        return np.full_like(a, model['probability'])
    if model['kind'] == 'monotone':
        return np.asarray(model['probabilities'])[_bins(a, np.asarray(model['bin_edges']))]
    if model['kind'] == 'logistic':
        intercept, slope = model['coefficients']
        return _sigmoid(intercept+slope*100*a)
    raise ValueError('Unknown fill model')


def in_training_support(model: dict, x: Sequence[float]) -> np.ndarray:
    """Observed min/max range only; this is not a complete support certificate."""
    a = np.asarray(x, float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite support query')
    low, high = model['support']
    return (a >= low) & (a <= high)


def in_occupied_bin_support(model: dict, x: Sequence[float]) -> np.ndarray:
    """Whether a predeclared bin contains training data; density is not certified."""
    a = np.asarray(x, float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Finite edge vector required')
    edges = np.asarray(model['training_bin_edges'])
    occupied = np.asarray(model['training_bin_counts']) > 0
    return (a >= edges[0]) & (a <= edges[-1]) & occupied[_bins(a, edges)]


def fit_markout_models(records: Sequence[dict], bin_edges: Sequence[float], *,
                       pseudocount: float = 20) -> dict:
    validate_final_records(records)
    filled = [r for r in records if r['won'] is True and r['markout'] is not None]
    if not filled or not math.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError('Observed fill markouts and nonnegative pseudocount required')
    edges = np.asarray(bin_edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Strictly increasing finite markout bins required')
    x = np.array([r['final_edge'] for r in filled])
    y = np.array([r['markout'] for r in filled])
    b = _bins(x, edges)
    counts = np.bincount(b, minlength=len(edges)-1)
    sums = np.bincount(b, weights=y, minlength=len(edges)-1)
    pooled = float(y.mean())
    denominators = counts + pseudocount
    means = np.divide(sums+pseudocount*pooled, denominators,
                      out=np.full_like(sums, pooled), where=denominators > 0)
    return {'pooled_mean': pooled, 'bin_edges': edges.tolist(),
            'conditional_means': means.tolist(), 'counts': counts.tolist(),
            'pseudocount': pseudocount, 'support': [float(x.min()), float(x.max())]}


def predict_markout(model: dict, x: Sequence[float], *, conditional: bool) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Finite edge vector required')
    if not conditional:
        return np.full_like(a, model['pooled_mean'])
    return np.asarray(model['conditional_means'])[_bins(a, np.asarray(model['bin_edges']))]


def bernoulli_losses(y: Sequence[float], p: Sequence[float]) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Equal nonempty observation/probability vectors required')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or np.any((y != 0) & (y != 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Binary observations and finite probabilities required')
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return {'brier': float(np.mean((y-p)**2)),
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))}
