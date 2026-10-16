"""Empirical public-feature fill estimates for one declared synthetic law.

This module supplies research predictions, not a competition model or a
uniform uncertainty certificate. A loss labels winning AND eligibility; it
does not identify the hidden cause. Unsupported queries return no prediction.
"""
from __future__ import annotations

import math
import numpy as np

from .calibration import decreasing_pava

DOMAIN = "controlled-sdk-mixture-v2"
EDGES = (0., .02, .04, .06, .08, .10, .12, .16, .20, .24, .28, .32)
FEATURE_NAMES = ("initial_edge", "current_edge", "customer_BUY", "product_B",
                 "quantity", "previous_increment", "assigned_theta", "float_boundary_ok")


def public_features(row):
    """Only quantities known before matching; future/latent labels are ignored.

    float_boundary_ok compares our quote with public reference +/- our own
    nominal spread. It does not inspect a customer's (unobserved) limit.
    Initial and current reference are equal in this controlled one-update law;
    accepting an arbitrary different current reference would extend the domain.
    """
    side = row["side"]
    if side not in ("BUY", "SELL"):
        raise ValueError("Unknown customer side")
    d = 1 if side == "BUY" else -1
    initial = float(row["matching_reference"])
    current = float(row.get("current_reference", initial))
    quote = float(row["own_quote"])
    edge = float(row["edge"])
    theta = float(row["theta"])
    quantity = float(row["quantity"])
    previous = float(row["previous_increment"])
    values = (initial, current, quote, edge, theta, quantity, previous)
    if not all(math.isfinite(v) for v in values) or min(initial, current, quote, quantity) <= 0:
        raise ValueError("Finite positive references/quote/quantity required")
    threshold = initial+d*edge
    float_ok = quote <= threshold if side == "BUY" else quote >= threshold
    return np.array([edge, d*(quote-current), side == "BUY", row["product_id"] == "B",
                     quantity, previous, theta, float_ok], float)


def validate_records(rows):
    seen = set()
    if not rows:
        raise ValueError("Nonempty final-RFQ observations required")
    for row in rows:
        key = (row["session_seed"], row["order_id"], row["theta"])
        if key in seen:
            raise ValueError("Repeated callback for a final RFQ/experimental arm")
        seen.add(key)
        public_features(row)
        if type(row["won"]) is not bool:
            raise ValueError("Only observed final own outcomes fit this model")


def _cell(row):
    x = public_features(row)
    # Nominal edge has a fixed Decimal quote convention in this synthetic law.
    return f"{x[0]:.2f}:{int(x[7])}"


def _sigmoid(z):
    return np.exp(-np.logaddexp(0., -z))


def _fit_logistic(x, y):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0.]+[1.]*(design.shape[1]-1))
    for _ in range(100):
        p = _sigmoid(design@beta)
        h = design.T@((p*(1-p))[:, None]*design)+penalty
        step = np.linalg.solve(h+1e-9*np.eye(len(beta)), design.T@(p-y)+penalty@beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return dict(coefficients=beta.tolist(), mean=mean.tolist(), scale=scale.tolist())


def fit_models(rows, *, endpoints=(.2, .8), minimum_cell_n=40):
    validate_records(rows)
    x = np.array([public_features(r) for r in rows])
    y = np.array([r["won"] for r in rows], float)
    counts = np.array([sum(r["edge"] == e for r in rows) for e in EDGES])
    wins = np.array([sum(r["won"] for r in rows if r["edge"] == e) for e in EDGES])
    if np.any(counts == 0) or not endpoints[0] < endpoints[1]:
        raise ValueError("All declared edges and ordered endpoint levels required")
    pooled = (wins+.5)/(counts+1.)
    mono = decreasing_pava(wins+.5, counts+1.)
    cells = {}
    for level in endpoints:
        for row in rows:
            if row["theta"] == level:
                key = f"{level:.1f}|{_cell(row)}"
                cell = cells.setdefault(key, dict(n=0, wins=0))
                cell["n"] += 1
                cell["wins"] += row["won"]
    for cell in cells.values():
        cell["probability"] = (cell["wins"]+.5)/(cell["n"]+1.)
    # The old misspecified feature family remains an explicit comparator.
    legacy = x[:, [0, 3, 2, 4, 5]]
    return dict(domain=DOMAIN, endpoints=list(endpoints), minimum_cell_n=minimum_cell_n,
                feature_names=list(FEATURE_NAMES), constant=float(y.mean()),
                pooled=pooled.tolist(), monotone=mono.tolist(),
                logistic_legacy=_fit_logistic(legacy, y),
                logistic_public=_fit_logistic(x, y), cells=cells,
                training_rfqs=len(rows), training_sessions=len({r["session_seed"] for r in rows}),
                empirical_uncertainty_only=True, runtime_enabled=False)


def predict(model, rows, method, *, domain=DOMAIN):
    """Return probability vector, supported mask; NaN is an abstention.

    Model choice/calibration never silently extends the law, edge grid, timing
    convention, intervention range or minimum empirical support.
    """
    if method not in ("constant", "pooled", "monotone", "logistic_legacy", "logistic_public", "conditional_finite"):
        raise ValueError("Unknown predeclared predictor")
    probabilities = np.full(len(rows), np.nan)
    supported = np.zeros(len(rows), bool)
    lo, hi = model["endpoints"]
    for i, row in enumerate(rows):
        x = public_features(row)
        if (domain != model["domain"] or row.get("shifted_family", False) or
                not lo <= x[6] <= hi or row["edge"] not in EDGES or
                row["product_id"] not in ("A", "B") or row["quantity"] not in (2, 4, 10) or
                abs(row["previous_increment"]) > .1400001 or
                row.get("current_reference", row["matching_reference"]) != row["matching_reference"]):
            continue
        if method == "constant":
            p = model[method]
        elif method in ("pooled", "monotone"):
            p = model[method][EDGES.index(row["edge"])]
        elif method.startswith("logistic"):
            fit = model[method]
            xx = x[[0, 3, 2, 4, 5]] if method == "logistic_legacy" else x
            design = np.r_[1., (xx-fit["mean"])/fit["scale"]]
            p = float(_sigmoid(design@fit["coefficients"]))
        else:
            keys = [f"{t:.1f}|{_cell(row)}" for t in (lo, hi)]
            if any(key not in model["cells"] or model["cells"][key]["n"] < model["minimum_cell_n"] for key in keys):
                continue
            w = (x[6]-lo)/(hi-lo)
            p = (1-w)*model["cells"][keys[0]]["probability"]+w*model["cells"][keys[1]]["probability"]
        supported[i], probabilities[i] = True, p
    return probabilities, supported


def markout_fit(rows):
    filled = [r for r in rows if r["won"]]
    if not filled:
        raise ValueError("No observed own fill markouts")
    design = np.array([[1., (-1 if r["side"] == "BUY" else 1)*r["previous_increment"],
                        r["product_id"] == "B"] for r in filled], float)
    coefficient = np.linalg.lstsq(design, [r["signed_markout"] for r in filled], rcond=None)[0]
    return dict(coefficients=coefficient.tolist(), features=["intercept", "signed_previous_increment", "product_B"],
                training_fills=len(filled), future_labels_are_features=False)


def markout_predict(model, rows):
    design = np.array([[1., (-1 if r["side"] == "BUY" else 1)*r["previous_increment"],
                        r["product_id"] == "B"] for r in rows], float)
    return design@model["coefficients"]
