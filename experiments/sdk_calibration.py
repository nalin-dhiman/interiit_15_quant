"""Training-only final-RFQ predictive calibration; no causal response claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_records(path, method="B1"):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    records = [r for r in records if r["method"] == method and r["own_quote"] is not None]
    seen = set()
    for r in records:
        key = r["block"], r["seed"], r.get("policy_id"), r["order_id"]
        if key in seen:
            raise ValueError("Repeated callback/sample for the same final RFQ/policy")
        seen.add(key)
    return records


def features(records):
    return np.array([[r["edge"], r["product_id"] == "B", r["side"] == "BUY", r["quantity"]]
                     for r in records], dtype=float)


def logistic_fit(x, y, ridge=1.):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.)
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    beta = np.zeros(design.shape[1])
    regularisation = np.diag([0.] + [ridge]*(design.shape[1]-1))
    for _ in range(100):
        p = 1/(1+np.exp(-np.clip(design@beta, -40, 40)))
        hessian = design.T @ ((p*(1-p))[:, None]*design) + regularisation
        delta = np.linalg.solve(hessian+1e-9*np.eye(len(beta)), design.T@(p-y)+regularisation@beta)
        beta -= delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return dict(coefficients=beta.tolist(), feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                feature_names=["edge", "product_B", "customer_BUY", "quantity"], ridge=ridge)


def logistic_predict(model, x):
    design = np.column_stack([np.ones(len(x)), (x-np.array(model["feature_mean"]))/model["feature_scale"]])
    return 1/(1+np.exp(-np.clip(design@model["coefficients"], -40, 40)))


def monotone_decreasing(values, weights):
    """Weighted pool-adjacent-violators fit for edge -> joint win probability."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i+1, float(value*weight), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2]/blocks[-2][3] < blocks[-1][2]/blocks[-1][3]:
            b = blocks.pop()
            blocks[-1][1] = b[1]
            blocks[-1][2] += b[2]
            blocks[-1][3] += b[3]
    result = np.empty(len(values))
    for lo, hi, total, weight in blocks:
        result[lo:hi] = total/weight
    return result


def metrics(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return dict(n=len(y), observed_win_rate=float(np.mean(y)), predicted_win_rate=float(np.mean(p)),
                brier=float(np.mean((p-y)**2)),
                log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))


def calibrate(train_path, validation_path, output_path):
    train, validation = read_records(train_path), read_records(validation_path)
    if not train or not validation:
        raise ValueError("Calibration needs nonempty B1 training and training-file validation RFQs")
    x, xv = features(train), features(validation)
    y, yv = np.array([r["won"] for r in train], dtype=float), np.array([r["won"] for r in validation], dtype=float)
    model = logistic_fit(x, y)
    p = logistic_predict(model, xv)
    edges = np.array([-.5, 0., .04, .08, .12, .16, .25, .5, 1e6])
    index = np.clip(np.searchsorted(edges, x[:, 0], side="right")-1, 0, len(edges)-2)
    counts = np.bincount(index, minlength=len(edges)-1)
    wins = np.bincount(index, weights=y, minlength=len(edges)-1)
    pooled = (wins+.5)/(counts+1.)
    mono = monotone_decreasing(pooled, counts+1.)
    vi = np.clip(np.searchsorted(edges, xv[:, 0], side="right")-1, 0, len(edges)-2)
    support, coverage = {}, []
    for product in ("A", "B"):
        for side in ("BUY", "SELL"):
            tr = [r["edge"] for r in train if r["product_id"] == product and r["side"] == side]
            va = [r["edge"] for r in validation if r["product_id"] == product and r["side"] == side]
            lo, hi = min(tr), max(tr)
            support[f"{product}_{side}"] = dict(training_n=len(tr), edge_min=lo, edge_max=hi,
                validation_n=len(va), validation_in_range=sum(lo <= e <= hi for e in va))
            coverage.extend(lo <= e <= hi for e in va)
    markout = {}
    for product in ("A", "B"):
        tr = [r["dealer_signed_one_tick_markout"] for r in train if r["won"] and r["product_id"] == product]
        va = [r["dealer_signed_one_tick_markout"] for r in validation if r["won"] and r["product_id"] == product]
        markout[product] = dict(training_fills=len(tr), validation_fills=len(va),
            training_mean=float(np.mean(tr)) if tr else None,
            validation_mean=float(np.mean(va)) if va else None,
            validation_zero_forecast_mse=float(np.mean(np.array(va)**2)) if va else None,
            validation_fitted_mean_mse=float(np.mean((np.array(va)-np.mean(tr))**2)) if tr and va else None)
    result = dict(status="NUMERICALLY_CHECKED_PREDICTIVE_DEMO_CALIBRATION_ONLY", deployment_enabled=False,
        training_input_sha256=hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        validation_input_sha256=hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
        training_final_rfqs=len(train), validation_final_rfqs=len(validation),
        independent_training_price_blocks=len({r["block"] for r in train}),
        independent_validation_price_blocks=len({r["block"] for r in validation}),
        model_target="Pr(winning AND customer eligibility) as observed own fill; losses are censored limit observations",
        causal_response_identified=False, future_features_used=False,
        constant=metrics(yv, np.full(len(yv), y.mean())), logistic=metrics(yv, p),
        pooled=metrics(yv, pooled[vi]), monotone=metrics(yv, mono[vi]),
        coefficients=model, edge_bins=edges.tolist(), pooled_probabilities=pooled.tolist(),
        monotone_probabilities=mono.tolist(), training_bin_counts=counts.tolist(),
        support=support, validation_support_coverage=float(np.mean(coverage)),
        markout=markout,
        limitations=["Repeated policy/seed observations share RFQ tapes and price blocks",
                    "Only six training and three validation price blocks; row-level independence not claimed",
                    "Calibration validates this demo opponent/customer configuration only",
                    "No hidden customer limits, losing rival quotes or production response are identified",
                    "Coefficients exported for review; runtime transfer remains disabled"])
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.train, args.validation, args.out)
    print(json.dumps({k: result[k] for k in ("training_final_rfqs", "validation_final_rfqs", "constant", "logistic", "monotone")}))
