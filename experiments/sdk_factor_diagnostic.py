"""Prospective train-only covariance selection; descriptive opened-holdout audit.

No policy tuning, no exchange actions, no change to the sealed SDK experiment.
"""
import argparse
from experiments.sdk_environment import KIT
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def read_prices(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    products = [key for key in rows[0] if key not in ("time_s", "round", "price_version")]
    return products, np.array([[float(row[pid]) for pid in products] for row in rows])


def covariance(increments):
    return np.atleast_2d(np.cov(np.asarray(increments, float), rowvar=False, ddof=1))


def low_rank_idiosyncratic(matrix, rank=1):
    """PCA common component plus nonnegative diagonal residual (symmetric PSD)."""
    matrix = np.asarray(matrix, float)
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    chosen = np.argsort(eigenvalues)[::-1][:rank]
    common = (vectors[:, chosen] * np.maximum(eigenvalues[chosen], 0)) @ vectors[:, chosen].T
    idiosyncratic = np.maximum(0., np.diag(matrix - common))
    return common + np.diag(idiosyncratic), common, idiosyncratic


def regularise(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    floor = max(1e-12, float(np.trace(matrix)) * 1e-10)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def qscore(increments, mean, matrix):
    """Gaussian quasi-likelihood score, not a Gaussian-market assertion."""
    matrix = regularise(matrix)
    residual = np.asarray(increments) - mean
    return float(np.linalg.slogdet(matrix)[1] +
                 np.mean(np.einsum("ni,ij,nj->n", residual, np.linalg.inv(matrix), residual)))


def chronological_folds(n, width=200, minimum=400):
    return [(0, end, end, end + width) for end in range(minimum, n - width + 1, width)]


def select_shrinkage(increments, alphas=(0., .25, .5, .75, 1.)):
    scores = []
    folds = chronological_folds(len(increments))
    if not folds:
        raise ValueError("Need at least 600 training increments for chronological selection")
    for alpha in alphas:
        block_scores = []
        for begin, end, vbegin, vend in folds:
            fitted = covariance(increments[begin:end])
            lowrank, _, _ = low_rank_idiosyncratic(fitted)
            matrix = alpha * lowrank + (1 - alpha) * np.diag(np.diag(fitted))
            block_scores.append(qscore(increments[vbegin:vend], increments[begin:end].mean(axis=0), matrix))
        scores.append({"alpha": alpha, "mean_qscore": float(np.mean(block_scores)),
                       "block_qscores": block_scores})
    selected = min(scores, key=lambda row: row["mean_qscore"])["alpha"]
    return selected, scores, folds


def portfolio_risk(quantities, matrix, multipliers, long_limits, short_limits):
    """Whole risk call includes multiplier projection and every individual limit."""
    q = np.asarray(quantities, float)
    exposure = q * np.asarray(multipliers, float)
    feasible = bool(np.all(q <= long_limits) and np.all(q >= -np.asarray(short_limits)))
    return {"variance_per_update": float(exposure @ matrix @ exposure), "within_individual_limits": feasible}


def transient_example(horizon=40):
    directed = np.array([[.8, 4.], [0., .8]])
    rows = []
    for h in range(horizon + 1):
        power = np.linalg.matrix_power(directed, h)
        rows.append({"horizon": h, "operator_norm": float(np.linalg.norm(power, 2)),
                     "unit_second_coordinate_norm": float(np.linalg.norm(power @ np.array([0., 1.]))),
                     "normal_comparator_norm": .8 ** h})
    return {"matrix": directed.tolist(), "eigenvalues": [float(x) for x in np.linalg.eigvals(directed)],
            "spectral_radius": .8, "rows": rows, "peak": max(rows, key=lambda row: row["operator_norm"]),
            "interpretation": "Synthetic deterministic linear map; not estimated market causality or impact."}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(KIT / "demo/demo_data.csv"))
    parser.add_argument("--holdout", default=str(KIT / "demo/demo_data_holdout.csv"))
    parser.add_argument("--rules", default=str(KIT / "demo/demo_settlement_rules.csv"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    utc = datetime.now(timezone.utc)
    out = Path(args.out or ("runs/" + utc.strftime("%Y%m%dT%H%M%SZ") + "_sdk_factors"))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "environment.json", {"python": sys.version, "numpy": np.__version__, "os": platform.platform()})
    config = {"stage": "T10 two-product development-data diagnostic", "data": args.data,
              "holdout": args.holdout, "rules": args.rules, "alphas": [0, .25, .5, .75, 1],
              "rank": 1, "fold_width_increments": 200, "minimum_training_increments": 400,
              "units": "arithmetic price increments per .5-second update", "no_policy_retuning": True,
              "holdout_status": "Already opened by parent comparative study; descriptive, not fresh confirmation",
              "demo_product_specs": {"A": {"max_long": 50, "max_short": 50, "multiplier": 1},
                                     "B": {"max_long": 25, "max_short": 25, "multiplier": 1}}}
    dump(out / "config.json", config)
    products, prices = read_prices(args.data)
    if products != ["A", "B"]:
        raise ValueError("This explicitly scoped diagnostic requires the published two-product demo")
    increments = np.diff(prices, axis=0)
    fitted = covariance(increments)
    lowrank, common, idio = low_rank_idiosyncratic(fitted)
    alpha, selection, folds = select_shrinkage(increments)
    models = {"diagonal": np.diag(np.diag(fitted)), "full": fitted,
              "low_rank_1_idiosyncratic": alpha * lowrank + (1 - alpha) * np.diag(np.diag(fitted))}
    lock = {"selected_alpha": alpha, "rank_fixed_before_evaluation": 1,
            "selection": selection, "folds": folds, "training_mean": increments.mean(axis=0).tolist(),
            "training_covariance": fitted.tolist(), "common_component": common.tolist(),
            "nonnegative_idiosyncratic": idio.tolist(),
            "models": {key: value.tolist() for key, value in models.items()},
            "source_sha256": sha(__file__), "training_sha256": sha(args.data), "rules_sha256": sha(args.rules)}
    # This lock is written before opening the held-out price file in this script.
    dump(out / "TRAINING_LOCK.json", lock)
    holdproducts, holdprices = read_prices(args.holdout)
    if holdproducts != products:
        raise ValueError("Holdout product columns differ")
    heldout = np.diff(holdprices, axis=0)
    realised = covariance(heldout)
    scores = {}
    for name, matrix in models.items():
        blocks = [qscore(heldout[i:i + 200], increments.mean(axis=0), matrix)
                  for i in range(0, len(heldout), 200) if len(heldout[i:i + 200]) == 200]
        scores[name] = {"heldout_qscore": qscore(heldout, increments.mean(axis=0), matrix),
                       "heldout_block_qscores": blocks,
                       "relative_covariance_frobenius_error": float(np.linalg.norm(matrix-realised)/np.linalg.norm(realised))}
    basis_vector = np.array([-2., 1.])
    basis = {"definition": "B - 2*A", "training_variance_per_update": float(basis_vector @ fitted @ basis_vector),
             "heldout_variance_per_update": float(basis_vector @ realised @ basis_vector),
             "exact_terminal_factor": "q_A*m_A + 2*q_B*m_B",
             "exact_terminal_basis_exposure": 0,
             "intraday_hedge_example": {"q_A": -2, "q_B": 1,
                                         "heldout_variance": float(basis_vector @ realised @ basis_vector)},
             "factor_loading_source": "Published demo B=2*A settlement, not a fitted causal coefficient"}
    basis["predicted_by_model"] = {
        name: {"variance_per_update": float(basis_vector @ matrix @ basis_vector),
               "ratio_to_heldout": float((basis_vector @ matrix @ basis_vector) /
                                         basis["heldout_variance_per_update"])}
        for name, matrix in models.items()}
    risk_rows = []
    for qa in (-50, -25, 0, 25, 50):
        for qb in (-25, -12, 0, 12, 25):
            for name, matrix in {**models, "realised_holdout": realised}.items():
                risk_rows.append({"q_A": qa, "q_B": qb, "model": name,
                                  **portfolio_risk([qa, qb], matrix, [1, 1], [50, 25], [50, 25])})
    dump(out / "risk_grid.json", risk_rows)
    transient = transient_example()
    dump(out / "synthetic_transient.json", transient)
    before = time.perf_counter()
    for _ in range(10000):
        portfolio_risk([-50, 25], models["low_rank_1_idiosyncratic"], [1, 1], [50, 25], [50, 25])
    profile = {"calls": 10000, "elapsed_seconds": time.perf_counter()-before,
               "whole_call_components": ["array conversion", "multiplier projection", "all individual limits", "quadratic risk"],
               "scope": "Two assets only; no 1000-asset scalability certification"}
    profile["seconds_per_call"] = profile["elapsed_seconds"] / profile["calls"]
    result = {"training_increments": len(increments), "heldout_increments": len(heldout),
              "realised_holdout_covariance": realised.tolist(), "scores": scores, "basis": basis,
              "profile": profile, "training_lock_sha256": sha(out / "TRAINING_LOCK.json"),
              "holdout_sha256": sha(args.holdout)}
    dump(out / "analysis.json", result)
    rows = "\n".join(f"| {name} | {row['heldout_qscore']:.6f} | {row['relative_covariance_frobenius_error']:.6f} |"
                     for name, row in scores.items())
    best = min(scores, key=lambda key: scores[key]["heldout_qscore"])
    (out / "REPORT.md").write_text(f"""# T10 two-product SDK factor diagnostic

No sealed policy, SDK Dealer or comparative harness was modified. There were {len(increments)} training increments and {len(heldout)} evaluation increments. Covariance is symmetric and uses price changes per 0.5-second update. Training-only chronological folds selected low-rank/diagonal mixture alpha={alpha}; rank one was fixed because this is a two-product diagnostic. The training lock was written before this script opened the holdout. That holdout was already opened by the parent study: these results are descriptive, not a new confirmatory test.

| Model | Held-out quasi-likelihood (lower better) | Relative covariance error |
|---|---:|---:|
{rows}

The lowest descriptive held-out score was {best}. Every candidate and every chronological score remains in TRAINING_LOCK.json; no losing fit was omitted. Gaussian quasi-likelihood is an evaluation criterion, not a claim of independent Gaussian market increments. Six 200-increment evaluation blocks are reported without pretending that their dependence has been removed. No financial superiority or hedge PnL is inferred.

The published final relation is B=2*A. Terminal factor exposure is q_A*m_A+2*q_B*m_B; multipliers are one in this demo. Individual limits remain A ±50 and B ±25 even when terminal factor exposure is zero. Intraday basis B-2*A has positive held-out increment variance {basis['heldout_variance_per_update']:.8f}; a position (-2 A,+1 B) therefore has intraday risk despite exact terminal factor cancellation. The rank-one-plus-idiosyncratic fit does not replace this exact settlement map.

**Negative diagnostic:** the low-rank-plus-idiosyncratic fit predicts only {basis['predicted_by_model']['low_rank_1_idiosyncratic']['ratio_to_heldout']:.3f} times the held-out variance of the hedged basis portfolio. It matches individual fitted variances but loses a residual cross-covariance important for hedged risk. The full fitted covariance predicts {basis['predicted_by_model']['full']['ratio_to_heldout']:.3f} times that variance; the diagonal estimate predicts {basis['predicted_by_model']['diagonal']['ratio_to_heldout']:.1f} times. A good aggregate covariance fit or positive semidefiniteness does not certify directional portfolio risk. No low-rank covariance is deployed by this diagnostic.

Synthetic directed matrix [[0.8,4],[0,0.8]] has spectral radius 0.8 but its finite-horizon operator norm peaks at {transient['peak']['operator_norm']:.6f}, horizon {transient['peak']['horizon']}. This proves that eigenvalue stability alone does not preclude transient amplification in that stated deterministic example. It is not an estimated market interaction, impact model or causal result.

The entire two-asset risk call (conversion, multipliers, individual limits and quadratic form) averaged {profile['seconds_per_call']:.9f} seconds over 10,000 calls. This cannot certify rank selection, hedge instruments or full computations for 1,000 products.

Permitted claims: S for supplied settlement/specification facts; D/PROVED_IN_STATED_MODEL for the explicit matrix-power counterexample and exposure identity; I/NUMERICALLY_CHECKED for covariance/limit implementation; EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT for the stated descriptive covariance scores only. Multi-asset factor/hedge selection and trading performance remain INCONCLUSIVE or BLOCKED_BY_MISSING_INPUT. No policy adaptation, no external trades and no deployment certification occurred in this diagnostic.
""")
    dump(out / "commands.log", {"argv": [sys.executable, "-m", "experiments.sdk_factor_diagnostic"] + list(sys.argv[1:]),
                                 "started_utc": utc.isoformat(), "elapsed_seconds": time.perf_counter()-started, "exit_code": 0})
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(sha(path) + "  " + str(path.relative_to(out)) + "\n" for path in files))
    print(json.dumps({"out": str(out), "selected_alpha": alpha, "best_descriptive_holdout_score": best,
                      "basis_variance": basis["heldout_variance_per_update"], "source_sha256": sha(__file__)}))
    return out


if __name__ == "__main__":
    main()
