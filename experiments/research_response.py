"""Prospective response-model completion; actual SDK matching, synthetic law.

Declare before execution. Fresh independent train/development/validation streams
are never taken from a supplied holdout or from previous response experiments.
"""
from __future__ import annotations
from experiments.sdk_environment import KIT, complete_source_hashes, source_key, source_path

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from resetmm.research_response import (DOMAIN, EDGES, fit_models, predict,
    public_features, markout_fit, markout_predict)
from experiments.sdk_intervention_response import (make_tape, run_tape,
    response_analysis, sdk_fill_probability)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("constant", "pooled", "monotone", "logistic_legacy", "logistic_public", "conditional_finite")
NAMESPACE = "research-completion-response/20261016T134827Z/v1"


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    names = ["resetmm/research_response.py", "experiments/research_response.py",
             "experiments/sdk_intervention_response.py", "experiments/sdk_calibration.py",
             "resetmm/calibration.py", "docs/RESEARCH_RESPONSE.md", "tests_sdk/test_research_response.py",
             "inputs/official/kit/IMC/rfq/rfq/engine.py"]
    return complete_source_hashes({name: sha(source_path(name)) for name in names})


def seed(phase, index):
    return int.from_bytes(hashlib.sha256(f"{NAMESPACE}|{phase}|{index}".encode()).digest(), "big")


def declare(out):
    out.mkdir(parents=True, exist_ok=False)
    config = dict(stage="T06/T08 empirical response completion", declared_utc=datetime.now(timezone.utc).isoformat(),
        namespace=NAMESPACE, intervals=120,
        splits=dict(train=dict(levels=[.2, .8], sessions_per_level=200),
                    development=dict(levels=[.35, .65], sessions_per_level=40),
                    validation=dict(levels=[.2, .35, .65, .8], sessions_per_level=100),
                    shifted=dict(levels=[.5], sessions_per_level=100)),
        response_sessions=80, response_levels=[.45, .5, .55],
        methods=list(METHODS), endpoint_interpolation=[.2, .8], minimum_training_cell_n=40,
        gate=dict(absolute_mean_error=.03, ece=.05, edge_error=.10,
                  public_cell_error=.15, supported_validation_bin_n=100, coverage=1.),
        model_selection="Lowest mean development Brier among full-support, all-level passing models; fixed method order breaks ties. If none pass, select best Brier for reporting only and export nothing.",
        final_validation_used_for_selection=False, source_hashes=sources(), runtime_enabled=False,
        shifted_family_withheld_by_domain_guard=True,
        independent_unit="new price/customer/opponent/exploration session; repeated response arms are paired",
        claimed_target="own winning AND eligibility under the declared synthetic Engine law",
        false_claims=["hidden customer limits identified", "actual market law", "uniform empirical certificate", "R3 trading superiority"])
    write(out/"config.json", config)
    write(out/"environment.json", dict(python=sys.version, numpy=np.__version__, executable=sys.executable,
                                      platform=platform.platform()))
    snap = out/"sources"
    for name in config["source_hashes"]:
        target = snap/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path(name).read_bytes())
    return config


def collect(split, spec, out, intervals):
    records, totals = [], []
    with gzip.open(out/f"{split}_final_rfqs.jsonl.gz", "wt") as handle:
        for level in spec["levels"]:
            for i in range(spec["sessions_per_level"]):
                session_seed = seed(f"{split}:{level}", i)
                rows, total = run_tape(make_tape(session_seed, intervals), level, shifted=split == "shifted")
                for row in rows:
                    row["current_reference"] = row["matching_reference"]
                    handle.write(json.dumps(row, sort_keys=True)+"\n")
                records.extend(rows)
                totals.append(total)
    write(out/f"{split}_sessions.json", totals)
    return records


def measure(rows, probability, support, gates):
    y = np.array([r["won"] for r in rows], float)
    coverage = float(np.mean(support))
    if not np.any(support):
        return dict(n=len(rows), coverage=coverage, gate_passed=False, brier=None,
                    reason="Domain/support guard abstained on every observation")
    yy, pp = y[support], probability[support]
    brier = float(np.mean((yy-pp)**2))
    probability_bins = np.minimum((pp*10).astype(int), 9)
    ece = float(sum(np.sum(probability_bins == i)*abs(yy[probability_bins == i].mean()-pp[probability_bins == i].mean())
                    for i in range(10) if np.any(probability_bins == i))/len(yy))
    edge_bins, cells = [], []
    supported_rows = [r for r, yes in zip(rows, support) if yes]
    for edge in EDGES:
        ix = np.array([r["edge"] == edge for r in supported_rows])
        if ix.any():
            edge_bins.append(dict(edge=edge, n=int(ix.sum()), observed=float(yy[ix].mean()),
                                  predicted=float(pp[ix].mean()), error=float(abs(yy[ix].mean()-pp[ix].mean()))))
    for edge in EDGES:
        for flag in (0, 1):
            ix = np.array([r["edge"] == edge and int(public_features(r)[7]) == flag for r in supported_rows])
            if ix.any():
                cells.append(dict(edge=edge, float_boundary_ok=flag, n=int(ix.sum()),
                                  observed=float(yy[ix].mean()), predicted=float(pp[ix].mean()),
                                  error=float(abs(yy[ix].mean()-pp[ix].mean()))))
    max_edge = max((r["error"] for r in edge_bins if r["n"] >= gates["supported_validation_bin_n"]), default=0.)
    max_cell = max((r["error"] for r in cells if r["n"] >= gates["supported_validation_bin_n"]), default=0.)
    mean_error = float(abs(yy.mean()-pp.mean()))
    gate = (coverage == gates["coverage"] and mean_error <= gates["absolute_mean_error"] and
            ece <= gates["ece"] and max_edge <= gates["edge_error"] and max_cell <= gates["public_cell_error"])
    return dict(n=len(rows), coverage=coverage, brier=brier, ece=ece, absolute_mean_error=mean_error,
                max_supported_edge_error=max_edge, max_supported_public_cell_error=max_cell,
                observed=float(yy.mean()), predicted=float(pp.mean()), gate_passed=gate,
                edge_bins=edge_bins, public_float_cells=cells)


def evaluate(models, rows, gates):
    levels = sorted({r["theta"] for r in rows})
    result = {}
    for method in METHODS:
        item = {}
        for theta in levels:
            subset = [r for r in rows if r["theta"] == theta]
            probability, supported = predict(models, subset, method)
            item[str(theta)] = measure(subset, probability, supported, gates)
        result[method] = dict(levels=item, all_levels_pass=all(r["gate_passed"] for r in item.values()),
                              mean_brier=float(np.mean([r["brier"] for r in item.values()]))
                              if all(r["brier"] is not None for r in item.values()) else None)
    return result


def session_bootstrap_loss(rows, first, second, iterations=2000):
    keys = sorted({r["session_seed"] for r in rows})
    losses = (first-np.array([r["won"] for r in rows]))**2-(second-np.array([r["won"] for r in rows]))**2
    grouped = {key: [] for key in keys}
    for r, loss in zip(rows, losses):
        grouped[r["session_seed"]].append(float(loss))
    means = np.array([np.mean(grouped[key]) for key in keys])
    rng = np.random.default_rng(seed("bootstrap", 0))
    ix = rng.integers(len(keys), size=(iterations, len(keys)))
    return dict(mean=float(means.mean()), session_bootstrap_95=np.quantile(means[ix].mean(1), [.025, .975]).tolist(),
                independent_sessions=len(keys), conditional_on_frozen_training_fit=True,
                empirical_not_uniform_certificate=True)


def markout_evaluate(model, rows):
    filled = [r for r in rows if r["won"]]
    y = np.array([r["signed_markout"] for r in filled])
    p = markout_predict(model, filled)
    return dict(fills=len(filled), zero_mse=float(np.mean(y*y)), fitted_mse=float(np.mean((y-p)**2)),
                improvement=float(np.mean(y*y-(y-p)**2)),
                gate_passed=bool(np.mean((y-p)**2) < np.mean(y*y)),
                deliberate_momentum_generator=True, transfer_to_other_price_laws_established=False)


def numerical_diagnosis(rows):
    differences = []
    for row in rows:
        p0, p1 = [sdk_fill_probability(row["edge"], z, row["matching_reference"], row["side"]) for z in (0, 1)]
        differences.append((row["edge"], int(public_features(row)[7]), (1-row["theta"])*p0+row["theta"]*p1))
    groups = {}
    for edge, flag, probability in differences:
        groups.setdefault((edge, flag), []).append(probability)
    return [dict(edge=e, float_boundary_ok=f, observations=len(v), oracle_mean=float(np.mean(v)))
            for (e, f), v in sorted(groups.items())]


def execute(out):
    config = json.loads((out/"config.json").read_text())
    if config["source_hashes"] != sources():
        raise RuntimeError("Frozen protocol/source changed before execution")
    if (out/"train_final_rfqs.jsonl.gz").exists():
        raise FileExistsError("Never overwrite executed response evidence")
    started = time.perf_counter()
    train = collect("train", config["splits"]["train"], out, config["intervals"])
    fit_started = time.perf_counter()
    model = fit_models(train, minimum_cell_n=config["minimum_training_cell_n"])
    markout = markout_fit(train)
    fit_seconds = time.perf_counter()-fit_started
    write(out/"fitted_models.json", model)
    write(out/"markout_model.json", markout)
    development = collect("development", config["splits"]["development"], out, config["intervals"])
    dev = evaluate(model, development, config["gate"])
    eligible = [name for name in METHODS if dev[name]["all_levels_pass"]]
    selected = min(eligible or METHODS, key=lambda name: (dev[name]["mean_brier"]
                   if dev[name]["mean_brier"] is not None else float("inf"), METHODS.index(name)))
    write(out/"development_analysis.json", dev)
    seal = dict(selected_model=selected, development_gate_passed=bool(eligible),
                frozen_model_sha256=sha(out/"fitted_models.json"), source_hashes=sources(),
                sealed_before_final_validation_utc=datetime.now(timezone.utc).isoformat())
    write(out/"MODEL_SEAL.json", seal)
    validation = collect("validation", config["splits"]["validation"], out, config["intervals"])
    result = evaluate(model, validation, config["gate"])
    shifted = collect("shifted", config["splits"]["shifted"], out, config["intervals"])
    outside = evaluate(model, shifted, config["gate"])
    selected_p, supported = predict(model, validation, selected)
    constant_p, _ = predict(model, validation, "constant")
    inference = session_bootstrap_loss(validation, selected_p, constant_p) if supported.all() else None
    write(out/"validation_analysis.json", result)
    write(out/"shifted_domain_analysis.json", outside)
    write(out/"selected_brier_inference.json", inference)
    write(out/"numerical_context_diagnosis.json", numerical_diagnosis(validation))
    marks = {"validation": markout_evaluate(markout, validation), "shifted": markout_evaluate(markout, shifted)}
    write(out/"markout_analysis.json", marks)
    # This lab subexperiment exposes its assigned primitive family to every
    # estimator. It does not claim such latent labels are observable in markets.
    baseline, paired, totals = [], [], {"plus": [], "minus": []}
    with gzip.open(out/"paired_response_final_rfqs.jsonl.gz", "wt") as handle:
        for i in range(config["response_sessions"]):
            tape = make_tape(seed("paired_response", i), config["intervals"])
            for theta in config["response_levels"]:
                rows, total = run_tape(tape, theta)
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True)+"\n")
                if theta == .5:
                    baseline.extend(rows)
                else:
                    paired.extend(rows)
                    totals["plus" if theta > .5 else "minus"].append(total)
    write(out/"paired_response_sessions.json", totals)
    response = response_analysis(baseline, totals["plus"], totals["minus"],
        [r for r in paired if r["theta"] > .5], [r for r in paired if r["theta"] < .5])
    response["assigned_primitive_family_is_lab_metadata_not_passive_market_feature"] = True
    response["equal_information_is_within_lab_experiment_only"] = True
    write(out/"response_analysis.json", response)
    export = bool(eligible and result[selected]["all_levels_pass"] and inference and inference["session_bootstrap_95"][1] < 0)
    if export:
        write(out/"validated_research_artifact.json", dict(model=model, selected_model=selected,
              markout=markout if marks["validation"]["gate_passed"] else None, domain=DOMAIN,
              empirical_validation_only=True, runtime_enabled=False, price_forecast_transfer_established=False,
              out_of_domain_action="abstain and use conventional baseline",
              source_model_sha256=sha(out/"fitted_models.json")))
    if sources() != config["source_hashes"]:
        write(out/"SOURCE_CHANGE_FAILURE.json", dict(expected=config["source_hashes"], actual=sources()))
        raise RuntimeError("Frozen sources changed during execution; no inference permitted")
    estimated = response["estimators"]
    summary = dict(stage=config["stage"], selected_model=selected, development_gate_passed=bool(eligible),
        final_fill_gate_passed=result[selected]["all_levels_pass"], research_export_written=export,
        training_rfqs=len(train), development_rfqs=len(development), validation_rfqs=len(validation),
        shifted_rfqs=len(shifted), response_rfqs=len(baseline)+len(paired),
        engine_sessions=sum(len({r["session_seed"] for r in rows}) for rows in (train, development, validation, shifted))
                        +3*config["response_sessions"],
        independent_primitive_sessions=sum(len({r["session_seed"] for r in rows}) for rows in (train, development, validation, shifted))
                        +config["response_sessions"],
        final_selected_mean_brier=result[selected]["mean_brier"], final_constant_mean_brier=result["constant"]["mean_brier"],
        selected_brier_difference=inference,
        tangent_direct_refit_mean_difference=estimated["estimated_finite_tangent"]["mean"]-estimated["direct_conditional_refit"]["mean"],
        model_fit_seconds=fit_seconds, elapsed_seconds=time.perf_counter()-started,
        source_unchanged=True, official_holdout_opened=False, runtime_R3_enabled=False,
        permitted_claim="EMPIRICALLY_SUPPORTED_IN_TESTED_ENVIRONMENT for passed in-domain fill calibration; no superiority or market-transport claim")
    write(out/"summary.json", summary)
    print(json.dumps(summary, indent=2))


def manifest(out):
    write(out/"SHA256SUMS.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
          if p.is_file() and p.name != "SHA256SUMS.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("declare", "execute"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "declare":
        declare(args.out)
        print(f"Declared {args.out} before execution")
    else:
        execute(args.out)
    manifest(args.out)


if __name__ == "__main__":
    main()
