"""Optional, reproducible diagnostics and LightGBM pilot recommendations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from copy import deepcopy
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from .cv.temporal import build_folds
from .data.validation import prepare_data
from .evaluation.lightgbm import score, selected_matrix, strict_params
from .execution.resources import ResourceManager
from .execution.workers import MemoryMonitor
from .optimization.initialization import normalize_subsets


@dataclass
class WarmupResult:
    settings_path: Path
    suggestions: dict
    diagnostics: dict
    pilots: dict
    reasons: dict
    changes: dict
    warnings: list[str]


def _plain(value):
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sizes(n: int, depth: str) -> list[int]:
    fractions = (.1, .25, .5, 1.0) if depth == "standard" else (.02, .05, .1, .2, .4, .6, .8, 1.0)
    return sorted({max(1, min(n, round(n * fraction))) for fraction in fractions})


def _candidates(depth: str, min_train: int, baseline: dict, rounds: int) -> list[dict]:
    leaf = max(5, min(300, round(min_train / 500), max(5, min_train // 4)))
    extras = {key: value for key, value in (baseline or {}).items()
              if key not in {"learning_rate", "num_leaves", "min_data_in_leaf"}}
    if depth == "standard":
        tuples = [(.1, 100, 31, leaf), (.05, 200, 31, leaf),
                  (.05, 400, 31, 2 * leaf), (.1, 200, 15, leaf),
                  (.05, 400, 15, leaf), (.05, 400, 63, 2 * leaf)]
    else:
        tuples = [(rate, {0.03: 600, 0.05: 400, 0.1: 200}[rate], leaves, size)
                  for rate in (.03, .05, .1) for leaves in (15, 31, 63)
                  for size in (leaf, 2 * leaf)]
    candidates = [{"params": {**extras, "learning_rate": rate, "num_leaves": leaves,
                               "min_data_in_leaf": size}, "num_boost_round": nrounds}
                  for rate, nrounds, leaves, size in tuples]
    original = {"params": dict(baseline or {}), "num_boost_round": rounds}
    if original not in candidates:
        candidates.insert(0, original)
    return candidates


def _fit(data, fold, columns, candidate, threads, seed):
    train, valid = fold.train, fold.valid
    cats = [i for i, col in enumerate(columns) if col in data.categorical_indices]
    params = strict_params(candidate["params"], seed=seed, threads=threads, n_rows=len(train))
    started = time.perf_counter()
    with MemoryMonitor() as monitor, threadpool_limits(limits=1):
        matrix = selected_matrix(data.X, train, columns)
        model = lgb.train(params, lgb.Dataset(matrix, label=data.y[train],
                          categorical_feature=cats), num_boost_round=candidate["num_boost_round"])
        predictions = model.predict(selected_matrix(data.X, valid, columns), num_threads=threads)
    result = {"fold": fold.label, "gini": score(data.y[valid], predictions)["gini"],
              "elapsed_seconds": time.perf_counter() - started,
              "peak_increment_bytes": monitor.peak_increment_bytes,
              "training_matrix_bytes": matrix.nbytes}
    return result, model.feature_importance(importance_type="gain")


def _diagnostics(data, folds, snapshot):
    months = []
    for month in np.unique(data.months):
        mask = data.months == month
        label = str(pd.Period(ordinal=int(month), freq="M"))
        months.append({"month": label, "rows": int(mask.sum()),
                       "events": int(data.y[mask].sum()),
                       "nonevents": int(mask.sum() - data.y[mask].sum())})
    missing = {}
    for i, name in enumerate(data.feature_names):
        column = data.X[:, i]
        missing[name] = int((column == -1).sum() if i in data.categorical_indices
                            else pd.isna(column).sum())
    return {"rows": data.n_rows, "predictors": data.n_features,
            "data_fingerprint": data.fingerprint,
            "events": int(data.y.sum()), "nonevents": int(len(data.y) - data.y.sum()),
            "categorical_predictors": len(data.categorical_indices),
            "missing_by_feature": missing,
            "months": months,
            "folds": [{"label": fold.label, "train_rows": len(fold.train),
                       "validation_rows": len(fold.valid),
                       "train_events": int(data.y[fold.train].sum()),
                       "validation_events": int(data.y[fold.valid].sum())} for fold in folds],
            "resources": asdict(snapshot)}


def warmup_settings_params(cls, X, y, *, depth, output_path, base_config, overwrite,
                           id_col, segment_cols, time_values, ids, feature_names, time_cv):
    if depth not in {"quick", "standard", "deep"}:
        raise ValueError("depth must be 'quick', 'standard', or 'deep'")
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Warm-up YAML already exists: {destination}; use overwrite=True to replace it")
    baseline = cls.from_config(base_config) if base_config is not None else cls()
    if time_cv is not None:
        baseline.time_cv = time_cv
    baseline._validate_options()
    if not isinstance(baseline.time_cv, str):
        raise ValueError("YAML warm-up requires a named time_cv column; callable CV cannot be saved in YAML")
    settings = dict(baseline.get_params(deep=False))
    for key in ("cv_config", "optimization", "evaluation", "resources", "selection"):
        settings[key] = asdict(getattr(baseline, key + "_"))
    if settings["segment_constraints"]:
        settings["segment_constraints"] = [asdict(c) for c in baseline.constraints_]
    original_settings = deepcopy(settings)
    data = prepare_data(X, y, baseline.time_cv, id_col, segment_cols,
                        baseline.allow_repeated_ids, time_values, ids, feature_names)
    for constraint in baseline.constraints_:
        if constraint.column not in data.segments:
            raise ValueError(f"Constraint column {constraint.column!r} must be in segment_cols")
    normalize_subsets(baseline.initial_subsets, data.feature_names, (1, data.n_features))
    folds = build_folds(data.months, data.y, baseline.cv_config_,
                        custom=baseline.time_cv if callable(baseline.time_cv) else None)
    selected_folds = [] if depth == "quick" else (folds[-3:] if depth == "standard" else folds)
    sizes = [] if depth == "quick" else _sizes(data.n_features, depth)
    candidates = [] if depth == "quick" else _candidates(depth, min(len(f.train) for f in selected_folds),
                                                           baseline.model_params, baseline.num_boost_round)
    fit_count = len(candidates) * len(selected_folds) + (len(sizes) - 1) * len(selected_folds)
    notice = (f"Warm-up depth={depth}: {fit_count} planned LightGBM fits on {data.n_rows} rows and "
              f"{data.n_features} predictors. Duration depends on data size and the selected depth; "
              "there is no time or fit-count cutoff.")
    warnings.warn(notice, UserWarning, stacklevel=3)
    messages = [notice]
    snapshot_estimate = max(64 * 1024**2, 3 * max(len(f.train) for f in folds) * data.n_features * data.X.dtype.itemsize)
    manager = ResourceManager(baseline.resources_, snapshot_estimate)
    plan = manager.plan()
    if plan.workers < 1:
        raise MemoryError("Insufficient memory headroom for warm-up pilot or later full-feature fit")
    diagnostics = _diagnostics(data, folds, plan.snapshot)
    diagnostics["planned_fits"] = fit_count
    diagnostics["pilot_folds"] = [f.label for f in selected_folds]
    messages.append("Worker and thread suggestions reflect this resource snapshot and should be reviewed on another machine.")
    pilots = {"model_candidates": [], "subset_scores": [], "subset_score_gaps": []}
    reasons = {"model_params": "Heuristic defaults without model fits" if depth == "quick" else "Chronological full-row pilot",
               "num_boost_round": "Fixed-round LightGBM setting from heuristic" if depth == "quick" else "Fixed-round setting from chronological pilot",
               "resources": "CPU and memory snapshot; caps may become stale if the environment changes",
               "selection": "Conservative 0.001 absolute Gini tolerance; review final Pareto score gaps",
               "feature_range": "Explore all valid subset sizes rather than forcing a minimum signal count"}
    settings["feature_range"] = (1, data.n_features)
    if depth == "quick":
        n_train = min(len(f.train) for f in folds)
        settings["model_params"] = {**(baseline.model_params or {}), "learning_rate": .05,
                                    "num_leaves": 31, "min_data_in_leaf": max(5, min(300, round(n_train / 500)))}
        settings["num_boost_round"] = 200
        messages.append("Quick model suggestions are heuristics and were not validated by pilot fits.")
    else:
        full_cols = np.arange(data.n_features)
        full_records = []
        for candidate in candidates:
            fold_records, importances = [], []
            for fold in selected_folds:
                record, importance = _fit(data, fold, full_cols, candidate, plan.threads_per_worker,
                                          baseline.evaluation_.estimator_seed)
                fold_records.append(record)
                importances.append(importance.tolist())
            summary = {"params": candidate["params"], "num_boost_round": candidate["num_boost_round"],
                       "folds": fold_records, "mean_gini": float(np.mean([r["gini"] for r in fold_records])),
                       "elapsed_seconds": float(sum(r["elapsed_seconds"] for r in fold_records))}
            pilots["model_candidates"].append(summary)
            full_records.append((summary, importances))
        best_mean = max(item[0]["mean_gini"] for item in full_records)
        close = [item for item in full_records if item[0]["mean_gini"] >= best_mean - .001]
        winner, rankings = min(close, key=lambda item: (
            item[0]["params"].get("num_leaves", 31), item[0]["num_boost_round"],
            item[0]["elapsed_seconds"]))
        settings["model_params"] = winner["params"]
        settings["num_boost_round"] = winner["num_boost_round"]
        pilots["selected_model"] = {"params": winner["params"],
                                    "num_boost_round": winner["num_boost_round"]}
        for size in sizes:
            records = []
            for fold, gains, full_record in zip(selected_folds, rankings, winner["folds"]):
                if size == data.n_features:
                    record = full_record
                else:
                    # Every ranking comes from this fold's training-only full model.
                    columns = np.sort(np.argsort(-np.asarray(gains), kind="stable")[:size])
                    record, _ = _fit(data, fold, columns, pilots["selected_model"], plan.threads_per_worker,
                                     baseline.evaluation_.estimator_seed)
                records.append(record)
            pilots["subset_scores"].append({"n_features": size,
                "mean_gini": float(np.mean([r["gini"] for r in records])), "folds": records})
        for smaller, larger in zip(pilots["subset_scores"], pilots["subset_scores"][1:]):
            pilots["subset_score_gaps"].append({"from": smaller["n_features"], "to": larger["n_features"],
                "mean_gini_gain": larger["mean_gini"] - smaller["mean_gini"]})
        observed_peak = max(r["peak_increment_bytes"] + r["training_matrix_bytes"]
                            for item in pilots["model_candidates"] for r in item["folds"])
        manager.calibrate(observed_peak)
        plan = manager.plan()
        if plan.workers < 1:
            raise MemoryError("Pilot memory measurement leaves insufficient headroom for a full-feature fit")
        if pilots["subset_scores"][0]["mean_gini"] >= max(r["mean_gini"] for r in pilots["subset_scores"]) - .001:
            messages.append("Pilot scores are near the smallest probed subset; review the final frontier before accepting a minimum-boundary selection.")
    resources = settings["resources"]
    resources["max_workers"] = plan.workers
    resources["threads_per_worker"] = plan.threads_per_worker
    reasons["resources"] += "; measured pilot memory included" if depth != "quick" else "; conservative matrix estimate used"
    selection = settings["selection"]
    if selection["policy"] in {"tolerance", "statistical"}:
        selection["gini_tolerance"] = min(float(selection["gini_tolerance"]), .001)
    if depth in {"standard", "deep"}:
        settings["optimization"]["rfe_seed_sizes"] = sizes
        reasons["optimization"] = "RFE checkpoints span the pilot subset sizes"
    if depth == "deep":
        optimization = settings["optimization"]
        optimization["population_size"] = min(80, max(20, 2 * data.n_features))
        optimization["max_generations"] = min(100, max(30, 4 * data.n_features))
        optimization["min_generations"] = min(10, optimization["max_generations"])
        optimization["patience"] = 12
        optimization["local_search_budget"] = min(40, max(8, data.n_features // 2))
        optimization["reduced_fold_count"] = min(2, len(folds))
        reasons["optimization"] = "Complete search configuration from predictor count, fold count, and pilot cardinalities; unchanged fields retain validated defaults"
    plain = _plain(settings)
    # A selector YAML cannot contain an executable callable or a non-YAML object.
    try:
        serialized = yaml.safe_dump(plain, sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as error:
        raise ValueError("Selector settings cannot be represented safely in YAML") from error
    cls(**plain)._validate_options()
    previous = _plain(original_settings)
    changes = {key: {"previous": previous.get(key), "proposed": value,
                     "reason": reasons.get(key, "Normalized and validated for the suggested selector YAML")}
               for key, value in plain.items() if previous.get(key) != value}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return WarmupResult(destination, plain, diagnostics, pilots, reasons, changes, messages)
