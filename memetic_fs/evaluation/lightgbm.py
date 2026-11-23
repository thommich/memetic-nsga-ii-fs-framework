"""Fixed-round LightGBM evaluation, with explicit sampling restrictions."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any
import hashlib
import time
import warnings

import lightgbm as lgb
from lightgbm.basic import _ConfigAliases
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from ..config import EvaluationConfig, ResourceConfig, SegmentConstraint
from ..data.storage import load_dataset
from ..execution.resources import ResourceManager
from ..execution.scheduler import ResourceScheduler
from ..persistence.cache import EvaluationCache
from ..persistence.checkpoints import stable_hash
from ..progress import NULL_PROGRESS, progress_from_descriptor, progress_activity

EVALUATOR_VERSION = "1.0.1"

def selected_matrix(X: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Copy only selected rows/columns into a LightGBM-compatible numeric dtype.

    Integer input is retained on disk; LightGBM requires float32/float64 matrices.
    Integers outside the exact float64 range are rejected rather than rounded.
    """
    values = X[np.ix_(rows, columns)]
    if values.dtype.kind in "iu":
        if np.any(values > 2**53) or (values.dtype.kind == "i" and np.any(values < -(2**53))):
            raise ValueError("Integer predictor values exceed the exact float64 range (2**53); explicitly encode or rescale these features without information loss before fitting")
        return values.astype(np.float64)
    if values.dtype.kind == "b" or values.dtype == np.float16:
        return values.astype(np.float32)
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        converted = values.astype(np.float64)
        if not np.array_equal(values, converted.astype(values.dtype), equal_nan=True):
            raise ValueError("Predictor dtype loses information when converted to LightGBM float64")
        return converted
    return values

def strict_params(params: dict | None, *, seed: int, threads: int, n_rows: int) -> dict:
    """Reject sampling, early stopping and overrides of managed training settings.

    All LightGBM aliases are checked, including aliases hidden by another key.
    Histogram construction uses every training row, rather than the 200k default.
    """
    supplied = dict(params or {})
    required = {"bagging_fraction": 1.0, "bagging_freq": 0,
                "pos_bagging_fraction": 1.0, "neg_bagging_fraction": 1.0,
                "feature_fraction": 1.0, "feature_fraction_bynode": 1.0,
                "data_sample_strategy": "bagging", "boosting": "gbdt",
                "objective": "binary", "device_type": "cpu",
                "deterministic": True, "force_col_wise": True, "force_row_wise": False,
                "tree_learner": "serial", "pred_early_stop": False}
    managed = {"num_iterations", "early_stopping_round", "bin_construct_sample_cnt",
               "num_threads", "seed", "data_random_seed", "feature_fraction_seed",
               "bagging_seed", "drop_seed", "extra_seed", "config"}
    for canonical, expected in required.items():
        for alias in _ConfigAliases.get(canonical):
            if alias in supplied and supplied[alias] != expected:
                raise ValueError(f"{alias}={supplied[alias]!r} conflicts with strict full-row setting {canonical}={expected!r}")
            supplied.pop(alias, None)
    for canonical in managed:
        bad = set(_ConfigAliases.get(canonical)) & supplied.keys()
        if bad:
            raise ValueError(f"Managed LightGBM parameter {sorted(bad)}: use the dedicated framework setting")
    unsupported = {"monotone_constraints", "feature_contri", "max_bin_by_feature",
                   "categorical_feature", "forcedsplits_filename", "forcedbins_filename",
                   "interaction_constraints", "cegb_penalty_feature_coupled", "cegb_penalty_feature_lazy",
                   "init_score", "weight_column", "label_column", "group_column", "ignore_column",
                   "data", "valid", "input_model", "num_machines", "machines", "machine_list_filename",
                   "parser_config_file", "predict_contrib", "predict_leaf_index"}
    for canonical in unsupported:
        bad = set(_ConfigAliases.get(canonical)) & supplied.keys()
        if bad:
            raise ValueError(f"{sorted(bad)} is not supported across changing feature subsets and framework-managed data")
    supplied.update(required)
    supplied.update(seed=seed, data_random_seed=seed, feature_fraction_seed=seed,
                    bagging_seed=seed, num_threads=threads,
                    bin_construct_sample_cnt=n_rows, verbosity=-1)
    return supplied

def score(y: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute binary AUC/Gini using every supplied observation."""
    if len(y) != len(predictions) or len(np.unique(y)) != 2:
        raise ValueError("AUC requires aligned predictions and both target classes")
    auc = float(roc_auc_score(y, predictions))
    return {"auc": auc, "gini": 2 * auc - 1, "n": len(y),
            "events": int(np.sum(y)), "nonevents": int(len(y) - np.sum(y))}

@lru_cache(maxsize=4)
def _load(directory: str, fingerprint: str):
    return load_dataset(directory)

def _segment_mask(values, value):
    if value is None or pd.isna(value):
        return np.asarray(pd.isna(values))
    return pd.Series(values).eq(value).fillna(False).to_numpy(dtype=bool)


def _worker(task: dict) -> dict:
    """Record failed attempts; successful tasks commit before any batch completes."""
    progress = progress_from_descriptor(task.get("progress"))
    try:
        return _evaluate_task(task, progress=progress)
    except Exception as error:
        EvaluationCache(task["cache_path"]).failed(task["key"], f"{type(error).__name__}: {error}")
        progress.error("Candidate evaluation failed", candidate=task["key"][:12],
                       fidelity=task["fidelity"], error_type=type(error).__name__)
        if isinstance(error, lgb.basic.LightGBMError) and any(
                marker in str(error).lower() for marker in ("bad_alloc", "bad allocation", "cannot allocate memory", "out of memory")):
            raise MemoryError(f"LightGBM full-fold allocation failed: {error}") from error
        raise
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()

def _evaluate_task(task: dict, progress=None) -> dict:
    """Load shared arrays by path; commit each successful task before returning."""
    progress = progress or NULL_PROGRESS
    started = time.perf_counter()
    cache = EvaluationCache(task["cache_path"])
    existing = cache.get(task["key"])
    if existing is not None:
        progress.debug("Worker evaluation cache hit", candidate=task["key"][:12], fidelity=task["fidelity"])
        return existing
    from ..execution.workers import MemoryMonitor
    data = _load(task["data_path"], task["dataset_fingerprint"])
    mask = np.asarray(task["mask"], dtype=bool)
    cols = np.flatnonzero(mask)
    progress.debug("Candidate evaluation started", candidate=task["key"][:12], fidelity=task["fidelity"],
                   n_features=len(cols), folds=len(task["folds"]), num_boost_round=task["num_boost_round"],
                   threads=task["threads"], reference=task.get("reference", False))
    cats = [i for i, c in enumerate(cols) if c in data.categorical_indices]
    fold_results, segment_results, ys, ps = [], [], [], []
    importance = np.zeros(data.X.shape[1], dtype=float)
    segment_values = {column: pd.unique(pd.Series(values, dtype=object).where(pd.notna(values), None))
                      for column, values in data.segments.items()}
    with MemoryMonitor() as monitor, threadpool_limits(limits=1):
        for fold in task["folds"]:
            fold_started = time.perf_counter()
            train = np.load(fold["train_path"], mmap_mode="r")
            valid = np.load(fold["valid_path"], mmap_mode="r")
            fold_log = progress.info if task.get("reference", False) else progress.debug
            fold_log("Temporal fold started", candidate=task["key"][:12], fold=fold["label"],
                     fidelity=task["fidelity"], train_n=len(train), validation_n=len(valid),
                     n_features=len(cols), num_boost_round=task["num_boost_round"])
            params = strict_params(task["model_params"], seed=task["seed"],
                                   threads=task["threads"], n_rows=len(train))
            # np.ix_ makes only the selected fold/columns copy, not all predictors.
            with progress_activity(progress, "LightGBM fold fit still running", phase="model",
                                   candidate=task["key"][:12], fold=fold["label"],
                                   n_features=len(cols), train_n=len(train)):
                model = lgb.train(params, lgb.Dataset(selected_matrix(data.X, train, cols),
                                  label=data.y[train], categorical_feature=cats),
                                  num_boost_round=task["num_boost_round"])
            predictions = model.predict(selected_matrix(data.X, valid, cols), num_threads=task["threads"])
            yy = data.y[valid]
            metrics = score(yy, predictions)
            metrics.update(fold=fold["label"], train_n=len(train), validation_n=len(valid))
            fold_results.append(metrics)
            ys.append(yy)
            ps.append(predictions)
            importance[cols] += model.feature_importance(importance_type="gain")
            for column, values in data.segments.items():
                # Different missing sentinels represent one descriptive segment.
                for raw_value in segment_values[column]:
                    value = None if pd.isna(raw_value) else raw_value
                    selected = _segment_mask(values[valid], value)
                    sy = yy[selected]
                    counts = {"n": len(sy), "events": int(np.sum(sy)), "nonevents": int(len(sy) - np.sum(sy))}
                    sm = score(sy, predictions[selected]) if len(np.unique(sy)) == 2 else {"auc": None, "gini": None, **counts}
                    segment_results.append({"column": column, "value": value,
                                            "fold": fold["label"], **sm})
            fold_log("Temporal fold completed", candidate=task["key"][:12], fold=fold["label"],
                     fidelity=task["fidelity"], train_n=len(train), validation_n=len(valid),
                     gini=metrics["gini"], elapsed_seconds=time.perf_counter() - fold_started)
    ginis = [f["gini"] for f in fold_results]
    constraints = []
    for constraint in task["active_constraints"]:
        gaps = []
        for sm in segment_results:
            if sm["column"] == constraint["column"] and sm["value"] == constraint["value"]:
                ref = next(r for r in task["reference_segments"] if r["column"] == sm["column"] and r["value"] == sm["value"] and r["fold"] == sm["fold"])
                if sm["gini"] is None or ref["gini"] is None:
                    raise ValueError("Eligible constrained segment has undefined Gini")
                gaps.append(ref["gini"] - sm["gini"] - constraint["max_gini_drop"])
        if not gaps:
            raise ValueError("Constrained segment is missing from evaluation")
        constraints.append(float(max(gaps)))
    result = {"key": task["key"], "mask": mask.tolist(), "n_features": int(mask.sum()),
              "fidelity": task["fidelity"], "mean_gini": float(np.mean(ginis)),
              "std_gini": float(np.std(ginis)), "min_gini": float(min(ginis)),
              "pooled_gini": score(np.concatenate(ys), np.concatenate(ps))["gini"],
              "folds": fold_results, "segments": segment_results, "constraints": constraints,
              "importance": importance.tolist(), "memory_peak_bytes": monitor.peak_private_bytes}
    cache.put(task["key"], result)
    progress.debug("Candidate evaluation completed", candidate=task["key"][:12], fidelity=task["fidelity"],
                   n_features=result["n_features"], mean_gini=result["mean_gini"],
                   elapsed_seconds=time.perf_counter() - started)
    return result

class Evaluator:
    """Cached evaluator with canonical reference, fidelity and resource admission."""
    def __init__(self, data_path: str | Path, folds: list, directory: str | Path,
                 model_params: dict | None = None, num_boost_round: int = 100,
                 config: EvaluationConfig | None = None, resources: ResourceConfig | None = None,
                 segment_constraints: list[SegmentConstraint] | None = None, cancelled=None,
                 expected_identity: str | None = None, progress=None):
        self.progress = progress or NULL_PROGRESS
        self.directory = Path(directory).resolve()
        self.data_path = str(Path(data_path).resolve())
        self.data = load_dataset(self.data_path)
        self.config = config or EvaluationConfig()
        self.num_boost_round = num_boost_round
        if not isinstance(num_boost_round, int) or isinstance(num_boost_round, bool) or num_boost_round < 1:
            raise ValueError("num_boost_round must be a positive integer")
        self.model_params = model_params
        strict_params(model_params, seed=self.config.estimator_seed, threads=1, n_rows=len(self.data.y))
        self.cache_path = str(self.directory / "evaluations.sqlite")
        self.cache = EvaluationCache(self.cache_path)
        self.all_results = {}
        self.fold_specs = []
        fold_dir = self.directory / "folds"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for i, fold in enumerate(folds):
            paths = {"label": str(fold.label)}
            for kind in ("train", "valid"):
                path = fold_dir / f"{i}_{kind}.npy"
                values = np.asarray(getattr(fold, kind))
                if path.exists():
                    if not np.array_equal(np.load(path, mmap_mode="r"), values):
                        raise ValueError(f"Persisted fold differs from the requested definition: {path.name}")
                else:
                    np.save(path, values)
                paths[f"{kind}_path"] = str(path)
            self.fold_specs.append(paths)
        self.fold_identity = [{kind: hashlib.sha256(np.asarray(getattr(f, kind), dtype="<i8").tobytes()).hexdigest()
                               for kind in ("train", "valid")} | {"label": str(f.label)} for f in folds]
        # Hash once; per-candidate keys use this compact digest.
        self.active_constraints = []
        self.excluded_constraints = []
        for constraint in segment_constraints or []:
            if constraint.column not in self.data.segments:
                raise ValueError(f"Constraint column {constraint.column!r} must be in segment_cols")
            if not pd.api.types.is_scalar(constraint.value):
                raise ValueError("Segment constraint value must be a scalar category, or None for missing values")
            normalized_constraint = asdict(constraint)
            if pd.isna(constraint.value):
                normalized_constraint["value"] = None
            selected = _segment_mask(self.data.segments[constraint.column], normalized_constraint["value"])
            failures = []
            for fold in folds:
                yy = self.data.y[fold.valid][selected[fold.valid]]
                ev = int(yy.sum())
                if ev < self.config.min_segment_events or len(yy) - ev < self.config.min_segment_nonevents:
                    failures.append(str(fold.label))
            if failures:
                info = {**normalized_constraint, "failed_folds": failures,
                        "min_events": self.config.min_segment_events, "min_nonevents": self.config.min_segment_nonevents}
                self.excluded_constraints.append(info)
                self.progress.warning("Segment constraint disabled for insufficient validation support",
                                      failed_folds=len(failures), min_events=self.config.min_segment_events,
                                      min_nonevents=self.config.min_segment_nonevents)
                warnings.warn(f"Disabling segment constraint {constraint.column}={constraint.value!r}: requires >= {self.config.min_segment_events} events and >= {self.config.min_segment_nonevents} non-events in folds {failures}", UserWarning, stacklevel=2)
            else:
                self.active_constraints.append(normalized_constraint)
        # Resolve estimator threads once. Admission may vary worker count, but
        # reference and candidate training configurations remain identical.
        estimate = int(self.data.X.nbytes * 5 + 128 * 1024**2)
        self.manager = ResourceManager(resources or ResourceConfig(), estimate, temp_dir=self.directory)
        self.training_threads = self.manager.plan().threads_per_worker
        self.manager.bound_threads = self.training_threads
        self.versions = {name: version(name) for name in ("lightgbm", "numpy", "pandas", "scikit-learn", "pymoo", "joblib", "pyarrow")}
        self.identity = stable_hash({"dataset": self.data.fingerprint, "folds": self.fold_identity,
                                    "params": model_params, "rounds": num_boost_round, "evaluation": asdict(self.config),
                                    "constraints": self.active_constraints, "versions": self.versions,
                                    "training_threads": self.training_threads, "implementation": EVALUATOR_VERSION})
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("Checkpoint incompatible with current evaluator, threads, data or settings")
        self.progress.info("Evaluator initialized", identity=self.identity[:12], folds=len(self.fold_specs),
                           n_rows=len(self.data.y), candidate_features=self.data.X.shape[1],
                           training_threads=self.training_threads, active_constraints=len(self.active_constraints),
                           excluded_constraints=len(self.excluded_constraints))
        self.scheduler = ResourceScheduler(self.manager, cancelled=cancelled, progress=self.progress)
        calibration_started = time.perf_counter()
        self.progress.milestone("Full-feature reference calibration started", n_features=self.data.X.shape[1],
                           folds=len(self.fold_specs), num_boost_round=self.num_boost_round)
        self.reference = self.evaluate_many([np.ones(self.data.X.shape[1], dtype=bool)], "full", reference=True)[0]
        self.manager.calibrate(self.reference["memory_peak_bytes"])
        self.progress.milestone("Full-feature reference calibration completed", mean_gini=self.reference["mean_gini"],
                           measured_private_peak_bytes=self.reference["memory_peak_bytes"],
                           estimated_task_bytes=self.manager.estimated_task_bytes,
                           elapsed_seconds=time.perf_counter() - calibration_started)

    def evaluate_many(self, masks, fidelity: str = "full", reference: bool = False) -> list[dict]:
        """Evaluate unique valid masks, with final fold always included at low fidelity."""
        started = time.perf_counter()
        if fidelity == "full":
            folds = self.fold_specs
        elif fidelity.startswith("last_"):
            count = int(fidelity[5:])
            if count < 1 or count > len(self.fold_specs):
                raise ValueError("Reduced fidelity must use between one and all available folds")
            folds = self.fold_specs[-count:]
        else:
            raise ValueError(f"Unknown evaluation fidelity {fidelity!r}")
        tasks, keys, cache_hits, reference_reuses, feature_counts = {}, [], set(), set(), set()
        pending_duplicates = 0
        plan = self.manager.plan()
        for values in masks:
            mask = np.asarray(values)
            if mask.shape != (self.data.X.shape[1],) or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError("Feature mask must be nonempty binary vector matching predictor count")
            mask = mask.astype(bool)
            feature_counts.add(int(mask.sum()))
            # Reference calculations have no self-comparison constraints.
            key = stable_hash({"evaluation": self.identity, "mask": mask.tolist(), "fidelity": fidelity,
                               "reference": reference, "threads": self.training_threads})
            keys.append(key)
            if key in tasks:
                pending_duplicates += 1
                continue
            if key in reference_reuses:
                continue
            value = self.cache.get(key)
            if value is not None:
                self.all_results[key] = value
                cache_hits.add(key)
                self.progress.debug("Evaluation cache hit", candidate=key[:12], fidelity=fidelity,
                                    n_features=int(mask.sum()))
                continue
            if not reference and fidelity == "full" and mask.all():
                # This is the same set of full-row model fits as the calibrated
                # reference; only the self-comparison constraint vector differs.
                value = deepcopy(self.reference)
                value.update(key=key, constraints=[-c["max_gini_drop"] for c in self.active_constraints])
                self.cache.put(key, value)
                self.all_results[key] = value
                reference_reuses.add(key)
                self.progress.debug("Full-feature candidate reused reference", candidate=key[:12], fidelity=fidelity)
                continue
            self.cache.pending(key)
            tasks[key] = {"key": key, "mask": mask.tolist(), "data_path": self.data_path,
                          "dataset_fingerprint": self.data.fingerprint,
                          "cache_path": self.cache_path, "folds": folds, "fidelity": fidelity,
                          "threads": self.training_threads, "model_params": self.model_params,
                          "reference": reference, "progress": self.progress.descriptor,
                          "seed": self.config.estimator_seed, "num_boost_round": self.num_boost_round,
                          "active_constraints": [] if reference else self.active_constraints,
                          "reference_segments": [] if reference else self.reference["segments"]}
        # A single reference task is serial while still passing resource admission.
        # Every successful task commits its cache entry before returning.
        self.progress.info("Evaluation batch prepared", requested_candidates=len(keys), unique_candidates=len(set(keys)),
                           new_evaluations=len(tasks), cache_hits=len(cache_hits), reference_reuses=len(reference_reuses),
                           duplicate_requests=len(keys) - len(set(keys)), pending_duplicates=pending_duplicates,
                           fidelity=fidelity, feature_counts=sorted(feature_counts))
        values = self.scheduler.map(_worker, list(tasks.values()))
        for value in values:
            self.all_results[value["key"]] = value
        self.progress.info("Evaluation batch completed", requested_candidates=len(keys), evaluated_candidates=len(values),
                           cache_hits=len(cache_hits), fidelity=fidelity, elapsed_seconds=time.perf_counter() - started)
        return [self.all_results[key] for key in keys]
