"""Public estimator-style API and structured checkpoint recovery."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import inspect
from importlib.metadata import version
import logging
import shutil
import threading
import time
from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import (CVConfig, OptimizationConfig, ResourceConfig, EvaluationConfig,
                     SelectionConfig, SegmentConstraint, from_dict)
from .data.validation import prepare_data
from .data.storage import load_dataset, save_dataset
from .cv.temporal import Fold, build_folds
from .evaluation.lightgbm import Evaluator, strict_params, selected_matrix
from .persistence.checkpoints import atomic_json, read_json, stable_hash
from .optimization.algorithm import SearchEngine
from .selection import select_candidate
from .progress import ProgressReporter, progress_activity

log = logging.getLogger(__name__)

def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

class MemeticNSGA2Selector(TransformerMixin, BaseEstimator):
    """Select development predictors using temporal, full-row memetic NSGA-II.

    ``time_cv`` is a DataFrame time column or a callable receiving normalized
    monthly ordinals and returning integer ``(train, valid)`` positions. For a
    callable, ``fit(time_values=...)`` is mandatory. No external OOT input exists.
    ``feature_range`` is inclusive; the upper bound is clipped to predictor count.
    """
    def __init__(self, model_params: dict | None = None, time_cv: str | Callable = "reference_month",
                 feature_range: tuple[int, int] = (10, 200), objectives: tuple[str, str] = ("gini", "n_features"),
                 segment_constraints: list[SegmentConstraint | dict] | None = None,
                 use_rfe_initialization: bool = True, resources: ResourceConfig | dict | str = "auto",
                 random_state: int = 42, checkpoint_dir: str | Path = "./runs/run_001",
                 cv_config: CVConfig | dict | None = None, optimization: OptimizationConfig | dict | None = None,
                 evaluation: EvaluationConfig | dict | None = None, selection: SelectionConfig | dict | None = None,
                 num_boost_round: int = 100, allow_repeated_ids: bool = False,
                 initial_subsets: list | None = None, feature_groups: dict | None = None,
                 interactive_plots: bool = True, verbose: int = 1,
                 output_format: str = "parquet") -> None:
        self.model_params = model_params
        self.time_cv = time_cv
        self.feature_range = feature_range
        self.objectives = objectives
        self.segment_constraints = segment_constraints
        self.use_rfe_initialization = use_rfe_initialization
        self.resources = resources
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.cv_config = cv_config
        self.optimization = optimization
        self.evaluation = evaluation
        self.selection = selection
        self.num_boost_round = num_boost_round
        self.allow_repeated_ids = allow_repeated_ids
        self.initial_subsets = initial_subsets
        self.feature_groups = feature_groups
        self.interactive_plots = interactive_plots
        self.verbose = verbose
        self.output_format = output_format

    def _settings(self):
        self._validate_options()
        self._cancel = threading.Event()
        self.directory_ = Path(self.checkpoint_dir).resolve()
        self.directory_.mkdir(parents=True, exist_ok=True)
        self.progress_ = ProgressReporter(self.directory_, verbose=self.verbose)
        self.log_path_ = self.progress_.path

    def _validate_options(self):
        for name in ("use_rfe_initialization", "allow_repeated_ids", "interactive_plots"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.output_format, str) or self.output_format not in {"csv", "json", "parquet"}:
            raise ValueError("output_format must be 'csv', 'json', or 'parquet'")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if tuple(self.objectives) != ("gini", "n_features"):
            raise ValueError("Supported objectives are ('gini', 'n_features')")
        if isinstance(self.resources, str) and self.resources != "auto":
            raise ValueError("resources must be 'auto', ResourceConfig, or a mapping")
        self.cv_config_ = from_dict(CVConfig, self.cv_config)
        self.optimization_ = from_dict(OptimizationConfig, self.optimization)
        self.resources_ = from_dict(ResourceConfig, None if self.resources == "auto" else self.resources)
        self.evaluation_ = from_dict(EvaluationConfig, self.evaluation)
        self.selection_ = from_dict(SelectionConfig, self.selection)
        self.constraints_ = [from_dict(SegmentConstraint, c) for c in self.segment_constraints or []]
        if not isinstance(self.num_boost_round, int) or isinstance(self.num_boost_round, bool) or self.num_boost_round < 1:
            raise ValueError("num_boost_round must be a positive integer")
        if len(self.feature_range) != 2 or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in self.feature_range):
            raise ValueError("feature_range must contain two integer bounds")
        if self.feature_range[0] < 1 or self.feature_range[1] < self.feature_range[0]:
            raise ValueError("feature_range must have positive ascending bounds")
        if not isinstance(self.time_cv, str) and not callable(self.time_cv):
            raise ValueError("time_cv must be a column name or callable")
        if not isinstance(self.verbose, int) or isinstance(self.verbose, bool) or self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2")
        strict_params(self.model_params, seed=self.evaluation_.estimator_seed, threads=1, n_rows=1)

    @classmethod
    def from_config(cls, path: str | Path, **overrides) -> MemeticNSGA2Selector:
        """Load selector or experiment YAML, then apply explicit keyword overrides."""
        from .config_io import load_selector_settings
        settings = load_selector_settings(path)
        allowed = set(inspect.signature(cls).parameters)
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unknown selector overrides: {sorted(unknown)}")
        for key, value in overrides.items():
            if key in {"model_params", "cv_config", "optimization", "evaluation", "resources", "selection"} and isinstance(value, dict):
                previous = settings.get(key)
                if hasattr(previous, "__dataclass_fields__"):
                    previous = asdict(previous)
                settings[key] = {**(previous if isinstance(previous, dict) else {}), **value}
            else:
                settings[key] = value
        selector = cls(**settings)
        selector._validate_options()
        return selector

    @classmethod
    def warmup_settings_params(cls, X, y, *, depth: str = "standard", output_path: str | Path,
                               base_config: str | Path | None = None, overwrite: bool = False,
                               id_col: str | None = None, segment_cols: Sequence[str] | None = None,
                               time_values=None, ids=None, feature_names: Sequence[str] | None = None,
                               time_cv: str | Callable | None = None):
        """Inspect development data and save separate suggested selector YAML."""
        from .warmup import warmup_settings_params
        return warmup_settings_params(cls, X, y, depth=depth, output_path=output_path,
                                      base_config=base_config, overwrite=overwrite,
                                      id_col=id_col, segment_cols=segment_cols,
                                      time_values=time_values, ids=ids, feature_names=feature_names,
                                      time_cv=time_cv)

    def fit(self, X: pd.DataFrame | np.ndarray, y: str | Sequence[int] | np.ndarray, *,
            id_col: str | None = None, segment_cols: Sequence[str] | None = None,
            time_values=None, ids=None, feature_names: Sequence[str] | None = None) -> MemeticNSGA2Selector:
        """Validate development data, optimize subsets, and export an experiment."""
        self._settings()
        shape = getattr(X, "shape", ())
        self.progress_.info("Validating development data", phase="data", rows=shape[0] if len(shape) else None,
                            columns=shape[1] if len(shape) > 1 else None)
        if (self.directory_ / "manifest.json").exists():
            raise ValueError("Experiment directory already contains a run; use MemeticNSGA2Selector.resume(directory) or a new directory")
        data = prepare_data(X, y, self.time_cv, id_col, segment_cols, self.allow_repeated_ids,
                            time_values, ids, feature_names)
        self._set_features(data)
        self.progress_.info("Development data validated", phase="data", rows=data.n_rows,
                            predictors=data.n_features, categorical_features=len(data.categorical_indices),
                            observed_months=len(np.unique(data.months)))
        self.folds_ = build_folds(data.months, data.y, self.cv_config_, custom=self.time_cv if callable(self.time_cv) else None)
        if self.optimization_.reduced_generations and self.optimization_.reduced_fold_count > len(self.folds_):
            raise ValueError("reduced_fold_count exceeds the available canonical fold count")
        from .optimization.initialization import normalize_subsets
        normalize_subsets(self.initial_subsets, data.feature_names, self.feature_range_)
        self.progress_.info("Temporal folds validated", phase="cv", folds=len(self.folds_),
                            training_rows=[len(f.train) for f in self.folds_],
                            validation_rows=[len(f.valid) for f in self.folds_])
        required_bytes = data.X.nbytes + data.y.nbytes + data.months.nbytes
        required_bytes += sum(f.train.nbytes + f.valid.nbytes for f in self.folds_)
        if shutil.disk_usage(self.directory_).free < required_bytes * 1.1 + 64 * 1024**2:
            raise OSError("Insufficient disk space for full shared data, materialized folds and checkpoint reserve")
        save_dataset(data, self.directory_ / "data")
        self.progress_.info("Read-only experiment data persisted", phase="storage", data_bytes=required_bytes)
        data_fingerprint = data.fingerprint
        # Workers reopen the persisted arrays; release the one-time conversion
        # before calibration instead of retaining a second complete matrix.
        del data
        self._build_evaluator()
        config = self._config_record()
        artifacts = [p for p in (self.directory_ / "data").iterdir() if p.is_file()]
        artifacts.extend((self.directory_ / "folds").glob("*.npy"))
        self.manifest_ = {"schema": 1, "config": config, "evaluation_identity": self.evaluator_.identity,
                          "data_fingerprint": data_fingerprint, "versions": self.evaluator_.versions,
                          "training_settings": strict_params(self.model_params, seed=self.evaluation_.estimator_seed,
                              threads=self.evaluator_.training_threads, n_rows=max(len(f.train) for f in self.folds_)),
                          "bin_construction": "bin_construct_sample_cnt equals each complete training fold length",
                          "folds": [{"label": f.label, "train": f"folds/{i}_train.npy", "valid": f"folds/{i}_valid.npy"} for i, f in enumerate(self.folds_)],
                          "files": {str(p.relative_to(self.directory_)): _file_digest(p) for p in artifacts},
                          "excluded_constraints": self.evaluator_.excluded_constraints}
        atomic_json(self.directory_ / "manifest.json", self.manifest_)
        atomic_json(self.directory_ / "config.json", config)
        return self._run()

    def _set_features(self, data):
        self.feature_names_in_ = np.asarray(data.feature_names, dtype=object)
        self.n_features_in_ = data.n_features
        if len(self.feature_range) != 2 or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in self.feature_range):
            raise ValueError("feature_range must contain two integer bounds")
        low, high = self.feature_range
        high = min(high, data.n_features)
        if low < 1 or high < low:
            raise ValueError(f"Invalid feature_range {self.feature_range} for {data.n_features} predictors; lower bound must be >=1 and <= upper")
        self.feature_range_ = (low, high)

    def _build_evaluator(self, expected_identity=None):
        self.evaluator_ = Evaluator(self.directory_ / "data", self.folds_, self.directory_, self.model_params,
                                    self.num_boost_round, self.evaluation_, self.resources_, self.constraints_,
                                    cancelled=self._cancel.is_set, expected_identity=expected_identity,
                                    progress=self.progress_)

    def _config_record(self):
        return {"model_params": self.model_params, "time_cv": self.time_cv if isinstance(self.time_cv, str) else "__persisted_custom_folds__",
                "feature_range": self.feature_range_, "objectives": self.objectives,
                "segment_constraints": [asdict(c) for c in self.constraints_],
                "use_rfe_initialization": self.use_rfe_initialization, "resources": asdict(self.resources_),
                "random_state": self.random_state, "cv_config": asdict(self.cv_config_),
                "optimization": asdict(self.optimization_), "evaluation": asdict(self.evaluation_),
                "selection": asdict(self.selection_), "num_boost_round": self.num_boost_round,
                "allow_repeated_ids": self.allow_repeated_ids, "initial_subsets": self.initial_subsets,
                "feature_groups": self.feature_groups, "interactive_plots": self.interactive_plots,
                "output_format": self.output_format,
                "verbose": self.verbose}

    def _run(self, state=None):
        started = time.monotonic()
        self.progress_.info("Starting feature subset search" if state is None else "Resuming feature subset search",
                            phase="optimization", generation=state["generation"] if state else 0,
                            population=self.optimization_.population_size,
                            max_generations=self.optimization_.max_generations,
                            feature_range=self.feature_range_)
        self.engine_ = SearchEngine(self.evaluator_, self.n_features_in_, self.feature_range_, self.optimization_,
                                   random_state=self.random_state, use_rfe_initialization=self.use_rfe_initialization,
                                   initial_subsets=self.initial_subsets, feature_names=list(self.feature_names_in_),
                                   feature_groups=self.feature_groups)
        try:
            result = self.engine_.run(checkpoint_callback=self._checkpoint, state=state, cancel=self._cancel.is_set)
        except (KeyboardInterrupt, InterruptedError):
            self.progress_.warning("Search interrupted; completed evaluations and the last checkpoint are reusable", phase="optimization")
            log.warning("Search interrupted; completed evaluations are durable. Resume the experiment to continue.")
            raise
        except Exception as error:
            self.progress_.error("Feature subset search failed; completed evaluations remain cached", phase="optimization", error_type=type(error).__name__, reason=str(error))
            raise
        self._state = result["state"]
        self.confirmed_ = result["confirmed"]
        self.provisional_ = result["provisional"]
        self.history_ = pd.DataFrame(result["history"])
        self.history_ = self.history_.reindex(sorted(self.history_.columns), axis=1)
        self.stop_reason_ = result["stop_reason"]
        self.rfe_trajectory_ = result.get("rfe_trajectory", [])
        self.pareto_front_ = pd.DataFrame(self.confirmed_)
        self.reference_ = self.evaluator_.reference
        if self.selection_.policy == "manual":
            self.best_features_ = None
            self.selected_candidate_ = None
            selected_id = self.manifest_.get("selected_candidate_id")
            if selected_id is not None and any(r["key"] == selected_id for r in self.confirmed_):
                self._select(selected_id)
            else:
                self.progress_.info("Manual selection awaits a confirmed candidate ID", phase="selection",
                                    candidates=len(self.confirmed_))
        else:
            self._select()
        self._export()
        self.save_checkpoint()
        self.progress_.info("Feature selection completed", phase="complete", stop_reason=self.stop_reason_,
                            confirmed_pareto_candidates=len(self.confirmed_), selected_features=len(self.best_features_ or []),
                            elapsed_seconds=round(time.monotonic() - started, 3), log=str(self.log_path_))
        self.progress_.close()
        return self

    def _select(self, candidate_id=None):
        self.selected_candidate_ = select_candidate(self.confirmed_, self.selection_, self.random_state, candidate_id)
        self.best_features_ = self.feature_names_in_[np.asarray(self.selected_candidate_["mask"], bool)].tolist()
        self.progress_.info("Final candidate selected", phase="selection", policy=self.selection_.policy,
                            features=len(self.best_features_), mean_gini=self.selected_candidate_["mean_gini"],
                            candidate=self.selected_candidate_["key"][:12])

    def select(self, policy: str = "tolerance", *, candidate_id: str | None = None, gini_tolerance: float | None = None) -> list[str]:
        """Change final selection among feasible confirmed Pareto candidates."""
        check_is_fitted(self, "confirmed_")
        settings = asdict(self.selection_)
        settings["policy"] = policy
        if gini_tolerance is not None:
            settings["gini_tolerance"] = gini_tolerance
        self.selection_ = SelectionConfig(**settings)
        self._select(candidate_id)
        self.manifest_["config"]["selection"] = asdict(self.selection_)
        self.manifest_["selected_candidate_id"] = self.selected_candidate_["key"]
        atomic_json(self.directory_ / "manifest.json", self.manifest_)
        atomic_json(self.directory_ / "config.json", self.manifest_["config"])
        self._export()
        self.save_checkpoint()
        self.progress_.close()
        return self.best_features_

    def _checkpoint(self, state):
        self._state = state
        self.save_checkpoint()

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically save portable optimizer state; completed fits live in SQLite."""
        if not hasattr(self, "manifest_"):
            raise ValueError("No initialized experiment is available to checkpoint")
        destination = Path(path) if path is not None else self.directory_ / "checkpoint.json"
        atomic_json(destination, {"schema": 1, "evaluation_identity": self.evaluator_.identity,
                                  "config_hash": stable_hash(self.manifest_["config"]),
                                  "state": getattr(self, "_state", None),
                                  "pending": self.evaluator_.cache.incomplete()})
        self.progress_.info("Checkpoint saved", phase="checkpoint",
                            generation=self._state.get("generation", 0) if getattr(self, "_state", None) else 0)
        return destination

    @classmethod
    def resume(cls, experiment_dir: str | Path, *, verbose: int | None = None) -> MemeticNSGA2Selector:
        """Verify persisted data, versions, settings and resume without source access."""
        directory = Path(experiment_dir).resolve()
        manifest = read_json(directory / "manifest.json")
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported experiment schema")
        if any(version(name) != recorded for name, recorded in manifest["versions"].items()):
            raise ValueError("Checkpoint incompatible with current dependency versions; use the original environment")
        for relative, digest in manifest["files"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or _file_digest(path) != digest:
                raise ValueError(f"Persisted input changed or invalid path: {relative}")
        settings = dict(manifest["config"])
        if verbose is not None:
            settings["verbose"] = verbose
        selector = cls(**settings, checkpoint_dir=directory)
        selector._settings()
        selector.progress_.info("Checkpoint inputs and dependency versions verified", phase="resume", directory=str(directory))
        data = load_dataset(directory / "data")
        selector._set_features(data)
        selector.folds_ = [Fold(np.load(directory / f["train"], mmap_mode="r"),
                               np.load(directory / f["valid"], mmap_mode="r"), f["label"]) for f in manifest["folds"]]
        selector._build_evaluator(expected_identity=manifest["evaluation_identity"])
        if selector.evaluator_.identity != manifest["evaluation_identity"]:
            raise ValueError("Checkpoint incompatible with current evaluator, dependency versions, data or settings")
        selector.manifest_ = manifest
        checkpoint_path = directory / "checkpoint.json"
        state = None
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            if checkpoint.get("schema") != 1 or checkpoint["evaluation_identity"] != selector.evaluator_.identity or checkpoint["config_hash"] != stable_hash(manifest["config"]):
                raise ValueError("Checkpoint configuration or evaluator identity differs from manifest")
            state = checkpoint["state"]
        selector.evaluator_.all_results = {r["key"]: r for r in selector.evaluator_.cache.completed()}
        return selector._run(state)

    def cancel(self) -> None:
        """Request cancellation at the next safe admission/generation boundary."""
        if hasattr(self, "_cancel"):
            self._cancel.set()
            self.progress_.warning("Cancellation requested; waiting for the next safe boundary", phase="cancellation")

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Return selected predictor columns in their original order."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Use select('manual', candidate_id=...) before transform")
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.best_features_]
        array = np.asanyarray(X)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Array predictor count differs from fitted data")
        return array[:, self.get_support()]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected predictor mask or positions."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Select a candidate first")
        mask = np.isin(self.feature_names_in_, self.best_features_)
        return np.flatnonzero(mask) if indices else mask

    def fit_final_model(self) -> lgb.Booster:
        """Fit selected predictors on all development rows and persist the Booster."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Select a candidate first")
        data = self.evaluator_.data
        started = time.monotonic()
        self.progress_.info("Fitting final model on all development rows", phase="final_model", rows=len(data.y),
                            features=len(self.best_features_), rounds=self.num_boost_round)
        cols = np.flatnonzero(self.get_support())
        cats = [i for i, col in enumerate(cols) if col in data.categorical_indices]
        plan = self.evaluator_.manager.plan()
        if plan.workers < 1:
            raise MemoryError("Insufficient memory headroom for final full-development fit")
        params = strict_params(self.model_params, seed=self.evaluation_.estimator_seed,
                               threads=plan.threads_per_worker, n_rows=len(data.y))
        matrix = selected_matrix(data.X, np.arange(len(data.y)), cols)
        with progress_activity(self.progress_, "Final development fit still running", phase="final_model",
                               rows=len(data.y), features=len(cols)):
            self.final_model_ = lgb.train(params, lgb.Dataset(matrix, label=data.y, categorical_feature=cats),
                                          num_boost_round=self.num_boost_round)
        # LightGBM uses this mapping to align DataFrame categories on predict
        # and includes it when saving/reloading the native model text file.
        self.final_model_.pandas_categorical = [data.categories[self.feature_names_in_[cols[i]]] for i in cats]
        self.final_model_.save_model(str(self.directory_ / "final_model.txt"))
        atomic_json(self.directory_ / "final_model_metadata.json", {"features": self.best_features_, "categories": data.categories, "parameters": params})
        self.progress_.info("Final development model saved", phase="final_model", elapsed_seconds=round(time.monotonic() - started, 3))
        self.progress_.close()
        return self.final_model_

    def close(self) -> None:
        """Flush and close this selector's log files; later operations reopen them."""
        if hasattr(self, "progress_"):
            self.progress_.close()

    def _export(self):
        self.progress_.info("Exporting experiment results", phase="reporting", interactive_plots=self.interactive_plots)
        from .reporting.exports import export_results
        export_results(self.directory_, results=list(self.evaluator_.all_results.values()),
                       pareto=self.confirmed_, feature_names=list(self.feature_names_in_),
                       best_features=self.best_features_ or [], history=self.history_.to_dict("records"),
                       reference=self.reference_, rfe_trajectory=self.rfe_trajectory_,
                       feature_groups=self.feature_groups,
                       segment_constraints=self.evaluator_.active_constraints,
                       gini_tolerance=self.selection_.gini_tolerance,
                       selected_candidate_id=self.selected_candidate_["key"] if self.selected_candidate_ else None,
                       interactive_plots=self.interactive_plots,
                       output_format=self.output_format)
        from .reporting.exports import selection_frequency
        self.selection_frequency_ = selection_frequency(self.confirmed_, list(self.feature_names_in_), self.feature_groups)
        chosen = self.selected_candidate_ or self.reference_
        self.temporal_performance_ = pd.DataFrame(chosen["folds"])
        self.segment_performance_ = pd.DataFrame(chosen["segments"])
        self.progress_.info("Result artifacts saved", phase="reporting", directory=str(self.directory_))
