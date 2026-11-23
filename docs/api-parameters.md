# Python public API parameter reference

This reference covers the exported selector and configuration classes, their public
methods, and the notebook and reporting helpers exposed by
`memetic_fs_standalone.py`. Import the selector and configuration classes from
`memetic_fs` (or the standalone module). Calls use **development data only**.
Defaults below are the values in the Python API. Full experiment YAML maps to
them through the CLI or `from_config()`; warm-up writes selector-only YAML for
`from_config()`.

## `MemeticNSGA2Selector(...)`

Constructing a selector stores options. Validation, data persistence, and the
search begin when `fit` is called. All parameters are optional.

| Parameter | Default | Description |
| --- | --- | --- |
| `model_params` | `None` | Mapping of LightGBM training parameters. `None` uses LightGBM defaults plus the framework's fixed binary, deterministic, CPU, full-row settings. Sampling, early stopping, device, thread, seed, boosting-round, and other framework-managed aliases cannot override those settings. Per-feature parameter arrays and external training-data options are unsupported; use `num_boost_round` for rounds and `evaluation.estimator_seed` for the model seed. |
| `time_cv` | `"reference_month"` | Name of a DataFrame time column (or a label used with explicit `time_values`), or a callable that receives an aligned NumPy array of normalized monthly ordinals and returns an iterable of `(train_positions, validation_positions)` pairs (or `Fold` objects). For a callable, pass `time_values` to `fit` explicitly. Positions are integer row offsets; every selected monthly interval must include all its rows. Training must precede validation, validation windows cannot overlap, and the last window must reach the latest observed month. |
| `feature_range` | `(10, 200)` | Inclusive `(minimum, maximum)` number of selected predictors; both bounds must be integers, with minimum at least 1. The maximum is clipped to the available predictor count; the minimum must remain no greater than that count and the clipped maximum. |
| `objectives` | `("gini", "n_features")` | Fixed two-objective tuple: maximize the unweighted mean temporal-fold Gini while minimizing feature count. This exact tuple is the only supported value. |
| `segment_constraints` | `None` | List of `SegmentConstraint` instances or dictionaries with the same fields. Each requires a segment column listed in `fit(segment_cols=...)`; it bounds the worst per-fold Gini loss versus the full-feature reference. Ineligible segments are disabled for the run and recorded. See `SegmentConstraint` below. |
| `use_rfe_initialization` | `True` | Boolean. If true, recursively removes features from a full-feature reference and evaluates checkpoint subsets under full temporal CV to seed the search. |
| `resources` | `"auto"` | `"auto"`, a `ResourceConfig`, or a dictionary of its fields. Auto uses the documented `ResourceConfig` defaults. Worker and thread counts are still bounded by observed CPU and memory capacity. |
| `random_state` | `42` | Nonnegative integer seed for optimizer proposals and selection resampling. Separate from the LightGBM `estimator_seed`. |
| `checkpoint_dir` | `"./runs/run_001"` | String or `Path` for the experiment's data copy, cache, manifest, checkpoint, logs, and reports. A new `fit` requires a directory without an existing `manifest.json`; use `resume` for an existing run. |
| `cv_config` | `None` | `CVConfig` instance or dictionary. `None` creates `CVConfig()`; controls observed-month training and validation windows. |
| `optimization` | `None` | `OptimizationConfig` instance or dictionary. `None` creates `OptimizationConfig()`; controls population, stopping, initialization, fidelity, and local search. |
| `evaluation` | `None` | `EvaluationConfig` instance or dictionary. `None` creates `EvaluationConfig()`; controls segment eligibility and the LightGBM seed. |
| `selection` | `None` | `SelectionConfig` instance or dictionary. `None` creates `SelectionConfig()`; controls the final choice from the feasible, confirmed Pareto frontier. Under the `manual` policy, choose a candidate with `select` before transforming or fitting a final model. |
| `num_boost_round` | `100` | Number of LightGBM boosting rounds used for each reference, candidate, and final-model fit. Supply rounds here rather than through a LightGBM `model_params` alias. |
| `allow_repeated_ids` | `False` | Boolean. By default, applicant IDs are required, nonmissing, and unique. If true, repeated IDs are allowed; it does not deduplicate rows or create grouped CV. It also permits omitting IDs. |
| `initial_subsets` | `None` | List of seed subsets. Each seed is either a collection of predictor names or a binary mask with one 0/1 value per predictor. Names must exist and be unique within a seed; each seed's size must fit `feature_range`. Duplicate masks are collapsed. |
| `feature_groups` | `None` | Optional mapping of predictor name to family label, or family label to a collection of predictor names/positions. It guides some structured swaps and groups feature-frequency reports. Use the name-to-family form for simple reporting and search. |
| `interactive_plots` | `True` | Boolean controlling self-contained interactive Plotly output in `report.html`. Tabular result files are still exported. |
| `output_format` | `"parquet"` | File format for each exported result table: `"csv"`, `"json"`, or `"parquet"`. Only the chosen format is written for each table. Structured JSON records for checkpoints and report rebuilding remain available. YAML uses `output.format`. |
| `verbose` | `1` | Console progress level: `0` warnings/errors, `1` milestones and a generation bar with worker counts, `2` routine batch and candidate/fold detail. Detailed log files are retained at every level. |

Configuration groups may be passed as dataclass instances or dictionaries;
unknown dictionary keys raise an error.

## `selector.fit(X, y, *, id_col=None, segment_cols=None, time_values=None, ids=None, feature_names=None)`

Fits the search, writes experiment artifacts, and returns the selector.

| Parameter | Description |
| --- | --- |
| `X` | Nonempty, two-dimensional pandas `DataFrame` or numeric NumPy array of development rows. A DataFrame may include target, time, ID, and segment metadata columns; recognized metadata is excluded from predictors. Predictor columns must have unique string names and numeric or pandas `category` dtypes. Arrays contain predictors only. Numeric missing values are allowed; infinities and lossy numeric conversion are rejected. Row order is significant. |
| `y` | For a DataFrame, a target column name or an aligned one-dimensional sequence/Series/array. For an array, an aligned sequence/array. Values must be nonmissing binary `0`/`1`, with both classes present. An external pandas Series must have exactly the same index and order as `X`. |
| `id_col` | DataFrame column name for applicant IDs. The column is excluded from predictors. Supply either this or `ids`, not both. With the default `allow_repeated_ids=False`, IDs are required, nonmissing, and unique. |
| `segment_cols` | Sequence of DataFrame column names used for descriptive segment scores and any `segment_constraints`. Columns are excluded from predictors; duplicate names are rejected. Array input cannot supply segment columns. |
| `time_values` | Aligned one-dimensional dates/datetimes or valid integer `YYYYMM` values. Required for NumPy input and callable `time_cv`; optional when `time_cv` names a DataFrame column. Date strings must be parsed before calling `fit`. An external pandas Series must match `X`'s index exactly. |
| `ids` | Aligned one-dimensional applicant IDs supplied separately from `X`, chiefly for array input. Cannot be combined with `id_col`; missing values are rejected, and duplicates require `allow_repeated_ids=True`. An external pandas Series must match `X`'s index exactly. |
| `feature_names` | Sequence of one unique string per array predictor; defaults to `x0`, `x1`, and so on. For a DataFrame, if supplied, names must exactly match the predictor columns after metadata removal, in order. |

Time windows count **distinct observed months**. At least six observed months
are required. Each train and validation fold must contain both target classes;
folds always use all eligible rows in their selected monthly intervals.

## Other selector calls

`MemeticNSGA2Selector.warmup_settings_params(X, y, *, depth="standard",
output_path, base_config=None, overwrite=False, id_col=None,
segment_cols=None, time_values=None, ids=None, feature_names=None,
time_cv=None)` returns a `WarmupResult` and writes separate selector-settings
YAML. It validates data and folds but does not run the NSGA-II search. `quick`
uses no model fits, `standard` pilots the last three folds, and `deep` uses all
folds. Pilot duration grows with data size and depth; there is no duration or
fit-count cutoff within a selected depth. `quick` model suggestions are
heuristic. For a YAML warm-up, `time_cv` must be a column name. An existing
`output_path` raises unless `overwrite=True`, and `output_path` cannot equal
`base_config`.

`MemeticNSGA2Selector.from_config(path, **overrides)` loads either a flat
selector-settings YAML or a full CLI experiment YAML. Inline overrides take
priority, including individual keys inside `model_params`, `cv_config`,
`optimization`, `evaluation`, `resources`, and `selection`. Unknown and invalid
settings that do not require data are rejected before `fit`. Data-dependent
checks, such as subset names and available predictor count, run during warm-up
or `fit`. A selector resolves the file once; `resume()` uses the effective
settings in the run manifest.

| Call and parameter | Description |
| --- | --- |
| `selector.select(policy="tolerance", *, candidate_id=None, gini_tolerance=None)` | Rechoose from feasible, full-CV confirmed Pareto candidates after `fit` or `resume`; returns selected predictor names and updates saved selection/report artifacts. `policy` is `"tolerance"`, `"statistical"`, `"knee"`, or `"manual"`. `candidate_id` is the candidate's exact `key` from `pareto_front_`, required for manual choice; it is ignored by the other policies. `gini_tolerance`, if supplied, overrides the current nonnegative absolute Gini tolerance for this selection; `None` retains it. |
| `selector.save_checkpoint(path=None)` | Atomically writes structured optimizer state and returns the checkpoint `Path`. `path` is an optional string/`Path`; `None` writes `checkpoint.json` inside `checkpoint_dir`. Requires an initialized experiment. Completed evaluations also persist in SQLite. |
| `MemeticNSGA2Selector.resume(experiment_dir, *, verbose=None)` | Reopens and continues an existing experiment. `experiment_dir` is its directory (string or `Path`); stored data, hashes, settings, and dependency versions are verified. `verbose=None` retains the saved level; `0`, `1`, or `2` overrides console verbosity for the resumed run. Returns a selector. |
| `selector.transform(X)` | Returns selected columns in original predictor order. `X` is a DataFrame containing the selected predictor names, or a two-dimensional array with the same number and order of predictors as fitted. Returns the same container kind. Requires a selected candidate. |
| `selector.get_support(indices=False)` | The Boolean `indices` parameter defaults to `False`, returning a mask over the fitted predictor list; `True` returns their zero-based positions. Metadata columns are outside this list. Requires a selected candidate. |
| `selector.cancel()` | Requests cancellation at the next safe worker-admission or generation boundary. No parameters. An active model fit may finish before stopping. |
| `selector.fit_final_model()` | Trains a LightGBM `Booster` on all development rows using the selected predictors and saves `final_model.txt` plus metadata. No parameters; requires a selected candidate. It performs no external holdout evaluation. |
| `selector.close()` | Flushes and closes the selector's log files. No parameters; later operations reopen logging as needed. |

## Configuration class parameters

### `CVConfig(...)`

All window sizes count distinct observed months. Auto folds use expanding
training windows, nonoverlapping validation windows, and end at the latest
observed month. Custom folds must still pass the same chronology and full-row
checks.

| Parameter | Default | Description |
| --- | --- | --- |
| `min_train_months` | `3` | Minimum distinct observed months in each training fold; at least 3. |
| `validation_months` | `1` | Number of distinct observed months per auto validation window; positive integer. |
| `step_months` | `1` | Month offset between auto validation starts; positive integer and at least `validation_months` to avoid repeated validation rows. |
| `min_total_months` | `6` | Minimum distinct observed months in the development data; at least 6. |
| `gap_months` | `0` | Gap between auto training and validation windows. Version 1 supports only `0`. |

### `OptimizationConfig(...)`

| Parameter | Default | Description |
| --- | --- | --- |
| `population_size` | `50` | Target number of NSGA-II subsets per generation; integer at least 2. The actual size is capped by the number of distinct valid subsets. |
| `max_generations` | `100` | Maximum generations; positive integer. |
| `min_generations` | `10` | Earliest generation at which stalled-hypervolume stopping can occur; positive integer. |
| `patience` | `12` | Number of consecutive generations without sufficient confirmed-hypervolume improvement before stopping, subject to `min_generations`; positive integer. |
| `min_improvement` | `0.0001` | Minimum increase in normalized confirmed-frontier hypervolume that resets the stall counter; finite, nonnegative number. |
| `local_search_interval` | `4` | Run local refinement every this many generations; positive integer. |
| `local_search_budget` | `40` | Maximum distinct neighbor subsets proposed per refinement; nonnegative integer, with `0` disabling proposals. |
| `local_search_candidates` | `4` | Maximum confirmed-frontier seed candidates spread across feature counts for each refinement; positive integer. |
| `exploration_probability` | `0.3` | Probability that a neighbor feature choice uses the random branch instead of importance guidance; in `(0, 1]`. |
| `inverse_exponent` | `1.3` | Positive exponent in inverse-cardinality sampling weights `(size + inverse_smoothing)^(-inverse_exponent)`. Larger values favor smaller subsets. |
| `inverse_smoothing` | `1.0` | Positive offset added to subset size before inverse-cardinality weighting. |
| `uniform_cardinality_probability` | `0.15` | Mixture weight for uniform subset-size proposals, preserving a nonzero chance for every allowed size; in `(0, 1]`. |
| `rfe_seed_sizes` | `(20, 40, 60, 80, 100, 140, 180, 200)` | Positive integer feature counts for RFE checkpoints. Values outside the actual `feature_range` are skipped; if none apply, range endpoints are used. Relevant only when `use_rfe_initialization=True`. |
| `reduced_generations` | `5` | Number of initial generations allowed to use reduced-fold provisional evaluations; nonnegative integer, with `0` using full CV throughout. Final frontier candidates are confirmed on full CV. |
| `reduced_fold_count` | `1` | Number of last chronological folds used for provisional evaluation in those early generations; positive integer, no greater than the available canonical fold count when reduced generations are enabled. |

### `ResourceConfig(...)`

| Parameter | Default | Description |
| --- | --- | --- |
| `memory_reserve_fraction` | `0.25` | Fraction of effective memory capacity held in reserve during worker admission; strictly between 0 and 1. |
| `max_ram_gb` | `None` | Optional positive memory cap in GiB (multiples of `1024**3` bytes). The scheduler also honors available physical/container memory. |
| `max_workers` | `None` | Optional positive integer cap on concurrent evaluation workers. `None` derives a safe count from CPU and memory. |
| `threads_per_worker` | `None` | Optional positive integer cap for LightGBM threads per worker. `None` derives a safe fixed thread budget from available CPU capacity. |
| `max_retries` | `2` | Nonnegative integer number of retries after recoverable worker/resource failures. |
| `memory_wait_seconds` | `30` | Nonnegative, finite wait in seconds before retrying when memory pressure blocks admission. |

### `EvaluationConfig(...)`

| Parameter | Default | Description |
| --- | --- | --- |
| `min_segment_events` | `30` | Minimum event (`1`) count for a segment in every validation fold to keep its hard constraint active; positive integer. |
| `min_segment_nonevents` | `30` | Minimum non-event (`0`) count under the same rule; positive integer. |
| `estimator_seed` | `1729` | Nonnegative integer LightGBM seed applied consistently to the reference, candidate, and final-model fits. |

### `SegmentConstraint(column, value, max_gini_drop=0.01, reference="full_feature_model")`

| Parameter | Description |
| --- | --- |
| `column` | Name of a DataFrame segment column also supplied in `fit(segment_cols=...)`. |
| `value` | Exact segment value to match in that column. `None`/missing selects missing values. |
| `max_gini_drop` | Maximum permitted absolute Gini loss for the segment relative to the full-feature reference on each canonical validation fold; default `0.01`, finite and nonnegative. The worst fold determines the violation. |
| `reference` | Baseline identifier; only `"full_feature_model"` is supported. |

### `SelectionConfig(...)`

| Parameter | Default | Description |
| --- | --- | --- |
| `policy` | `"tolerance"` | `"tolerance"` selects the smallest feasible subset within `gini_tolerance` of the best confirmed mean Gini; `"statistical"` uses an approximate paired temporal-block upper loss; `"knee"` selects a normalized frontier bend; `"manual"` waits for `select(..., candidate_id=...)`. |
| `gini_tolerance` | `0.005` | Finite, nonnegative allowable absolute Gini loss; used by tolerance and statistical selection. |
| `bootstrap_repeats` | `1000` | Positive integer count of paired temporal-block resamples for statistical selection. |
| `confidence` | `0.95` | Quantile of the resampled paired Gini loss used as the upper bound in statistical selection; strictly between 0 and 1. This is a heuristic, not a formal post-selection equivalence test. |

## Standalone and example helpers

The standalone module exports the same selector/configuration API plus these
functions. `run_selector` is the one-call notebook entry point.

| Call and parameter | Description |
| --- | --- |
| `run_selector(X, y, *, id_col=None, segment_cols=None, time_values=None, ids=None, feature_names=None, **selector_options)` | Creates `MemeticNSGA2Selector(**selector_options)`, calls `fit` with the other arguments, and returns the fitted selector. `X`, `y`, and the named metadata arguments have the exact `fit` meanings above; each key in `selector_options` must be a constructor parameter listed above. |
| `make_synthetic_data(n_rows=2400, n_features=20, n_months=12, random_state=42, *, n_signal_features=3, n_redundant_features=1, n_interaction_features=2, n_segmentation_specific_features=1)` | Returns an example DataFrame with `n_features` predictors plus `target`, `reference_month`, `applicant_id`, `customer_type`, and `age_group` metadata. `n_rows` is the total row count; `n_months` is the number of observed months starting January 2023; `random_state` seeds NumPy. The four keyword-only counts create `signal_*` direct predictors, `redundant_*` correlated copies, `interaction_*` predictors that enter through products, and `segment_signal_*` predictors active within a customer type. Every remaining predictor is `noise_*`. Counts must be nonnegative integers and their sum must be **strictly less** than `n_features`, leaving at least one noise predictor. Require at least 6 months and 8 rows per month. Also available as `memetic_fs.synthetic.make_synthetic_data`. |
| `load_experiment(path)` | Reads a strict YAML experiment configuration. `path` is a string/`Path` to the YAML file; relative data and output paths resolve from its parent directory. Returns `(data_settings, selector_kwargs)` for CLI-style execution. |
| `main(argv=None)` | CLI entry point. `argv` is an optional list of command-line argument strings, such as `['run', '--config', 'experiment.yaml']`; `None` reads process arguments. Returns an exit code or uses argparse's exit behavior for errors. |
| `rebuild_report(directory)` | Recreates the interactive report from an existing experiment's saved results. `directory` is its string/`Path`; returns the report `Path`. |
| `aggregate_run_frequencies(directories)` | Takes a nonempty sequence of saved experiment directory strings/`Path`s and returns a DataFrame of pooled feature counts and per-run inclusion frequency. Runs must have identical predictor names in the same order. |

For YAML keys and CLI commands, see [configuration.md](configuration.md).
