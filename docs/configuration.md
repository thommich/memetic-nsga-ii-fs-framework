# Configuration and API

For every Python call parameter, default, and accepted value, see the
[public API parameter reference](api-parameters.md).

`MemeticNSGA2Selector.warmup_settings_params(...)` optionally creates a
separate, flat selector-settings YAML from in-memory development data. Set
`depth` to `quick`, `standard`, or `deep`, supply `output_path`, and pass the
same metadata arguments used by `fit`. `base_config` may name an existing
selector-settings or full experiment YAML. The source is never edited; the
new YAML contains the proposed active values and requires an unused path unless
`overwrite=True`. The returned `WarmupResult` includes diagnostics, chronological
pilot scores, reasons, changes, warnings, `settings_path`, and `log_path`. Model pilots
fit complete training and validation folds; their chosen settings remain
recommendations rather than a promise about final feature count.

`MemeticNSGA2Selector.from_config(path, **overrides)` accepts the generated
flat YAML or the full experiment YAML below. Explicit overrides win, with
`model_params` and typed configuration groups merged by key. The resolved
settings are captured on construction and then in the run manifest; later
changes to the YAML do not change an initialized selector or resumed run.

The constructor accepts `model_params`, `time_cv`, inclusive `feature_range`,
`segment_constraints`, `use_rfe_initialization`, `resources`, `random_state`,
`checkpoint_dir`, `cv_config`, `optimization`, `evaluation`, `selection`,
`num_boost_round`, `allow_repeated_ids`, `initial_subsets`, `feature_groups`,
`interactive_plots`, `output_format`, and `verbose`. Objectives are fixed at `('gini', 'n_features')`.
Advanced groups accept typed dataclasses or dictionaries; unknown fields fail.

`CVConfig`: minimum three training and six total observed months, validation
horizon one, step one, gap zero. v1 supports nonoverlapping validation windows
and no calendar gap. See [validation](validation.md) for missing-month behavior.

`OptimizationConfig`: population 50, at most 100 generations, minimum 10 before
convergence, patience 12, normalized hypervolume improvement threshold 0.0001.
Local search every four generations proposes up to 40 neighbors across four
cardinality regions. Exploration probability 0.3; inverse-cardinality exponent
1.3, smoothing 1, uniform exploration mixture 0.15. RFE checkpoints default to
20/40/60/80/100/140/180/200 and adapt to bounds. Initial five generations may use
the last one fold; `reduced_generations=0` disables this. Actual row counts never
change within selected folds. `initial_subsets` accepts names or binary masks.
`feature_groups` accepts feature-name to family mapping for structured swaps and
report aggregation.

`EvaluationConfig`: independent estimator seed 1729 and minimum segment support
30 events and 30 non-events in every canonical validation fold. Constraints
are `SegmentConstraint(column, value, max_gini_drop=0.01)`, with full-feature
reference. The worst fold's segment drop must satisfy the bound. An ineligible
constraint is disabled for the entire run and recorded with one warning per
constraint at initialization. All segments remain in descriptive reporting;
undefined AUC is represented by null and never enters optimizer constraints.

`ResourceConfig`: reserve 25% of effective capacity; optional `max_ram_gb`,
`max_workers`, `threads_per_worker`; two retries and 30-second pressure wait.
`resources='auto'` uses these defaults. Limits also honor current available RAM,
CPU affinity and cgroup restrictions. User limits cannot enable oversubscription.

`verbose=1` provides milestones and a generation bar with worker status; 2 includes each candidate/fold,
and 0 keeps warnings/errors on the built-in console. Detailed per-process files
are retained under `logs/` at every verbosity. YAML uses `output.verbose` and CLI
supports `--verbose` / `--quiet`. Long fits emit periodic heartbeats. See
[logging](logging.md) for application logger integration and file lifecycle.

`SelectionConfig`: policy `tolerance`, Gini tolerance 0.005, paired temporal block
bootstrap repeats 1000 and upper-bound confidence 0.95. Statistical selection
accepts a candidate only when the approximate upper paired Gini loss is within
the explicit tolerance. This is an uncertainty heuristic on fold metrics, not a
row-iid bootstrap or formal post-selection equivalence test.

LightGBM's own valid defaults are retained except for strict full-row,
deterministic CPU settings. Fixed rounds default to the documented `train`
default 100. Set rounds through `num_boost_round`, not `model_params` aliases.
Managed thread, seed, sampling, device, and early-stop aliases are rejected when
they conflict or bypass the dedicated setting. Per-feature parameter arrays
and forced split files are unsupported across changing subsets.

YAML sections mirror these settings. `optimization` additionally holds
`feature_range`, `random_state` and `use_rfe_initialization`. `model` holds
`estimator: lightgbm`, `params`, `num_boost_round`. `output` requires
`experiment_dir`; checkpoint persistence is always enabled. `data` requires
`path`, `target`, `time_column`, and normally `id_column`. Booleans for row and
feature sampling and sample weights must remain false. YAML cannot import a
custom CV callable; use the Python protocol.

Set `output.format` to `csv`, `json`, or `parquet` (default) to choose one
format for each exported result table. In Python, use `output_format` on the
selector constructor. Structured JSON records used by checkpoints and report
rebuilding remain available in every mode.

Fitted attributes: `pareto_front_` (DataFrame of feasible full-CV records),
`best_features_`, `history_`, `selection_frequency_`, `temporal_performance_`,
`segment_performance_`, `reference_`, `confirmed_`, `provisional_`,
`selected_candidate_`, `stop_reason_`, and optionally `final_model_`.
`get_support(indices=False)` exposes selected predictors relative to the
metadata-excluded input feature list. Manual policy leaves `best_features_`
unset until a valid candidate is selected. Transformation does not impute,
engineer features or evaluate a target.
