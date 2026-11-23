# Memetic NSGA-II feature selection

A reusable Python package that trades mean temporal validation Gini against
feature count using LightGBM and pymoo NSGA-II, with periodic add/remove/swap
local search. Every selected temporal fold uses all eligible rows. Final
frontiers and convergence use a common, complete CV specification.

For a SageMaker notebook without installing this project as a package, copy
only [memetic_fs_standalone.py](memetic_fs_standalone.py) beside the notebook.
Install its third-party dependencies in your environment, then use:

```python
from memetic_fs_standalone import run_selector, OptimizationConfig, ResourceConfig

selector = run_selector(
    df, "target", time_cv="reference_month", id_col="applicant_id",
    segment_cols=["customer_type", "age_group"],
    feature_range=(10, 200),
    optimization=OptimizationConfig(population_size=50, max_generations=100),
    resources=ResourceConfig(max_workers=4, threads_per_worker=4),
    checkpoint_dir="./runs/risk_selection", verbose=1,
)
print(selector.best_features_)
print(selector.pareto_front_[["n_features", "mean_gini"]])

# Recover an interrupted run in a later notebook session:
from memetic_fs_standalone import MemeticNSGA2Selector
selector = MemeticNSGA2Selector.resume("./runs/risk_selection")
```

The standalone file also exposes `MemeticNSGA2Selector.warmup_settings_params()`
and `MemeticNSGA2Selector.from_config()` for notebook workflows. It writes its
data copy, cache, checkpoints, logs and reports under
`checkpoint_dir`. To refresh it after editing package sources, run
`python tools/build_standalone.py`.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python examples/quickstart.py
```

Python 3.11+ and CPU LightGBM 4.5+ are supported by the dependency constraints;
the development environment was verified on Python 3.14.6, LightGBM 4.7.0,
pymoo 0.6.2, NumPy 2.5.5, pandas 2.3.3, joblib 1.6.0 and psutil 7.2.2.
The release wheel is pure Python; LightGBM/pymoo provide their own native wheels.

```python
from memetic_fs import MemeticNSGA2Selector, OptimizationConfig, ResourceConfig
from memetic_fs.synthetic import make_synthetic_data

df = make_synthetic_data(n_months=8, n_rows=800, n_features=16)
selector = MemeticNSGA2Selector(
    time_cv="reference_month", feature_range=(2, 12),
    optimization=OptimizationConfig(population_size=12, max_generations=6,
                                    local_search_budget=12),
    resources=ResourceConfig(max_workers=2, threads_per_worker=1),
    num_boost_round=25, checkpoint_dir="runs/my_experiment",
)
selector.fit(df, "target", id_col="applicant_id",
             segment_cols=["customer_type", "age_group"])
print(selector.best_features_)
print(selector.pareto_front_[["n_features", "mean_gini"]])
selected_frame = selector.transform(df)
```

An optional warm-up can inspect the data and write reviewable selector settings
before starting a search:

```python
result = MemeticNSGA2Selector.warmup_settings_params(
    df, "target", depth="standard", output_path="settings/warmup.yaml",
    id_col="applicant_id", segment_cols=["customer_type", "age_group"],
)
print(result.diagnostics, result.pilots, result.changes)
selector = MemeticNSGA2Selector.from_config(
    result.settings_path, checkpoint_dir="runs/warmed_experiment",
    optimization={"max_generations": 30},  # inline keys override YAML keys
)
selector.fit(df, "target", id_col="applicant_id",
             segment_cols=["customer_type", "age_group"])
```

`depth="quick"` uses diagnostics only; `"standard"` pilots three recent temporal
folds; `"deep"` pilots all folds and proposes the full optimization config.
The warm-up announces its planned number of fits and can take substantial time
on large data. Its recommendations are exploratory, and a selection near the
minimum feature count still deserves review. `base_config=` accepts an existing
selector or experiment YAML, leaving that file unchanged; `from_config()` can
load either format. The generated selector YAML expects data to be passed to
`fit()` and is not a CLI experiment file.

Use a fresh experiment directory for a new fit. Predictor names and pandas
categorical mappings are preserved; explicitly cast categorical predictors to
`category`. Missing numeric values are supported without imputation. Numeric
precision is preserved; lossy conversion to LightGBM's floating representation
raises an actionable error. Metadata columns are excluded from predictors.
An ID column is required by default, because uniqueness cannot otherwise be
verified. `allow_repeated_ids=True` explicitly permits repeats without dropping
rows or adding group CV.

NumPy/memmap input works with `fit(X, y, time_values=months, ids=ids,
feature_names=names)`; use a string time label as `time_cv`. For a custom
callable, pass `time_values` explicitly. Parquet and CSV loading are available
through the CLI. Only development data belongs in this interface.

```bash
python -c "from memetic_fs.synthetic import make_synthetic_data; make_synthetic_data(n_features=16).to_parquet('examples/synthetic.parquet', index=False)"
memetic-fs run --config examples/experiment.yaml
memetic-fs resume --experiment-dir runs/example
memetic-fs report --experiment-dir runs/example
```

All relative YAML paths resolve from the configuration file's directory. See
[the complete example](examples/experiment.yaml), [configuration guide](docs/configuration.md),
and [Python API parameter reference](docs/api-parameters.md).
The [complete synthetic notebook](examples/complete_pipeline.ipynb)
demonstrates warm-up through checkpoint recovery. Its saved outputs were cleared
after the API update; run its cells in order to generate current results.
Self-contained Plotly output lives in `report.html`.

The default policy chooses the smallest feasible confirmed model within 0.005
absolute Gini of the best confirmed candidate. `selector.select('knee')`,
`selector.select('statistical')` and `selector.select('manual', candidate_id=...)`
offer alternatives. `fit_final_model()` optionally fits all development rows
and saves a native LightGBM model. It never evaluates external data.

`selector.cancel()` requests a safe stop; Ctrl+C also preserves completed
evaluations. `MemeticNSGA2Selector.resume(directory)` verifies persisted inputs,
configuration and evaluator versions and restores optimizer state. Checkpoints
are structured JSON, and results are transactionally cached in SQLite.

Read [architecture](docs/architecture.md), [temporal evaluation](docs/validation.md),
[optimization](docs/optimization.md), [execution](docs/execution.md), and
[results](docs/results.md) for details; see [SageMaker operations](docs/sagemaker.md)
for deployment troubleshooting and [logging](docs/logging.md) for console/file
progress. `verbose=1` gives concise updates, `verbose=2` includes candidate/fold
details, and `verbose=0` keeps warnings/errors. Detailed logs persist under
`EXPERIMENT/logs/`. Large-scale evaluation is opt-in via
`python benchmarks/full_row.py --rows 1500000 --features 250 --rounds 100`.
Do not infer 1.5-million-row feasibility from the small automated tests.

There is no global-optimum guarantee. Repeated search on development validation
introduces optimism. The paired temporal block uncertainty heuristic is weak
with few folds and does not prove equivalence. Independent OOT assessment is
the user's separate responsibility.
