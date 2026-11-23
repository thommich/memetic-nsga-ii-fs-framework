# Architecture

For the end-to-end processing diagram and concise explanation of each step,
see [algorithm-flow.md](algorithm-flow.md).

The Python entry point is `MemeticNSGA2Selector`. Typed dataclasses group CV,
optimization, evaluation, execution and selection settings. YAML maps to the same
objects, rejects unknown keys, and never imports executable callables.

The optional `warmup_settings_params()` entry point validates development data,
builds the same temporal folds, inspects resources, and compares fixed LightGBM
settings at the selected depth. It writes separate selector-only YAML and returns
pilot diagnostics. `from_config()` resolves that file or a full experiment YAML
before `fit`; it does not start a run.

`data` validates once, excludes metadata and persists read-only arrays. `cv`
materializes full-month folds before training. `evaluation` trains fixed-round
CPU LightGBM models and writes compact metrics transactionally. `optimization`
uses pymoo NSGA-II, constrained survival and non-dominated sorting; custom code
handles initialization, swaps, local search and fidelity transitions. `execution`
admits bounded process batches using current resource headroom. `persistence`
stores JSON checkpoints and SQLite results. `reporting` consumes result records.

Every cached identity includes data content, normalized folds, fidelity, model,
metric settings, seeds and library versions. Reduced-fold populations are
reevaluated when fidelity changes; final archive comparisons and convergence use
only canonical full-CV results. Calibration uses the full-feature reference over
full folds. Checkpoints contain structured optimizer/RNG state, not pickled code.

Scope decisions: windows count observed months, not uninterrupted calendar
history; overlapping validation windows and temporal gaps are rejected in v1.
Statistical selection uses paired temporal block resampling of fold Gini
differences; few folds imply weak uncertainty estimates. No distributed backend,
external OOT input or feature engineering is added. Warm-up compares a fixed,
finite set of model settings; it does not optimize LightGBM jointly with NSGA-II.

API verification (2026-11-21): pymoo 0.6.2, LightGBM 4.7.0,
joblib 1.6.0, psutil 7.2.2. Sources:
https://pymoo.org/algorithms/moo/nsga2.html,
https://pymoo.org/customization/binary.html,
https://pymoo.org/operators/repair.html,
https://pymoo.org/interface/termination.html,
https://lightgbm.readthedocs.io/en/stable/Parameters.html,
https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html,
https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html,
https://psutil.readthedocs.io/stable/.
