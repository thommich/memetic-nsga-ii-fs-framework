# Selector output table columns

Saved table filenames use the configured `output_format` extension: `.csv`,
`.json`, or `.parquet` (default).

These four attributes are pandas DataFrames populated after a successful
`fit()` or `MemeticNSGA2Selector.resume()`. All scores come from development
validation folds. A call to `selector.select(...)` refreshes the performance
tables for the newly selected candidate; it does not rerun the search.

| Attribute | One row represents | Source |
| --- | --- | --- |
| `selector.temporal_performance_` | One validation fold for the selected candidate | `selected_candidate_["folds"]` |
| `selector.segment_performance_` | One segment value in one validation fold for the selected candidate | `selected_candidate_["segments"]` |
| `selector.selection_frequency_` | One predictor across the distinct confirmed Pareto subsets | `confirmed_` |
| `selector.history_` | One completed optimization generation | Search history |

If `selection.policy="manual"` and no candidate has been selected yet, the
temporal and segment tables show the **full-feature reference model** instead.
After `select("manual", candidate_id=...)`, they show that candidate. The
reference model can include more predictors than `feature_range` allows.

## `selector.temporal_performance_`

The selected model is fitted separately on each fold's complete training
interval and scored on every row in its validation interval. Rows follow
chronological fold order. The table contains these columns:

| Column | Meaning |
| --- | --- |
| `fold` | Validation month label (`YYYY-MM`) or interval label (`YYYY-MM/YYYY-MM`). |
| `train_n` | Number of development rows used to train the fold's LightGBM model. |
| `validation_n` | Number of rows in the fold's validation interval. |
| `n` | Number of scored validation rows; equals `validation_n` here. |
| `events` | Count of validation target values equal to `1`. |
| `nonevents` | Count of validation target values equal to `0`; `events + nonevents = n`. |
| `auc` | ROC AUC of the fold's predictions on all validation rows. |
| `gini` | Fold discrimination score, calculated as `2 * auc - 1`. The selector's `mean_gini` is the unweighted mean of these fold Gini values. |

The selected candidate always has full-CV metrics; provisional reduced-fold
results are not used for this table. Each accepted fold contains both target
classes, so its AUC and Gini are defined. The saved `fold_performance` table
covers **all evaluated candidates** and adds `key` and `fidelity`; it is not
the same single-candidate view as this
attribute.

## `selector.segment_performance_`

For every column passed to `fit(segment_cols=...)`, the evaluator creates a
row for each distinct segment value in every validation fold. Rows describe
the selected candidate (or the full-feature reference while manual selection
is pending).

| Column | Meaning |
| --- | --- |
| `column` | Name of the segment metadata column. Segment columns are excluded from candidate predictors. |
| `value` | Segment value within `column`; missing values are grouped as `None`/null. |
| `fold` | Validation month or interval label, matching `temporal_performance_.fold`. |
| `n` | Validation rows in this fold with this segment value. Can be zero when a value observed elsewhere is absent from this fold. |
| `events` | Count of target `1` rows in the segment and fold. |
| `nonevents` | Count of target `0` rows in the segment and fold; `events + nonevents = n`. |
| `auc` | ROC AUC for this segment and fold, or null when it has fewer than two target classes. |
| `gini` | `2 * auc - 1`, or null when AUC is undefined. |

This table remains descriptive even when a segment constraint is ineligible.
An eligible hard constraint compares each fold's segment Gini with the
full-feature reference, not with the mean of this table. If no `segment_cols`
were supplied, `segment_performance_` is an empty DataFrame with no columns.
The saved `segment_performance` table contains rows for **all
evaluated candidates**, with extra `key` and `fidelity` columns.

## `selector.selection_frequency_`

This table counts predictor inclusion across distinct, feasible, full-CV
Pareto subsets in `selector.confirmed_`. It has one row per candidate predictor,
including predictors never selected. Rows are sorted by descending frequency,
then ascending feature name.

| Column | Meaning |
| --- | --- |
| `feature` | Predictor name from `feature_names_in_`; metadata columns are excluded. |
| `count` | Number of distinct confirmed Pareto subsets containing the predictor. Each subset contributes at most one. |
| `frequency` | `count / n_solutions`; zero when there are no confirmed subsets. This is a proportion from 0 to 1. |
| `n_solutions` | Total number of distinct confirmed Pareto subsets used as the denominator; repeated on every predictor row. |
| `family` | Optional feature-group label supplied via `feature_groups`. Present only when a nonempty group mapping is supplied; unassigned predictors show `"ungrouped"`. |

For example, if a predictor appears in 3 of 5 distinct confirmed subsets,
`count=3`, `n_solutions=5`, and `frequency=0.6`. These frequencies summarize
searched solutions; they are not feature importance, selection probabilities,
or significance tests. The saved `selection_frequency` table
has the same columns. The separate `high_performing_frequency` export uses
only confirmed subsets within the configured Gini tolerance of the best.

## `selector.history_`

Each row records a completed generation, beginning at generation 1. The
DataFrame's columns are sorted alphabetically by name. “Confirmed” refers to
feasible, nondominated subsets evaluated on every canonical validation fold.
“Provisional” refers to the current reduced-fold basis when enabled.

| Column | Meaning |
| --- | --- |
| `generation` | One-based completed generation number. |
| `fidelity` | Evaluation basis for that generation: `"full"` or `"last_N"` for the last `N` chronological folds. Provisional results are confirmed on full CV before entering the confirmed frontier. |
| `confirmed_hypervolume` | Normalized hypervolume of the feasible confirmed frontier after this generation; `0.0` if none exists. Fixed Gini and feature-count bounds make values comparable across generations of the same run. |
| `best_confirmed_gini` | Highest mean temporal-fold Gini on the confirmed frontier so far; null when it is empty. |
| `confirmed_count` | Number of feasible, nondominated full-CV subsets on the confirmed frontier. |
| `provisional_count` | Number of feasible, nondominated subsets on the current `last_N` basis; `0` for a `"full"` generation. |
| `unique_evaluations` | Number of distinct `(fidelity, feature mask)` results tracked by the search so far. The same subset at reduced and full fidelity counts twice. |
| `feature_count_min` | Fewest selected predictors among the surviving population after this generation. |
| `feature_count_mean` | Arithmetic mean selected-predictor count among that population. |
| `feature_count_max` | Most selected predictors among that population. |
| `local_search_evaluations` | Number of distinct local-search neighbor subsets generated in this generation; zero if local search did not run or found none. A generated neighbor can reuse a cached evaluation. |
| `stalled_generations` | Consecutive generations without a confirmed-hypervolume increase greater than `optimization.min_improvement`. |
| `converged` | `True` only on a generation that meets both `optimization.min_generations` and `optimization.patience` for the stalled-hypervolume stopping rule; otherwise `False`. |

`history_` is numerical search history, not a timing log. It can stop at
`max_generations`, because the search space is exhausted, after a safe
cancellation, or at convergence. The saved `history` export
contains the same recorded fields.

See [results.md](results.md) for the other saved artifacts and
[api-parameters.md](api-parameters.md) for input settings.
