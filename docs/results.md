# Results and diagnostic artifacts

For the columns and row meanings of `temporal_performance_`,
`segment_performance_`, `selection_frequency_`, and `history_`, see the
[selector output table reference](output-tables.md).

Every completed experiment writes `results.json`, a compact, versioned metrics
record, and `report.html`, a self-contained interactive Plotly report that works
offline. The report consumes metrics only and can be rebuilt with
`memetic-fs report --experiment-dir RUN_DIRECTORY` without loading training data.

`selected_features.json` contains the final selected names. `pareto.json`
contains every confirmed Pareto subset, its mask, names, fold metrics, segment
metrics and constraint violations. `feature_masks.json` includes provisional and
confirmed evaluations with explicit fidelity tags. `evaluations`,
`confirmed_masks`, `fold_performance`, `segment_performance`,
`selection_frequency`, `high_performing_frequency`, `history` and
`rfe_trajectory` and `segment_constraints` each use the selected `output_format`
(`parquet` by default; `csv` and `json` are also supported). Each table is written
once. JSON table files contain an array of row objects. Feature-mask table columns
use the `mask::` prefix. The caller's experiment configuration and checkpoint
are managed by the experiment API.

The report includes the observed Pareto frontier, all confirmed and provisional
evaluations, a separately marked full-feature reference, best discovered Gini
by cardinality with RFE initialization when used, inclusion frequencies,
co-selection, temporal validation, segment comparisons and evolution history.
The co-selection map is limited to the 30 most frequently selected features.
Plotly supports zoom, hover, filtering through legend clicks, and image export.
The segment chart averages defined fold-level metrics equally; complete support
counts and undefined metrics remain available in the exported segment tables.
Hard constraints apply to **each relevant fold**, so a segment mean alone cannot
establish feasibility. Any plotted mean constraint floor is descriptive. The
report includes a per-fold constraint table that highlights violations, showing
selected, best and reference comparisons plus up to 500 additional violations.
The complete constraint-comparison table includes support counts and fidelity.

`selection_frequency` describes distinct subsets across the confirmed frontier.
`high_performing_frequency` restricts that family to models within the configured
reporting `gini_tolerance` (default 0.005 absolute Gini) of its best candidate.
This threshold does not alter the search objectives. The experiment API passes
the final-selection tolerance through to reporting. When feature-to-family
metadata is supplied, `family_frequency` reports how often at least one member
of a family is present.

Use `memetic_fs.reporting.aggregate_run_frequencies([run_1, run_2, ...])` to
combine independent runs with identical ordered candidate names. `frequency`
pools all distinct Pareto subsets within each run, preserving independent-run
contributions even when masks match. `run_frequency` counts runs where the
feature appears in at least one confirmed Pareto subset.

Sparse search trajectories are evidence about discovered models, not the true
optimal frontier. Inclusion frequencies and co-selection are descriptive, not
tests of causal relevance or statistical significance. Repeated feature search
introduces optimism. Keep final OOT assessment outside this package.
