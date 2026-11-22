# Temporal evaluation and interpretation

Dates and integer YYYYMM values normalize to monthly pandas Period ordinals.
Integers must encode valid months 01–12. Six distinct observed months are
required. Expanding auto folds start with at least three observed months;
validation follows training strictly. Missing calendar months are skipped:
January/March/June are three observed months, not six contiguous months.
Training uses every row through the last training month. Validation includes
every row in its interval. The final interval always ends in the maximum
observed development month. A final window may shift to end there when the
configured step otherwise misses the tail; overlapping windows are rejected.

Custom protocol:

```python
def folds(months):
    # months is an aligned ndarray of normalized monthly ordinals
    observed = sorted(set(months))
    import numpy as np
    for month in observed[3:]:
        yield np.flatnonzero(months < month), np.flatnonzero(months == month)

# selector = MemeticNSGA2Selector(time_cv=folds, ...)
# selector.fit(df, 'target', time_values=df['reference_month'], ...)
```

Indices are row positions, not DataFrame labels. Empty, duplicate, out-of-range,
partial-month, nonchronological, overlapping or leaking indices raise errors.
Custom training windows may roll, but each interval must be complete and contain
at least three observed months. Both training and validation must have classes
0 and 1. Normalized folds persist before optimization; resume never reexecutes
a custom callable.

Each fold's AUC uses all validation rows. Gini is `2*AUC-1`, and the primary
discrimination objective is the unweighted arithmetic mean of fold Ginis.
Fold standard deviation, minimum, AUC and pooled out-of-fold Gini are diagnostics.
Pooling predictions from different fitted fold models can change rankings and
does not replace the fold-mean objective. Validation windows do not overlap.

Reduced evaluations select fewer entire folds, always including the final fold.
They have distinct cache keys and provisional archives. Parents are reevaluated
on fidelity changes. Provisional front members and final surviving candidates
are confirmed using the complete canonical folds before final dominance or
convergence comparisons. The two minimization objectives are negative mean
Gini and feature count. Temporal stability is diagnostic only.

Sampling prevention covers canonical LightGBM settings and their aliases:
bagging fractions one, bagging frequency zero, GOSS disabled, feature fractions
one, fixed CPU GBDT boosting. `bin_construct_sample_cnt` equals each training
fold's entire row count. This avoids LightGBM's usual 200,000-row bin-estimation
sample and can materially increase construction time and private memory. See
the official [parameter reference](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

Segment hard constraints compare the candidate against a full-feature model
trained with the same settings on the same folds. Constraint violations use
the maximum segment Gini drop across folds. Small segments with insufficient
class support are disabled once at initialization and documented in metadata.
Descriptive metrics retain event/non-event counts and undefined scores as null.

The confirmed frontier approximates a search result, not a theoretical optimum.
Tolerance selection is a transparent business rule. Knee selection is a
normalized geometric heuristic. Statistical selection resamples adjacent fold
pairs as temporal blocks and bounds the paired mean loss; with only three
folds its uncertainty is especially unreliable. Lack of significance does not
establish equivalence. Adaptive reuse of validation outcomes produces selection
optimism. Evaluate generalization on independent OOT data outside this package.
