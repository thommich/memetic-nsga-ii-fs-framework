"""Reproducible development-only data for examples, tests and scale benchmarks."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic_data(n_rows: int = 2400, n_features: int = 20,
                        n_months: int = 12, random_state: int = 42, *,
                        n_signal_features: int = 3, n_redundant_features: int = 1,
                        n_interaction_features: int = 2,
                        n_segmentation_specific_features: int = 1) -> pd.DataFrame:
    """Construct temporal binary risk data with configurable predictor types.

    Signal features affect the target directly (the third has temporal drift).
    Redundant features are correlated copies of signals; when there are no
    signals, they share an informative latent source. Interaction features act
    in pairs, with an unpaired final feature interacting with time. Segmentation
    features act within one customer type. All remaining predictors are noise.
    Metadata consists of ``target``, ``reference_month`` (YYYYMM), unique
    ``applicant_id``, ``customer_type`` and ``age_group``. No independent OOT
    data is generated. At least eight rows per observed month are required.
    The generator does not promise an exactly recoverable feature subset.
    """
    counts = (n_signal_features, n_redundant_features, n_interaction_features,
              n_segmentation_specific_features)
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (n_rows, n_features, n_months, *counts)):
        raise ValueError("Synthetic data dimensions and feature counts must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("Feature-type counts must be nonnegative")
    if n_months < 6 or n_rows < 8*n_months:
        raise ValueError("Require >=6 months and >=8 rows per month")
    if sum(counts) >= n_features:
        raise ValueError("Signal + redundant + interaction + segmentation-specific feature counts must be less than n_features, leaving at least one noise feature")
    rng = np.random.default_rng(random_state)
    months = np.minimum(np.arange(n_rows)*n_months//n_rows, n_months-1)
    matrix = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    n_signal, n_redundant, n_interaction, n_segment = counts
    redundant_start = n_signal
    interaction_start = redundant_start + n_redundant
    segment_start = interaction_start + n_interaction
    names = ([f"signal_{i}" for i in range(n_signal)]
             + [f"redundant_{i}" for i in range(n_redundant)]
             + [f"interaction_{i}" for i in range(n_interaction)]
             + [f"segment_signal_{i}" for i in range(n_segment)]
             + [f"noise_{i}" for i in range(n_features-segment_start-n_segment)])
    for i in range(n_redundant):
        source = matrix[:, i % n_signal] if n_signal else matrix[:, redundant_start]
        if n_signal or i:
            matrix[:, redundant_start + i] = source + rng.normal(scale=.06, size=n_rows).astype(np.float32)
    prospect = np.zeros(n_rows, dtype=bool)
    for month in range(n_months):
        positions = np.flatnonzero(months == month)
        prospect[rng.choice(positions, max(2, len(positions)//8), replace=False)] = True
    trend = 2*months/max(1, n_months-1) - 1
    logits = np.full(n_rows, -.45)
    for i in range(n_signal):
        strength = 2.8 if i == 0 else (1.2 if i == 1 else .7 / (i-1))
        logits += strength * matrix[:, i] * (trend if i == 2 else 1)
    if not n_signal and n_redundant:
        logits += 1.2 * matrix[:, redundant_start]
    for i in range(0, n_interaction, 2):
        left = matrix[:, interaction_start + i]
        right = matrix[:, interaction_start + i + 1] if i + 1 < n_interaction else trend
        logits += 1.2 * left * right
    for i in range(n_segment):
        group = prospect if i % 2 == 0 else ~prospect
        logits += (2.0 / (i + 1)) * matrix[:, segment_start + i] * group
    probability = 1/(1+np.exp(-np.clip(logits, -30, 30)))
    target = rng.binomial(1, probability).astype(np.int8)
    # Avoid a classless demonstration fold on tiny examples while preserving
    # the Bernoulli mechanism; this is synthetic generation, never fit sampling.
    for month in range(n_months):
        positions = np.flatnonzero(months == month)
        while len(np.unique(target[positions])) != 2:
            target[positions] = rng.binomial(1, probability[positions])
    frame = pd.DataFrame(matrix, columns=names)
    periods = pd.period_range("2023-01", periods=n_months, freq="M")
    calendar = np.array([period.year*100+period.month for period in periods])
    frame["target"] = target
    frame["reference_month"] = calendar[months]
    frame["applicant_id"] = np.arange(n_rows, dtype=np.int64)
    frame["customer_type"] = np.where(prospect, "prospect", "client")
    frame["age_group"] = rng.choice(["18-30", "31-45", "46-60", "60+"], n_rows)
    return frame
