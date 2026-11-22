"""Contract checks for configurable synthetic predictor families."""
import pandas as pd
import pytest

from memetic_fs.synthetic import make_synthetic_data


def test_requested_feature_counts_and_noise_remainder():
    options = dict(n_rows=240, n_months=6, n_features=12, random_state=9,
                   n_signal_features=2, n_redundant_features=3,
                   n_interaction_features=3, n_segmentation_specific_features=2)
    frame = make_synthetic_data(**options)
    predictors = frame.columns[:options["n_features"]].tolist()
    assert predictors == (["signal_0", "signal_1"]
                          + [f"redundant_{i}" for i in range(3)]
                          + [f"interaction_{i}" for i in range(3)]
                          + [f"segment_signal_{i}" for i in range(2)]
                          + ["noise_0", "noise_1"])
    pd.testing.assert_frame_equal(frame, make_synthetic_data(**options))
    assert frame["signal_0"].corr(frame["redundant_0"]) > .99
    assert frame.groupby("reference_month")["target"].nunique().eq(2).all()


def test_zero_counts_and_default_noise_remainder():
    frame = make_synthetic_data(n_rows=48, n_months=6, n_features=4,
                                n_signal_features=0, n_redundant_features=0,
                                n_interaction_features=0,
                                n_segmentation_specific_features=0)
    assert frame.columns[:4].tolist() == [f"noise_{i}" for i in range(4)]
    default = make_synthetic_data(n_rows=48, n_months=6, n_features=8)
    assert default.columns[:8].tolist()[-1] == "noise_0"


@pytest.mark.parametrize("counts", [
    dict(n_signal_features=3, n_redundant_features=1,
         n_interaction_features=2, n_segmentation_specific_features=2),
    dict(n_signal_features=4, n_redundant_features=1,
         n_interaction_features=2, n_segmentation_specific_features=2),
])
def test_feature_counts_must_leave_noise(counts):
    with pytest.raises(ValueError, match="less than n_features"):
        make_synthetic_data(n_rows=48, n_months=6, n_features=8, **counts)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_feature_counts_must_be_nonnegative_integers(value):
    with pytest.raises(ValueError, match="counts"):
        make_synthetic_data(n_signal_features=value)
