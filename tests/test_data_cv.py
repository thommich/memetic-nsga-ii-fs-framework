from datetime import date

import numpy as np
import pandas as pd
import pytest

from memetic_fs.config import CVConfig
from memetic_fs.cv import Fold, build_folds, normalize_months
from memetic_fs.data import load_dataset, prepare_data, save_dataset


@pytest.fixture
def frame():
    n = 72
    return pd.DataFrame({"id": np.arange(n), "target": np.tile([0, 1], n // 2),
                         "month": np.repeat(np.arange(202601, 202607), 12),
                         "segment": np.tile(["a", "b"], n // 2),
                         "numeric": np.arange(n, dtype=np.float32),
                         "category": pd.Categorical(np.tile(["red", "blue", None], n // 3))})


def test_integer_datetime_and_python_dates():
    expected = normalize_months([202401, 202402, 202412])
    assert np.array_equal(expected, normalize_months(pd.to_datetime(["2024-01-03", "2024-02-20", "2024-12-31"])))
    assert np.array_equal(expected, normalize_months([date(2024, 1, 3), date(2024, 2, 20), date(2024, 12, 31)]))
    assert np.array_equal(expected, normalize_months(pd.to_datetime(["2024-01-03", "2024-02-20", "2024-12-31"]).tz_localize("UTC")))


@pytest.mark.parametrize("values", [[202400], [202413], [0], [202401.0], [None], ["2024-01-01"]])
def test_invalid_month(values):
    with pytest.raises(ValueError):
        normalize_months(values)


def test_default_folds_are_full_row_and_chronological(frame):
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    folds = build_folds(ds.months, ds.y, CVConfig())
    assert [len(f.train) for f in folds] == [36, 48, 60]
    assert [len(f.valid) for f in folds] == [12, 12, 12]
    assert folds[-1].label == "2026-06"
    for fold in folds:
        assert ds.months[fold.train].max() < ds.months[fold.valid].min()
        assert Fold.from_dict(fold.to_dict()).label == fold.label
        assert np.array_equal(Fold.from_dict(fold.to_dict()).valid, fold.valid)


def test_missing_calendar_months_count_observed_months(frame):
    frame["month"] = frame["month"].replace({202603: 202610, 202604: 202611, 202605: 202612, 202606: 202701})
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    folds = build_folds(ds.months, ds.y, CVConfig())
    assert folds[-1].label == "2027-01"
    assert len(folds[0].train) == 36


def test_minimum_history(frame):
    with pytest.raises(ValueError, match="six"):
        CVConfig(min_total_months=5)
    with pytest.raises(ValueError, match="three"):
        CVConfig(min_train_months=2)
    with pytest.raises(ValueError, match="distinct observed months"):
        build_folds(normalize_months(frame.month[:60]), frame.target[:60].to_numpy(), CVConfig())


def test_custom_protocol_and_validation(frame):
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    def valid(months):
        yield np.arange(36), np.arange(60, 72)
    result = build_folds(ds.months, ds.y, CVConfig(), valid)
    assert len(result) == 1
    bad = [
        (np.arange(36), np.arange(24, 72)),  # overlap/leakage
        (np.arange(24), np.arange(60, 72)),  # short train
        (np.arange(35), np.arange(60, 72)),  # omitted training row
        (np.arange(36), np.arange(60, 71)),  # omitted final validation row
        (np.arange(36), np.arange(48, 60)),  # absent last month
        (np.arange(36), np.array([72])),
        (np.arange(36), np.array([], dtype=int)),
        (np.arange(36), np.array([60., 61.])),
    ]
    for split in bad:
        with pytest.raises(ValueError):
            build_folds(ds.months, ds.y, CVConfig(), lambda months: [split])
    with pytest.raises(ValueError, match="no folds"):
        build_folds(ds.months, ds.y, CVConfig(), lambda months: [])


def test_long_horizon_tail_does_not_repeat_validation_rows():
    months = normalize_months(np.repeat(np.arange(202601, 202609), 4))
    y = np.tile([0, 1], 16)
    folds = build_folds(months, y, CVConfig(validation_months=2, step_months=2))
    assert months[folds[-1].valid].max() == months.max()
    all_valid = np.concatenate([f.valid for f in folds])
    assert len(all_valid) == len(np.unique(all_valid))


def test_repeated_ids_and_override(frame):
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="Repeated applicant"):
        prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"], allow_repeated_ids=True)
    assert ds.n_rows == len(frame)
    with pytest.raises(ValueError, match="uniqueness cannot be verified"):
        prepare_data(frame, "target", "month", segment_cols=["segment"])


@pytest.mark.parametrize("target", [[0, 1, 2], [0, 1, np.nan], [0, 0, 0]])
def test_binary_target_validation(target):
    with pytest.raises(ValueError, match="Target"):
        prepare_data(np.ones((3, 2)), target, "month", time_values=[202601, 202602, 202603], allow_repeated_ids=True)


def test_metadata_excluded_categories_and_storage(frame, tmp_path):
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    assert ds.feature_names == ["numeric", "category"]
    assert ds.categorical_indices == [1]
    assert ds.X.dtype == np.float32
    assert ds.categories == {"category": ["blue", "red"]}
    assert (ds.X[:, 1] == -1).sum() == 24
    path = save_dataset(ds, tmp_path / "data")
    recovered = load_dataset(path)
    assert isinstance(recovered.X, np.memmap)
    assert not recovered.X.flags.writeable
    assert not recovered.y.flags.writeable
    assert np.array_equal(recovered.X, ds.X)
    assert recovered.fingerprint == ds.fingerprint
    assert recovered.categories == ds.categories
    assert np.array_equal(recovered.segments["segment"], ds.segments["segment"])


def test_array_memmap_is_not_copied(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.arange(24, dtype=np.float32).reshape(12, 2))
    array = np.load(path, mmap_mode="r")
    ds = prepare_data(array, np.tile([0, 1], 6), "month", time_values=np.repeat(np.arange(202601, 202607), 2), ids=np.arange(12))
    assert ds.X is array


def test_hash_changes_with_data_metadata_and_category_order(frame):
    kwargs = dict(y="target", time_cv="month", id_col="id", segment_cols=["segment"])
    baseline = prepare_data(frame, **kwargs)
    changed = frame.copy()
    changed.loc[0, "numeric"] += 1
    assert prepare_data(changed, **kwargs).fingerprint != baseline.fingerprint
    changed = frame.copy()
    changed.loc[0, "segment"] = "c"
    assert prepare_data(changed, **kwargs).fingerprint != baseline.fingerprint
    changed = frame.copy()
    changed["category"] = changed.category.cat.reorder_categories(["red", "blue"])
    assert prepare_data(changed, **kwargs).fingerprint != baseline.fingerprint


def test_aligned_target_and_names(frame):
    with pytest.raises(ValueError, match="index"):
        prepare_data(frame, frame.target.iloc[::-1], "month", id_col="id", segment_cols=["segment"])
    duplicated = pd.DataFrame(np.ones((12, 2)), columns=["same", "same"])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_data(duplicated, np.tile([0, 1], 6), "month", allow_repeated_ids=True)
    with pytest.raises(ValueError, match="unique string"):
        prepare_data(np.ones((12, 2)), np.tile([0, 1], 6), "month", time_values=np.repeat(np.arange(202601, 202607), 2), allow_repeated_ids=True, feature_names=["a", "a"])


def test_callable_named_series_excludes_time_column(frame):
    ds = prepare_data(frame, "target", lambda months: [], time_values=frame.month, id_col="id", segment_cols=["segment"])
    assert "month" not in ds.feature_names
    with pytest.raises(ValueError, match="time_values"):
        prepare_data(frame, "target", lambda months: [], id_col="id", segment_cols=["segment"])


def test_unnamed_metadata_vectors_do_not_leak_original_columns(frame):
    frame["numeric"] += .25
    ds = prepare_data(frame, frame.target.to_numpy(), lambda months: [], time_values=frame.month.to_numpy(),
                      ids=frame.id.to_numpy(), segment_cols=["segment"])
    assert ds.feature_names == ["numeric", "category"]
    frame["almost_target"] = frame.target
    frame.loc[71, "almost_target"] = 1 - frame.loc[71, "almost_target"]
    ds = prepare_data(frame, frame.target.to_numpy(), lambda months: [], time_values=frame.month.to_numpy(),
                      ids=frame.id.to_numpy(), segment_cols=["segment"])
    assert "almost_target" in ds.feature_names  # full equality, not a sampled check


def test_object_segment_fingerprint_distinguishes_integer_and_string_values(frame):
    frame["segment"] = pd.Series([1] * len(frame), dtype=object)
    numeric_segment = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    frame["segment"] = pd.Series(["1"] * len(frame), dtype=object)
    string_segment = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    assert numeric_segment.fingerprint != string_segment.fingerprint


def test_nullable_integer_precision_and_float32_metadata(frame):
    frame = frame.drop(columns="category")
    frame["nullable"] = pd.Series([2**53 + 1] * len(frame), dtype="Int64")
    frame.loc[0, "nullable"] = pd.NA
    with pytest.raises(ValueError, match="integer precision"):
        prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    frame["nullable"] = pd.Series([1.25] * len(frame), dtype="Float32")
    frame.loc[0, "nullable"] = pd.NA
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    assert ds.X.dtype == np.float32
    assert np.isnan(ds.X[0, 1])


def test_mixed_signed_unsigned_precision(frame):
    frame = frame.drop(columns=["numeric", "category"])
    frame["unsigned"] = np.full(len(frame), 2**53 + 1, dtype=np.uint64)
    frame["signed"] = np.full(len(frame), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="integer precision"):
        prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    frame["unsigned"] = np.full(len(frame), 2**54, dtype=np.uint64)
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    assert ds.X[0, 0] == 2**54  # exactly representable large integer remains valid


def test_missing_segments_preserved_for_reporting(frame, tmp_path):
    frame.loc[0, "segment"] = None
    frame.loc[1, "segment"] = np.nan
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    recovered = load_dataset(save_dataset(ds, tmp_path / "data"))
    assert pd.isna(recovered.segments["segment"][:2]).all()


def test_mixed_segment_types_round_trip_without_coercing_labels(frame, tmp_path):
    frame["segment"] = pd.Series(np.tile([1, "unknown", None], len(frame) // 3), dtype=object)
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    recovered = load_dataset(save_dataset(ds, tmp_path / "data"))
    assert recovered.segments["segment"].tolist() == ds.segments["segment"].tolist()
    assert isinstance(recovered.segments["segment"][0], int)
    assert isinstance(recovered.segments["segment"][1], str)


def test_category_order_and_datetime_labels_preserved(frame, tmp_path):
    labels = pd.to_datetime(["2020-01-01", "2020-02-01"])
    frame["category"] = pd.Categorical(np.tile(labels, len(frame) // 2), categories=labels, ordered=True)
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    recovered = load_dataset(save_dataset(ds, tmp_path / "data"))
    assert recovered.categories == ds.categories
    assert recovered.category_ordered == {"category": True}


def test_preserves_double_precision_and_rejects_integer_precision_loss():
    n = 12
    values = pd.DataFrame({"x": np.full(n, 1.123456789), "large": np.full(n, 2**53 + 1, dtype=np.int64)})
    kwargs = dict(y=np.tile([0, 1], 6), time_cv="month", time_values=np.repeat(np.arange(202601, 202607), 2), allow_repeated_ids=True)
    with pytest.raises(ValueError, match="integer precision"):
        prepare_data(values, **kwargs)
    ds = prepare_data(values.drop(columns="large"), **kwargs)
    assert ds.X.dtype == np.float64
    assert ds.X[0, 0] == 1.123456789


def test_fold_rejects_single_class_validation(frame):
    frame.loc[frame.month == 202606, "target"] = 1
    ds = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    with pytest.raises(ValueError, match="both training and validation"):
        build_folds(ds.months, ds.y, CVConfig())
