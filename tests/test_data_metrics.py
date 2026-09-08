import numpy as np
import pandas as pd
import pytest

from ts_forecasting.data import (
    add_calendar_features,
    chronological_split,
    generate_synthetic_demand,
    make_supervised_windows,
    rolling_origin_folds,
    time_series_diagnostics,
)
from ts_forecasting.metrics import (
    forecast_metrics,
    mase,
    mean_error,
    per_horizon_metrics,
    rank_models,
    wape,
)


def test_synthetic_demand_is_reproducible() -> None:
    first = generate_synthetic_demand(periods=24 * 21, seed=4)
    second = generate_synthetic_demand(periods=24 * 21, seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert {"timestamp", "load", "temperature"}.issubset(first.columns)


def test_calendar_features_add_periodic_columns() -> None:
    frame = add_calendar_features(generate_synthetic_demand(periods=24 * 14))

    assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"}.issubset(frame.columns)
    assert frame["hour_sin"].between(-1, 1).all()


def test_time_series_diagnostics_report_gaps_and_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 01:00",
                    "2024-01-01 04:00",
                ]
            ),
            "load": [10.0, 11.0, np.nan, 13.0],
        }
    )

    report = time_series_diagnostics(frame, "timestamp", "load")

    assert report["rows"] == 4
    assert report["duplicate_timestamps"] == 1
    assert report["missing_target_rows"] == 1
    assert report["irregular_steps"] == 1


def test_chronological_split_preserves_order() -> None:
    frame = generate_synthetic_demand(periods=1000)
    split = chronological_split(frame, validation_size=0.2, test_size=0.2)

    assert len(split.train) == 600
    assert len(split.validation) == 200
    assert len(split.test) == 200
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_rolling_origin_folds_expand_training_window() -> None:
    folds = rolling_origin_folds(
        rows=100,
        initial_train_size=50,
        validation_size=10,
        step_size=20,
        max_folds=3,
    )

    assert [fold.fold for fold in folds] == [1, 2, 3]
    assert [fold.train_end for fold in folds] == [50, 70, 90]
    assert folds[0].validation_start == 50
    assert folds[-1].validation_end == 100


def test_supervised_windows_have_expected_shape() -> None:
    frame = add_calendar_features(generate_synthetic_demand(periods=24 * 20))
    columns = ["load", "temperature", "hour_sin", "hour_cos"]
    dataset = make_supervised_windows(
        frame,
        feature_columns=columns,
        target_column="load",
        lookback=24,
        horizon=6,
    )

    assert dataset.x.shape[1:] == (24, len(columns))
    assert dataset.y.shape[1] == 6
    assert dataset.target_index == 0


def test_metrics_report_forecast_errors() -> None:
    true = np.array([[10.0, 12.0], [14.0, 16.0]])
    pred = np.array([[11.0, 11.0], [13.0, 17.0]])
    report = forecast_metrics(true, pred, insample=np.arange(20.0), seasonality=1)

    assert report["mae"] == 1.0
    assert report["rmse"] == 1.0
    assert report["mean_error"] == 0.0
    assert report["wape"] == pytest.approx(400 / 52)
    assert report["mase"] == 1.0
    assert len(per_horizon_metrics(true, pred)) == 2


def test_forecast_bias_and_wape_are_scale_aware() -> None:
    true = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 21.0, 33.0])

    assert mean_error(true, pred) == pytest.approx(2.0)
    assert wape(true, pred) == pytest.approx(10.0)


def test_rank_models_orders_by_primary_metric() -> None:
    rows = rank_models(
        {
            "lstm": {"mae": 1.8, "rmse": 2.0, "wape": 4.0, "mean_error": -0.1},
            "ridge": {"mae": 1.2, "rmse": 1.6, "wape": 3.0, "mean_error": 0.2},
        }
    )

    assert rows[0]["model"] == "ridge"
    assert rows[0]["rank"] == 1
    assert rows[1]["rank_value"] == 1.8


def test_mase_rejects_short_insample() -> None:
    with pytest.raises(ValueError, match="too short"):
        mase([1.0], [1.0], insample=[1.0], seasonality=2)
