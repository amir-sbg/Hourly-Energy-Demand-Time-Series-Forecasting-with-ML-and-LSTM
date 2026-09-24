import numpy as np

from ts_forecasting.data import SplitFrames, generate_synthetic_demand
from ts_forecasting.pipeline import (
    ScaleInfo,
    inverse_target,
    model_skill_scores,
    scale_splits,
    select_feature_columns,
)


def test_scale_splits_fit_only_on_train_partition() -> None:
    train = generate_synthetic_demand(periods=24 * 14)
    validation = train.copy()
    validation["load"] = validation["load"] + 1000
    test = validation.copy()
    columns = ["load", "temperature"]

    scaled, scale = scale_splits(
        SplitFrames(train=train, validation=validation, test=test),
        columns,
    )

    assert abs(float(scaled.train["load"].mean())) < 1e-6
    assert abs(float(scaled.validation["load"].mean())) > 10
    assert scale.feature_columns == columns


def test_inverse_target_restores_scaled_values() -> None:
    scale = ScaleInfo(feature_columns=["load", "temperature"], mean=[50.0, 10.0], scale=[2.0, 5.0])
    restored = inverse_target(np.array([[0.0, 1.0]], dtype=np.float32), scale, "load")

    np.testing.assert_allclose(restored, np.array([[50.0, 52.0]], dtype=np.float32))


def test_feature_selection_allows_target_only_series() -> None:
    frame = generate_synthetic_demand(periods=24 * 14).drop(columns=["temperature"])
    frame["hour_sin"] = 0.0
    frame["hour_cos"] = 1.0
    frame["dow_sin"] = 0.0
    frame["dow_cos"] = 1.0
    frame["is_weekend"] = 0.0

    columns = select_feature_columns(frame, "load")

    assert columns == ["load", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def test_feature_selection_keeps_available_exogenous_columns() -> None:
    frame = generate_synthetic_demand(periods=24 * 14)
    frame["hour_sin"] = 0.0
    frame["hour_cos"] = 1.0
    frame["dow_sin"] = 0.0
    frame["dow_cos"] = 1.0
    frame["is_weekend"] = 0.0

    columns = select_feature_columns(frame, "load")

    assert "temperature" in columns


def test_model_skill_scores_use_seasonal_reference() -> None:
    y_true = np.array([[10.0, 12.0]])
    predictions = {
        "seasonal_naive": np.array([[8.0, 10.0]]),
        "ridge": np.array([[10.0, 13.0]]),
    }

    scores = model_skill_scores(y_true, predictions)

    assert scores["seasonal_naive"]["skill_vs_seasonal_naive_mae"] == 0.0
    assert scores["ridge"]["skill_vs_seasonal_naive_mae"] > 0.0
