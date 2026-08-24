import numpy as np

from ts_forecasting.data import SplitFrames, generate_synthetic_demand
from ts_forecasting.pipeline import ScaleInfo, inverse_target, scale_splits


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
