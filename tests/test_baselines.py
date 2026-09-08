import numpy as np
import pytest

from ts_forecasting.baselines import (
    RidgeForecaster,
    flatten_windows,
    moving_average_from_windows,
    persistence_from_windows,
    seasonal_naive_from_windows,
)


def toy_windows() -> np.ndarray:
    values = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    return values


def test_flatten_windows_preserves_sample_count() -> None:
    flat = flatten_windows(toy_windows())

    assert flat.shape == (2, 12)


def test_seasonal_naive_uses_recent_seasonal_values() -> None:
    predictions = seasonal_naive_from_windows(
        toy_windows(),
        target_index=0,
        horizon=3,
        seasonality=2,
    )

    np.testing.assert_array_equal(predictions[0], np.array([8, 10, 8], dtype=np.float32))


def test_persistence_repeats_last_target_value() -> None:
    predictions = persistence_from_windows(toy_windows(), target_index=0, horizon=4)

    np.testing.assert_array_equal(predictions[:, 0], toy_windows()[:, -1, 0])
    assert predictions.shape == (2, 4)


def test_moving_average_repeats_recent_mean() -> None:
    predictions = moving_average_from_windows(
        toy_windows(),
        target_index=0,
        horizon=3,
        window_size=3,
    )

    np.testing.assert_array_equal(
        predictions[0],
        np.array([(6 + 8 + 10) / 3] * 3, dtype=np.float32),
    )


def test_ridge_forecaster_learns_simple_mapping() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 4, 2)).astype(np.float32)
    y = x[:, -1, 0:1] * 2.0
    model = RidgeForecaster(alpha=0.0).fit(x, y)

    predictions = model.predict(x)

    assert predictions.shape == y.shape
    assert np.mean(np.abs(predictions - y)) < 1e-4


def test_baselines_reject_bad_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        flatten_windows(np.zeros((3, 4), dtype=np.float32))
