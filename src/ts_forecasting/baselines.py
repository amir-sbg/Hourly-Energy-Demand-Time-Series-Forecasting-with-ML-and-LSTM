from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeForecaster:
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must not be negative")
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        features = flatten_windows(x)
        targets = _as_2d_targets(y)
        self.model.fit(features, targets)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(flatten_windows(x))
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        return predictions


def flatten_windows(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("x must have shape (samples, lookback, features)")
    return values.reshape(values.shape[0], values.shape[1] * values.shape[2])


def seasonal_naive_from_windows(
    x: np.ndarray,
    target_index: int,
    horizon: int,
    seasonality: int = 24,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("x must have shape (samples, lookback, features)")
    if horizon < 1 or seasonality < 1:
        raise ValueError("horizon and seasonality must be positive")
    if not 0 <= target_index < values.shape[2]:
        raise ValueError("target_index is out of bounds")

    target_history = values[:, :, target_index]
    lookback = target_history.shape[1]
    predictions = []
    for step in range(horizon):
        offset = lookback - seasonality + (step % seasonality)
        if offset < 0 or offset >= lookback:
            offset = lookback - 1
        predictions.append(target_history[:, offset])
    return np.stack(predictions, axis=1)


def persistence_from_windows(
    x: np.ndarray,
    target_index: int,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("x must have shape (samples, lookback, features)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    last_value = values[:, -1, target_index]
    return np.repeat(last_value[:, None], horizon, axis=1)


def moving_average_from_windows(
    x: np.ndarray,
    target_index: int,
    horizon: int,
    window_size: int = 24,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("x must have shape (samples, lookback, features)")
    if horizon < 1 or window_size < 1:
        raise ValueError("horizon and window_size must be positive")
    if not 0 <= target_index < values.shape[2]:
        raise ValueError("target_index is out of bounds")

    target_history = values[:, :, target_index]
    window_size = min(window_size, target_history.shape[1])
    averages = target_history[:, -window_size:].mean(axis=1)
    return np.repeat(averages[:, None], horizon, axis=1)


def _as_2d_targets(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("y must have shape (samples, horizon)")
    return values
