from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), epsilon))) * 100)


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(true) + np.abs(pred), epsilon)
    return float(np.mean(2 * np.abs(pred - true) / denominator) * 100)


def mase(y_true, y_pred, insample, seasonality: int = 1) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    insample = np.asarray(insample, dtype=np.float64).reshape(-1)
    if seasonality < 1:
        raise ValueError("seasonality must be positive")
    if len(insample) <= seasonality:
        raise ValueError("insample series is too short for the requested seasonality")
    scale = np.mean(np.abs(insample[seasonality:] - insample[:-seasonality]))
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(true - pred)) / scale)


def forecast_metrics(y_true, y_pred, insample=None, seasonality: int = 24) -> dict[str, float]:
    report = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if insample is not None:
        report["mase"] = mase(y_true, y_pred, insample, seasonality=seasonality)
    return report


def per_horizon_metrics(y_true, y_pred) -> list[dict[str, float | int]]:
    true, pred = _as_matching_arrays(y_true, y_pred)
    if true.ndim == 1:
        true = true[:, None]
        pred = pred[:, None]
    return [
        {
            "horizon": index + 1,
            "mae": mae(true[:, index], pred[:, index]),
            "rmse": rmse(true[:, index], pred[:, index]),
            "smape": smape(true[:, index], pred[:, index]),
        }
        for index in range(true.shape[1])
    ]


def _as_matching_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} != {pred.shape}")
    if true.size == 0:
        raise ValueError("forecast arrays must not be empty")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("forecast arrays must contain finite values")
    return true, pred
