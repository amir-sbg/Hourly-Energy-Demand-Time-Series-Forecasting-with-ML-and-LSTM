from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mean_error(y_true, y_pred) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(pred - true))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), epsilon))) * 100)


def wape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true, pred = _as_matching_arrays(y_true, y_pred)
    denominator = max(float(np.sum(np.abs(true))), epsilon)
    return float(np.sum(np.abs(true - pred)) / denominator * 100)


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
        "mean_error": mean_error(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if insample is not None:
        report["mase"] = mase(y_true, y_pred, insample, seasonality=seasonality)
    return report


def rank_models(
    metric_report: dict[str, dict[str, float]],
    primary_metric: str = "mae",
) -> list[dict[str, float | int | str]]:
    rows = []
    for model_name, metrics in metric_report.items():
        if primary_metric not in metrics:
            raise ValueError(f"{model_name} is missing metric {primary_metric}")
        value = float(metrics[primary_metric])
        if not np.isfinite(value):
            value = float("inf")
        rows.append(
            {
                "model": model_name,
                "rank_metric": primary_metric,
                "rank_value": value,
                "mae": float(metrics.get("mae", np.nan)),
                "rmse": float(metrics.get("rmse", np.nan)),
                "wape": float(metrics.get("wape", np.nan)),
                "mean_error": float(metrics.get("mean_error", np.nan)),
            }
        )
    rows = sorted(rows, key=lambda row: (float(row["rank_value"]), str(row["model"])))
    return [{**row, "rank": index + 1} for index, row in enumerate(rows)]


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
            "mean_error": mean_error(true[:, index], pred[:, index]),
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
