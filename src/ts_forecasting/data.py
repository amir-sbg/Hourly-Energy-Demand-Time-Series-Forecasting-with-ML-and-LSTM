from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class WindowedDataset:
    x: np.ndarray
    y: np.ndarray
    target_index: int
    feature_columns: list[str]


def generate_synthetic_demand(
    periods: int = 24 * 180,
    frequency: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    if periods < 24 * 14:
        raise ValueError("periods should cover at least two weeks")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=periods, freq=frequency)
    step = np.arange(periods)
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    daily = 8.0 * np.sin(2 * np.pi * (hour - 7) / 24)
    evening_peak = 10.0 * np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    weekly = 5.0 * np.sin(2 * np.pi * day_of_week / 7)
    trend = 0.0025 * step
    weekend_discount = np.where(day_of_week >= 5, -6.0, 0.0)
    temperature = 18 + 8 * np.sin(2 * np.pi * step / (24 * 365)) + rng.normal(0, 2, periods)
    temperature_effect = 0.55 * np.maximum(temperature - 22, 0) + 0.25 * np.maximum(10 - temperature, 0)
    noise = rng.normal(0, 2.2, periods)
    load = 52 + daily + evening_peak + weekly + weekend_discount + trend + temperature_effect + noise

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": load.astype(np.float32),
            "temperature": temperature.astype(np.float32),
        }
    )


def load_time_series_csv(
    path: Path,
    timestamp_column: str = "timestamp",
    target_column: str = "load",
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_time_series_frame(frame, timestamp_column, target_column)
    frame = frame.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])
    return frame.sort_values(timestamp_column).reset_index(drop=True)


def validate_time_series_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> None:
    missing = {timestamp_column, target_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("time-series frame is empty")
    if not pd.api.types.is_numeric_dtype(frame[target_column]):
        raise TypeError(f"{target_column} must be numeric")


def add_calendar_features(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    values = frame.copy()
    timestamp = pd.to_datetime(values[timestamp_column])
    hour = timestamp.dt.hour.to_numpy()
    day_of_week = timestamp.dt.dayofweek.to_numpy()
    values["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    values["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    values["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    values["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    values["is_weekend"] = (day_of_week >= 5).astype(float)
    return values


def chronological_split(
    frame: pd.DataFrame,
    validation_size: float,
    test_size: float,
) -> SplitFrames:
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must sum to less than 1")

    n_rows = len(frame)
    test_rows = int(round(n_rows * test_size))
    validation_rows = int(round(n_rows * validation_size))
    train_rows = n_rows - validation_rows - test_rows
    if min(train_rows, validation_rows, test_rows) <= 0:
        raise ValueError("not enough rows for the requested split sizes")

    return SplitFrames(
        train=frame.iloc[:train_rows].reset_index(drop=True),
        validation=frame.iloc[train_rows : train_rows + validation_rows].reset_index(drop=True),
        test=frame.iloc[train_rows + validation_rows :].reset_index(drop=True),
    )


def make_supervised_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> WindowedDataset:
    if lookback < 1 or horizon < 1 or stride < 1:
        raise ValueError("lookback, horizon, and stride must be positive")
    missing = set(feature_columns + [target_column]).difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns for windowing: {sorted(missing)}")

    values = frame[feature_columns].to_numpy(dtype=np.float32)
    target = frame[target_column].to_numpy(dtype=np.float32)
    if len(frame) < lookback + horizon:
        raise ValueError("not enough rows to build one supervised window")

    x_rows = []
    y_rows = []
    for start in range(0, len(frame) - lookback - horizon + 1, stride):
        end = start + lookback
        x_rows.append(values[start:end])
        y_rows.append(target[end : end + horizon])

    return WindowedDataset(
        x=np.stack(x_rows),
        y=np.stack(y_rows),
        target_index=feature_columns.index(target_column),
        feature_columns=feature_columns,
    )
