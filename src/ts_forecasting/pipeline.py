from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .baselines import (
    RidgeForecaster,
    drift_from_windows,
    moving_average_from_windows,
    persistence_from_windows,
    seasonal_naive_from_windows,
)
from .config import ForecastConfig
from .data import (
    SplitFrames,
    add_calendar_features,
    chronological_split,
    generate_synthetic_demand,
    load_time_series_csv,
    make_supervised_windows,
    rolling_origin_folds,
    time_series_diagnostics,
)
from .metrics import forecast_metrics, per_horizon_metrics, rank_models, skill_score
from .models import (
    LSTMForecaster,
    TorchTrainConfig,
    make_torch_loader,
    predict_lstm,
    train_lstm_forecaster,
)


@dataclass(frozen=True)
class ScaleInfo:
    feature_columns: list[str]
    mean: list[float]
    scale: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def scale_splits(
    splits: SplitFrames,
    feature_columns: list[str],
) -> tuple[SplitFrames, ScaleInfo]:
    scaler = StandardScaler()
    scaler.fit(splits.train[feature_columns])

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        scaled = frame.copy()
        scaled[feature_columns] = scaler.transform(frame[feature_columns]).astype(np.float32)
        return scaled

    return (
        SplitFrames(
            train=transform(splits.train),
            validation=transform(splits.validation),
            test=transform(splits.test),
        ),
        ScaleInfo(
            feature_columns=list(feature_columns),
            mean=scaler.mean_.astype(float).tolist(),
            scale=scaler.scale_.astype(float).tolist(),
        ),
    )


def inverse_target(
    values: np.ndarray,
    scale: ScaleInfo,
    target_column: str,
) -> np.ndarray:
    if target_column not in scale.feature_columns:
        raise ValueError(f"{target_column} was not scaled")
    target_index = scale.feature_columns.index(target_column)
    return np.asarray(values, dtype=np.float32) * scale.scale[target_index] + scale.mean[target_index]


def select_feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    calendar_columns = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    optional_exogenous = [
        column
        for column in ("temperature",)
        if column in frame.columns and column != target_column
    ]
    columns = [target_column, *optional_exogenous, *calendar_columns]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns after feature engineering: {missing}")
    return columns


def model_skill_scores(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_name: str = "seasonal_naive",
) -> dict[str, dict[str, float]]:
    if baseline_name not in predictions:
        raise ValueError(f"missing baseline predictions: {baseline_name}")
    baseline = predictions[baseline_name]
    return {
        name: {
            f"skill_vs_{baseline_name}_mae": skill_score(
                y_true,
                values,
                baseline,
                metric="mae",
            ),
            f"skill_vs_{baseline_name}_rmse": skill_score(
                y_true,
                values,
                baseline,
                metric="rmse",
            ),
        }
        for name, values in predictions.items()
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    config = ForecastConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        validation_size=args.validation_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        patience=args.patience,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
    )
    set_seed(config.seed)
    device = choose_device(args.device)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv is not None:
        raw = load_time_series_csv(args.input_csv, config.timestamp_column, config.target_column)
    else:
        raw = generate_synthetic_demand(periods=args.periods, frequency=config.frequency, seed=config.seed)
    data_diagnostics = time_series_diagnostics(
        raw,
        config.timestamp_column,
        config.target_column,
    )
    frame = add_calendar_features(raw, config.timestamp_column)
    feature_columns = select_feature_columns(frame, config.target_column)
    splits = chronological_split(frame, config.validation_size, config.test_size)
    backtest_folds = rolling_origin_folds(
        rows=len(frame),
        initial_train_size=len(splits.train),
        validation_size=len(splits.validation),
        step_size=max(len(splits.validation), 1),
        max_folds=3,
    )
    scaled_splits, scale_info = scale_splits(splits, feature_columns)

    train_windows = make_supervised_windows(
        scaled_splits.train,
        feature_columns,
        config.target_column,
        config.lookback,
        config.horizon,
    )
    validation_windows = make_supervised_windows(
        scaled_splits.validation,
        feature_columns,
        config.target_column,
        config.lookback,
        config.horizon,
    )
    test_windows = make_supervised_windows(
        scaled_splits.test,
        feature_columns,
        config.target_column,
        config.lookback,
        config.horizon,
    )

    y_test = inverse_target(test_windows.y, scale_info, config.target_column)
    train_insample = splits.train[config.target_column].to_numpy(dtype=np.float32)
    predictions: dict[str, np.ndarray] = {}

    predictions["persistence"] = inverse_target(
        persistence_from_windows(test_windows.x, test_windows.target_index, config.horizon),
        scale_info,
        config.target_column,
    )
    predictions["drift"] = inverse_target(
        drift_from_windows(test_windows.x, test_windows.target_index, config.horizon),
        scale_info,
        config.target_column,
    )
    predictions["moving_average"] = inverse_target(
        moving_average_from_windows(
            test_windows.x,
            test_windows.target_index,
            config.horizon,
            window_size=args.moving_average_window,
        ),
        scale_info,
        config.target_column,
    )
    predictions["seasonal_naive"] = inverse_target(
        seasonal_naive_from_windows(test_windows.x, test_windows.target_index, config.horizon, seasonality=24),
        scale_info,
        config.target_column,
    )

    ridge = RidgeForecaster(alpha=args.ridge_alpha).fit(train_windows.x, train_windows.y)
    predictions["ridge"] = inverse_target(
        ridge.predict(test_windows.x),
        scale_info,
        config.target_column,
    )

    history = []
    training_summary = None
    if not args.skip_lstm:
        train_loader = make_torch_loader(
            train_windows.x,
            train_windows.y,
            config.batch_size,
            shuffle=True,
        )
        validation_loader = make_torch_loader(
            validation_windows.x,
            validation_windows.y,
            config.batch_size,
            shuffle=False,
        )
        model = LSTMForecaster(
            input_size=len(feature_columns),
            horizon=config.horizon,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        model, history, training_summary = train_lstm_forecaster(
            model,
            train_loader,
            validation_loader,
            TorchTrainConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                patience=config.patience,
            ),
            device=device,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "scale": scale_info.__dict__,
                "config": config.to_dict(),
            },
            config.artifact_dir / "lstm_forecaster.pt",
        )
        predictions["lstm"] = inverse_target(
            predict_lstm(model, test_windows.x, device, batch_size=config.batch_size),
            scale_info,
            config.target_column,
        )

    metrics = {
        name: forecast_metrics(y_test, values, insample=train_insample, seasonality=24)
        for name, values in predictions.items()
    }
    skill_scores = model_skill_scores(y_test, predictions, baseline_name="seasonal_naive")
    model_ranking = rank_models(metrics, primary_metric="mae")
    horizon_rows = []
    for name, values in predictions.items():
        for row in per_horizon_metrics(y_test, values):
            horizon_rows.append({"model": name, **row})

    _save_json(
        {
            "config": config.to_dict(),
            "device": str(device),
            "rows": {
                "raw": len(raw),
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
            "data_diagnostics": data_diagnostics,
            "rolling_origin_folds": [fold.__dict__ for fold in backtest_folds],
            "scale": scale_info.__dict__,
            "metrics": metrics,
            "skill_scores": skill_scores,
            "model_ranking": model_ranking,
            "best_model": model_ranking[0]["model"],
            "training_summary": training_summary,
        },
        config.report_dir / "run_summary.json",
    )
    pd.DataFrame(model_ranking).to_csv(config.report_dir / "model_ranking.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(config.report_dir / "per_horizon_metrics.csv", index=False)
    _save_predictions(y_test, predictions, config.report_dir / "predictions.csv")
    if history:
        pd.DataFrame(history).to_csv(config.artifact_dir / "lstm_training_history.csv", index=False)
        _plot_learning_curve(history, config.report_dir / "learning_curve.png")
    _plot_forecasts(y_test, predictions, config.report_dir / "forecast_comparison.png")
    return {
        "metrics": metrics,
        "model_ranking": model_ranking,
        "training_summary": training_summary,
    }


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _save_predictions(y_true: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    rows = []
    for window_index in range(y_true.shape[0]):
        for horizon_index in range(y_true.shape[1]):
            row = {
                "window": window_index,
                "horizon": horizon_index + 1,
                "actual": float(y_true[window_index, horizon_index]),
            }
            for name, values in predictions.items():
                row[name] = float(values[window_index, horizon_index])
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_learning_curve(history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frame["epoch"], frame["train_loss"], label="train")
    axis.plot(frame["epoch"], frame["validation_loss"], label="validation")
    axis.set_title("LSTM training history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_forecasts(y_true: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    steps = min(160, y_true.shape[0])
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(y_true[:steps, 0], label="actual", linewidth=2)
    for name, values in predictions.items():
        axis.plot(values[:steps, 0], label=name, alpha=0.85)
    axis.set_title("One-step-ahead forecast comparison")
    axis.set_xlabel("Test window")
    axis.set_ylabel("Load")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the time-series forecasting lab.")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--periods", type=int, default=24 * 180)
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--moving-average-window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
