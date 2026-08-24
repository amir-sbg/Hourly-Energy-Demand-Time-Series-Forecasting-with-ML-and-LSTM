# Time Series Forecasting Lab

End-to-end forecasting project that compares classical ML baselines with a PyTorch sequence model on a realistic hourly demand series.

The repo is intentionally practical: it covers data preparation, time-aware splitting, lag-window construction, baseline modeling, deep learning, evaluation, plots, and reproducible reports without turning into a giant framework.

## What the project does

The default run generates a synthetic hourly energy-demand dataset with daily seasonality, weekly seasonality, trend, temperature effects, weekend behavior, and noise. You can also pass your own CSV with at least:

```text
timestamp, load
```

If a `temperature` column exists, the pipeline uses it as an exogenous feature. If not, it still works using the target history and calendar features.

## Forecasting workflow

1. Load a CSV or generate the included hourly demand series.
2. Add calendar features: hour-of-day, day-of-week, and weekend indicators.
3. Split chronologically into train, validation, and test sets.
4. Fit scaling only on the training partition.
5. Build supervised lookback windows for multi-step forecasting.
6. Train/evaluate:
   - persistence baseline
   - seasonal naive baseline
   - Ridge regression on flattened lag windows
   - PyTorch LSTM direct multi-horizon forecaster
7. Save metrics, per-horizon errors, predictions, plots, and model artifacts.

## Models

The classical ML model is a direct multi-output Ridge regressor trained on flattened lookback windows. It gives a strong, fast baseline and makes it easy to compare against the sequence model.

The deep learning model is an LSTM encoder with a small MLP forecast head. It predicts the full horizon directly instead of recursively predicting one step at a time. Training uses AdamW, MSE loss, gradient clipping, validation tracking, and early stopping.

## Setup

```bash
git clone https://github.com/amir-sbg/Time-series-forecasting.git
cd Time-series-forecasting

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Run the default experiment

```bash
python -m ts_forecasting.pipeline
```

For a quick smoke run without the LSTM:

```bash
python -m ts_forecasting.pipeline \
  --periods 1000 \
  --lookback 48 \
  --horizon 12 \
  --skip-lstm
```

For a custom CSV:

```bash
python -m ts_forecasting.pipeline \
  --input-csv data/raw/my_series.csv \
  --lookback 168 \
  --horizon 24
```

## Main options

```bash
python -m ts_forecasting.pipeline \
  --lookback 168 \
  --horizon 24 \
  --epochs 12 \
  --batch-size 64 \
  --hidden-size 64 \
  --learning-rate 0.001 \
  --device auto
```

`lookback` controls how much history each model sees. `horizon` controls how many future time steps are predicted at once.

## Outputs

```text
reports/
├── run_summary.json
├── per_horizon_metrics.csv
├── predictions.csv
├── forecast_comparison.png
└── learning_curve.png

artifacts/
├── lstm_forecaster.pt
└── lstm_training_history.csv
```

`run_summary.json` contains MAE, RMSE, MAPE, SMAPE, and MASE for each model. `per_horizon_metrics.csv` shows how error changes from short-range to longer-range forecasts. `predictions.csv` keeps actual values and model forecasts in a flat format for review.

## Project structure

```text
src/ts_forecasting/
├── baselines.py   # persistence, seasonal naive, Ridge window model
├── config.py      # experiment configuration
├── data.py        # synthetic series, CSV loading, splits, windows
├── metrics.py     # forecasting metrics
├── models.py      # PyTorch LSTM forecaster and training loop
└── pipeline.py    # end-to-end experiment runner

tests/
├── test_baselines.py
├── test_data_metrics.py
├── test_models.py
└── test_pipeline.py
```

This is a small but complete forecasting pipeline: the code tests the data logic, model shapes, metrics, baselines, scaling behavior, and target-only CSV support.
