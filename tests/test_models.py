import numpy as np
import torch

from ts_forecasting.models import (
    LSTMForecaster,
    TorchTrainConfig,
    make_torch_loader,
    predict_lstm,
    train_lstm_forecaster,
)


def test_lstm_forecaster_outputs_full_horizon() -> None:
    model = LSTMForecaster(input_size=3, horizon=5, hidden_size=8)
    output = model(torch.zeros(4, 12, 3))

    assert output.shape == (4, 5)


def test_torch_loader_rejects_sample_mismatch() -> None:
    x = np.zeros((3, 4, 2), dtype=np.float32)
    y = np.zeros((4, 2), dtype=np.float32)

    try:
        make_torch_loader(x, y, batch_size=2, shuffle=False)
    except ValueError as error:
        assert "same number" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_lstm_training_smoke() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(24, 6, 2)).astype(np.float32)
    y = x[:, -1, :1] + 0.1
    train_loader = make_torch_loader(x[:18], y[:18], batch_size=6, shuffle=True)
    validation_loader = make_torch_loader(x[18:], y[18:], batch_size=6, shuffle=False)
    model = LSTMForecaster(input_size=2, horizon=1, hidden_size=8)

    trained, history, summary = train_lstm_forecaster(
        model,
        train_loader,
        validation_loader,
        TorchTrainConfig(epochs=2, batch_size=6, patience=2),
        torch.device("cpu"),
    )
    predictions = predict_lstm(trained, x[:3], torch.device("cpu"), batch_size=2)

    assert trained is model
    assert len(history) == 2
    assert summary["best_epoch"] >= 1
    assert predictions.shape == (3, 1)
