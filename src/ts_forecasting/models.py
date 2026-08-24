from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_size < 1 or horizon < 1:
            raise ValueError("input_size and horizon must be positive")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, lookback, features)")
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1])


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.patience < 1 or self.gradient_clip <= 0:
            raise ValueError("patience and gradient_clip must be positive")


def make_torch_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if x_tensor.ndim != 3 or y_tensor.ndim != 2:
        raise ValueError("expected x=(samples, lookback, features), y=(samples, horizon)")
    if x_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def train_lstm_forecaster(
    model: LSTMForecaster,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TorchTrainConfig,
    device: torch.device,
) -> tuple[LSTMForecaster, list[dict[str, float]], dict[str, float | int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device, config.gradient_clip)
        validation_loss = _run_epoch(model, validation_loader, loss_fn, None, device, config.gradient_clip)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= config.patience:
            break

    model.load_state_dict(best_state)
    summary = {
        "epochs_requested": config.epochs,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    return model, history, summary


def predict_lstm(model: LSTMForecaster, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    loader = make_torch_loader(x, np.zeros((len(x), model.head[-1].out_features), dtype=np.float32), batch_size, False)
    model.eval()
    rows = []
    with torch.inference_mode():
        for features, _ in loader:
            rows.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradient_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = loss_fn(prediction, target)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        total_loss += loss.item() * len(features)
        total_items += len(features)
    return total_loss / max(total_items, 1)
