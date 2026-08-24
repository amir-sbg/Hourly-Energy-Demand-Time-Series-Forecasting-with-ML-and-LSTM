from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    timestamp_column: str = "timestamp"
    target_column: str = "load"
    frequency: str = "h"
    lookback: int = 168
    horizon: int = 24
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.10
    patience: int = 5

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")

    def __post_init__(self) -> None:
        if not self.timestamp_column or not self.target_column:
            raise ValueError("timestamp_column and target_column must not be empty")
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not 0 < self.validation_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("validation_size and test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 0.8:
            raise ValueError("validation_size + test_size leaves too little training data")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.patience < 1:
            raise ValueError("patience must be positive")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["artifact_dir"] = str(self.artifact_dir)
        payload["report_dir"] = str(self.report_dir)
        return payload
