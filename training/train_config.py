"""
Training configuration persistence for reproducibility and audit.

Every training run saves a train_config.json with dataset choice,
hyperparameters, and split info so runs are fully reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.io import load_json, save_json


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    seed: int
    num_clients: int
    num_transactions: int
    test_size: float
    train_clients: int
    test_clients: int | None
    feature_count: int
    suspicious_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            dataset=str(d["dataset"]),
            seed=int(d["seed"]),
            num_clients=int(d["num_clients"]),
            num_transactions=int(d["num_transactions"]),
            test_size=float(d["test_size"]),
            train_clients=int(d["train_clients"]),
            test_clients=int(d["test_clients"]) if d.get("test_clients") is not None else None,
            feature_count=int(d["feature_count"]),
            suspicious_rate=float(d["suspicious_rate"]),
        )


def save_train_config(config: TrainConfig, output_dir: str | Path) -> None:
    path = Path(output_dir) / "train_config.json"
    save_json(config.to_dict(), path)


def load_train_config(output_dir: str | Path) -> TrainConfig | None:
    path = Path(output_dir) / "train_config.json"
    if not path.exists():
        return None
    return TrainConfig.from_dict(load_json(path))
