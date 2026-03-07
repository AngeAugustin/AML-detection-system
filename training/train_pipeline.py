from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic_generator import SyntheticGenerationParams, generate_synthetic_dataset
from features.feature_engineering import build_feature_frame
from models.autoencoder import AutoEncoderScorer
from models.isolation_forest import IsolationForestScorer
from models.xgboost_model import XGBoostAMLModel
from utils.registry import ModelArtifacts, save_artifacts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AML detection models and save artifacts.")
    p.add_argument("--output", type=str, default="artifacts", help="Output directory for model artifacts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=2000)
    p.add_argument("--min-tx", type=int, default=20)
    p.add_argument("--max-tx", type=int, default=250)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    params = SyntheticGenerationParams(
        num_clients=int(args.num_clients),
        min_tx_per_client=int(args.min_tx),
        max_tx_per_client=int(args.max_tx),
        seed=int(args.seed),
    )

    clients, tx, labels = generate_synthetic_dataset(params=params)
    X, feature_names, _ = build_feature_frame(transactions=tx, clients=clients)

    y = labels.set_index("client_id").loc[X.index, "label_suspicious"].astype(int)

    # Train Isolation Forest on all clients (unsupervised)
    iso = IsolationForestScorer.fit(X, seed=args.seed)

    # Train XGBoost supervised model
    xgb = XGBoostAMLModel.fit(X, y=y, seed=args.seed)

    # Train AutoEncoder primarily on normal clients to learn baseline behavior
    X_normal = X[y == 0]
    if len(X_normal) < 50:
        X_normal = X  # fallback
    ae = AutoEncoderScorer.fit(X_normal, seed=args.seed)

    artifacts = ModelArtifacts(
        isolation_forest=iso,
        xgboost=xgb,
        autoencoder=ae,
        feature_names=feature_names,
    )
    save_artifacts(artifacts, output_dir=args.output)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Save a small metadata file for traceability
    (out / "train_summary.txt").write_text(
        "\n".join(
            [
                f"num_clients={len(clients)}",
                f"num_transactions={len(tx)}",
                f"features={len(feature_names)}",
                f"suspicious_rate={float(y.mean()):.4f}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved artifacts to: {out.resolve()}")


if __name__ == "__main__":
    main()

