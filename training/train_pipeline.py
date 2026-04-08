"""
AML model training pipeline.

- Recommended: --dataset synthetic (aligns with feature engineering; reproducible).
- Optional: --dataset creditcard for benchmark (pseudo-clients; see README).
- Validates all inputs; persists train_config.json for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.dataset_loader import load_dataset
from features.feature_engineering import build_feature_frame
from models.autoencoder import AutoEncoderScorer
from models.isolation_forest import IsolationForestScorer
from models.xgboost_model import XGBoostAMLModel
from training.train_config import TrainConfig, save_train_config
from utils.registry import ModelArtifacts, save_artifacts
from utils.validation import (
    safe_output_path,
    validate_positive_int,
    validate_seed,
    validate_test_size,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AML detection models and save artifacts. Use synthetic for optimal results."
    )
    p.add_argument("--output", type=str, default="artifacts", help="Output directory for model artifacts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=("synthetic", "creditcard"),
        help="Dataset: synthetic (recommended) or creditcard (benchmark).",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV (required when --dataset=creditcard).",
    )
    p.add_argument(
        "--max-transactions",
        type=int,
        default=None,
        help="Cap transactions for creditcard (default: use all).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test set (default 0.2). 0 = no split.",
    )
    p.add_argument("--num-clients", type=int, default=2000, help="For synthetic only.")
    p.add_argument("--min-tx", type=int, default=20, help="For synthetic only.")
    p.add_argument("--max-tx", type=int, default=250, help="For synthetic only.")
    return p.parse_args()


def _compute_test_metrics(y_test: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Compute accuracy, F1, ROC-AUC; handle single-class safely."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = (proba >= threshold).astype(int)
    lines = [
        f"  accuracy={accuracy_score(y_test, y_pred):.4f}",
        f"  f1={f1_score(y_test, y_pred, zero_division=0):.4f}",
    ]
    try:
        auc = roc_auc_score(y_test, proba)
        lines.append(f"  roc_auc={auc:.4f}")
    except ValueError:
        # Single class in y_test
        lines.append("  roc_auc=N/A (single class in test set)")
    return lines


def main() -> None:
    args = _parse_args()

    # Input validation
    try:
        out_dir = safe_output_path(args.output)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    validate_seed(args.seed)
    test_size = validate_test_size(args.test_size)
    if args.dataset == "creditcard":
        if not args.data_path:
            raise SystemExit("--dataset=creditcard requires --data-path=<path to CSV>")
        if args.max_transactions is not None:
            validate_positive_int(args.max_transactions, "max_transactions", max_val=10_000_000)
    else:
        validate_positive_int(args.num_clients, "num_clients", max_val=500_000)
        validate_positive_int(args.min_tx, "min_tx", max_val=10_000)
        validate_positive_int(args.max_tx, "max_tx", max_val=100_000)
        if args.min_tx > args.max_tx:
            raise SystemExit("--min-tx must be <= --max-tx")

    clients, tx, labels = load_dataset(
        dataset=args.dataset,
        num_clients=args.num_clients,
        min_tx_per_client=args.min_tx,
        max_tx_per_client=args.max_tx,
        seed=args.seed,
        data_path=args.data_path,
        max_transactions=args.max_transactions,
    )

    X, feature_names, _ = build_feature_frame(transactions=tx, clients=clients)
    y = labels.set_index("client_id").loc[X.index, "label_suspicious"].astype(int)

    use_split = args.test_size > 0 and len(X) >= 100
    if use_split:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, stratify=y, random_state=args.seed
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed
            )
        n_train, n_test = len(X_train), len(X_test)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        n_train, n_test = len(X), 0

    iso = IsolationForestScorer.fit(X_train, seed=args.seed)
    xgb = XGBoostAMLModel.fit(X_train, y=y_train, seed=args.seed)
    X_normal = X_train[y_train == 0]
    if len(X_normal) < 50:
        X_normal = X_train
    ae = AutoEncoderScorer.fit(X_normal, seed=args.seed)

    artifacts = ModelArtifacts(
        isolation_forest=iso,
        xgboost=xgb,
        autoencoder=ae,
        feature_names=feature_names,
    )
    save_artifacts(artifacts, output_dir=out_dir)

    summary_lines = [
        f"dataset={args.dataset}",
        f"num_clients={len(clients)}",
        f"num_transactions={len(tx)}",
        f"features={len(feature_names)}",
        f"suspicious_rate={float(y.mean()):.4f}",
    ]
    if X_test is not None and len(X_test) > 0:
        proba = xgb.predict_proba(X_test)
        summary_lines.extend([""] + ["test_metrics:"] + _compute_test_metrics(y_test, proba))

    (out_dir / "train_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    train_config = TrainConfig(
        dataset=args.dataset,
        seed=args.seed,
        num_clients=len(clients),
        num_transactions=len(tx),
        test_size=args.test_size,
        train_clients=n_train,
        test_clients=n_test if use_split else None,
        feature_count=len(feature_names),
        suspicious_rate=float(y.mean()),
    )
    save_train_config(train_config, out_dir)

    print(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
