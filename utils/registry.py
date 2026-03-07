from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from models.autoencoder import AutoEncoderScorer
from models.isolation_forest import IsolationForestScorer
from models.xgboost_model import XGBoostAMLModel
from utils.io import load_joblib, load_json, save_joblib, save_json


@dataclass(frozen=True)
class ModelArtifacts:
    isolation_forest: IsolationForestScorer
    xgboost: XGBoostAMLModel
    autoencoder: AutoEncoderScorer
    feature_names: list[str]


def save_artifacts(artifacts: ModelArtifacts, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_joblib(artifacts.isolation_forest, out / "isolation_forest.joblib")
    save_joblib(artifacts.xgboost, out / "xgboost.joblib")
    save_joblib(artifacts.autoencoder, out / "autoencoder.joblib")
    save_json({"feature_names": artifacts.feature_names}, out / "schema.json")


def load_artifacts(output_dir: str | Path) -> ModelArtifacts:
    out = Path(output_dir)
    iso = load_joblib(out / "isolation_forest.joblib")
    xgb = load_joblib(out / "xgboost.joblib")
    ae = load_joblib(out / "autoencoder.joblib")
    schema = load_json(out / "schema.json")
    return ModelArtifacts(
        isolation_forest=iso,
        xgboost=xgb,
        autoencoder=ae,
        feature_names=list(schema["feature_names"]),
    )


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align a feature frame to the training schema (missing -> 0, extra -> drop)."""
    X2 = X.copy()
    for f in feature_names:
        if f not in X2.columns:
            X2[f] = 0.0
    X2 = X2[feature_names]
    X2 = X2.fillna(0.0)
    return X2

