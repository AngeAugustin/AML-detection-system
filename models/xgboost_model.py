from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass
class XGBoostAMLModel:
    model: XGBClassifier
    feature_names: list[str]

    @staticmethod
    def fit(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> "XGBoostAMLModel":
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X.values, y.values)
        return XGBoostAMLModel(model=model, feature_names=list(X.columns))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X.values)[:, 1]
        return np.clip(proba.astype(float), 0.0, 1.0)

    def feature_importance(self) -> dict[str, float]:
        imp = self.model.feature_importances_
        return {name: float(val) for name, val in zip(self.feature_names, imp)}

