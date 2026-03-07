from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestScorer:
    model: IsolationForest
    q_low: float
    q_high: float

    @staticmethod
    def fit(X: pd.DataFrame, seed: int = 42) -> "IsolationForestScorer":
        model = IsolationForest(
            n_estimators=400,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X.values)
        raw = -model.decision_function(X.values)  # higher => more anomalous
        q_low = float(np.quantile(raw, 0.05))
        q_high = float(np.quantile(raw, 0.95))
        return IsolationForestScorer(model=model, q_low=q_low, q_high=q_high)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raw = -self.model.decision_function(X.values)
        # Robust scaling to [0, 1]
        denom = (self.q_high - self.q_low) if (self.q_high - self.q_low) != 0 else 1.0
        scaled = (raw - self.q_low) / denom
        return np.clip(scaled, 0.0, 1.0)

