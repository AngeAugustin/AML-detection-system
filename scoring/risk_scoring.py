from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.config import AMLConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class ScoreBreakdown:
    anomaly_score: float
    xgboost_score: float
    graph_score: float
    autoencoder_score: float
    aml_risk_score: float
    risk_level: str


def fuse_scores(
    anomaly_score: float,
    xgboost_score: float,
    graph_score: float,
    autoencoder_score: float,
    config: AMLConfig = DEFAULT_CONFIG,
) -> ScoreBreakdown:
    s = (
        config.w_isolation_forest * anomaly_score
        + config.w_xgboost * xgboost_score
        + config.w_graph * graph_score
        + config.w_autoencoder * autoencoder_score
    )
    s = float(np.clip(s, 0.0, 1.0))
    if s < config.low_threshold:
        lvl = "LOW"
    elif s < config.high_threshold:
        lvl = "MEDIUM"
    else:
        lvl = "HIGH"
    return ScoreBreakdown(
        anomaly_score=float(anomaly_score),
        xgboost_score=float(xgboost_score),
        graph_score=float(graph_score),
        autoencoder_score=float(autoencoder_score),
        aml_risk_score=s,
        risk_level=lvl,
    )

