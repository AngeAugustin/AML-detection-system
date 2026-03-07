from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from actions.action_engine import recommend_actions
from explainability.shap_explainer import explain_with_shap_or_importance
from features.feature_engineering import build_behavior_features, build_feature_frame
from graph.graph_builder import build_transaction_graph
from graph.graph_features import compute_graph_features, graph_risk_score_from_features
from scoring.risk_scoring import fuse_scores
from utils.config import DEFAULT_CONFIG
from utils.registry import align_features, load_artifacts


class TransactionIn(BaseModel):
    transaction_id: str
    client_id: str
    sender_account: str
    receiver_account: str
    amount: float
    currency: str
    timestamp: str
    country_origin: str
    country_destination: str
    transaction_type: str
    channel: str
    merchant_category: str
    is_cash_transaction: bool
    device_id: str


class AnalyzeClientRequest(BaseModel):
    client_id: str = Field(..., description="Client identifier")
    transaction_history: list[TransactionIn] = Field(..., description="Historical transactions for the client")


class AnalyzeClientResponse(BaseModel):
    client_id: str
    aml_risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    explanations: list[str]
    detected_patterns: list[str]
    recommended_actions: list[str]


ARTIFACT_DIR = Path("artifacts")
app = FastAPI(title="AML Detection System", version="0.1.0")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirige vers la documentation interactive."""
    return RedirectResponse(url="/docs")


def _load_models() -> Any:
    if not (ARTIFACT_DIR / "schema.json").exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train first with: python -m training.train_pipeline --output artifacts"
        )
    return load_artifacts(ARTIFACT_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-client", response_model=AnalyzeClientResponse)
def analyze_client(req: AnalyzeClientRequest) -> AnalyzeClientResponse:
    try:
        artifacts = _load_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if not req.transaction_history:
        raise HTTPException(status_code=400, detail="transaction_history must be non-empty")

    # Convert to DataFrame
    tx = pd.DataFrame([t.model_dump() for t in req.transaction_history])
    tx = tx[tx["client_id"].astype(str) == str(req.client_id)].copy()
    if tx.empty:
        raise HTTPException(status_code=400, detail="No transactions found for given client_id in transaction_history")

    # Feature engineering
    X, _, patterns_by_client = build_feature_frame(transactions=tx, clients=None, config=DEFAULT_CONFIG)
    X = align_features(X, artifacts.feature_names)

    x_row = X.iloc[[0]]

    # Graph score + graph patterns
    g = build_transaction_graph(tx).graph
    gf = compute_graph_features(g, config=DEFAULT_CONFIG)
    graph_score = graph_risk_score_from_features(gf.features)

    # Model scores
    anomaly_score = float(artifacts.isolation_forest.score(x_row)[0])
    xgb_score = float(artifacts.xgboost.predict_proba(x_row)[0])
    ae_score = float(artifacts.autoencoder.score(x_row)[0])

    breakdown = fuse_scores(
        anomaly_score=anomaly_score,
        xgboost_score=xgb_score,
        graph_score=graph_score,
        autoencoder_score=ae_score,
        config=DEFAULT_CONFIG,
    )

    # Explanations (XGBoost-centric) + behavioral/graph hints
    exp = explain_with_shap_or_importance(
        xgb_model=artifacts.xgboost.model,
        X_row=x_row,
        feature_names=artifacts.feature_names,
        top_k=DEFAULT_CONFIG.top_k_explanations,
    ).explanations

    # Human-friendly behavioral explanations (rule-based)
    beh = build_behavior_features(tx, config=DEFAULT_CONFIG)
    rule_exps: list[str] = []
    if beh.features["sudden_activity_spike"] >= 3.0:
        rule_exps.append("Unusual increase in transaction volume (daily spike).")
    if beh.features["transfers_to_high_risk_countries"] >= 0.15:
        rule_exps.append("Frequent transfers to high-risk countries.")
    if beh.features["rapid_transaction_sequences"] >= 0.35:
        rule_exps.append("Multiple rapid transfers over short time windows.")
    if beh.features["structuring_indicator"] >= 0.6:
        rule_exps.append("Repeated cash deposits slightly below reporting thresholds (possible structuring).")
    if gf.features.get("transaction_cycles", 0.0) >= 1:
        rule_exps.append("Detected transaction loops between linked accounts (potential layering).")

    explanations = (rule_exps + exp)[: max(3, DEFAULT_CONFIG.top_k_explanations)]

    detected_patterns = list(
        dict.fromkeys(
            patterns_by_client.get(str(req.client_id), [])
            + beh.detected_patterns
            + gf.detected_patterns
            + (["transaction_layering"] if gf.features.get("transaction_cycles", 0.0) >= 1 else [])
        )
    )

    actions = recommend_actions(breakdown.risk_level).recommended_actions

    return AnalyzeClientResponse(
        client_id=str(req.client_id),
        aml_risk_score=round(breakdown.aml_risk_score, 4),
        risk_level=breakdown.risk_level,  # type: ignore[arg-type]
        explanations=explanations,
        detected_patterns=detected_patterns,
        recommended_actions=actions,
    )

