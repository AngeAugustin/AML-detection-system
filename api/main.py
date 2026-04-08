from __future__ import annotations

from pathlib import Path
from typing import Any
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from actions.action_engine import recommend_actions
from api.external_ingestion import ExternalIngestionConfig, ExternalTransactionIngestor
from explainability.shap_explainer import explain_with_shap_or_importance
from features.feature_engineering import build_behavior_features, build_feature_frame
from graph.graph_builder import build_transaction_graph
from graph.graph_features import compute_graph_features, graph_risk_score_from_features
from scoring.risk_scoring import fuse_scores
from utils.config import DEFAULT_CONFIG
from utils.registry import align_features, load_artifacts
from utils.validation import MAX_AMOUNT, MAX_LIST_TRANSACTIONS_API, MAX_STRING_LENGTH


class TransactionIn(BaseModel):
    transaction_id: str = Field(..., max_length=MAX_STRING_LENGTH)
    client_id: str = Field(..., max_length=MAX_STRING_LENGTH)
    sender_account: str = Field(..., max_length=MAX_STRING_LENGTH)
    receiver_account: str = Field(..., max_length=MAX_STRING_LENGTH)
    amount: float = Field(..., ge=0.0, le=MAX_AMOUNT, description="Transaction amount")
    currency: str = Field(..., max_length=16)
    timestamp: str = Field(..., max_length=64)
    country_origin: str = Field(..., max_length=8)
    country_destination: str = Field(..., max_length=8)
    transaction_type: str = Field(..., max_length=32)
    channel: str = Field(..., max_length=32)
    merchant_category: str = Field(..., max_length=64)
    is_cash_transaction: bool
    device_id: str = Field(..., max_length=MAX_STRING_LENGTH)


class AnalyzeClientRequest(BaseModel):
    client_id: str = Field(..., description="Client identifier", max_length=MAX_STRING_LENGTH)
    client_name: str = Field("Unknown Client", description="Display name", max_length=MAX_STRING_LENGTH)
    client_type: str = Field("INDIVIDUAL", description="Client type", max_length=32)
    client_identifier: str = Field("", description="External identifier", max_length=MAX_STRING_LENGTH)
    transaction_history: list[TransactionIn] = Field(
        ...,
        description="Historical transactions for the client",
        min_length=1,
        max_length=MAX_LIST_TRANSACTIONS_API,
    )


class SuspiciousOperationOut(BaseModel):
    operationDate: str
    operationType: str
    amount: float
    currency: str
    description: str
    suspicionReason: str


class AnalyzeClientResponse(BaseModel):
    linkedClient: str
    clientName: str
    clientType: str
    clientIdentifier: str
    suspicionTypologies: list[str]
    suspicionDescription: str
    suspiciousOperations: list[SuspiciousOperationOut]


class ExternalIngestionStartRequest(BaseModel):
    base_url: str = Field(..., description="Base URL of external transactions API")
    endpoint: str = Field("/transactions", description="Endpoint returning transaction events")
    auth_token: str = Field("", description="Bearer token if required")
    poll_interval_seconds: float = Field(5.0, ge=0.2, le=300.0)
    limit: int = Field(200, ge=1, le=5000)
    timeout_seconds: float = Field(15.0, ge=1.0, le=120.0)
    cursor_param: str = Field("cursor", max_length=64)
    limit_param: str = Field("limit", max_length=64)
    initial_cursor: str = Field("", max_length=256)


ARTIFACT_DIR = Path("artifacts")
app = FastAPI(title="AML Detection System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_MAX_PER_CLIENT = 5000
FINDINGS_MAX = 2000
_store_lock = Lock()
_history_by_client: dict[str, list[dict[str, Any]]] = {}
_seen_transaction_ids: set[str] = set()
_findings: list[dict[str, Any]] = []
_ingestor: ExternalTransactionIngestor | None = None


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


def _to_iso_ms(value: str) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return "1970-01-01T00:00:00.000Z"
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _map_typologies(patterns: list[str], beh_features: dict[str, float]) -> list[str]:
    out: list[str] = []
    if "possible_structuring" in patterns:
        out.append("SMURFING")
    if "transaction_layering" in patterns:
        out.append("LAYERING")
    if beh_features.get("number_of_cash_deposits", 0.0) >= 3:
        out.append("UNUSUAL_CASH")
    if "high_risk_geography" in patterns or "cross_border_high_risk" in patterns:
        out.append("HIGH_RISK_GEO")
    if "rapid_transaction_sequences" in patterns:
        out.append("RAPID_MOVEMENT")
    return list(dict.fromkeys(out)) or ["UNUSUAL_BEHAVIOR"]


def _build_suspicious_operations(tx: pd.DataFrame) -> list[SuspiciousOperationOut]:
    if tx.empty:
        return []
    work = tx.copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work["timestamp_dt"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.sort_values("timestamp_dt")
    work["delta_s"] = work["timestamp_dt"].diff().dt.total_seconds().fillna(999999.0)

    def _score_row(row: pd.Series) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        if bool(row.get("is_cash_transaction", False)):
            score += 1
            reasons.append("Transaction en espèces")
        if float(row.get("amount", 0.0)) >= 7000.0:
            score += 1
            reasons.append("Montant inhabituel")
        if float(row.get("delta_s", 999999.0)) <= 15 * 60:
            score += 1
            reasons.append("Séquence rapide")
        if str(row.get("country_origin", "")) != str(row.get("country_destination", "")):
            score += 1
            reasons.append("Flux transfrontalier")
        return score, reasons

    scored = []
    for _, row in work.iterrows():
        score, reasons = _score_row(row)
        scored.append((score, float(row["amount"]), row, reasons))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [x for x in scored if x[0] > 0][:3] or scored[:1]

    out: list[SuspiciousOperationOut] = []
    for _, _, row, reasons in top:
        out.append(
            SuspiciousOperationOut(
                operationDate=_to_iso_ms(str(row.get("timestamp", ""))),
                operationType=str(row.get("transaction_type", "TRANSFER")).upper(),
                amount=round(float(row.get("amount", 0.0)), 2),
                currency=str(row.get("currency", "EUR")),
                description=f"Operation via {str(row.get('channel', 'unknown')).lower()}",
                suspicionReason="; ".join(reasons) if reasons else "Comportement atypique",
            )
        )
    return out


def _score_to_finding(
    req_client_id: str,
    tx: pd.DataFrame,
    client_name: str = "Unknown Client",
    client_type: str = "INDIVIDUAL",
    client_identifier: str = "",
) -> AnalyzeClientResponse:
    artifacts = _load_models()
    X, _, patterns_by_client = build_feature_frame(transactions=tx, clients=None, config=DEFAULT_CONFIG)
    X = align_features(X, artifacts.feature_names)
    x_row = X.iloc[[0]]

    g = build_transaction_graph(tx).graph
    gf = compute_graph_features(g, config=DEFAULT_CONFIG)
    graph_score = graph_risk_score_from_features(gf.features)
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

    exp = explain_with_shap_or_importance(
        xgb_model=artifacts.xgboost.model,
        X_row=x_row,
        feature_names=artifacts.feature_names,
        top_k=DEFAULT_CONFIG.top_k_explanations,
    ).explanations

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
            patterns_by_client.get(str(req_client_id), [])
            + beh.detected_patterns
            + gf.detected_patterns
            + (["transaction_layering"] if gf.features.get("transaction_cycles", 0.0) >= 1 else [])
        )
    )
    _ = recommend_actions(breakdown.risk_level).recommended_actions
    suspicion_description = explanations[0] if explanations else "Activités inhabituelles détectées sur une courte période."
    typologies = _map_typologies(detected_patterns, beh.features)
    suspicious_ops = _build_suspicious_operations(tx)
    return AnalyzeClientResponse(
        linkedClient=str(req_client_id),
        clientName=client_name,
        clientType=client_type.upper(),
        clientIdentifier=client_identifier or str(req_client_id),
        suspicionTypologies=typologies,
        suspicionDescription=suspicion_description,
        suspiciousOperations=suspicious_ops,
    )


def _ingest_external_batch(batch: list[dict[str, Any]]) -> None:
    validated: list[TransactionIn] = []
    for obj in batch:
        try:
            tx = TransactionIn(**obj)
        except Exception:
            continue
        with _store_lock:
            if tx.transaction_id in _seen_transaction_ids:
                continue
            _seen_transaction_ids.add(tx.transaction_id)
        validated.append(tx)

    clients_to_score: set[str] = set()
    with _store_lock:
        for tx in validated:
            cid = str(tx.client_id)
            clients_to_score.add(cid)
            hist = _history_by_client.setdefault(cid, [])
            hist.append(tx.model_dump())
            if len(hist) > HISTORY_MAX_PER_CLIENT:
                _history_by_client[cid] = hist[-HISTORY_MAX_PER_CLIENT:]
        # Keep dedup set bounded
        if len(_seen_transaction_ids) > 300_000:
            _seen_transaction_ids.clear()
            for hist in _history_by_client.values():
                for row in hist[-1000:]:
                    _seen_transaction_ids.add(str(row.get("transaction_id", "")))

    # score outside lock
    for cid in clients_to_score:
        with _store_lock:
            history = list(_history_by_client.get(cid, []))
        if not history:
            continue
        tx_df = pd.DataFrame(history)
        try:
            finding = _score_to_finding(req_client_id=cid, tx=tx_df)
        except Exception:
            continue
        with _store_lock:
            _findings.insert(0, finding.model_dump())
            if len(_findings) > FINDINGS_MAX:
                del _findings[FINDINGS_MAX:]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-client", response_model=AnalyzeClientResponse)
def analyze_client(req: AnalyzeClientRequest) -> AnalyzeClientResponse:
    try:
        _ = _load_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    tx = pd.DataFrame([t.model_dump() for t in req.transaction_history])
    tx = tx[tx["client_id"].astype(str) == str(req.client_id)].copy()
    if tx.empty:
        raise HTTPException(status_code=400, detail="No transactions found for given client_id in transaction_history")
    return _score_to_finding(
        req_client_id=str(req.client_id),
        tx=tx,
        client_name=req.client_name,
        client_type=req.client_type,
        client_identifier=req.client_identifier,
    )


@app.post("/external/start")
def external_start(req: ExternalIngestionStartRequest) -> dict[str, Any]:
    global _ingestor
    try:
        _ = _load_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    cfg = ExternalIngestionConfig(
        base_url=req.base_url,
        endpoint=req.endpoint,
        auth_token=req.auth_token,
        poll_interval_seconds=req.poll_interval_seconds,
        limit=req.limit,
        timeout_seconds=req.timeout_seconds,
        cursor_param=req.cursor_param,
        limit_param=req.limit_param,
        initial_cursor=req.initial_cursor,
    )
    if _ingestor is not None:
        _ingestor.stop()
    _ingestor = ExternalTransactionIngestor(cfg, on_batch=_ingest_external_batch)
    _ingestor.start()
    return {"ok": True, "message": "External ingestion started", "status": _ingestor.status()}


@app.post("/external/stop")
def external_stop() -> dict[str, Any]:
    global _ingestor
    if _ingestor is None:
        return {"ok": True, "message": "External ingestion already stopped"}
    _ingestor.stop()
    _ingestor = None
    return {"ok": True, "message": "External ingestion stopped"}


@app.get("/external/status")
def external_status() -> dict[str, Any]:
    if _ingestor is None:
        return {"running": False}
    return _ingestor.status()


@app.get("/findings")
def get_findings(limit: int = 50, client_id: str | None = None) -> dict[str, Any]:
    take = max(1, min(500, int(limit)))
    with _store_lock:
        data = list(_findings)
    if client_id:
        data = [f for f in data if str(f.get("linkedClient", "")) == str(client_id)]
    return {"count": len(data[:take]), "items": data[:take]}

