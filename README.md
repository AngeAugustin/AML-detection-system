# AML Detection System (Production-Grade Blueprint)

This repository implements a modular Anti-Money Laundering (AML) detection system designed like a modern fintech AML platform:

- Feature engineering (client behavioral, temporal, geographic, AML indicators, network features)
- Isolation Forest anomaly score
- XGBoost supervised score (synthetic labels + optional public dataset adapter)
- AutoEncoder reconstruction score (deep anomaly detection)
- Graph analytics score (centrality, pagerank, communities, loops, optional Node2Vec)
- Score fusion + explainable AI (SHAP + feature importance fallback)
- Real-time inference via FastAPI

## Repository layout

```
aml_detection_system/
  __init__.py
data/
  synthetic_generator.py
features/
  feature_engineering.py
models/
  isolation_forest.py
  xgboost_model.py
  autoencoder.py
graph/
  graph_builder.py
  graph_features.py
scoring/
  risk_scoring.py
explainability/
  shap_explainer.py
actions/
  action_engine.py
api/
  main.py
training/
  train_pipeline.py
utils/
  config.py
  io.py
  registry.py
artifacts/
requirements.txt
README.md
```

## Quickstart

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Train models (synthetic AML dataset)

This generates synthetic client + transaction data, builds feature matrices, trains all models, fits normalizers, and writes artifacts into `./artifacts`.

```bash
python -m training.train_pipeline --output artifacts --seed 42 --num-clients 2000
```

### 3) Start the API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Ouvrez dans le navigateur **http://localhost:8000** (ou http://127.0.0.1:8000) — ne pas utiliser `0.0.0.0` dans l’URL. La doc interactive : http://localhost:8000/docs

### 4) Call the API

```bash
curl -X POST http://localhost:8000/analyze-client ^
  -H "Content-Type: application/json" ^
  -d "{\"client_id\":\"C0001\",\"transaction_history\":[{\"transaction_id\":\"T1\",\"client_id\":\"C0001\",\"sender_account\":\"A1\",\"receiver_account\":\"A2\",\"amount\":1200.0,\"currency\":\"EUR\",\"timestamp\":\"2026-03-01T10:11:12Z\",\"country_origin\":\"FR\",\"country_destination\":\"FR\",\"transaction_type\":\"transfer\",\"channel\":\"online\",\"merchant_category\":\"misc\",\"is_cash_transaction\":false,\"device_id\":\"D1\"}]}"
```

Expected response shape:

```json
{
  "client_id": "C0001",
  "aml_risk_score": 0.87,
  "risk_level": "HIGH",
  "explanations": [
    "Unusual increase in transaction volume",
    "Frequent transfers to high-risk countries",
    "Multiple rapid transfers between linked accounts"
  ],
  "detected_patterns": [
    "possible_structuring",
    "transaction_layering"
  ],
  "recommended_actions": [
    "flag_for_manual_review",
    "enhanced_due_diligence",
    "monitor_future_transactions"
  ]
}
```

## Installation notes

- Run from the **project root** so that `data`, `features`, `models`, etc. resolve as packages.
- If `shap` fails to install (e.g. on some Windows/Python versions), omit it; the explainer falls back to XGBoost feature importance.
- Graph analytics use NetworkX (degree centrality, PageRank, community detection, cycle detection). Node2Vec is optional and not required for the current pipeline.

## Notes (production readiness)

- Artifacts are versioned by filename and include model parameters + feature schema.
- The API is stateless and supports real-time scoring from request payloads.
- Explainability uses SHAP for the XGBoost model when available; otherwise falls back to feature importance + rule-based explanations.

