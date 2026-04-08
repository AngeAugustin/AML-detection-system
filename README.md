# AML Detection System (Production-Grade Blueprint)

This repository implements a modular Anti-Money Laundering (AML) detection system designed like a modern fintech AML platform:

- Feature engineering (client behavioral, temporal, geographic, AML indicators, network features)
- Isolation Forest anomaly score
- XGBoost supervised score (synthetic labels + **Credit Card Fraud** public dataset)
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
  creditcard_adapter.py
  dataset_loader.py
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
  train_config.py
utils/
  config.py
  io.py
  registry.py
  validation.py
artifacts/
requirements.txt
README.md
```

## Quickstart

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Train models

**Option A — Données synthétiques (recommandé)**  
Génère clients et transactions avec patterns AML, entraîne les modèles et écrit les artefacts. Avec `--test-size 0.2`, un split train/test est appliqué et les métriques test sont enregistrées.

```bash
python -m training.train_pipeline --output artifacts --seed 42 --num-clients 2000 --test-size 0.2
```

**Option B — Dataset public Credit Card Fraud (ULB)**  
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (MLG-ULB) : transactions de cartes bancaires anonymisées (~284k transactions, labels fraude). Référence encore disponible sur Kaggle.

1. Téléchargez le CSV sur Kaggle : [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (fichier `creditcard.csv`).
2. Placez le fichier dans le projet (ex. `data/creditcard.csv`).
3. Entraînement avec split train/test (80/20) et métriques sur l’ensemble de test :

```bash
python -m training.train_pipeline --dataset creditcard --data-path data/creditcard.csv --output artifacts --test-size 0.2
```

Pour un premier test rapide, limitez le nombre de transactions :

```bash
python -m training.train_pipeline --dataset creditcard --data-path data/creditcard.csv --max-transactions 50000 --output artifacts --test-size 0.2
```

Les métriques (accuracy, F1, ROC-AUC) sont enregistrées dans `artifacts/train_summary.txt`.

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

### 5) Branch an external transactions API (streaming findings)

Start external ingestion (polling with cursor):

```bash
curl -X POST http://localhost:8000/external/start ^
  -H "Content-Type: application/json" ^
  -d "{\"base_url\":\"https://your-external-api.com\",\"endpoint\":\"/transactions\",\"auth_token\":\"YOUR_TOKEN\",\"poll_interval_seconds\":5,\"limit\":200,\"cursor_param\":\"cursor\",\"limit_param\":\"limit\",\"initial_cursor\":\"\"}"
```

Check ingestion status:

```bash
curl http://localhost:8000/external/status
```

Read findings generated continuously:

```bash
curl "http://localhost:8000/findings?limit=50"
```

Stop ingestion:

```bash
curl -X POST http://localhost:8000/external/stop
```

External API expected response (one of):

- `[{...transaction...}, {...transaction...}]`
- `{"transactions":[{...}, {...}], "next_cursor":"abc123"}`

Each transaction object should match the same fields as `TransactionIn` (`transaction_id`, `client_id`, `amount`, `timestamp`, etc.).

### 6) Mock external API for integration tests

Run a local external API with an enlarged dataset (6000 tx, mixed normal + suspicious):

```bash
python -m uvicorn api.mock_external:app --host 0.0.0.0 --port 8010
```

Quick check:

```bash
curl "http://localhost:8010/transactions?cursor=&limit=5"
```

Then connect AML API to this mock source:

```bash
curl -X POST http://localhost:8000/external/start ^
  -H "Content-Type: application/json" ^
  -d "{\"base_url\":\"http://localhost:8010\",\"endpoint\":\"/transactions\",\"poll_interval_seconds\":2,\"limit\":200,\"cursor_param\":\"cursor\",\"limit_param\":\"limit\",\"initial_cursor\":\"\"}"
```

Read findings as they are produced:

```bash
curl "http://localhost:8000/findings?limit=20"
```

## Stratégie dataset idéal

Pour un **entraînement et une efficacité optimaux**, le pipeline est conçu pour le **dataset synthétique** (`--dataset synthetic`), qui est le choix par défaut et recommandé :

- **Alignement parfait** : le feature engineering (comportement client, géographie, structuring, layering, boucles) exploite des champs que le générateur produit explicitement (pays, types de tx, montants, séquences). Les modèles apprennent des signaux AML cohérents.
- **Reproductibilité** : même seed → mêmes données ; split train/test possible (`--test-size 0.2`) pour évaluation stricte.
- **Contrôle du déséquilibre** : `suspicious_rate` configurable (défaut 12 %) pour éviter un déséquilibre extrême.

Le **dataset Credit Card (ULB)** est proposé comme **benchmark public** uniquement : pas d’identifiant client (pseudo-clients par blocs), pas de géographie ni de types de transaction réels. Le pipeline applique le même schéma, mais le signal est surtout montant/temps ; utiliser le synthétique pour des résultats représentatifs.

| Source | Rôle | Usage |
|--------|------|--------|
| **Synthetic** | **Recommandé** — idéal pour ce pipeline | `--dataset synthetic` (défaut), optionnel `--test-size 0.2` |
| **Credit Card (ULB)** | Benchmark public | `--dataset creditcard --data-path <fichier.csv>` |

## Reproductibilité

Chaque entraînement enregistre dans `artifacts/` :

- `train_config.json` : dataset, seed, nombre de clients/transactions, test_size, taux de fraude. Permet de refaire exactement la même run.
- `train_summary.txt` : métriques (dont test si split activé).

L’argument `--output` est validé (pas de path traversal) ; les arguments numériques (seed, test_size, num_clients, etc.) sont bornés.

## Sécurité

- **CLI** : chemins de sortie limités au répertoire du projet ; bornes sur seed, test_size, num_clients, max_transactions.
- **API** : validation Pydantic sur les requêtes (montant ≥ 0 et ≤ seuil, longueur des chaînes, nombre max de transactions par requête). Aucune désérialisation non sécurisée (pas de `pickle` chargé depuis l’extérieur).

## Installation notes

- Run from the **project root** so that `data`, `features`, `models`, etc. resolve as packages.
- If `shap` fails to install (e.g. on some Windows/Python versions), omit it; the explainer falls back to XGBoost feature importance.
- Graph analytics use NetworkX (degree centrality, PageRank, community detection, cycle detection). Node2Vec is optional and not required for the current pipeline.

## Notes (production readiness)

- Artifacts are versioned by filename and include model parameters + feature schema.
- The API is stateless and supports real-time scoring from request payloads.
- Explainability uses SHAP for the XGBoost model when available; otherwise falls back to feature importance + rule-based explanations.

