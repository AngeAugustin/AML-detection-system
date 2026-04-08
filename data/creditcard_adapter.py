"""
Credit Card Fraud (ULB) public dataset adapter — benchmark only.

Dataset: anonymized credit card transactions, fraud labels.
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Columns: Time, V1..V28, Amount, Class (0=normal, 1=fraud).

Limitation: no client_id in the data; we form pseudo-clients (consecutive blocks).
Geography and transaction type are placeholders; feature signal is mainly amount/time.
Use synthetic data for optimal alignment with the AML pipeline (see README).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Schema constants
DEFAULT_TRANSACTIONS_PER_CLIENT = 80
DEFAULT_BASE_DATE = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def load_creditcard_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load ULB Credit Card Fraud CSV. Validates path exists and required columns."""
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Credit card CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Credit card CSV must have columns {required}. Missing: {missing}. "
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    return df


def creditcard_to_aml_schema(
    raw: pd.DataFrame,
    base_date: datetime | None = None,
    transactions_per_client: int = DEFAULT_TRANSACTIONS_PER_CLIENT,
    max_transactions: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert ULB Credit Card DataFrame to (clients, transactions, labels) in AML schema.

    Pseudo-clients: consecutive blocks of transactions_per_client rows.
    Label = 1 if any transaction in the block has Class==1.
    """
    base_date = base_date or DEFAULT_BASE_DATE

    df = raw.copy()
    if max_transactions is not None and len(df) > max_transactions:
        df = df.sample(n=max_transactions, random_state=seed).sort_values("Time").reset_index(drop=True)

    df["amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    time_seconds = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0).astype(float)
    ts_index = pd.Timestamp(base_date) + pd.to_timedelta(time_seconds, unit="s")
    df["timestamp"] = ts_index.strftime("%Y-%m-%dT%H:%M:%SZ")

    block_id = (df.index // transactions_per_client).astype(int)
    df["client_id"] = "CC_" + block_id.astype(str)

    # Vectorized transaction table (no iterrows)
    tx = pd.DataFrame(
        {
            "transaction_id": "CC_" + df.index.astype(str),
            "client_id": df["client_id"],
            "sender_account": "CARD",
            "receiver_account": "MERCHANT",
            "amount": df["amount"],
            "currency": "EUR",
            "timestamp": df["timestamp"],
            "country_origin": "XX",
            "country_destination": "XX",
            "transaction_type": "card_payment",
            "channel": "online",
            "merchant_category": "misc",
            "is_cash_transaction": 0,
            "device_id": "D0",
        }
    )

    client_fraud = df.groupby("client_id")["Class"].max().astype(int)
    clients_df = client_fraud.reset_index()
    clients_df.columns = ["client_id", "label_suspicious"]
    clients = clients_df[["client_id"]].copy()
    clients["age"] = 0
    clients["pep_flag"] = 0
    clients["risk_country_flag"] = 0
    clients["income_range"] = "mid"

    labels = clients_df[["client_id", "label_suspicious"]].copy()
    return clients, tx, labels
