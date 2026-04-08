"""
PaySim public dataset adapter.

PaySim is a synthetic financial dataset for fraud detection, generated from
real mobile money transaction patterns. Widely used in research.
Source: https://www.kaggle.com/datasets/ntnu-testimon/paysim1
Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# PaySim transaction type -> our transaction_type
TYPE_MAP = {
    "CASH_IN": "cash_deposit",
    "CASH_OUT": "cash_withdrawal",
    "DEBIT": "card_payment",
    "PAYMENT": "card_payment",
    "TRANSFER": "transfer",
}


def load_paysim_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load PaySim CSV; supports both full and subset files."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"PaySim CSV must have columns {required}. Missing: {missing}. "
            "Download from: https://www.kaggle.com/datasets/ntnu-testimon/paysim1"
        )
    return df


def paysim_to_aml_schema(
    raw: pd.DataFrame,
    base_date: datetime | None = None,
    max_transactions: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert PaySim DataFrame to (clients, transactions, labels) in AML pipeline schema.

    - base_date: start of timeline; step 1 = base_date + 1 hour. Default: 2024-01-01 UTC.
    - max_transactions: cap total rows (for quick runs). None = use all.
    - seed: for sampling when max_transactions is set.
    """
    if base_date is None:
        base_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    df = raw.copy()
    if max_transactions is not None and len(df) > max_transactions:
        df = df.sample(n=max_transactions, random_state=seed).sort_values("step").reset_index(drop=True)

    df["transaction_type"] = df["type"].astype(str).str.upper().map(TYPE_MAP).fillna("transfer")
    df["is_cash"] = df["type"].astype(str).str.upper().isin(["CASH_IN", "CASH_OUT"])

    # Timestamp: step is typically 1 hour
    df["timestamp"] = df["step"].apply(
        lambda s: (base_date + timedelta(hours=int(s))).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Our schema columns
    tx_rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        tx_rows.append(
            {
                "transaction_id": f"PS_{i}",
                "client_id": str(row["nameOrig"]),
                "sender_account": str(row["nameOrig"]),
                "receiver_account": str(row["nameDest"]),
                "amount": float(row["amount"]),
                "currency": "EUR",
                "timestamp": row["timestamp"],
                "country_origin": "XX",
                "country_destination": "XX",
                "transaction_type": row["transaction_type"],
                "channel": "online",
                "merchant_category": "misc",
                "is_cash_transaction": int(row["is_cash"]),
                "device_id": "D0",
                "_is_fraud": int(row["isFraud"]),
            }
        )

    tx = pd.DataFrame(tx_rows)

    # Clients = unique nameOrig with label = 1 if any transaction is fraud
    client_fraud = tx.groupby("client_id")["_is_fraud"].max()
    clients_df = pd.DataFrame(
        {"client_id": client_fraud.index.astype(str), "label_suspicious": client_fraud.values.astype(int)}
    ).drop_duplicates(subset=["client_id"])

    # Minimal client table (no static attributes in PaySim)
    clients = clients_df[["client_id"]].copy()
    clients["age"] = 0
    clients["pep_flag"] = 0
    clients["risk_country_flag"] = 0
    clients["income_range"] = "mid"

    labels = clients_df[["client_id", "label_suspicious"]].copy()
    tx = tx.drop(columns=["_is_fraud"])

    return clients, tx, labels
