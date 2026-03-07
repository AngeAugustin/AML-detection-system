from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.config import AMLConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class FeatureResult:
    feature_names: list[str]
    features: dict[str, float]
    detected_patterns: list[str]


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def build_behavior_features(
    transactions: pd.DataFrame,
    config: AMLConfig = DEFAULT_CONFIG,
) -> FeatureResult:
    """Compute client-level behavioral features from transaction history."""
    detected: list[str] = []
    if transactions.empty:
        features = {
            "total_transactions": 0.0,
            "avg_transaction_amount": 0.0,
            "max_transaction_amount": 0.0,
            "transaction_variance": 0.0,
            "daily_transaction_frequency": 0.0,
            "night_transactions": 0.0,
            "weekend_transactions": 0.0,
            "sudden_activity_spike": 0.0,
            "number_of_cash_deposits": 0.0,
            "international_transfer_ratio": 0.0,
            "structuring_indicator": 0.0,
            "rapid_transaction_sequences": 0.0,
            "transfers_to_high_risk_countries": 0.0,
            "number_of_destination_countries": 0.0,
            "unique_devices": 0.0,
            "unique_counterparties": 0.0,
        }
        return FeatureResult(feature_names=list(features.keys()), features=features, detected_patterns=detected)

    df = transactions.copy()

    required = {
        "amount",
        "timestamp",
        "country_origin",
        "country_destination",
        "transaction_type",
        "is_cash_transaction",
        "sender_account",
        "receiver_account",
        "device_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"transactions missing required columns: {sorted(missing)}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    df["timestamp"] = _to_datetime_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    total_tx = float(len(df))
    avg_amt = float(df["amount"].mean()) if len(df) else 0.0
    max_amt = float(df["amount"].max()) if len(df) else 0.0
    var_amt = float(df["amount"].var(ddof=0)) if len(df) else 0.0

    # Daily frequency
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date")["amount"].size()
    daily_freq = float(daily_counts.mean()) if len(daily_counts) else 0.0

    # Temporal patterns
    hours = df["timestamp"].dt.hour
    night = float((hours.isin(list(config.night_hours))).mean()) if len(df) else 0.0
    weekend = float((df["timestamp"].dt.dayofweek >= 5).mean()) if len(df) else 0.0

    # Sudden activity spike: ratio of max daily tx count to mean daily count
    spike = 0.0
    if len(daily_counts) >= 3:
        spike = float(_safe_div(float(daily_counts.max()), float(daily_counts.mean())))
    if spike >= 3.0:
        detected.append("sudden_activity_spike")

    # AML indicators
    is_cash = df["is_cash_transaction"].astype(int) == 1
    cash_deposits = float((df["transaction_type"] == "cash_deposit").sum())

    intl = df["country_origin"].astype(str) != df["country_destination"].astype(str)
    intl_ratio = float(intl.mean()) if len(df) else 0.0

    # Structuring indicator: many cash deposits just below threshold + high count
    near_thresh = df["amount"].between(config.cash_structuring_threshold * 0.70, config.cash_structuring_threshold * 0.999)
    structuring_hits = int(((df["transaction_type"] == "cash_deposit") & near_thresh).sum())
    structuring_indicator = float(min(1.0, structuring_hits / 8.0))
    if structuring_indicator >= 0.6:
        detected.append("possible_structuring")

    # Rapid transaction sequences: count of consecutive transactions within short window
    df_sorted = df.sort_values("timestamp")
    deltas = df_sorted["timestamp"].diff().dt.total_seconds().fillna(1e9)
    rapid = float((deltas <= 15 * 60).mean()) if len(deltas) else 0.0
    if rapid >= 0.35:
        detected.append("rapid_transaction_sequences")

    # Geographic risk
    high_risk = df["country_destination"].astype(str).isin(set(config.high_risk_countries))
    transfers_to_high_risk = float(high_risk.mean()) if len(df) else 0.0
    if transfers_to_high_risk >= 0.15:
        detected.append("high_risk_geography")
    n_dest_countries = float(df["country_destination"].astype(str).nunique())

    # Device / counterparty diversity
    unique_devices = float(df["device_id"].astype(str).nunique())
    counterparties = pd.concat([df["sender_account"].astype(str), df["receiver_account"].astype(str)], axis=0)
    unique_counterparties = float(counterparties.nunique())

    features = {
        "total_transactions": total_tx,
        "avg_transaction_amount": avg_amt,
        "max_transaction_amount": max_amt,
        "transaction_variance": var_amt,
        "daily_transaction_frequency": daily_freq,
        "night_transactions": night,
        "weekend_transactions": weekend,
        "sudden_activity_spike": spike,
        "number_of_cash_deposits": cash_deposits,
        "international_transfer_ratio": intl_ratio,
        "structuring_indicator": structuring_indicator,
        "rapid_transaction_sequences": rapid,
        "transfers_to_high_risk_countries": transfers_to_high_risk,
        "number_of_destination_countries": n_dest_countries,
        "unique_devices": unique_devices,
        "unique_counterparties": unique_counterparties,
    }

    # Extra pattern hints
    if intl_ratio >= 0.35 and transfers_to_high_risk >= 0.10:
        detected.append("cross_border_high_risk")

    return FeatureResult(feature_names=list(features.keys()), features=features, detected_patterns=detected)


def build_feature_frame(
    transactions: pd.DataFrame,
    clients: pd.DataFrame | None = None,
    config: AMLConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Build a model-ready feature matrix keyed by client_id.

    Returns:
        (X, feature_names, patterns_by_client)
    """
    if transactions.empty:
        return pd.DataFrame(), [], {}

    if "client_id" not in transactions.columns:
        raise ValueError("transactions must include client_id")

    patterns_by_client: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for client_id, tx_c in transactions.groupby("client_id"):
        fr = build_behavior_features(tx_c, config=config)
        row = {"client_id": str(client_id), **fr.features}
        rows.append(row)
        patterns_by_client[str(client_id)] = list(dict.fromkeys(fr.detected_patterns))

    X = pd.DataFrame(rows).set_index("client_id")
    feature_names = list(X.columns)

    # Optional: join client static attributes (lightweight encoding)
    if clients is not None and not clients.empty and "client_id" in clients.columns:
        c = clients.set_index("client_id").copy()
        # Include numeric flags, age; encode income_range ordinal
        income_map = {"low": 0, "lower_mid": 1, "mid": 2, "upper_mid": 3, "high": 4}
        if "income_range" in c.columns:
            c["income_range_ord"] = c["income_range"].astype(str).map(income_map).fillna(1).astype(float)
        keep_cols = [col for col in ["age", "pep_flag", "risk_country_flag", "income_range_ord"] if col in c.columns]
        if keep_cols:
            X = X.join(c[keep_cols], how="left").fillna(0.0)
            feature_names = list(X.columns)

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return X, feature_names, patterns_by_client

