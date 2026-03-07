from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from utils.config import AMLConfig, DEFAULT_CONFIG


TransactionType = Literal["transfer", "cash_deposit", "cash_withdrawal", "card_payment", "wire"]
Channel = Literal["ATM", "online", "branch"]


@dataclass(frozen=True)
class SyntheticGenerationParams:
    num_clients: int = 2000
    min_tx_per_client: int = 20
    max_tx_per_client: int = 250
    start_date: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date: datetime = datetime(2026, 3, 1, tzinfo=timezone.utc)
    suspicious_rate: float = 0.12
    seed: int = 42


def _choice(rng: np.random.Generator, items: list[str], p: list[float] | None = None) -> str:
    return items[int(rng.choice(len(items), p=p))]


def _rand_ts(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = rng.integers(0, int(delta.total_seconds()))
    return start + timedelta(seconds=int(seconds))


def _make_account_id(rng: np.random.Generator) -> str:
    return f"A{rng.integers(10_000_000, 99_999_999)}"


def generate_synthetic_clients(params: SyntheticGenerationParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)

    countries_eu = ["FR", "DE", "ES", "IT", "BE", "NL", "PT", "IE", "AT", "GR"]
    countries_af = ["NG", "KE", "GH", "CI", "SN", "ZA", "MA", "TN", "CM", "UG"]
    all_countries = countries_eu + countries_af

    occupations = ["employee", "self_employed", "student", "unemployed", "retired", "merchant", "consultant"]
    income_ranges = ["low", "lower_mid", "mid", "upper_mid", "high"]
    genders = ["M", "F"]

    rows: list[dict[str, Any]] = []
    for i in range(params.num_clients):
        client_id = f"C{i:05d}"
        age = int(rng.integers(18, 80))
        gender = _choice(rng, genders)
        nationality = _choice(rng, all_countries)
        residence = _choice(rng, all_countries)
        occupation = _choice(rng, occupations)
        income_range = _choice(rng, income_ranges, p=[0.18, 0.26, 0.30, 0.18, 0.08])
        open_days_ago = int(rng.integers(30, 3650))
        account_open_date = (params.end_date - timedelta(days=open_days_ago)).date().isoformat()

        pep_flag = bool(rng.random() < 0.03)
        risk_country_flag = bool(residence in DEFAULT_CONFIG.high_risk_countries or nationality in DEFAULT_CONFIG.high_risk_countries)

        rows.append(
            dict(
                client_id=client_id,
                age=age,
                gender=gender,
                nationality=nationality,
                country_residence=residence,
                occupation=occupation,
                income_range=income_range,
                account_open_date=account_open_date,
                pep_flag=int(pep_flag),
                risk_country_flag=int(risk_country_flag),
            )
        )

    return pd.DataFrame(rows)


def generate_synthetic_transactions(
    clients: pd.DataFrame,
    params: SyntheticGenerationParams,
    config: AMLConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate transaction table and client-level synthetic AML labels.

    Returns:
        (transactions_df, labels_df) where labels_df has columns: client_id, label_suspicious (0/1)
    """
    rng = np.random.default_rng(params.seed)

    currencies = ["EUR", "USD", "XOF", "XAF", "KES", "ZAR"]
    merchant_categories = ["groceries", "fuel", "electronics", "travel", "services", "misc"]

    tx_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    suspicious_clients = set(
        rng.choice(
            clients["client_id"].to_numpy(),
            size=max(1, int(params.num_clients * params.suspicious_rate)),
            replace=False,
        ).tolist()
    )

    # Global pool of counterpart accounts (to simulate a network)
    counterparty_pool = [_make_account_id(rng) for _ in range(max(50_000, params.num_clients * 30))]

    for _, c in clients.iterrows():
        client_id = str(c["client_id"])
        is_suspicious = client_id in suspicious_clients
        label_rows.append({"client_id": client_id, "label_suspicious": int(is_suspicious)})

        n_tx = int(rng.integers(params.min_tx_per_client, params.max_tx_per_client + 1))
        base_account = _make_account_id(rng)

        # Baseline behavior
        income_range = str(c["income_range"])
        income_scale = {"low": 0.8, "lower_mid": 1.0, "mid": 1.4, "upper_mid": 2.0, "high": 3.0}[income_range]
        mean_amount = 200 * income_scale
        std_amount = 400 * income_scale

        # Pattern knobs for suspicious clients
        structuring = is_suspicious and (rng.random() < 0.45)
        layering = is_suspicious and (rng.random() < 0.35)
        mule = is_suspicious and (rng.random() < 0.25)
        loops = is_suspicious and (rng.random() < 0.20)
        tf = is_suspicious and (rng.random() < 0.15)

        # Build a small "linked accounts" neighborhood for layering/loops
        linked_accounts = [_make_account_id(rng) for _ in range(int(rng.integers(3, 12)))]
        if layering or loops:
            linked_accounts += [_make_account_id(rng) for _ in range(int(rng.integers(5, 20)))]

        for j in range(n_tx):
            ts = _rand_ts(rng, params.start_date, params.end_date)
            currency = _choice(rng, currencies, p=[0.46, 0.28, 0.12, 0.06, 0.04, 0.04])
            channel: Channel = _choice(rng, ["ATM", "online", "branch"], p=[0.18, 0.68, 0.14])  # type: ignore[assignment]

            # transaction type distribution
            tx_type: TransactionType = _choice(
                rng,
                ["transfer", "card_payment", "cash_deposit", "cash_withdrawal", "wire"],
                p=[0.48, 0.28, 0.10, 0.08, 0.06],
            )  # type: ignore[assignment]

            is_cash = tx_type in {"cash_deposit", "cash_withdrawal"}

            # Amount model
            amount = float(max(1.0, rng.normal(mean_amount, std_amount)))

            # International transfers
            origin = str(c["country_residence"])
            dest = origin
            if rng.random() < (0.08 + (0.10 if is_suspicious else 0.0)):
                dest = _choice(rng, list(set([origin] + list(config.high_risk_countries) + ["GB", "CH", "AE", "TR", "QA"])))

            # Suspicious pattern injections
            if structuring and is_cash and rng.random() < 0.65:
                # Many deposits slightly below threshold
                amount = float(rng.uniform(config.cash_structuring_threshold * 0.75, config.cash_structuring_threshold * 0.99))
                tx_type = "cash_deposit"
                channel = _choice(rng, ["ATM", "branch"], p=[0.72, 0.28])  # type: ignore[assignment]

            if tf and rng.random() < 0.35:
                dest = _choice(rng, list(config.high_risk_countries))
                tx_type = "wire"
                amount = float(amount * rng.uniform(1.5, 4.0))

            if layering and rng.random() < 0.55:
                tx_type = _choice(rng, ["transfer", "wire"], p=[0.7, 0.3])  # type: ignore[assignment]
                # Rapid sequences and hops
                ts = ts.replace(tzinfo=timezone.utc) + timedelta(minutes=int(rng.integers(1, 40)))
                amount = float(amount * rng.uniform(0.9, 1.1))

            if mule and rng.random() < 0.40:
                # Pass-through: many in/out with similar amounts
                tx_type = "transfer"
                amount = float(amount * rng.uniform(2.0, 6.0))

            # Sender/receiver account selection
            if tx_type in {"card_payment"}:
                sender = base_account
                receiver = _choice(rng, counterparty_pool)
            elif tx_type in {"cash_deposit"}:
                sender = "CASH"
                receiver = base_account
            elif tx_type in {"cash_withdrawal"}:
                sender = base_account
                receiver = "CASH"
            else:
                sender = base_account
                receiver = _choice(rng, counterparty_pool)

            # Use linked accounts to form layering/loops
            if layering and tx_type in {"transfer", "wire"} and rng.random() < 0.70:
                receiver = _choice(rng, linked_accounts)
            if loops and tx_type in {"transfer", "wire"} and rng.random() < 0.25:
                # Create a small chance of closing a cycle
                receiver = base_account if rng.random() < 0.35 else _choice(rng, linked_accounts)

            device_id = f"D{rng.integers(1, 12000)}"
            mcc = _choice(rng, merchant_categories)

            tx_rows.append(
                dict(
                    transaction_id=f"T{client_id}_{j:05d}",
                    client_id=client_id,
                    sender_account=sender,
                    receiver_account=receiver,
                    amount=round(amount, 2),
                    currency=currency,
                    timestamp=ts.isoformat().replace("+00:00", "Z"),
                    country_origin=origin,
                    country_destination=dest,
                    transaction_type=tx_type,
                    channel=channel,
                    merchant_category=mcc,
                    is_cash_transaction=int(is_cash),
                    device_id=device_id,
                )
            )

    tx_df = pd.DataFrame(tx_rows)
    labels_df = pd.DataFrame(label_rows)

    # Optional: shuffle transactions by time for realism
    tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], utc=True)
    tx_df = tx_df.sort_values(["client_id", "timestamp"]).reset_index(drop=True)
    tx_df["timestamp"] = tx_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return tx_df, labels_df


def generate_synthetic_dataset(
    params: SyntheticGenerationParams = SyntheticGenerationParams(),
    config: AMLConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clients = generate_synthetic_clients(params)
    tx, labels = generate_synthetic_transactions(clients, params=params, config=config)
    return clients, tx, labels

