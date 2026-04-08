"""
Unified dataset loader for training: synthetic or Credit Card Fraud (ULB) public dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from data.creditcard_adapter import load_creditcard_csv, creditcard_to_aml_schema
from data.synthetic_generator import (
    SyntheticGenerationParams,
    generate_synthetic_dataset,
)

DatasetType = Literal["synthetic", "creditcard"]


def load_dataset(
    dataset: DatasetType = "synthetic",
    *,
    # Synthetic params
    num_clients: int = 2000,
    min_tx_per_client: int = 20,
    max_tx_per_client: int = 250,
    seed: int = 42,
    # Credit card (public) params
    data_path: str | Path | None = None,
    max_transactions: int | None = None,
    transactions_per_client: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load (clients, transactions, labels) for the AML pipeline.

    - dataset="synthetic": use built-in synthetic generator (data_path ignored).
    - dataset="creditcard": load ULB Credit Card Fraud CSV from data_path; optional max_transactions to cap size.
    """
    if dataset == "synthetic":
        params = SyntheticGenerationParams(
            num_clients=num_clients,
            min_tx_per_client=min_tx_per_client,
            max_tx_per_client=max_tx_per_client,
            seed=seed,
        )
        return generate_synthetic_dataset(params=params)

    if dataset == "creditcard":
        if not data_path:
            raise ValueError("dataset='creditcard' requires data_path pointing to Credit Card Fraud CSV")
        raw = load_creditcard_csv(data_path)
        return creditcard_to_aml_schema(
            raw,
            max_transactions=max_transactions,
            transactions_per_client=transactions_per_client,
            seed=seed,
        )

    raise ValueError(f"Unknown dataset: {dataset}. Use 'synthetic' or 'creditcard'.")