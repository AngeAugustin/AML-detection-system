from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Mock External Transactions API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_dataset(seed: int = 42, n_clients: int = 60, n_tx: int = 6000) -> list[dict[str, Any]]:
    """
    Generate an enlarged synthetic transactions stream for integration tests.
    Includes normal behavior and suspicious patterns (smurfing, rapid movement).
    """
    rng = Random(seed)
    base = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    currencies = ["EUR", "XOF", "USD"]
    countries = ["SN", "FR", "CI", "NG", "GH", "MA"]
    tx_types = ["transfer", "cash_deposit", "cash_withdrawal", "card_payment", "wire"]
    channels = ["online", "branch", "ATM"]
    mcc = ["services", "retail", "travel", "cash_services", "misc"]

    clients = [f"C{i:04d}" for i in range(1, n_clients + 1)]
    suspicious_clients = set(clients[: max(3, n_clients // 8)])

    rows: list[dict[str, Any]] = []
    for i in range(n_tx):
        cid = clients[i % n_clients]
        suspicious = cid in suspicious_clients
        ts = base + timedelta(minutes=i * 3)

        tx_type = rng.choice(tx_types)
        amount = round(rng.uniform(8.0, 5000.0), 2)
        origin = rng.choice(countries)
        dest = origin if rng.random() < 0.85 else rng.choice(countries)
        channel = rng.choice(channels)
        is_cash = tx_type in {"cash_deposit", "cash_withdrawal"}

        # Inject suspicious behavior
        if suspicious and i % 13 in (0, 1, 2):
            # Structuring-like deposits below threshold
            tx_type = "cash_deposit"
            channel = "branch"
            amount = round(rng.uniform(7200.0, 9950.0), 2)
            is_cash = True
            if i % 13 == 2:
                # Rapid sequence + cross-border occasionally
                ts = ts - timedelta(minutes=10)
                dest = rng.choice(["NG", "GH", "CI"])

        rows.append(
            {
                "transaction_id": f"EXT_{i:08d}",
                "client_id": cid,
                "sender_account": f"A{rng.randint(10_000_000, 99_999_999)}",
                "receiver_account": f"A{rng.randint(10_000_000, 99_999_999)}",
                "amount": amount,
                "currency": rng.choice(currencies),
                "timestamp": _iso(ts),
                "country_origin": origin,
                "country_destination": dest,
                "transaction_type": tx_type,
                "channel": channel,
                "merchant_category": rng.choice(mcc),
                "is_cash_transaction": is_cash,
                "device_id": f"D{rng.randint(1, 9999):04d}",
            }
        )

    rows.sort(key=lambda x: x["timestamp"])
    return rows


DATASET: list[dict[str, Any]] = _build_dataset()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/transactions")
def transactions(
    cursor: str = Query("", description="Offset cursor"),
    limit: int = Query(200, ge=1, le=5000),
) -> dict[str, Any]:
    """
    Cursor-based pagination.
    - cursor: integer offset as string
    - returns transactions + next_cursor
    """
    try:
        offset = int(cursor) if cursor else 0
    except ValueError:
        offset = 0
    offset = max(0, min(offset, len(DATASET)))
    end = min(offset + limit, len(DATASET))
    batch = DATASET[offset:end]
    next_cursor = str(end) if end < len(DATASET) else ""
    return {
        "transactions": batch,
        "next_cursor": next_cursor,
        "count": len(batch),
        "total": len(DATASET),
    }

