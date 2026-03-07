from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


DEFAULT_CASH_STRUCTURING_THRESHOLD: Final[float] = 10000.0


@dataclass(frozen=True)
class AMLConfig:
    """Central configuration for feature engineering and scoring."""

    # Countries list is intentionally short and configurable; in production this
    # should come from FATF / internal risk taxonomy + sanctions lists.
    high_risk_countries: tuple[str, ...] = (
        "IR",
        "KP",
        "SY",
        "AF",
        "MM",
        "SD",
        "SS",
        "SO",
        "LY",
        "CD",
        "CF",
        "ML",
        "NE",
        "NG",
    )

    cash_structuring_threshold: float = DEFAULT_CASH_STRUCTURING_THRESHOLD
    night_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    # Graph analysis
    cycle_max_length: int = 6
    community_min_size: int = 3

    # Score fusion weights (must sum to 1.0)
    w_isolation_forest: float = 0.30
    w_xgboost: float = 0.30
    w_graph: float = 0.20
    w_autoencoder: float = 0.20

    # Explainability
    top_k_explanations: int = 5

    # Risk thresholds
    low_threshold: float = 0.30
    high_threshold: float = 0.70


DEFAULT_CONFIG: Final[AMLConfig] = AMLConfig()

