from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplanationResult:
    explanations: list[str]


def explain_with_shap_or_importance(
    xgb_model: object,
    X_row: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> ExplanationResult:
    """Produce human-readable explanations for a single client row.

    Tries SHAP for tree models; falls back to feature importances and per-row magnitudes.
    """
    # SHAP path (TreeExplainer)
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_row.values)
        vals = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values
        vals = np.asarray(vals).reshape(-1)
        top_idx = np.argsort(np.abs(vals))[::-1][:top_k]

        explanations: list[str] = []
        for i in top_idx:
            name = feature_names[int(i)]
            direction = "increases" if vals[int(i)] > 0 else "decreases"
            explanations.append(f"{name} {direction} risk (SHAP impact: {float(vals[int(i)]):.3f})")
        return ExplanationResult(explanations=explanations)
    except Exception:
        pass

    # Fallback: importance * normalized value magnitude
    explanations = []
    try:
        importances = getattr(xgb_model, "feature_importances_", None)
        if importances is None:
            return ExplanationResult(explanations=["Model explainability not available."])
        importances = np.asarray(importances).reshape(-1)
        x = X_row.values.reshape(-1)
        score = np.abs(x) * importances
        top_idx = np.argsort(score)[::-1][:top_k]
        for i in top_idx:
            name = feature_names[int(i)]
            explanations.append(f"High contribution from {name} (importance-weighted value: {float(score[int(i)]):.3f})")
        return ExplanationResult(explanations=explanations)
    except Exception:
        return ExplanationResult(explanations=["Model explainability not available."])

