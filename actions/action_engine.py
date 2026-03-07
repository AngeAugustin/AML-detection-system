from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionRecommendation:
    recommended_actions: list[str]


def recommend_actions(risk_level: str) -> ActionRecommendation:
    if risk_level == "LOW":
        return ActionRecommendation(
            recommended_actions=[
                "monitor_future_transactions",
            ]
        )
    if risk_level == "MEDIUM":
        return ActionRecommendation(
            recommended_actions=[
                "enhanced_due_diligence",
                "monitor_future_transactions",
            ]
        )
    return ActionRecommendation(
        recommended_actions=[
            "flag_for_manual_review",
            "enhanced_due_diligence",
            "file_suspicious_activity_report",
            "monitor_future_transactions",
        ]
    )

