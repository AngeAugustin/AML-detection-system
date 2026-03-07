from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from utils.config import AMLConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class GraphFeatureResult:
    features: dict[str, float]
    detected_patterns: list[str]


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(values))


def _count_cycles_limited(g: nx.DiGraph, max_len: int) -> int:
    """Count simple cycles with a cutoff to avoid runaway costs.

    For real-time inference we keep it small; for offline investigations, compute
    cycles with specialized graph tooling.
    """
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return 0
    # networkx.simple_cycles can be expensive; limit by early exit if too many
    cycles = 0
    for cyc in nx.simple_cycles(g):
        if len(cyc) <= max_len:
            cycles += 1
        if cycles >= 50:
            break
    return cycles


def compute_graph_features(
    g: nx.DiGraph,
    config: AMLConfig = DEFAULT_CONFIG,
) -> GraphFeatureResult:
    """Compute client-level graph risk features.

    The returned dict contains:
      - number_of_connected_accounts
      - graph_centrality_score
      - pagerank_score
      - transaction_cycles
      - community_count
    """
    if g.number_of_nodes() == 0:
        return GraphFeatureResult(
            features=dict(
                number_of_connected_accounts=0.0,
                graph_centrality_score=0.0,
                pagerank_score=0.0,
                transaction_cycles=0.0,
                community_count=0.0,
            ),
            detected_patterns=[],
        )

    n = g.number_of_nodes()
    detected: list[str] = []

    # Degree centrality (mean/max)
    deg_cent = nx.degree_centrality(g)
    deg_vals = list(deg_cent.values())

    # PageRank
    try:
        pr = nx.pagerank(g, alpha=0.85)
        pr_vals = list(pr.values())
    except Exception:
        pr_vals = [0.0] * n

    # Community detection (undirected projection)
    try:
        und = g.to_undirected()
        try:
            communities = list(nx.community.greedy_modularity_communities(und))
        except AttributeError:
            communities = list(nx.algorithms.community.greedy_modularity_communities(und))
        community_count = float(len(communities))
    except Exception:
        community_count = 0.0

    cycles = float(_count_cycles_limited(g, max_len=config.cycle_max_length))
    if cycles >= 1:
        detected.append("transaction_loop")

    # A simple graph-risk score proxy: high centrality + cycles + fragmented communities
    graph_centrality_score = float(min(1.0, (_safe_max(deg_vals) * 0.65 + _safe_mean(pr_vals) * 0.35)))

    features = dict(
        number_of_connected_accounts=float(n),
        graph_centrality_score=graph_centrality_score,
        pagerank_score=float(_safe_max(pr_vals)),
        transaction_cycles=cycles,
        community_count=community_count,
    )

    return GraphFeatureResult(features=features, detected_patterns=detected)


def graph_risk_score_from_features(features: dict[str, float]) -> float:
    """Map graph features into a normalized risk score in [0, 1]."""
    # heuristic mapping; in production this should be calibrated.
    n = float(features.get("number_of_connected_accounts", 0.0))
    cycles = float(features.get("transaction_cycles", 0.0))
    cent = float(features.get("graph_centrality_score", 0.0))
    pr = float(features.get("pagerank_score", 0.0))
    comm = float(features.get("community_count", 0.0))

    # Normalize components
    n_score = min(1.0, np.log1p(n) / np.log1p(200.0))
    cycles_score = min(1.0, cycles / 10.0)
    comm_score = min(1.0, comm / 10.0)

    score = 0.30 * n_score + 0.25 * cycles_score + 0.25 * cent + 0.15 * pr + 0.05 * comm_score
    return float(max(0.0, min(1.0, score)))

