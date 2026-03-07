from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class GraphBuildResult:
    graph: nx.DiGraph


def build_transaction_graph(transactions: pd.DataFrame) -> GraphBuildResult:
    """Build a directed transaction graph (accounts as nodes, transactions as edges).

    Expected columns:
        sender_account, receiver_account, amount, timestamp, client_id

    Notes:
        - Uses a simple DiGraph with aggregated edge attributes for scalability.
        - For highly multi-edge datasets, consider MultiDiGraph + edge aggregation offline.
    """
    g = nx.DiGraph()

    if transactions.empty:
        return GraphBuildResult(graph=g)

    cols = {"sender_account", "receiver_account", "amount", "timestamp", "client_id"}
    missing = cols - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions missing required columns: {sorted(missing)}")

    # Add edges with aggregation
    for row in transactions.itertuples(index=False):
        u = getattr(row, "sender_account")
        v = getattr(row, "receiver_account")
        amt = float(getattr(row, "amount"))
        if u is None or v is None:
            continue
        if u not in g:
            g.add_node(u)
        if v not in g:
            g.add_node(v)
        if g.has_edge(u, v):
            g[u][v]["count"] += 1
            g[u][v]["total_amount"] += amt
            g[u][v]["max_amount"] = max(g[u][v]["max_amount"], amt)
        else:
            g.add_edge(u, v, count=1, total_amount=amt, max_amount=amt)

    return GraphBuildResult(graph=g)

