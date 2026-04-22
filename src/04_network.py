"""Temporal co-activity network: 30-minute sliding window edge construction."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml


WINDOW_MINUTES = 30


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_coactivity_edges(df: pd.DataFrame, window_minutes: int = WINDOW_MINUTES) -> pd.DataFrame:
    """For every message, find all other messages within ±window_minutes.
    Each unique (A, B) pair that co-occurs in a window contributes +1 to edge weight.
    Self-loops are excluded. Returns an undirected weighted edge list.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].values.astype("datetime64[ns]")
    senders = df["sender_code"].values
    window = np.timedelta64(window_minutes, "m")

    pair_counts: dict[tuple[str, str], int] = {}

    for i in range(len(df)):
        if pd.isna(senders[i]):
            continue
        t = timestamps[i]
        # Binary search bounds for the window
        left = np.searchsorted(timestamps, t - window, side="left")
        right = np.searchsorted(timestamps, t + window, side="right")
        window_senders = set()
        for j in range(left, right):
            if j != i and not pd.isna(senders[j]):
                window_senders.add(senders[j])
        for other in window_senders:
            if other == senders[i]:
                continue
            a, b = tuple(sorted([senders[i], other]))
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["source", "target", "weight"])

    # Weights were double-counted (once from each direction), halve them
    rows = [{"source": a, "target": b, "weight": w // 2}
            for (a, b), w in pair_counts.items()]
    return pd.DataFrame(rows)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def compute_node_metrics(
    df: pd.DataFrame,
    edges: pd.DataFrame,
    label: str,
) -> list[dict]:
    """Compute per-node centrality and activity metrics for one time slice."""
    G = nx.Graph()
    for _, e in edges.iterrows():
        if e["weight"] > 0:
            G.add_edge(e["source"], e["target"], weight=int(e["weight"]))

    if G.number_of_nodes() == 0:
        return []

    deg_c = nx.degree_centrality(G)
    bet_c = nx.betweenness_centrality(G, weight="weight", normalized=True)
    try:
        eig_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig_c = {n: np.nan for n in G.nodes}

    msg_counts = df.groupby("sender_code").size().to_dict()
    gini_val = gini(np.array(list(msg_counts.values()), dtype=float))

    rows = []
    for node in G.nodes:
        rows.append({
            "sender_code": node,
            "phase": label,
            "degree_centrality": round(deg_c.get(node, 0.0), 6),
            "betweenness_centrality": round(bet_c.get(node, 0.0), 6),
            "eigenvector_centrality": round(eig_c.get(node, np.nan), 6),
            "message_count": msg_counts.get(node, 0),
            "co_activity_partners": G.degree(node),
            "gini_coefficient": round(gini_val, 6),
        })
    return rows


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["_year"] = df["timestamp"].dt.year.astype(str)
    return {year: sub.drop(columns="_year") for year, sub in df.groupby("_year")}


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])

    msgs = (
        df[df["message_type"] != "system"]
        .dropna(subset=["timestamp", "sender_code"])
        .copy()
    )

    # Full-period graph for GraphML export
    all_edges = build_coactivity_edges(msgs)
    G = nx.Graph()
    for _, e in all_edges.iterrows():
        if e["weight"] > 0:
            G.add_edge(e["source"], e["target"], weight=int(e["weight"]))

    graph_path = Path(settings["paths"]["network_graph"])
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graph_path)

    # Per-year metrics
    all_rows = []
    for year, sub in split_by_year(msgs).items():
        edges = build_coactivity_edges(sub)
        all_rows.extend(compute_node_metrics(sub, edges, label=year))

    metrics_df = pd.DataFrame(all_rows)

    out = Path(settings["paths"]["network_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out, index=False)
    print(f"[04_network] {len(G.nodes)} nodes, {len(G.edges)} edges → {graph_path}")
    print(f"[04_network] Wrote {len(metrics_df)} metric rows → {out}")


if __name__ == "__main__":
    main()
