"""Directed reply network: centralities, Gini, membership timeline."""
from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

REPLY_QUOTE_RE = re.compile(r"^>\s*(\S+):", re.MULTILINE)


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_reply_target(message: str) -> str | None:
    if not isinstance(message, str):
        return None
    m = REPLY_QUOTE_RE.search(message)
    return m.group(1) if m else None


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    edges = []
    for i, row in df.iterrows():
        target = extract_reply_target(row["message"])
        if target and pd.notna(row["sender_code"]):
            edges.append({"source": row["sender_code"], "target": target,
                          "phase": row["phase"], "timestamp": row["timestamp"]})
            continue
        if i > 0 and pd.notna(row["sender_code"]):
            prev = df.iloc[i - 1]
            if pd.notna(prev["sender_code"]) and prev["sender_code"] != row["sender_code"]:
                edges.append({"source": row["sender_code"],
                              "target": prev["sender_code"],
                              "phase": row["phase"],
                              "timestamp": row["timestamp"]})
    return pd.DataFrame(edges)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def phase_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, sub in edges.groupby("phase"):
        G = nx.DiGraph()
        for _, e in sub.iterrows():
            if G.has_edge(e["source"], e["target"]):
                G[e["source"]][e["target"]]["weight"] += 1
            else:
                G.add_edge(e["source"], e["target"], weight=1)
        if G.number_of_nodes() == 0:
            continue
        bet = nx.betweenness_centrality(G)
        deg = dict(G.degree())
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eig = {n: np.nan for n in G.nodes}
        for node in G.nodes:
            rows.append({"phase": phase, "node": node,
                         "betweenness": bet[node], "degree": deg[node],
                         "eigenvector": eig.get(node, np.nan)})
    return pd.DataFrame(rows)


def message_share_gini(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for phase, sub in df.groupby("phase"):
        counts = sub.groupby("sender_code").size().values
        out.append({"phase": phase, "gini_message_share": gini(counts)})
    return pd.DataFrame(out)


def membership_timeline(df: pd.DataFrame) -> pd.DataFrame:
    sys_msgs = df[df["message_type"] == "system"].copy()
    events = []
    for _, row in sys_msgs.iterrows():
        msg = row["message"] or ""
        event = None
        if "added" in msg.lower():
            event = "added"
        elif "left" in msg.lower():
            event = "left"
        elif "removed" in msg.lower():
            event = "removed"
        elif "joined" in msg.lower():
            event = "joined"
        elif "changed the group" in msg.lower():
            event = "group_changed"
        if event:
            events.append({"timestamp": row["timestamp"],
                           "event": event, "detail": msg})
    return pd.DataFrame(events)


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])

    msgs = df[df["message_type"] != "system"].dropna(subset=["timestamp"])
    edges = build_edges(msgs)

    G = nx.DiGraph()
    for _, e in edges.iterrows():
        if G.has_edge(e["source"], e["target"]):
            G[e["source"]][e["target"]]["weight"] += 1
        else:
            G.add_edge(e["source"], e["target"], weight=1)

    graph_path = Path(settings["paths"]["network_graph"])
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graph_path)

    metrics = phase_metrics(edges)
    ginis = message_share_gini(msgs)
    members = membership_timeline(df)

    metrics = metrics.assign(kind="centrality")
    ginis = ginis.assign(kind="gini")
    members = members.assign(kind="membership")
    combined = pd.concat([metrics, ginis, members], ignore_index=True, sort=False)

    out = Path(settings["paths"]["network_metrics"])
    combined.to_csv(out, index=False)
    print(f"[04_network] Wrote {out} and {graph_path}")


if __name__ == "__main__":
    main()
