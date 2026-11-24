"""Shared helpers for the R2 #19 / #20 network and phase re-analyses (Jheeng archive only).

Reads ONLY the archived Jheeng data under data/Jheeng/. Writes nothing itself.
The co-activity and Gini functions replicate src/04_network.py exactly so that
original numbers can be reproduced; alternatives are added alongside.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
JH = ROOT / "data" / "Jheeng"
OUT = JH / "revision" / "network_phase"
OUT.mkdir(parents=True, exist_ok=True)

SETTINGS = yaml.safe_load(open(JH / "config" / "settings.yaml"))
PHASES = {p["name"]: (pd.Timestamp(p["start"]), pd.Timestamp(p["end"]) + pd.Timedelta(hours=23, minutes=59, seconds=59))
          for p in SETTINGS["phases"]}
PHASE_NAMES = list(PHASES)
STUDY_START = pd.Timestamp("2024-09-22")
STUDY_END = pd.Timestamp("2026-04-11 23:59:59")
TEMPLATE_BREAK = pd.Timestamp("2026-02-02")  # empirically identified in phase_change.py


def phase_days(name: str) -> int:
    s, e = PHASES[name]
    return (e.normalize() - s.normalize()).days + 1


def load_messages() -> pd.DataFrame:
    df = pd.read_csv(JH / "interim" / "messages.csv", parse_dates=["timestamp"])
    df["phase"] = df["phase"].astype(str)
    df["row"] = np.arange(len(df))  # export order, used to break same-minute ties
    return df


def engagement_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Base used by src/03_engagement.py: non-system rows, sender may be empty."""
    return df[df["message_type"] != "system"].dropna(subset=["timestamp"]).copy()


def sender_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Base used by src/04_network.py: non-system rows with a sender code."""
    return df[df["message_type"] != "system"].dropna(subset=["timestamp", "sender_code"]).copy()


# ---------------------------------------------------------------- edges
def coactivity_edges(df: pd.DataFrame, window_minutes: float = 30) -> pd.DataFrame:
    """Exact replica of src/04_network.build_coactivity_edges (window parameterised)."""
    df = df.sort_values(["timestamp", "row"]).reset_index(drop=True)
    ts = df["timestamp"].values.astype("datetime64[ns]")
    snd = df["sender_code"].values
    w = np.timedelta64(int(window_minutes * 60), "s")
    pair: dict[tuple[str, str], int] = {}
    for i in range(len(df)):
        if pd.isna(snd[i]):
            continue
        lo = np.searchsorted(ts, ts[i] - w, side="left")
        hi = np.searchsorted(ts, ts[i] + w, side="right")
        others = {snd[j] for j in range(lo, hi) if j != i and not pd.isna(snd[j])}
        for o in others:
            if o == snd[i]:
                continue
            k = tuple(sorted((snd[i], o)))
            pair[k] = pair.get(k, 0) + 1
    rows = [{"source": a, "target": b, "weight": c // 2} for (a, b), c in pair.items()]
    return pd.DataFrame(rows, columns=["source", "target", "weight"])


def sequential_edges(df: pd.DataFrame, max_gap_minutes: float | None = None) -> pd.DataFrame:
    """Edge A-B +1 each time a message by B immediately follows a message by A (A != B).
    Undirected; optional cap on the gap between the two consecutive messages."""
    d = df.sort_values(["timestamp", "row"]).reset_index(drop=True)
    s = d["sender_code"].values
    t = d["timestamp"].values
    pair: dict[tuple[str, str], int] = {}
    for i in range(1, len(d)):
        if s[i] == s[i - 1]:
            continue
        if max_gap_minutes is not None and (t[i] - t[i - 1]) > np.timedelta64(int(max_gap_minutes * 60), "s"):
            continue
        k = tuple(sorted((s[i], s[i - 1])))
        pair[k] = pair.get(k, 0) + 1
    rows = [{"source": a, "target": b, "weight": c} for (a, b), c in pair.items()]
    return pd.DataFrame(rows, columns=["source", "target", "weight"])


def graph_from_edges(edges: pd.DataFrame, nodes: list | None = None) -> nx.Graph:
    G = nx.Graph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    for a, b, w in edges[["source", "target", "weight"]].itertuples(index=False):
        if w > 0:
            G.add_edge(a, b, weight=float(w), dist=1.0 / float(w))
    return G


# ---------------------------------------------------------------- inequality
def gini(values) -> float:
    """Exact replica of src/04_network.gini."""
    v = np.sort(np.asarray(list(values), dtype=float))
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def gini_corrected(values) -> float:
    """Small-sample corrected Gini, G * n/(n-1), so the maximum is 1 for any n."""
    v = np.asarray(list(values), dtype=float)
    n = v.size
    return gini(v) * n / (n - 1) if n > 1 else float("nan")


# ---------------------------------------------------------------- summaries
def network_summary(msgs: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Original-style summary (isolates excluded by construction) plus with-isolate variants."""
    G = graph_from_edges(edges)
    counts = msgs.groupby("sender_code").size()
    active = list(counts.index)
    n, m = G.number_of_nodes(), G.number_of_edges()
    isolates = sorted(set(active) - set(G.nodes))
    n_all = len(active)
    out = {
        "active_senders": n_all,
        "nodes": n,
        "edges": m,
        "density": m / (n * (n - 1) / 2) if n > 1 else np.nan,
        "mean_partners": 2 * m / n if n else np.nan,
        "density_incl_isolates": m / (n_all * (n_all - 1) / 2) if n_all > 1 else np.nan,
        "mean_partners_incl_isolates": 2 * m / n_all if n_all else np.nan,
        "n_isolates": len(isolates),
        "isolates": ";".join(isolates),
        "gini": gini(counts.values),
        "gini_corrected": gini_corrected(counts.values),
        "total_weight": float(edges["weight"].sum()) if len(edges) else 0.0,
    }
    if n > 1:
        b_orig = nx.betweenness_centrality(G, weight="weight", normalized=True)  # as in src/04
        b_unw = nx.betweenness_centrality(G, weight=None, normalized=True)
        b_inv = nx.betweenness_centrality(G, weight="dist", normalized=True)
        for tag, b in [("orig", b_orig), ("unw", b_unw), ("invw", b_inv)]:
            top = max(b, key=lambda k: (b[k], k))
            out[f"top_betweenness_{tag}"] = f"{top} ({b[top]:.2f})"
            out[f"top_betweenness_{tag}_value"] = b[top]
        out["n_components"] = nx.number_connected_components(G)
    return out


def node_centralities(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() < 2:
        return pd.DataFrame()
    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    b_unw = nx.betweenness_centrality(G, weight=None, normalized=True)
    b_invw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    b_orig = nx.betweenness_centrality(G, weight="weight", normalized=True)
    try:
        e_unw = nx.eigenvector_centrality(G, weight=None, max_iter=5000)
    except nx.PowerIterationFailedConvergence:
        e_unw = {k: np.nan for k in G}
    try:
        e_w = nx.eigenvector_centrality(G, weight="weight", max_iter=5000)
    except nx.PowerIterationFailedConvergence:
        e_w = {k: np.nan for k in G}
    rows = []
    for k in G.nodes:
        rows.append({"sender_code": k, "degree": deg[k], "strength": strength[k],
                     "betweenness_unweighted": b_unw[k], "betweenness_inverse_weight": b_invw[k],
                     "betweenness_weight_as_distance_original": b_orig[k],
                     "eigenvector_unweighted": e_unw[k], "eigenvector_weighted": e_w[k]})
    return pd.DataFrame(rows)


def in_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))]


def silence_periods(eng: pd.DataFrame, hours: float = 48) -> pd.DataFrame:
    """Replica of src/03_engagement.silence_periods on the full series."""
    ts = eng["timestamp"].sort_values().reset_index(drop=True)
    gaps = ts.diff().dt.total_seconds() / 3600
    mask = gaps > hours
    return pd.DataFrame({"gap_start": ts.shift(1)[mask].values, "gap_end": ts[mask].values,
                         "gap_hours": gaps[mask].values})


def velocity_spikes(eng: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Replica of src/03_engagement.velocity_spikes on the full series."""
    daily = eng.set_index("timestamp").resample("D").size().rename("count")
    roll = daily.rolling(window=window_days, min_periods=1)
    mean, sd = roll.mean(), roll.std().fillna(0)
    return pd.DataFrame({"date": daily.index, "count": daily.values, "spike": (daily > mean + 2 * sd).values})


def fmt(x, nd=2):
    if isinstance(x, (float, np.floating)):
        return "NA" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def flag_case_reports(df: pd.DataFrame) -> pd.Series:
    """True for messages that are structured case-dispatch reports: either text identical to a
    raw_message in case_reports.csv, or template-formatted (a 'gender' field plus a 'case' or
    'date' field). The second rule recovers reposted or unparsed reports."""
    cr = pd.read_csv(JH / "interim" / "case_reports.csv")
    txt = df["message"].fillna("")
    low = txt.str.lower()
    rule = low.str.contains("gender") & (low.str.contains("case") | low.str.contains("date")) & (txt.str.count(":") >= 2)
    return txt.isin(set(cr["raw_message"])) | rule
