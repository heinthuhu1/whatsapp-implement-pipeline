"""
Stage 09 – Cross-domain Table 1: structural overview and domain-level metrics by phase.

Reads pipeline outputs from earlier stages and produces a publication-ready
Table 1 with rows grouped by corpus structure and the four analytic domains
(temporal, relational, content, behavioural), and columns for the overall
period and each phase from config/settings.yaml.

Outputs:
  data/processed/table1_overview.csv  (tidy: section, metric, one column per phase)
  data/processed/table1_overview.txt  (fixed-width text, copy-paste ready)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import join_voice_translations, load_settings

FOOTNOTES = [
    "Content-bearing messages: free text plus voice notes with an English "
    "translation (joined by media filename). Media placeholders, deleted-message "
    "notices, untranslated voice notes, system notices and structured case-report "
    "messages are excluded from sentiment and lexicon denominators.",
    "Sentiment: mean DistilBERT SST-2 score (-1 to +1) over content-bearing "
    "messages; n = messages scored.",
    "Lexicon rates: % of content-bearing messages with a keyword at a word start "
    "(case-insensitive; 'monitor' matches 'monitoring', 'now' does not match "
    "'know'); shown as % (flagged/denominator).",
    "ToC coverage: % of messages mentioning a component (word-start match, voice "
    "notes replaced by their translation), shown as % (hits/messages); Overall "
    "pools hits and messages across phases. 'Excl. case reports' drops "
    "structured case-report messages from numerator and denominator.",
    "Gini of message counts per sender; corrected = Gini x n/(n-1) for n "
    "senders; top-sender share = largest sender's share of messages.",
    "Response latency: a turn is a maximal run of consecutive messages by one "
    "sender (system notices excluded); latency runs from the last message of a "
    "turn to the first message of the next turn (always another sender). "
    "Median (IQR) over turns with a next turn; n = such turns.",
    "Urgency-flagged messages: content-bearing messages matching an urgency "
    "keyword. Urgency-response latency: median time from a flagged message "
    "to the next message by a different sender (n = flagged messages with a "
    "later reply).",
]


# ── helpers ────────────────────────────────────────────────────────────────

def _norm_phase(v) -> str:
    """Normalise '2024.0' / 2024 / np.float to '2024'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return re.sub(r"\.0+$", "", str(v).strip())


def _months_touched(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _gini(values) -> float:
    v = np.sort(np.asarray(list(values), dtype=float))
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def _fmt_duration_hours(hours: float) -> str:
    if pd.isna(hours):
        return "–"
    if hours < 1:
        return f"{int(round(hours * 60))} min"
    return f"{hours:.1f} h"


def _safe_load_csv(path, **kwargs) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return None


# ── core computation ───────────────────────────────────────────────────────
def compute_cells(settings: dict) -> tuple[list, list[str]]:
    """Build the table rows and column list.

    Returns (rows, cols) where rows is a list of (label, vals_dict_or_None);
    None as the second element marks a section header. cols is the ordered
    list of column names (Overall + phase names from settings).
    """
    paths = settings["paths"]
    phase_defs = settings.get("phases", [])
    phase_names = [str(p["name"]) for p in phase_defs]
    cols = ["Overall"] + phase_names

    months_per_phase: dict[str, int] = {}
    for p in phase_defs:
        months_per_phase[str(p["name"])] = _months_touched(
            pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        )
    months_per_phase["Overall"] = sum(months_per_phase.values())

    # ── load source tables ─────────────────────────────────────────────────
    msg = _safe_load_csv(paths["messages_interim"], parse_dates=["timestamp"])
    if msg is None:
        raise SystemExit(f"messages.csv not found at {paths['messages_interim']}")
    msg = msg.dropna(subset=["timestamp"]).copy()
    msg["phase_clean"] = msg["phase"].map(_norm_phase)

    cases = _safe_load_csv(paths.get("case_reports", "data/interim/case_reports.csv"),
                           parse_dates=["timestamp"], low_memory=False)
    if cases is not None and "phase" in cases.columns:
        cases["phase_clean"] = cases["phase"].map(_norm_phase)

    vn = _safe_load_csv(paths.get("voice_notes_output",
                                  "data/interim/voice_notes_transcribed.csv"))
    if vn is not None and "date" in vn.columns:
        vn["date"] = pd.to_datetime(vn["date"], errors="coerce")
        vn["phase_clean"] = vn["date"].dt.year.map(_norm_phase)
        # Restrict analysis to the configured phase window. The full set of
        # transcripts stays in voice_notes.csv; notes dated outside the phases
        # (e.g. pre-2023) are excluded here so Overall == sum of phase columns.
        vn = vn[vn["phase_clean"].isin(phase_names)].copy()

    eng = _safe_load_csv(paths["engagement_metrics"])
    net = _safe_load_csv(paths["network_metrics"])
    if net is not None:
        net["phase_clean"] = net["phase"].map(_norm_phase)
    sen = _safe_load_csv(paths["sentiment_metrics"])
    if sen is not None and "phase" in sen.columns:
        sen["phase_clean"] = sen["phase"].map(_norm_phase)
    fid = _safe_load_csv(paths["fidelity_metrics"])
    if fid is not None and "phase" in fid.columns:
        fid["phase_clean"] = fid["phase"].map(_norm_phase)

    # ── slicers ────────────────────────────────────────────────────────────
    def slice_msg(p):
        return msg if p == "Overall" else msg[msg["phase_clean"] == p]

    def text_only(df):
        return df[df["message_type"] == "text"]

    def non_system(df):
        return df[df["message_type"] != "system"]

    def slice_case(p):
        if cases is None:
            return None
        return cases if p == "Overall" else cases[cases["phase_clean"] == p]

    def slice_vn(p):
        if vn is None:
            return None
        return vn if p == "Overall" else vn[vn["phase_clean"] == p]

    def slice_net(p):
        if net is None:
            return None
        return net if p == "Overall" else net[net["phase_clean"] == p]

    def slice_eng(metric):
        return eng[eng["metric"] == metric] if eng is not None else None

    def slice_sen(kind):
        return sen[sen["kind"] == kind] if sen is not None else None

    def slice_fid(kind):
        return fid[fid["kind"] == kind] if fid is not None else None

    def _full_graph():
        try:
            import networkx as nx
            return nx.read_graphml(paths["network_graph"])
        except Exception:
            return None

    # ── row builders ───────────────────────────────────────────────────────
    rows: list[tuple[str, dict | None]] = []

    def section(name):
        rows.append((name, None))

    def row(label, fn):
        rows.append((label, {p: fn(p) for p in cols}))

    # Corpus structure ─────────────────────────────────────────────────────
    section("Corpus structure")

    row("Months observed",
        lambda p: months_per_phase.get(p, "–"))
    row("Active senders (n)",
        lambda p: int(non_system(slice_msg(p))["sender_code"].dropna().nunique()))
    row("Text messages and system notices (n)",
        lambda p: f"{len(slice_msg(p)):,}")
    row("Mean messages per month",
        lambda p: f"{len(slice_msg(p)) / months_per_phase[p]:.0f}"
                  if p in months_per_phase and months_per_phase[p] else "–")
    row("Voice notes posted (n)",
        lambda p: int((slice_msg(p)["message_type"] == "voice_note").sum()))

    def vn_transcribed(p):
        s = slice_vn(p)
        if s is None:
            return "– (voice_notes.csv missing)"
        return int((s["status"] == "success").sum()) if "status" in s.columns else "–"

    row("Voice notes successfully transcribed (n)", vn_transcribed)

    vn_linked = join_voice_translations(
        non_system(msg)[non_system(msg)["message_type"] == "voice_note"],
        paths.get("voice_notes_output"))

    def vn_joined(p):
        s = vn_linked if p == "Overall" else vn_linked[vn_linked["phase_clean"] == p]
        return int(s["translation_en"].notna().sum())

    row("Voice notes linked to an English translation (n)", vn_joined)

    def vn_duration(p):
        s = slice_vn(p)
        if s is None or "duration_seconds" not in s.columns:
            return "–"
        v = s["duration_seconds"].dropna()
        return f"{v.mean():.1f}" if not v.empty else "–"

    row("Voice note mean duration (s)", vn_duration)

    def mean_words(p):
        sub = text_only(slice_msg(p))
        if sub.empty:
            return "–"
        wc = sub["message"].fillna("").astype(str).str.split().str.len()
        return f"{wc.mean():.1f}"

    row("Mean text-message length (words)", mean_words)

    row("Structured case reports extracted (n)",
        lambda p: f"{len(slice_case(p)):,}" if slice_case(p) is not None
                  else "– (case_reports.csv missing)")

    # Temporal domain ──────────────────────────────────────────────────────
    section("Temporal domain")

    def weekly_range(p):
        sub = non_system(slice_msg(p))
        if sub.empty:
            return "–"
        weekly = sub.set_index("timestamp").resample("W").size()
        rolling = weekly.rolling(4, min_periods=1).mean()
        return f"{rolling.min():.0f}–{rolling.max():.0f}"

    row("Weekly volume, four-week rolling mean (range)", weekly_range)

    def velocity_count(p):
        v = slice_eng("velocity")
        if v is None or v.empty or "spike" not in v.columns:
            return "–"
        v = v.copy()
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        spikes = v[v["spike"].astype(bool)]
        if p == "Overall":
            return int(len(spikes))
        return int((spikes["date"].dt.year.astype("Int64").map(_norm_phase) == p).sum())

    row("Velocity-spike days (n)", velocity_count)

    def silence_count(p):
        s = slice_eng("silence")
        if s is None or s.empty:
            return 0
        if p == "Overall":
            return int(len(s))
        if "gap_start" not in s.columns:
            return "–"
        s = s.copy()
        s["gap_start"] = pd.to_datetime(s["gap_start"], errors="coerce")
        return int((s["gap_start"].dt.year.astype("Int64").map(_norm_phase) == p).sum())

    row("Silence periods (n)", silence_count)

    # Relational domain ────────────────────────────────────────────────────
    section("Relational domain")

    def net_nodes(p):
        s = slice_net(p)
        if s is None:
            return "–"
        if p == "Overall":
            G = _full_graph()
            return G.number_of_nodes() if G else int(s["sender_code"].nunique())
        return int(s["sender_code"].nunique())

    row("Co-activity network nodes (n)", net_nodes)

    def net_edges(p):
        s = slice_net(p)
        if s is None or "co_activity_partners" not in s.columns:
            return "–"
        if p == "Overall":
            G = _full_graph()
            return G.number_of_edges() if G else "–"
        return int(s["co_activity_partners"].sum() // 2)

    row("Co-activity network edges (n)", net_edges)

    def net_density(p):
        s = slice_net(p)
        if s is None:
            return "–"
        if p == "Overall":
            G = _full_graph()
            if not G:
                return "–"
            try:
                import networkx as nx
                return f"{nx.density(G):.2f}"
            except Exception:
                return "–"
        n = int(s["sender_code"].nunique())
        if n < 2 or "co_activity_partners" not in s.columns:
            return "–"
        edges = s["co_activity_partners"].sum() / 2
        return f"{edges / (n * (n - 1) / 2):.2f}"

    row("Network density", net_density)

    def gini_coef(p):
        if p == "Overall":
            counts = non_system(msg).groupby("sender_code").size().values
            return f"{_gini(counts):.2f}"
        s = slice_net(p)
        if s is None or "gini_coefficient" not in s.columns:
            return "–"
        v = s["gini_coefficient"].dropna()
        return f"{v.iloc[0]:.2f}" if not v.empty else "–"

    row("Gini coefficient of message counts", gini_coef)

    def _sender_counts(p):
        sub = non_system(slice_msg(p))
        return sub.groupby("sender_code").size().values

    def gini_corrected(p):
        if p != "Overall":
            s = slice_net(p)
            if s is not None and "gini_corrected" in s.columns:
                v = s["gini_corrected"].dropna()
                return f"{v.iloc[0]:.2f}" if not v.empty else "–"
        counts = _sender_counts(p)
        n = len(counts)
        return f"{_gini(counts) * n / (n - 1):.2f}" if n > 1 else "–"

    row("Gini coefficient, small-sample corrected", gini_corrected)

    def top_share(p):
        if p != "Overall":
            s = slice_net(p)
            if s is not None and "top_sender_share" in s.columns:
                v = s["top_sender_share"].dropna()
                return f"{v.iloc[0] * 100:.1f}" if not v.empty else "–"
        counts = _sender_counts(p)
        return f"{counts.max() / counts.sum() * 100:.1f}" if counts.sum() else "–"

    row("Top-sender share of messages (%)", top_share)

    def partners_mean(p):
        if p == "Overall":
            G = _full_graph()
            if G:
                degs = [G.degree(n) for n in G.nodes]
                return f"{np.mean(degs):.1f}" if degs else "–"
        s = slice_net(p)
        if s is None or "co_activity_partners" not in s.columns:
            return "–"
        return f"{s['co_activity_partners'].mean():.1f}"

    row("Mean co-activity partners per sender", partners_mean)

    # Content domain ───────────────────────────────────────────────────────
    section("Content domain")

    def sen_row(p):
        """Phase (or overall) summary row from 05_sentiment, or None."""
        kind = "overall_summary" if p == "Overall" else "phase_summary"
        ph = slice_sen(kind)
        if ph is None or ph.empty:
            return None
        sub = ph if p == "Overall" else ph[ph["phase_clean"] == p]
        return None if sub.empty else sub.iloc[0]

    def sentiment_mean(p):
        r = sen_row(p)
        if r is None or "sentiment_n" not in r.index or pd.isna(r["sentiment_mean"]):
            return "–"
        return f"{r['sentiment_mean']:+.2f} (n={int(r['sentiment_n']):,})"

    row("Mean sentiment score, content-bearing messages", sentiment_mean)

    def rate(name):
        def f(p):
            r = sen_row(p)
            if r is None or f"{name}_n" not in r.index or not r["n_content"]:
                return "–"
            k, n = int(r[f"{name}_n"]), int(r["n_content"])
            return f"{k / n * 100:.1f} ({k}/{n:,})"
        return f

    row("Urgency-language rate, % (n/N)", rate("urgency"))
    row("Peer-support language rate, % (n/N)", rate("peer_support"))

    def toc_extreme(kind, which):
        def f(p):
            cov = slice_fid(kind)
            if cov is None or cov.empty or "n_messages" not in cov.columns:
                return "–"
            if p == "Overall":
                g = cov.groupby("component")
                sub = pd.DataFrame({"hits": g["hits"].sum(),
                                    "n_messages": g["n_messages"].sum()}).reset_index()
            else:
                sub = cov[cov["phase_clean"] == p]
            if sub.empty:
                return "–"
            pct = sub["hits"] / sub["n_messages"] * 100
            r = sub.loc[pct.idxmax() if which == "max" else pct.idxmin()]
            comp = str(r["component"]).replace("_", " ").capitalize()
            return (f"{comp} {r['hits'] / r['n_messages'] * 100:.1f} "
                    f"({int(r['hits'])}/{int(r['n_messages']):,})")
        return f

    row("Highest ToC coverage, all messages, % (n/N)", toc_extreme("coverage", "max"))
    row("Lowest ToC coverage, all messages, % (n/N)", toc_extreme("coverage", "min"))
    row("Highest ToC coverage, excl. case reports, % (n/N)",
        toc_extreme("coverage_excl_case_reports", "max"))
    row("Lowest ToC coverage, excl. case reports, % (n/N)",
        toc_extreme("coverage_excl_case_reports", "min"))

    # Behavioural domain ───────────────────────────────────────────────────
    section("Behavioural domain")

    def lat_row(p):
        lat = slice_eng("turn_latency")
        if lat is None or lat.empty or "latency_median_min" not in lat.columns:
            return None
        sub = lat[lat["phase"].map(_norm_phase) == p]
        return None if sub.empty else sub.iloc[0]

    def latency_median(p):
        r = lat_row(p)
        if r is None or pd.isna(r["latency_median_min"]):
            return "–"
        return (f"{r['latency_median_min']:.0f} min ({r['latency_q1_min']:.0f}–"
                f"{r['latency_q3_min']:.0f})")

    row("Response latency between turns, median (IQR)", latency_median)
    row("Turns followed by another sender's turn (n)",
        lambda p: f"{int(lat_row(p)['n_turns']):,}" if lat_row(p) is not None else "–")
    row("Turns answered within 1 h (%)",
        lambda p: f"{lat_row(p)['share_le_60min'] * 100:.1f}"
                  if lat_row(p) is not None and pd.notna(lat_row(p)["share_le_60min"]) else "–")

    def urgency_count(p):
        r = sen_row(p)
        if r is None or "urgency_n" not in r.index:
            return "–"
        return int(r["urgency_n"])

    row("Urgency-flagged messages (n)", urgency_count)

    def urgency_latency(p):
        resp = slice_fid("responsiveness")
        if resp is None or resp.empty or "latency_minutes" not in resp.columns:
            return "–"
        sub = resp if p == "Overall" else resp[resp["phase_clean"] == p]
        if sub.empty:
            return "– (n=0)"
        v = sub["latency_minutes"].median()
        return f"{_fmt_duration_hours(v / 60.0)} (n={len(sub)})"

    row("Urgency-response latency, median (n)", urgency_latency)

    def after_hours(p):
        start_h = settings.get("after_hours_start", 22)
        end_h = settings.get("after_hours_end", 7)
        sub = non_system(slice_msg(p))
        if sub.empty:
            return "–"
        hours = sub["timestamp"].dt.hour
        mask = (hours >= start_h) | (hours < end_h)
        return f"{mask.mean() * 100:.1f}"

    row("After-hours activity rate (%)", after_hours)

    return rows, cols


# ── output rendering ───────────────────────────────────────────────────────
def to_dataframe(rows, cols) -> pd.DataFrame:
    """Convert rows/cols into a tidy DataFrame for CSV export."""
    records = []
    section = ""
    for label, vals in rows:
        if vals is None:
            section = label
            continue
        rec = {"section": section, "metric": label}
        rec.update({c: vals[c] for c in cols})
        records.append(rec)
    return pd.DataFrame(records)


def to_text(rows, cols) -> str:
    """Render a fixed-width text table for terminal display and copy-paste."""
    label_w = max(len(label) for label, _ in rows) + 4
    col_w = max([22] + [len(str(v)) + 3 for _, vals in rows if vals
                        for v in vals.values()])

    lines = ["Table 1 cell values".ljust(label_w)
             + "".join(c.rjust(col_w) for c in cols)]
    lines.append("─" * (label_w + col_w * len(cols)))
    for label, vals in rows:
        if vals is None:
            lines.extend(["", ("  " + label).upper(), ""])
        else:
            line = ("  " + label).ljust(label_w)
            line += "".join(str(vals[c]).rjust(col_w) for c in cols)
            lines.append(line)
    lines.append("")
    lines.append("Notes")
    lines.extend(f"  {i}. {note}" for i, note in enumerate(FOOTNOTES, start=1))
    lines.append("")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    settings = load_settings()
    rows, cols = compute_cells(settings)

    csv_path = Path(settings["paths"].get(
        "table1_overview", "data/processed/table1_overview.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = to_dataframe(rows, cols)
    df.to_csv(csv_path, index=False)
    print(f"[09_table1] Wrote {csv_path}")

    text = to_text(rows, cols)
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"[09_table1] Wrote {txt_path}")

    print()
    print(text)


if __name__ == "__main__":
    main()
