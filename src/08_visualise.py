"""
Stage 08 — Publication-ready figures for methods paper.
Produces 5 figures + 1 summary table, saved to data/processed/figures/.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import networkx as nx
import yaml

# ── config ──────────────────────────────────────────────────────────────────
with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)

PHASES = cfg.get("phases", [])
MILESTONES = cfg.get("milestones", {})

OUT_DIR = "data/processed/figures"
os.makedirs(OUT_DIR, exist_ok=True)

PHASE_COLORS = {
    "baseline":       "#d4e6f1",
    "implementation": "#d5f5e3",
    "sustainment":    "#fdebd0",
}
DEFAULT_COLOR = "#f2f3f4"

STYLE = {
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
}
plt.rcParams.update(STYLE)

def phase_color(p):
    return PHASE_COLORS.get(str(p).lower(), DEFAULT_COLOR)

def shade_phases(ax, phases, y0=0, y1=1, transform="axes"):
    """Draw translucent phase bands on an axis."""
    for ph in phases:
        try:
            x0 = pd.Timestamp(ph["start"])
            x1 = pd.Timestamp(ph["end"])
            ax.axvspan(x0, x1, alpha=0.12, color=phase_color(ph["name"]),
                       zorder=0, label=f'_{ph["name"]}')
        except Exception:
            pass

def add_phase_legend(ax, phases):
    handles = [mpatches.Patch(facecolor=phase_color(p["name"]), alpha=0.4,
                               label=p["name"].capitalize())
               for p in phases]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.6)


# ── Figure 1: Weekly message volume ─────────────────────────────────────────
def fig_engagement():
    eng_all = pd.read_csv("data/processed/engagement_metrics.csv")
    eng = eng_all[eng_all["metric"] == "weekly_volume"].copy()
    if eng.empty:
        print("  [08] engagement: no weekly_volume rows found")
        return
    week_col  = "timestamp"
    count_col = "message_count"

    eng[week_col] = pd.to_datetime(eng[week_col])
    eng = eng.sort_values(week_col)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    shade_phases(ax, PHASES)

    ax.bar(eng[week_col], eng[count_col], width=6, color="#2980b9", alpha=0.7,
           label="Weekly messages")

    # rolling mean
    rolling = eng[count_col].rolling(4, center=True).mean()
    ax.plot(eng[week_col], rolling, color="#e74c3c", linewidth=1.5,
            label="4-week rolling mean")

    # milestone lines
    milestone_styles = {"training_start": ("Training", "#8e44ad"),
                        "launch":         ("Launch",   "#27ae60"),
                        "mid_review":     ("Mid-review","#f39c12"),
                        "final_review":   ("Final review","#c0392b")}
    for key, (label, color) in milestone_styles.items():
        if key in MILESTONES:
            try:
                ax.axvline(pd.Timestamp(MILESTONES[key]), color=color,
                           linestyle="--", linewidth=1, alpha=0.7, label=label)
            except Exception:
                pass

    ax.set_xlabel("Week")
    ax.set_ylabel("Messages per week")
    ax.set_title("Figure 1  Weekly message volume with phase periods", fontweight="bold", pad=10)
    add_phase_legend(ax, PHASES)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_engagement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] Saved {path}")


# ── Figure 2: Network graph — implementation vs sustainment ─────────────────
def fig_network():
    try:
        G_full = nx.read_graphml("data/processed/network_edges.graphml")
    except Exception as e:
        print(f"  [08] network graph: {e}")
        return

    msg = pd.read_csv("data/interim/messages.csv")
    msg["timestamp"] = pd.to_datetime(msg["timestamp"])

    phase_defs = {p["name"]: (pd.Timestamp(p["start"]), pd.Timestamp(p["end"]))
                  for p in PHASES}

    target_phases = [p for p in ["implementation", "sustainment", "baseline"]
                     if p in phase_defs][:2]
    if not target_phases:
        target_phases = list(phase_defs.keys())[:2]

    n_panels = len(target_phases)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, phase_name in zip(axes, target_phases):
        start, end = phase_defs[phase_name]
        phase_msg = msg[(msg["timestamp"] >= start) & (msg["timestamp"] <= end)]
        senders = set(phase_msg["sender_code"].dropna().unique())

        G = nx.Graph()
        for u, v, d in G_full.edges(data=True):
            if u in senders and v in senders:
                G.add_edge(u, v, weight=float(d.get("weight", 1)))
        for s in senders:
            G.add_node(s)

        if len(G.nodes) == 0:
            ax.set_title(f"{phase_name.capitalize()}\n(no data)", fontweight="bold")
            ax.axis("off")
            continue

        pos = nx.spring_layout(G, seed=42, k=1.5)
        degrees = dict(G.degree())
        node_sizes = [300 + degrees.get(n, 0) * 120 for n in G.nodes]
        edge_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 3 * w / max_w for w in edge_weights]

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               alpha=0.4, edge_color="#7f8c8d")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color="#2980b9", alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white",
                                font_weight="bold")

        density = nx.density(G) if len(G.nodes) > 1 else 0
        ax.set_title(
            f"{phase_name.capitalize()}\n{len(G.nodes)} nodes · {len(G.edges)} edges · density={density:.2f}",
            fontweight="bold", fontsize=10
        )
        ax.axis("off")

    fig.suptitle("Figure 2  Co-activity network by study phase\n"
                 "(node size = degree; edge width = co-activity frequency)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig2_network.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] Saved {path}")


# ── Figure 3: Sentiment trends ───────────────────────────────────────────────
def fig_sentiment():
    sent_all = pd.read_csv("data/processed/sentiment_metrics.csv")

    # split into daily time-series and phase summaries
    daily = sent_all[sent_all["kind"] == "rolling_sentiment"].copy()
    summary = sent_all[sent_all["kind"] == "phase_summary"].copy()

    if daily.empty:
        print("  [08] sentiment: no rolling_sentiment rows")
        return

    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    daily = daily.sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    # panel A — rolling sentiment time series
    col_14d = "sentiment_14d" if "sentiment_14d" in daily.columns else "sentiment_7d"
    col_7d  = "sentiment_7d"  if "sentiment_7d"  in daily.columns else col_14d
    ax = axes[0]
    shade_phases(ax, PHASES)
    if col_14d in daily.columns:
        ax.plot(daily["timestamp"], daily[col_14d], color="#2c3e50", linewidth=1.5,
                label="14-day rolling")
    if col_7d in daily.columns and col_7d != col_14d:
        ax.plot(daily["timestamp"], daily[col_7d], color="#7f8c8d", linewidth=0.8,
                alpha=0.6, label="7-day rolling")
    ax.axhline(0, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Sentiment score\n(negative ← 0 → positive)")
    ax.set_title("A  Sentiment over time", fontweight="bold", loc="left")
    ax.legend(fontsize=8, framealpha=0.6)

    # panel B — urgency & peer support per year (bar chart from phase summaries)
    ax2 = axes[1]
    def _norm_phase(v):
        try: return str(int(float(v)))
        except: return str(v)
    summary["phase_label"] = summary["phase"].apply(_norm_phase)
    summary = summary.sort_values("phase_label")

    x = np.arange(len(summary))
    w = 0.35
    ax2.bar(x - w/2, summary["urgency_rate"] * 100, width=w,
            color="#e74c3c", alpha=0.8, label="Urgency %")
    ax2.bar(x + w/2, summary["peer_support_rate"] * 100, width=w,
            color="#27ae60", alpha=0.8, label="Peer support %")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary["phase_label"])
    ax2.set_ylabel("Rate (%)")
    ax2.set_xlabel("Year")
    ax2.set_title("B  Urgency and peer support rates by year", fontweight="bold", loc="left")
    ax2.legend(fontsize=8, framealpha=0.6)

    fig.suptitle("Figure 3  Sentiment and communication style over time",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig3_sentiment.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] Saved {path}")


# ── Figure 4: Fidelity heatmap ───────────────────────────────────────────────
def fig_fidelity():
    fid = pd.read_csv("data/processed/fidelity_metrics.csv")
    cov = fid[fid["kind"] == "coverage"].copy()
    if cov.empty:
        print("  [08] fidelity: no coverage rows")
        return

    pivot = cov.pivot_table(index="component", columns="phase",
                            values="coverage_pct", aggfunc="mean")

    # order components consistently
    comp_order = ["training", "coordination", "clinical_protocol",
                  "equipment", "supervision"]
    pivot = pivot.reindex([c for c in comp_order if c in pivot.index])
    pivot.index = [c.replace("_", " ").title() for c in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(2.5, np.nanmax(data)))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c).capitalize() for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 1.5 else "white")

    plt.colorbar(im, ax=ax, label="Coverage (proportion of weeks with mention)")
    ax.set_title("Figure 4  Theory of Change fidelity by component and phase\n"
                 "(green = strong coverage; red = gap)",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_fidelity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] Saved {path}")


# ── Figure 5: Case report profile ───────────────────────────────────────────
def fig_case_reports():
    try:
        cr = pd.read_csv("data/interim/case_reports.csv")
    except FileNotFoundError:
        print("  [08] case_reports.csv not found — skipping Fig 5")
        return

    complaint_col = next((c for c in cr.columns
                          if "complaint" in c.lower() or "initial" in c.lower()), None)
    gender_col    = next((c for c in cr.columns if "gender" in c.lower()), None)

    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # panel A — top complaints
    if complaint_col:
        top = cr[complaint_col].value_counts().head(10)
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top)))[::-1]
        bars = ax1.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        for bar, val in zip(bars, top.values[::-1]):
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=8)
        ax1.set_xlabel("Number of cases")
        ax1.set_title("A  Top 10 presenting complaints", fontweight="bold", loc="left")
        ax1.grid(axis="y", alpha=0)

    # panel B — gender pie
    if gender_col:
        # normalise variants
        g = cr[gender_col].str.lower().str.strip()
        g = g.replace({"male child": "child", "children": "child",
                       "mal": "male", "old female": "female"})
        counts = g.value_counts()
        # keep top 4
        top_g = counts.head(4)
        wedge_colors = ["#5dade2", "#ec407a", "#66bb6a", "#ffa726"]
        ax2.pie(top_g.values, labels=[l.title() for l in top_g.index],
                autopct="%1.0f%%", colors=wedge_colors[:len(top_g)],
                startangle=140, textprops={"fontsize": 9})
        ax2.set_title("B  Patient gender", fontweight="bold", loc="left")

    fig.suptitle(f"Figure 5  Case report profile  (n={len(cr)})",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_case_reports.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] Saved {path}")


# ── Table 1: Implementation summary ─────────────────────────────────────────
def table_summary():
    try:
        impl = pd.read_csv("data/processed/implementation_summary.csv")
    except FileNotFoundError:
        print("  [08] implementation_summary.csv not found")
        return

    net  = pd.read_csv("data/processed/network_metrics.csv")
    sent = pd.read_csv("data/processed/sentiment_metrics.csv")
    fid  = pd.read_csv("data/processed/fidelity_metrics.csv")

    # build a clean display table
    rows = []
    for phase in impl["phase"].unique():
        r = {"Phase": str(phase).capitalize()}

        # messages
        msg = pd.read_csv("data/interim/messages.csv")
        phase_msg = msg[msg["phase"] == phase] if "phase" in msg.columns else pd.DataFrame()
        r["Messages (n)"] = len(phase_msg) if not phase_msg.empty else "—"

        # network
        net_p = net[net["phase"].astype(str) == str(phase)]
        if not net_p.empty:
            r["Network density (Gini)"] = f"{net_p['gini_coefficient'].iloc[0]:.2f}"
            r["Avg co-activity partners"] = f"{net_p['co_activity_partners'].mean():.1f}"
        else:
            r["Network density (Gini)"] = "—"
            r["Avg co-activity partners"] = "—"

        # sentiment — normalise float phases (2024.0 → "2024") before matching
        def _norm(v):
            try: return str(int(float(v)))
            except: return str(v)
        sent_p = sent[sent["phase"].apply(_norm) == _norm(phase)] if "phase" in sent.columns else pd.DataFrame()
        def _fmt_sent(df):
            sm = df["sentiment_mean"].mean()
            ur = df["urgency_rate"].mean() * 100
            ps = df["peer_support_rate"].mean() * 100
            return (f"{sm:.3f}" if pd.notna(sm) else "—",
                    f"{ur:.1f}%" if pd.notna(ur) else "—",
                    f"{ps:.1f}%" if pd.notna(ps) else "—")
        if not sent_p.empty and sent_p["sentiment_mean"].notna().any():
            r["Sentiment (mean)"], r["Urgency rate"], r["Peer support rate"] = _fmt_sent(sent_p)
        else:
            r["Sentiment (mean)"] = "—"
            r["Urgency rate"] = "—"
            r["Peer support rate"] = "—"

        # fidelity — best and worst
        fid_p = fid[(fid["phase"].astype(str) == str(phase)) & (fid["kind"] == "coverage")]
        if not fid_p.empty:
            best = fid_p.loc[fid_p["coverage_pct"].idxmax()]
            worst = fid_p.loc[fid_p["coverage_pct"].idxmin()]
            r["Strongest ToC component"] = f"{best['component'].replace('_',' ')} ({best['coverage_pct']:.2f})"
            r["Weakest ToC component"] = f"{worst['component'].replace('_',' ')} ({worst['coverage_pct']:.2f})"
        else:
            r["Strongest ToC component"] = "—"
            r["Weakest ToC component"] = "—"

        rows.append(r)

    tbl = pd.DataFrame(rows).set_index("Phase")

    # render as figure
    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.6 * (len(tbl) + 1)))
    ax.axis("off")
    t = ax.table(cellText=tbl.values,
                 colLabels=tbl.columns,
                 rowLabels=tbl.index,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.auto_set_column_width(range(len(tbl.columns) + 1))

    # header style (column headers are row 0, cols 0..n-1)
    for j in range(len(tbl.columns)):
        cell = t[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    # row label style (row labels are col -1 for rows 1..n)
    for i in range(1, len(tbl) + 1):
        if (i, -1) in t._cells:
            t[i, -1].set_facecolor("#eaf2ff")
            t[i, -1].set_text_props(fontweight="bold")
    # alternating row colours
    for i in range(1, len(tbl) + 1):
        for j in range(len(tbl.columns)):
            if i % 2 == 0:
                t[i, j].set_facecolor("#f8f9fa")

    ax.set_title("Table 1  Pipeline output summary by study phase",
                 fontweight="bold", fontsize=11, pad=14)
    fig.tight_layout()
    path = f"{OUT_DIR}/table1_summary.png"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  [08] Saved {path}")

    # also write as CSV for paper
    csv_path = f"{OUT_DIR}/table1_summary.csv"
    tbl.to_csv(csv_path)
    print(f"  [08] Saved {csv_path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[08_visualise] Generating figures...")
    fig_engagement()
    fig_network()
    fig_sentiment()
    fig_fidelity()
    fig_case_reports()
    table_summary()
    print(f"[08_visualise] Done. Figures saved to {OUT_DIR}/")
