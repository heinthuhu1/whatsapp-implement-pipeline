"""Revised Figure 4: co-activity network by study phase (Jheeng archive only).

Ties and weights follow the revised Methods (net_common.coactivity_edges, a replica of src/04_network):
per message, every distinct other sender within +/- 30 minutes is counted once; pairwise totals are halved.
Changes from the submitted figure:
  - each member keeps the same position on a fixed circle in all three panels,
  - senders who posted in the phase but had no co-active partner (isolates) are drawn as hollow nodes,
  - panel subtitles give connected senders out of all senders and the number of ties; density is left to
    the caption and Supplementary Material 3 because it is reported for description only.

Output: data/Jheeng/revision/drafts/Figure4_revised.png / .pdf (1 x 3) and Figure4_revised_vertical.png / .pdf (3 x 1)
Usage : venv/bin/python revision/analyses/fig4_network.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_common import JH, PHASE_NAMES, coactivity_edges, load_messages, sender_frame  # noqa: E402

OUT = JH / "revision" / "drafts"
NODE, EDGE, INK, INK2, MUTED = "#2a78d6", "#9a9890", "#0b0b0b", "#52514e", "#898781"


def main() -> None:
    df = sender_frame(load_messages())
    graphs, senders = {}, {}
    for ph in PHASE_NAMES:
        d = df[df.phase == ph]
        senders[ph] = sorted(d.sender_code.unique())
        G = nx.Graph()
        for _, e in coactivity_edges(d).iterrows():
            if e.weight > 0:
                G.add_edge(e.source, e.target, weight=int(e.weight))
        graphs[ph] = G

    # one layout for everyone who appears in any phase
    pooled = nx.Graph()
    for G in graphs.values():
        for a, b, w in G.edges(data="weight"):
            pooled.add_edge(a, b, weight=pooled.get_edge_data(a, b, {"weight": 0})["weight"] + w)
    for ph in PHASE_NAMES:
        pooled.add_nodes_from(senders[ph])
    # fixed circle, ordered by pooled number of partners so the busiest members are spread around it
    order = sorted(pooled, key=lambda n: (-pooled.degree(n), n))
    ang = {n: np.pi / 2 - 2 * np.pi * i / len(order) for i, n in enumerate(order)}
    pos = {n: (np.cos(a), np.sin(a)) for n, a in ang.items()}

    max_w = max(w for G in graphs.values() for _, _, w in G.edges(data="weight"))
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "text.color": INK})
    for vertical in (False, True):
        draw(graphs, senders, pos, ang, max_w, vertical)


def draw(graphs, senders, pos, ang, max_w, vertical: bool) -> None:
    if vertical:
        fig, axes = plt.subplots(3, 1, figsize=(3.6, 9.0))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    for ax, ph in zip(axes, PHASE_NAMES):
        G = graphs[ph]
        iso = [s for s in senders[ph] if s not in G]
        widths = [0.6 + 4.4 * w / max_w for _, _, w in G.edges(data="weight")]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=EDGE, alpha=0.55)
        deg = dict(G.degree())
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G), node_size=[120 + 45 * deg[n] for n in G],
                               node_color=NODE, edgecolors="white", linewidths=1.2)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=iso, node_size=90, node_color="white",
                               edgecolors=MUTED, linewidths=1.2)
        for n in list(G) + iso:
            a = ang[n]
            ax.text(1.33 * np.cos(a), 1.33 * np.sin(a), n, ha="center", va="center", fontsize=6.5,
                    color=INK2 if n in G else MUTED)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{ph}\n{G.number_of_nodes()} of {len(senders[ph])} senders connected, "
                     f"{G.number_of_edges()} ties", fontsize=8.5, fontweight="bold", color=INK, linespacing=1.4)
        if not vertical: print(ph, "nodes", G.number_of_nodes(), "senders", len(senders[ph]), "edges", G.number_of_edges(),
              "density", round(nx.density(G), 2), "isolates", iso)

    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", ms=7, mfc=NODE, mec="white"),
                Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=MUTED),
                Line2D([], [], color=EDGE, lw=2.5, alpha=0.7)],
               ["Connected sender (size = number of partners)", "Sender with no co-active partner",
                "Tie (width = co-activity weight)"],
               loc="lower center", ncol=1 if vertical else 3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01 if vertical else -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1) if vertical else (0, 0.07, 1, 1), h_pad=1.5, w_pad=1.0)
    name = "Figure4_revised_vertical" if vertical else "Figure4_revised"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
