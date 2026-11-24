"""Revised Figure 3: topical coverage by component and phase (Jheeng archive only).

Values are the pipeline's own output (src/06_fidelity.py, word-start keyword matching, voice notes represented by
their English translations), so they match Table 1 exactly.
Panel A: all messages.  Panel B: excluding structured case reports (the panel suggested in the R1 #4 response).
Keeps the look of the submitted figure (RdYlGn heat map, values in cells, colour bar on the right); adds the
case-report panel on a shared colour scale.

Input : data/Jheeng/rerun/processed/fidelity_metrics.csv
Output: data/Jheeng/revision/drafts/Figure3_revised.png / .pdf
Usage : venv/bin/python revision/analyses/fig3_topical.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "Jheeng" / "rerun" / "processed" / "fidelity_metrics.csv"
OUT = ROOT / "data" / "Jheeng" / "revision" / "drafts"
COMPS = ["training", "coordination", "clinical_protocol", "equipment", "supervision"]
LABELS = ["Training", "Coordination", "Clinical Protocol", "Equipment", "Supervision"]
PHASES = ["2024", "2025", "2026"]
CMAP = plt.get_cmap("RdYlGn")
VMAX = 45


def text_colour(v):
    r, g, b, _ = CMAP(v / VMAX)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def grid(df: pd.DataFrame, kind: str):
    d = df[df.kind == kind].copy()
    d["phase"] = d.phase.astype(str).str.replace(r"\.0$", "", regex=True)
    pct = d.pivot(index="component", columns="phase", values="coverage_pct").loc[COMPS, PHASES]
    hits = d.pivot(index="component", columns="phase", values="hits").loc[COMPS, PHASES]
    n = d.groupby("phase").n_messages.first().loc[PHASES]
    return pct.values, hits.values.astype(int), n.astype(int).values


def main() -> None:
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (kind, title) in zip(axes, [("coverage", "A  All messages"),
                                        ("coverage_excl_case_reports", "B  Excluding structured case reports")]):
        pct, hits, n = grid(df, kind)
        im = ax.imshow(pct, cmap=CMAP, vmin=0, vmax=VMAX, aspect="auto")
        for i in range(len(COMPS)):
            for j in range(len(PHASES)):
                ax.text(j, i, f"{pct[i, j]:.2f}", ha="center", va="center", fontsize=10, color=text_colour(pct[i, j]))
        ax.set_xticks(range(3)); ax.set_xticklabels(PHASES, fontsize=11)
        ax.set_yticks(range(len(COMPS))); ax.set_yticklabels(LABELS, fontsize=11)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        print(kind, dict(zip(PHASES, n.tolist())), np.round(pct, 2).tolist())
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Coverage (% of messages)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Figure3_revised.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("written", OUT / "Figure3_revised.png")


if __name__ == "__main__":
    main()
