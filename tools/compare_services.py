#!/usr/bin/env python3
"""Cross-service comparison figures for the three MASEEHA programmes.

Produces:
  - fig1_3panel_engagement_comparison.png : weekly message volume (Fig 1 style),
    one panel per service, shared calendar x-axis.
  - fig4_3panel_fidelity_comparison.png   : ToC fidelity heatmap (Fig 4 style),
    one panel per service, shared year columns + shared colour scale.

Reads each service's archived outputs under data/<Service>/.
"""
from __future__ import annotations
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SERVICES = [
    ("Jheeng",      "data/Jheeng"),
    ("Pind Begwal", "data/Pind_Begwal"),
    ("Tumair",      "data/Tumair"),
]
OUT_DIR = Path("data/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# consistent pastel colour per calendar year (used for phase shading)
YEAR_COLORS = {
    "2021": "#d0ece7", "2022": "#d4e6f1", "2023": "#d5f5e3",
    "2024": "#fdebd0", "2025": "#f9ebea", "2026": "#e8daef",
}
ALL_YEARS = ["2022", "2023", "2024", "2025", "2026"]
COMP_ORDER = ["training", "coordination", "clinical_protocol", "equipment", "supervision"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})


def load_phases(service_dir: str):
    cfg = yaml.safe_load(open(f"{service_dir}/config/settings.yaml"))
    return cfg.get("phases", [])


# ── Figure 1 (3 panels): weekly message volume ───────────────────────────────
def engagement_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    xmins, xmaxs = [], []

    for ax, (name, d) in zip(axes, SERVICES):
        eng = pd.read_csv(f"{d}/processed/engagement_metrics.csv")
        eng = eng[eng["metric"] == "weekly_volume"].copy()
        eng["timestamp"] = pd.to_datetime(eng["timestamp"])
        eng = eng.sort_values("timestamp")
        xmins.append(eng["timestamp"].min()); xmaxs.append(eng["timestamp"].max())

        # phase shading (this service's own phases, coloured by year)
        for ph in load_phases(d):
            ax.axvspan(pd.Timestamp(ph["start"]), pd.Timestamp(ph["end"]),
                       alpha=0.13, color=YEAR_COLORS.get(str(ph["name"]), "#f2f3f4"), zorder=0)

        ax.bar(eng["timestamp"], eng["message_count"], width=6,
               color="#2980b9", alpha=0.7, label="Weekly messages")
        rolling = eng["message_count"].rolling(4, center=True).mean()
        ax.plot(eng["timestamp"], rolling, color="#e74c3c", linewidth=1.5,
                label="4-week rolling mean")

        ax.set_ylabel("Messages / week")
        ax.set_title(name, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    axes[-1].set_xlabel("Week")
    axes[0].set_xlim(min(xmins) - pd.Timedelta(days=14), max(xmaxs) + pd.Timedelta(days=14))

    # shared year-colour legend
    handles = [mpatches.Patch(facecolor=YEAR_COLORS[y], alpha=0.5, label=y) for y in ALL_YEARS]
    fig.legend(handles=handles, loc="lower center", ncol=len(ALL_YEARS),
               fontsize=8, framealpha=0.6, title="Phase (calendar year)",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Supplementary Material 1  Weekly message volume by service, with phase periods",
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    path = OUT_DIR / "supplementary_material_1_engagement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 4 (3 panels): ToC fidelity heatmap ────────────────────────────────
def fidelity_comparison():
    # first pass: build each service's pivot + find global max for a shared scale
    pivots = {}
    gmax = 2.5
    for name, d in SERVICES:
        fid = pd.read_csv(f"{d}/processed/fidelity_metrics.csv")
        cov = fid[fid["kind"] == "coverage"].copy()
        piv = cov.pivot_table(index="component", columns="phase",
                              values="coverage_pct", aggfunc="mean")
        piv.columns = [str(c) for c in piv.columns]
        piv = piv.reindex(index=[c for c in COMP_ORDER if c in piv.index],
                          columns=ALL_YEARS)
        pivots[name] = piv
        if np.isfinite(np.nanmax(piv.values)):
            gmax = max(gmax, float(np.nanmax(piv.values)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#eeeeee")   # blank cells for years a service didn't operate
    im = None
    for ax, (name, d) in zip(axes, SERVICES):
        piv = pivots[name]
        data = np.ma.masked_invalid(piv.values)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=gmax)
        ax.set_xticks(range(len(ALL_YEARS))); ax.set_xticklabels(ALL_YEARS, fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([c.replace("_", " ").title() for c in piv.index], fontsize=9)
        ax.set_title(name, fontweight="bold", loc="left")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=8, color="black" if v < gmax * 0.6 else "white")
        ax.set_ylim(len(piv.index) - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Coverage (proportion of weeks with mention)")
    fig.suptitle("Supplementary Material 2  Theory of Change fidelity by service, component and year\n"
                 "(green = strong coverage; red = gap; grey = service not yet operating)",
                 fontweight="bold", y=0.995)
    path = OUT_DIR / "supplementary_material_2_fidelity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    engagement_comparison()
    fidelity_comparison()
