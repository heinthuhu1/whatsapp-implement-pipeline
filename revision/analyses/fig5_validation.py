"""Revised Figure 5: content measures against human coding.

Panel A: sentiment as judged by the coders (negative / neutral / positive) and as labelled by the
         DistilBERT SST-2 model (negative / positive; it has no neutral class), overall and by
         language group.
Panel B: peer-support language by phase: rate flagged by the keyword lexicon (Wilson 95% CI) and
         design-weighted prevalence from human coding (stratified bootstrap 95% CI).

Input : data/Jheeng/revision/clinical_repro/coding/results/sentiment_model.csv, merged_codes.csv
Output: data/Jheeng/revision/clinical_repro/coding/results/fig5_validation.png / .pdf
Usage : venv/bin/python revision/analyses/fig5_validation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "Jheeng" / "revision" / "clinical_repro" / "coding" / "results"
NEG, NEU, POS = "#e34948", "#d6d5cf", "#2a78d6"
LEX, HUM = "#eb6834", "#2a78d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LEXICON = {"2024": (15, 299), "2025": (14, 1019), "2026": (0, 282)}   # Table 1, content-bearing messages


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0), c + h


def human_prevalence(seed=20261120, reps=1000):
    m = pd.read_csv(RES / "merged_codes.csv", usecols=["source", "phase", "stratum", "design_weight", "peer_support"])
    s = m[(m.source == "sample") & m.peer_support.isin(["y", "n"])]
    rng = np.random.default_rng(seed)
    out = {}
    for ph, d in s.groupby("phase"):
        est = lambda x: x.design_weight[x.peer_support == "y"].sum() / x.design_weight.sum()
        groups = [g for _, g in d.groupby("stratum")]
        bs = [est(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(reps)]
        out[str(ph)] = (est(d), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d))
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "text.color": INK})
    sm = pd.read_csv(RES / "sentiment_model.csv").set_index("group")
    rows = [("all coded", "All coded items"), ("language English text", "English text"),
            ("language Roman Urdu text", "Roman Urdu text"), ("language voice note", "Voice notes")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel A
    y, ticks, labels = 0.0, [], []
    for key, lab in rows:
        r = sm.loc[key]
        hn, hneu = r.human_negative_share, r.human_neutral_share
        hp = 1 - hn - hneu
        mn = r.model_negative_share
        a.text(0, y, f"{lab} (n = {int(r.n)})", fontsize=8, fontweight="bold", color=INK, va="center", ha="left")
        y += 0.62
        for who, segs in (("Coders", [(hn, NEG), (hneu, NEU), (hp, POS)]), ("Model", [(mn, NEG), (0, NEU), (1 - mn, POS)])):
            left = 0
            for w, col in segs:
                if w > 0:
                    a.barh(y, w, left=left, height=0.6, color=col, edgecolor="white", linewidth=1)
                    if w >= 0.09:
                        a.text(left + w / 2, y, f"{w * 100:.0f}%", ha="center", va="center", fontsize=7,
                               color="white" if col != NEU else INK)
                left += w
            ticks.append(y); labels.append(who)
            y += 0.7
        y += 0.35
    a.set_yticks(ticks); a.set_yticklabels(labels)
    a.set_xlim(0, 1); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    a.set_ylim(y - 0.3, -0.5); style(a); a.spines["left"].set_visible(False)
    a.set_title("A  Sentiment: human coders and the model", loc="left", fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NEG, NEU, POS)]
    a.legend(handles, ["Negative", "Neutral (coders only)", "Positive"], loc="upper center", bbox_to_anchor=(0.5, -0.1),
             ncol=3, frameon=False, fontsize=7.5, handlelength=1.2)

    # Panel B
    hum = human_prevalence()
    phases = ["2024", "2025", "2026"]
    x = np.arange(3)
    for off, col, name, vals in (
            (-0.08, LEX, "Keyword lexicon", [wilson(*LEXICON[p]) for p in phases]),
            (0.08, HUM, "Human coding (weighted)", [hum[p][:3] for p in phases])):
        est = np.array([v[0] for v in vals]) * 100
        lo = np.array([v[1] for v in vals]) * 100
        hi = np.array([v[2] for v in vals]) * 100
        b.plot(x + off, est, color=col, linewidth=1.5, zorder=2)
        b.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor=col, elinewidth=1.2, capsize=0, zorder=2)
        b.scatter(x + off, est, s=36, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        b.text(x[-1] + off + 0.14, est[-1] + (0.35 if est[-1] < 0.5 else 0), name.split(" (")[0], color=INK2, fontsize=7.5, va="center")
    b.set_xticks(x); b.set_xticklabels(phases); b.set_xlim(-0.4, 3.3)
    b.set_ylim(0, 9); b.set_ylabel("Content-bearing messages (%)", color=INK2, fontsize=8)
    b.yaxis.grid(True, color=GRID, linewidth=0.6); b.set_axisbelow(True)
    style(b)
    b.set_title("B  Peer-support language by phase", loc="left", fontsize=9, fontweight="bold")
    from matplotlib.lines import Line2D
    b.legend([Line2D([], [], color=LEX, marker="o", lw=1.5), Line2D([], [], color=HUM, marker="o", lw=1.5)],
             ["Keyword lexicon (Wilson 95% CI)", "Human coding, design-weighted (bootstrap 95% CI)"],
             loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=7.5, ncol=1)

    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"fig5_validation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("written", RES / "fig5_validation.png", {p: tuple(round(v * 100, 1) for v in hum[p][:3]) for p in phases})


if __name__ == "__main__":
    main()
