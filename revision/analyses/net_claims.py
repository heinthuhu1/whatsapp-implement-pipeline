"""Summarise whether each originally reported relational claim survives every specification.
Reads the CSVs written by net_sensitivity.py, net_roster.py and phase_change.py.
Output: A8_claims_robustness.csv, fig_A4_exposure_rates.png"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from net_common import OUT

P = ["2024", "2025", "2026"]


def pattern(v):
    a, b, c = v
    s = lambda x, y: "up" if y > x + 1e-9 else ("down" if y < x - 1e-9 else "flat")
    return f"{s(a, b)}/{s(b, c)}"


rows = []
win = pd.read_csv(OUT / "A1_window_sensitivity.csv", dtype={"phase": str})
seq = pd.read_csv(OUT / "A2_sequential.csv", dtype={"phase": str})
for w, g in win.groupby("window_min"):
    g = g.set_index("phase")
    for m, lab in [("mean_partners", "isolates excluded"), ("mean_partners_incl_isolates", "isolates included")]:
        v = [g.loc[p, m] for p in P]
        rows.append({"claim": "Mean co-activity partners declined 2024>2025>2026", "specification": f"{w}-min window, {lab}",
                     "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[0] > v[1] > v[2]})
    for m, lab in [("density", "isolates excluded"), ("density_incl_isolates", "isolates included")]:
        v = [g.loc[p, m] for p in P]
        rows.append({"claim": "Density 2026 lowest of the three phases", "specification": f"{w}-min window, {lab}",
                     "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[2] < min(v[:2])})
    for m, lab in [("top_betweenness_orig_value", "weight as distance (original code)"),
                   ("top_betweenness_unw_value", "unweighted"), ("top_betweenness_invw_value", "distance = 1/weight")]:
        v = [g.loc[p, m] for p in P]
        who = [g.loc[p, m.replace("_value", "")].split(" ")[0] for p in P]
        rows.append({"claim": "Top betweenness rose across phases", "specification": f"{w}-min window, {lab} [{'/'.join(who)}]",
                     "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[0] < v[1] < v[2]})
for spec, g in seq.groupby("specification"):
    g = g.set_index("phase")
    v = [g.loc[p, "mean_partners"] for p in P]
    rows.append({"claim": "Mean co-activity partners declined 2024>2025>2026", "specification": spec,
                 "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[0] > v[1] > v[2]})
    v = [g.loc[p, "density"] for p in P]
    rows.append({"claim": "Density 2026 lowest of the three phases", "specification": spec,
                 "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[2] < min(v[:2])})
    v = [g.loc[p, "top_betweenness_invw_value"] for p in P]
    rows.append({"claim": "Top betweenness rose across phases", "specification": f"{spec}, distance = 1/weight",
                 "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[0] < v[1] < v[2]})
gv = pd.read_csv(OUT / "A7_gini_variants.csv", dtype={"phase": str})
for var, g in gv.groupby("variant", sort=False):
    g = g.set_index("phase")
    for m in ["gini", "gini_corrected"]:
        v = [g.loc[p, m] for p in P]
        rows.append({"claim": "Gini rose 2024<2025<=2026", "specification": f"{var} ({m})",
                     "2024": v[0], "2025": v[1], "2026": v[2], "pattern": pattern(v), "claim_holds": v[0] < v[1] <= v[2]})
rp = pd.read_csv(OUT / "A4_roster_phase.csv", dtype={"phase": str}).set_index("phase")
for m in ["gini_roster_incl_silent", "gini_roster_corrected", "gini_rate_per_exposure_active",
          "gini_rate_per_exposure_roster", "gini_roster_plus_one_silent"]:
    v = [rp.loc[p, m] for p in P]
    rows.append({"claim": "Gini rose 2024<2025<=2026", "specification": m, "2024": v[0], "2025": v[1], "2026": v[2],
                 "pattern": pattern(v), "claim_holds": v[0] < v[1] <= v[2]})
mw = pd.read_csv(OUT / "B2_matched_windows.csv").set_index("window")
for a, b in [("Sep-Dec 2024", "Sep-Dec 2025"), ("Jan-Apr 2025", "Jan-Apr 2026")]:
    for m, claim, rule in [("gini", "Gini rose 2024<2025<=2026", lambda x, y: y > x),
                           ("mean_partners", "Mean co-activity partners declined 2024>2025>2026", lambda x, y: y < x),
                           ("mean_partners_incl_isolates", "Mean co-activity partners declined 2024>2025>2026", lambda x, y: y < x),
                           ("density", "Density 2026 lowest of the three phases", lambda x, y: y < x)]:
        x, y = mw.loc[a, m], mw.loc[b, m]
        rows.append({"claim": claim, "specification": f"matched calendar {a} vs {b}: {m}", "2024": np.nan,
                     "2025": x, "2026": y, "pattern": "up" if y > x else ("down" if y < x else "flat"),
                     "claim_holds": bool(rule(x, y))})
res = pd.DataFrame(rows)
res.to_csv(OUT / "A8_claims_robustness.csv", index=False)
print(res.groupby("claim").claim_holds.agg(["sum", "count"]))
pd.set_option("display.width", 250)
print(res.round(3).to_string())

# per-sender exposure-normalised message rates
ex = pd.read_csv(OUT / "A4_exposure_by_sender_phase.csv", dtype={"phase": str})
ex = ex[ex.phase.isin(P)]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for s, g in ex.groupby("sender_code"):
    g = g.set_index("phase").reindex(P)
    y = g["messages_per_30d_exposure"].replace(0, 0.05)
    ax.plot(P, y, marker="o", lw=1.6 if s == "P024" else 0.9, color="#b2182b" if s == "P024" else "#777777")
    last = y.last_valid_index()
    ax.annotate(s, (last, y[last]), fontsize=7, xytext=(4, 0), textcoords="offset points")
ax.set_yscale("log")
ax.set_ylabel("Messages per 30 days in group (log; 0 plotted at 0.05)")
ax.set_title("Exposure-normalised posting rate by sender and phase", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_A4_exposure_rates.png", dpi=200)
