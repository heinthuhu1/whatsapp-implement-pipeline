"""Reproduce the originally reported relational, temporal and topical numbers from the Jheeng archive."""
from __future__ import annotations

import pandas as pd

from net_common import (OUT, PHASE_NAMES, SETTINGS, coactivity_edges, engagement_frame, load_messages,
                        network_summary, sender_frame, silence_periods, velocity_spikes, md_table)

REPORTED = {
    "nodes": {"2024": 8, "2025": 7, "2026": 8, "Overall": 11},
    "edges": {"2024": 18, "2025": 14, "2026": 14, "Overall": 27},
    "density": {"2024": 0.64, "2025": 0.67, "2026": 0.50, "Overall": 0.49},
    "gini": {"2024": 0.63, "2025": 0.80, "2026": 0.81, "Overall": 0.81},
    "mean_partners": {"2024": 4.5, "2025": 4.0, "2026": 3.5, "Overall": 4.9},
    "silences": {"2024": 17, "2025": 26, "2026": 4, "Overall": 47},
    "spike_days": {"2024": 5, "2025": 11, "2026": 1, "Overall": 17},
    "cov_coordination": {"2026": 19.3}, "cov_clinical_protocol": {"2026": 19.0},
    "cov_supervision": {"2026": 19.0}, "cov_training": {"2024": 0.37, "2025": 0.0, "2026": 0.0},
}


def main():
    df = load_messages()
    msgs, eng = sender_frame(df), engagement_frame(df)
    rows = []
    sil = silence_periods(eng, SETTINGS["silence_threshold_hours"])
    spk = velocity_spikes(eng, SETTINGS["rolling_window_days"])
    for ph in PHASE_NAMES + ["Overall"]:
        sub = msgs if ph == "Overall" else msgs[msgs["phase"] == ph]
        s = network_summary(sub, coactivity_edges(sub, 30))
        r = {"phase": ph, "nodes": s["nodes"], "edges": s["edges"], "density": round(s["density"], 2),
             "gini": round(s["gini"], 2), "mean_partners": round(s["mean_partners"], 1)}
        if ph == "Overall":
            r["silences"], r["spike_days"] = len(sil), int(spk["spike"].sum())
        else:
            r["silences"] = int((pd.to_datetime(sil["gap_start"]).dt.year.astype(str) == ph).sum())
            r["spike_days"] = int(spk.loc[spk["spike"], "date"].dt.year.astype(str).eq(ph).sum())
            e = eng[eng["phase"] == ph]
            text = e["message"].fillna("").astype(str).str.lower()
            # src/06_fidelity.py builds text_combined from a RangeIndex Series of None and assigns it to a
            # frame whose index has gaps (system rows removed): the last k rows (k = removed system rows)
            # receive NaN text and are never matched. Replicated here to reproduce the published values.
            bug_text = text.where(e.index < len(eng), "")
            for comp, kws in SETTINGS["toc_component_keywords"].items():
                hit = text.apply(lambda t: any(k.lower() in t for k in kws))
                hit_bug = bug_text.apply(lambda t: any(k.lower() in t for k in kws))
                r[f"cov_{comp}"] = round(100 * hit_bug.mean(), 2)
                r[f"cov_{comp}_corrected"] = round(100 * hit.mean(), 2)
        rows.append(r)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "00_reproduction.csv", index=False)
    checks = []
    for metric, vals in REPORTED.items():
        for ph, v in vals.items():
            got = rep.loc[rep["phase"] == ph, metric].iloc[0]
            tol = 0.051 if isinstance(v, float) else 0
            checks.append({"metric": metric, "phase": ph, "reported": v, "reproduced": got,
                           "match": abs(float(got) - float(v)) <= tol})
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "00_reproduction_check.csv", index=False)
    print(rep.to_string())
    print(md_table(ck))
    print("ALL MATCH:", ck["match"].all())


if __name__ == "__main__":
    main()
