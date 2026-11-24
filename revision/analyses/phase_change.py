"""R2 #19 / R1 #4: phase-length handling, matched-calendar comparisons, template revision date,
topical-fidelity decomposition and a segmented (interrupted) time series. Jheeng archive only.

Outputs (data/Jheeng/revision/network_phase/):
  B1_rate_normalised.csv, B2_matched_windows.csv, B2_rolling_windows.csv, B3_template_labels.csv,
  B3_template_monthly.csv, B4_topical_coverage.csv, B5_its_results.csv, B5_weekly_series.csv,
  fig_B2_rolling_windows.png, fig_B5_its.png, fig_B4_topical.png
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from net_common import (JH, OUT, PHASE_NAMES, PHASES, SETTINGS, STUDY_END, STUDY_START, TEMPLATE_BREAK,
                        coactivity_edges, engagement_frame, flag_case_reports, gini, gini_corrected,
                        load_messages, network_summary, phase_days, sender_frame, sequential_edges,
                        silence_periods, velocity_spikes)

TOC = SETTINGS["toc_component_keywords"]
COMPS = list(TOC)
NEW_LABELS = ["pickup location", "pickup time", "destination", "arrival time", "complaint"]
OLD_LABELS = ["initial complaint", "location", "name of hospital", "time hospital", "time patien", "emp personnel name"]


def prep():
    df = load_messages()
    eng = engagement_frame(df)
    eng["case_report"] = flag_case_reports(eng)
    # voice-note translations (English) joined on date + WhatsApp sequence number
    vn = pd.read_csv(JH / "processed" / "voice_notes.csv")
    vn["key"] = vn["date"].astype(str) + "_" + vn["sequence"].astype(int).astype(str)
    tr = dict(zip(vn["key"], vn["translation_en"].fillna("")))
    seq = eng["message"].fillna("").str.extract(r"WA(\d+)\.opus")[0]
    key = eng["timestamp"].dt.strftime("%Y-%m-%d") + "_" + seq.fillna("-1").astype(int).astype(str)
    eng["vn_translation"] = np.where(eng["message_type"] == "voice_note", key.map(tr).fillna(""), "")
    eng["text_orig"] = eng["message"].fillna("").astype(str).str.lower()
    eng["text_with_vn"] = np.where(eng["message_type"] == "voice_note", eng["vn_translation"].str.lower(), eng["text_orig"])
    for c in COMPS:
        kws = [k.lower() for k in TOC[c]]
        eng[f"hit_{c}"] = eng["text_orig"].apply(lambda t: any(k in t for k in kws))
        eng[f"hitvn_{c}"] = eng["text_with_vn"].apply(lambda t: any(k in t for k in kws))
    eng["hit_any"] = eng[[f"hit_{c}" for c in COMPS]].any(axis=1)
    eng["hitvn_any"] = eng[[f"hitvn_{c}" for c in COMPS]].any(axis=1)
    return df, eng


def window_metrics(eng, msgs, sil, spk, start, end, label):
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    e = eng[(eng.timestamp >= start) & (eng.timestamp <= end)]
    m = msgs[(msgs.timestamp >= start) & (msgs.timestamp <= end)]
    days = (end.normalize() - start.normalize()).days + 1
    s = network_summary(m, coactivity_edges(m, 30))
    sq = network_summary(m, sequential_edges(m, None))
    cnt = m.groupby("sender_code").size()
    r = {"window": label, "start": start.date(), "end": end.date(), "days": days,
         "messages": len(e), "messages_per_30d": len(e) / days * 30, "case_reports": int(e.case_report.sum()),
         "case_reports_per_30d": e.case_report.sum() / days * 30,
         "non_report_messages_per_30d": (~e.case_report).sum() / days * 30,
         "active_senders": s["active_senders"], "nodes": s["nodes"], "edges": s["edges"], "density": s["density"],
         "mean_partners": s["mean_partners"], "mean_partners_incl_isolates": s["mean_partners_incl_isolates"],
         "isolates": s["isolates"], "seq_density": sq["density"], "seq_mean_partners": sq["mean_partners"],
         "gini": s["gini"], "gini_corrected": s["gini_corrected"],
         "top_sender": cnt.idxmax() if len(cnt) else "", "top_sender_share": cnt.max() / cnt.sum() if len(cnt) else np.nan,
         "gini_excl_top_sender": gini(cnt.drop(cnt.idxmax()).values) if len(cnt) > 2 else np.nan,
         "silences": int(((sil.gap_start >= start) & (sil.gap_start <= end)).sum()),
         "silences_per_30d": ((sil.gap_start >= start) & (sil.gap_start <= end)).sum() / days * 30,
         "spike_days": int(spk[(spk.date >= start.normalize()) & (spk.date <= end) & spk.spike].shape[0])}
    for c in COMPS + ["any"]:
        r[f"cov_{c}"] = 100 * e[f"hit_{c}"].mean() if len(e) else np.nan
        ne = e[~e.case_report]
        r[f"cov_excl_reports_{c}"] = 100 * ne[f"hit_{c}"].mean() if len(ne) else np.nan
    return r


def b1_b2(eng, msgs):
    sil = silence_periods(eng, SETTINGS["silence_threshold_hours"])
    spk = velocity_spikes(eng, SETTINGS["rolling_window_days"])
    rows = [window_metrics(eng, msgs, sil, spk, PHASES[p][0], PHASES[p][1].normalize(), f"phase {p}") for p in PHASE_NAMES]
    b1 = pd.DataFrame(rows)
    b1["spike_days_per_30d"] = b1.spike_days / b1.days * 30
    b1["messages_per_active_sender_per_30d"] = b1.messages_per_30d / b1.active_senders
    b1["edges_per_active_sender"] = b1.edges / b1.active_senders
    b1.to_csv(OUT / "B1_rate_normalised.csv", index=False)
    wins = [("2025-01-01", "2025-04-11", "Jan-Apr 2025"), ("2026-01-01", "2026-04-11", "Jan-Apr 2026"),
            ("2024-09-22", "2024-12-31", "Sep-Dec 2024"), ("2025-09-22", "2025-12-31", "Sep-Dec 2025"),
            ("2026-01-01", "2026-02-01", "2026 pre-template (1 Jan-1 Feb)"),
            ("2026-02-02", "2026-04-11", "2026 post-template (2 Feb-11 Apr)"),
            ("2025-11-22", "2026-02-01", "70 days pre-template (22 Nov 2025-1 Feb 2026)")]
    b2 = pd.DataFrame([window_metrics(eng, msgs, sil, spk, a, b, lab) for a, b, lab in wins])
    b2.to_csv(OUT / "B2_matched_windows.csv", index=False)
    # rolling 101-day windows (the length of the 2024 and 2026 phases), 14-day step
    roll = []
    st = STUDY_START
    while st + pd.Timedelta(days=100) <= STUDY_END.normalize():
        en = st + pd.Timedelta(days=100)
        roll.append(window_metrics(eng, msgs, sil, spk, st, en, f"{st:%Y-%m-%d}"))
        st += pd.Timedelta(days=14)
    roll = pd.DataFrame(roll)
    roll.to_csv(OUT / "B2_rolling_windows.csv", index=False)
    return b1, b2, roll


def b3_template():
    cr = pd.read_csv(JH / "interim" / "case_reports.csv", parse_dates=["timestamp"])
    lab = re.compile(r"^[\s\*‎‏]*([A-Za-z][A-Za-z /\.\-\(\)#]{1,40}?)\s*[:：]", re.M)
    recs = []
    for ts, m in zip(cr.timestamp, cr.raw_message.fillna("")):
        labels = {re.sub(r"\s+", " ", x.strip().lower()) for x in lab.findall(m)}
        new = any(l in labels for l in NEW_LABELS)
        old = any(any(l.startswith(o) for o in OLD_LABELS) for l in labels)
        fa_vitals = bool(re.search(r"first aid[^:\n]*:[^\n]*(vital|monitor)", m, re.I))
        recs.append({"timestamp": ts, "new_template": new, "old_template": old and not new, "first_aid_vitals_monitor": fa_vitals})
    r = pd.DataFrame(recs)
    first_new = r.loc[r.new_template, "timestamp"].min()
    last_old = r.loc[r.old_template, "timestamp"].max()
    # label first/last appearance table (labels only, no field values)
    rows = []
    for L in NEW_LABELS + ["initial complaint", "location", "name of hospital", "time hospital was", "time patien",
                           "emp personnel name", "emp personnel", "first aid", "case no", "gender", "date"]:
        hits = [ts for ts, m in zip(cr.timestamp, cr.raw_message.fillna(""))
                if L in {re.sub(r"\s+", " ", x.strip().lower()) for x in lab.findall(m)}]
        rows.append({"field_label": L, "reports_using_label": len(hits),
                     "first_seen": min(hits) if hits else pd.NaT, "last_seen": max(hits) if hits else pd.NaT})
    labels = pd.DataFrame(rows)
    labels.to_csv(OUT / "B3_template_labels.csv", index=False)
    r["month"] = r.timestamp.dt.to_period("M").astype(str)
    monthly = r.groupby("month").agg(reports=("timestamp", "size"), new_template=("new_template", "sum"),
                                     old_template=("old_template", "sum"),
                                     first_aid_vitals_monitor=("first_aid_vitals_monitor", "sum")).reset_index()
    monthly.to_csv(OUT / "B3_template_monthly.csv", index=False)
    summary = {"first_new_template": first_new, "last_old_template": last_old,
               "new_after": int(r.loc[r.timestamp >= first_new, "new_template"].sum()),
               "reports_after": int((r.timestamp >= first_new).sum()),
               "old_after": int(r.loc[r.timestamp >= first_new, "old_template"].sum())}
    return labels, monthly, summary


def b4_topical(eng):
    rows = []
    def add(scope, label, sub, prefix="hit"):
        r = {"scope": scope, "subset": label, "n_messages": len(sub)}
        for c in COMPS + ["any"]:
            r[c] = 100 * sub[f"{prefix}_{c}"].mean() if len(sub) else np.nan
            r[f"{c}_hits"] = int(sub[f"{prefix}_{c}"].sum())
        rows.append(r)
    for ph in PHASE_NAMES + ["Overall"]:
        sub = eng if ph == "Overall" else eng[eng.phase == ph]
        add(ph, "(a) all messages, original denominator", sub)
        add(ph, "(a2) text messages only", sub[sub.message_type == "text"])
        add(ph, "(b) excluding structured case reports", sub[~sub.case_report])
        add(ph, "(b2) structured case reports only", sub[sub.case_report])
        add(ph, "(d) voice-note translations substituted", sub, "hitvn")
        add(ph, "(d2) voice-note translations, excluding case reports", sub[~sub.case_report], "hitvn")
    for lab, sub in [("pre-template (22 Sep 2024-1 Feb 2026)", eng[eng.timestamp < TEMPLATE_BREAK]),
                     ("post-template (2 Feb-11 Apr 2026)", eng[eng.timestamp >= TEMPLATE_BREAK]),
                     ("2026 pre-template (1 Jan-1 Feb)", eng[(eng.phase == "2026") & (eng.timestamp < TEMPLATE_BREAK)]),
                     ("2026 post-template (2 Feb-11 Apr)", eng[eng.timestamp >= TEMPLATE_BREAK])]:
        add(f"(c) {lab}", "all messages", sub)
        add(f"(c) {lab}", "excluding structured case reports", sub[~sub.case_report])
        add(f"(c) {lab}", "structured case reports only", sub[sub.case_report])
        add(f"(c) {lab}", "voice-note translations, excluding case reports", sub[~sub.case_report], "hitvn")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "B4_topical_coverage.csv", index=False)
    return res


def b5_its(eng):
    e = eng[(eng.timestamp >= "2024-09-23") & (eng.timestamp < "2026-04-06")].copy()
    e["week"] = e.timestamp.dt.to_period("W-SUN").dt.start_time
    wk = e.groupby("week").agg(messages=("timestamp", "size"), case_reports=("case_report", "sum"),
                               any_hits=("hit_any", "sum"),
                               non_report_any_hits=("hit_any", lambda s: s[~e.loc[s.index, "case_report"]].sum()))
    full = pd.date_range("2024-09-23", "2026-03-30", freq="7D")
    wk = wk.reindex(full, fill_value=0)
    wk.index.name = "week"
    wk["non_report_messages"] = wk.messages - wk.case_reports
    wk["coverage_any_pct"] = 100 * wk.any_hits / wk.messages.replace(0, np.nan)
    wk["coverage_any_excl_reports_pct"] = 100 * wk.non_report_any_hits / wk.non_report_messages.replace(0, np.nan)
    wk["t"] = np.arange(len(wk))
    wk["post"] = (wk.index >= TEMPLATE_BREAK).astype(int)
    t0 = wk.loc[wk.post == 1, "t"].min()
    wk["t_post"] = np.where(wk.post == 1, wk.t - t0, 0)
    wk.to_csv(OUT / "B5_weekly_series.csv")
    out = []
    for y in ["messages", "non_report_messages", "case_reports", "coverage_any_pct", "coverage_any_excl_reports_pct"]:
        d = wk.dropna(subset=[y])
        X = sm.add_constant(d[["t", "post", "t_post"]])
        ols = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        resid = sm.OLS(d[y], X).fit().resid
        lb = acorr_ljungbox(resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]
        gl = sm.GLSAR(d[y], X, rho=1).iterative_fit(maxiter=10)
        for term in ["post", "t_post", "t"]:
            ci = ols.conf_int().loc[term]
            gci = gl.conf_int().loc[term]
            out.append({"outcome": y, "term": {"post": "level change", "t_post": "slope change (per week)",
                                               "t": "pre-break slope (per week)"}[term],
                        "estimate_ols_hac": ols.params[term], "ci_lo": ci[0], "ci_hi": ci[1], "p": ols.pvalues[term],
                        "estimate_ar1": gl.params[term], "ar1_ci_lo": gci[0], "ar1_ci_hi": gci[1], "ar1_rho": gl.model.rho[0],
                        "durbin_watson": durbin_watson(resid), "ljung_box_p_lag4": lb,
                        "n_weeks": len(d), "post_weeks": int(d.post.sum()), "pre_mean": d.loc[d.post == 0, y].mean(),
                        "post_mean": d.loc[d.post == 1, y].mean()})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "B5_its_results.csv", index=False)
    return wk, res


def figures(roll, wk, its, b4):
    # rolling windows
    roll = roll.copy()
    roll["mid"] = pd.to_datetime(roll["start"]) + pd.Timedelta(days=50)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    specs = [("messages_per_30d", "Messages per 30 days"), ("active_senders", "Active senders"),
             ("top_sender_share", "Top sender's share of messages"), ("gini", "Gini (active senders)"),
             ("density", "Co-activity density (30 min, isolates excluded)"), ("mean_partners", "Mean co-activity partners")]
    for ax, (m, lab) in zip(axes.flat, specs):
        ax.plot(roll["mid"], roll[m], color="#333333", lw=1.4)
        if m == "density":
            ax.plot(roll["mid"], roll["seq_density"], color="#1f78b4", lw=1, ls="--", label="sequential")
            ax.legend(frameon=False, fontsize=8)
        if m == "mean_partners":
            ax.plot(roll["mid"], roll["mean_partners_incl_isolates"], color="#e31a1c", lw=1, ls="--", label="incl. isolates")
            ax.legend(frameon=False, fontsize=8)
        if m == "gini":
            ax.plot(roll["mid"], roll["gini_excl_top_sender"], color="#e31a1c", lw=1, ls="--", label="excl. top sender")
            ax.legend(frameon=False, fontsize=8)
        for b in ["2025-01-01", "2026-01-01"]:
            ax.axvline(pd.Timestamp(b), color="grey", lw=.6, ls=":")
        ax.axvline(TEMPLATE_BREAK, color="#b2182b", lw=.8, ls="-.")
        ax.set_title(lab, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("Rolling 101-day windows (14-day step), plotted at window midpoint; dotted lines: phase boundaries; dash-dot: template revision", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_B2_rolling_windows.png", dpi=200)
    plt.close(fig)
    # ITS
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, y, lab in zip(axes, ["messages", "case_reports", "coverage_any_pct"],
                          ["Weekly messages", "Weekly structured case reports",
                           "Weekly topical coverage, any component (%)\nall messages (fitted) vs excluding case reports (orange)"]):
        d = wk.dropna(subset=[y])
        ax.plot(d.index, d[y], color="#999999", lw=.9, marker=".", ms=3)
        X = sm.add_constant(d[["t", "post", "t_post"]])
        fit = sm.OLS(d[y], X).fit()
        pre, post = d.post == 0, d.post == 1
        ax.plot(d.index[pre], fit.fittedvalues[pre], color="#2166ac", lw=1.8)
        ax.plot(d.index[post], fit.fittedvalues[post], color="#b2182b", lw=1.8)
        if y == "coverage_any_pct":
            ax.plot(wk.index, wk["coverage_any_excl_reports_pct"], color="#ff7f00", lw=.9, ls="--")
        ax.axvline(TEMPLATE_BREAK, color="#b2182b", lw=.8, ls="-.")
        ax.set_title(lab, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "fig_B5_its.png", dpi=200)
    plt.close(fig)
    # topical decomposition
    sel = b4[b4.scope.isin(PHASE_NAMES) & b4.subset.isin(["(a) all messages, original denominator",
                                                          "(b) excluding structured case reports",
                                                          "(d2) voice-note translations, excluding case reports"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for ax, (sub, g) in zip(axes, sel.groupby("subset", sort=False)):
        x = np.arange(len(COMPS))
        for i, ph in enumerate(PHASE_NAMES):
            v = g[g.scope == ph][COMPS].values.ravel()
            ax.bar(x + (i - 1) * 0.27, v, width=0.27, label=ph, color=["#1b9e77", "#d95f02", "#7570b3"][i])
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in COMPS], fontsize=8, rotation=20)
        ax.set_title(sub, fontsize=9)
    axes[0].set_ylabel("Coverage (% of messages)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_B4_topical.png", dpi=200)
    plt.close(fig)


def main():
    df, eng = prep()
    msgs = sender_frame(df)
    b1, b2, roll = b1_b2(eng, msgs)
    labels, monthly, tsum = b3_template()
    b4 = b4_topical(eng)
    wk, its = b5_its(eng)
    figures(roll, wk, its, b4)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("voice notes with translation matched:", int((eng.message_type.eq("voice_note") & eng.vn_translation.ne("")).sum()),
          "of", int(eng.message_type.eq("voice_note").sum()))
    print(b1.round(3).T.to_string())
    print(b2.round(3).T.to_string())
    print(labels.to_string())
    print(monthly.tail(10).to_string())
    print(tsum)
    print(b4.round(2).to_string())
    print(its.round(3).to_string())
    print(roll[["window", "messages_per_30d", "active_senders", "gini", "gini_excl_top_sender", "top_sender_share",
                "density", "mean_partners", "mean_partners_incl_isolates", "seq_density"]].round(2).to_string())


if __name__ == "__main__":
    main()
