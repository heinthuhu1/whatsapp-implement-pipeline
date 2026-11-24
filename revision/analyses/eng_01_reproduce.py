"""Step 1: reproduce the originally reported engagement numbers (Table 1) from
the archived Jheeng messages.csv using the exact logic of src/03_engagement.py,
src/05_sentiment.py, src/06_fidelity.py and src/09_table1.py, then report the
number of qualifying events per phase and per sender.

Outputs (data/Jheeng/revision/engagement/):
  01_reproduction.csv          original vs reproduced cell values
  01_latency_pairs_original.csv  every prompt -> next-other-sender pair (original rule)
  01_n_events_by_phase_sender.csv  n of latency pairs by phase x sender
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eng_common import MESSAGES, OUT, PHASES, load_raw, match_any, settings, voice_note_lookup


def original_latency_pairs(df: pd.DataFrame, kind: str = "quicksort") -> pd.DataFrame:
    """Verbatim logic of src/03_engagement.response_latency, before grouping.
    kind="quicksort" is the pandas default used by the original code (unstable:
    it reorders messages that share a minute); "mergesort" keeps export order."""
    df2 = df.sort_values("timestamp", kind=kind).reset_index(drop=True)
    recs = []
    for i in range(len(df2) - 1):
        sender = df2.at[i, "sender_code"]
        phase = df2.at[i, "phase"]
        for j in range(i + 1, len(df2)):
            if df2.at[j, "sender_code"] != sender and pd.notna(df2.at[j, "sender_code"]):
                recs.append({"i": i, "j": j, "prompt_ts": df2.at[i, "timestamp"],
                             "reply_ts": df2.at[j, "timestamp"],
                             "sender_code": sender, "responder": df2.at[j, "sender_code"],
                             "phase": phase,
                             "latency_minutes": (df2.at[j, "timestamp"] - df2.at[i, "timestamp"]).total_seconds() / 60})
                break
    return pd.DataFrame(recs)


def original_urgency_pairs(df: pd.DataFrame, kws: list, kind: str = "quicksort") -> pd.DataFrame:
    """Verbatim logic of src/06_fidelity.responsiveness. In the original run
    messages.csv has no translation_en column, so text_combined == message."""
    df2 = df.sort_values("timestamp", kind=kind).reset_index(drop=True)
    df2["text_combined"] = df2["message"].fillna("").astype(str)
    rows = []
    for i, row in df2.iterrows():
        if not match_any(row["text_combined"], kws):
            continue
        for j in range(i + 1, len(df2)):
            o = df2.iloc[j]
            if pd.notna(o["sender_code"]) and o["sender_code"] != row["sender_code"]:
                rows.append({"flag_timestamp": row["timestamp"], "phase": row["phase"],
                             "flagger": row["sender_code"], "responder": o["sender_code"],
                             "latency_minutes": (o["timestamp"] - row["timestamp"]).total_seconds() / 60})
                break
    return pd.DataFrame(rows)


def original_urgency_flags(df: pd.DataFrame, kws: list) -> pd.Series:
    """Verbatim logic of src/05_sentiment: translation joined on media_filename."""
    d = df.copy()
    vn = voice_note_lookup()[["filename", "translation_en"]]
    d["translation_en"] = None
    d["media_filename"] = d["media_filename"].astype(str)
    d = d.merge(vn, left_on="media_filename", right_on="filename", how="left")
    d["translation_en"] = d["translation_en_y"].combine_first(d["translation_en_x"])
    n_joined = int(d["translation_en_y"].notna().sum())
    text = d["translation_en"].fillna(d["message"]).fillna("").astype(str)
    d["urgency"] = text.apply(lambda t: match_any(t, kws))
    return d, n_joined


def h(x):
    return x / 60.0


def main() -> None:
    st = settings()
    kws = st["urgency_keywords"]
    df = load_raw()
    msg_all = pd.read_csv(MESSAGES, parse_dates=["timestamp"])
    msg_all["phase"] = msg_all["phase"].astype(str)

    # ---- response latency (src/03 -> src/09) -------------------------------
    pairs = original_latency_pairs(df)
    pairs.to_csv(OUT / "01_latency_pairs_original.csv", index=False)
    cell = pairs.groupby(["sender_code", "phase"])["latency_minutes"].mean().reset_index()
    rep = {}
    rep["Mean response latency (h), as published: mean of sender x phase means"] = {
        "Overall": h(cell["latency_minutes"].mean()),
        **{p: h(cell.loc[cell.phase == p, "latency_minutes"].mean()) for p in PHASES}}
    rep["Mean response latency (h), pooled over all pairs (not published)"] = {
        "Overall": h(pairs["latency_minutes"].mean()),
        **{p: h(pairs.loc[pairs.phase == p, "latency_minutes"].mean()) for p in PHASES}}
    rep["Sender x phase cells averaged (n)"] = {
        "Overall": len(cell), **{p: int((cell.phase == p).sum()) for p in PHASES}}
    named = pairs[pairs.sender_code.notna()]
    rep["Latency pairs entering the mean (n, named prompt senders)"] = {
        "Overall": len(named), **{p: int((named.phase == p).sum()) for p in PHASES}}
    rep["Latency pairs incl. prompts from empty-sender rows (n)"] = {
        "Overall": len(pairs), **{p: int((pairs.phase == p).sum()) for p in PHASES}}

    # ---- urgency-response latency (src/06 -> src/09) ------------------------
    up = original_urgency_pairs(df, kws)
    up.to_csv(OUT / "01_urgency_pairs_original.csv", index=False)
    rep["Mean urgency-response latency (h), as published"] = {
        "Overall": h(up["latency_minutes"].mean()),
        **{p: h(up.loc[up.phase == p, "latency_minutes"].mean()) for p in PHASES}}
    rep["Urgency-response pairs (n)"] = {
        "Overall": len(up), **{p: int((up.phase == p).sum()) for p in PHASES}}

    # ---- sensitivity: identical rules, stable sort (export order kept) -------
    n_moved = int((df.sort_values("timestamp").index.to_numpy()
                   != df.sort_values("timestamp", kind="mergesort").index.to_numpy()).sum())
    rep["Rows reordered by the original unstable sort (n)"] = {"Overall": n_moved}
    ps = original_latency_pairs(df, kind="mergesort")
    ps.to_csv(OUT / "01_latency_pairs_stable.csv", index=False)
    cs = ps.groupby(["sender_code", "phase"])["latency_minutes"].mean().reset_index()
    rep["Mean response latency (h), published rule but stable sort"] = {
        "Overall": h(cs["latency_minutes"].mean()),
        **{p: h(cs.loc[cs.phase == p, "latency_minutes"].mean()) for p in PHASES}}
    us = original_urgency_pairs(df, kws, kind="mergesort")
    us.to_csv(OUT / "01_urgency_pairs_stable.csv", index=False)
    rep["Mean urgency-response latency (h), published rule but stable sort"] = {
        "Overall": h(us["latency_minutes"].mean()),
        **{p: h(us.loc[us.phase == p, "latency_minutes"].mean()) for p in PHASES}}

    # ---- urgency-language rate (src/05 -> src/09) ---------------------------
    d, n_joined = original_urgency_flags(df, kws)
    rates = d.groupby("phase")["urgency"].mean()
    counts_all = msg_all.groupby("phase").size()  # src/09 weights by all rows incl. system
    overall = sum(rates[p] * counts_all[p] for p in PHASES) / counts_all.sum()
    rep["Urgency-language rate (%), as published"] = {
        "Overall": overall * 100, **{p: rates[p] * 100 for p in PHASES}}
    rep["Urgency-flagged messages (n), original flagging"] = {
        "Overall": int(d["urgency"].sum()), **{p: int(d.loc[d.phase == p, "urgency"].sum()) for p in PHASES}}
    rep["Voice-note translations joined in original src/05 (n)"] = {"Overall": n_joined}

    # ---- after-hours (src/09) -----------------------------------------------
    s, e = st["after_hours_start"], st["after_hours_end"]
    hr = df["timestamp"].dt.hour
    df["after"] = (hr >= s) | (hr < e)
    rep["After-hours activity rate (%), as published"] = {
        "Overall": df["after"].mean() * 100,
        **{p: df.loc[df.phase == p, "after"].mean() * 100 for p in PHASES}}

    published = {
        "Mean response latency (h), as published: mean of sender x phase means": [13.1, 6.3, 19.8, 11.6],
        "Mean urgency-response latency (h), as published": [5.9, 4.6, 16.3, 40 / 60],
        "After-hours activity rate (%), as published": [9.0, 8.9, 9.7, 7.2],
        "Urgency-language rate (%), as published": [0.2, 0.7, 0.1, 0.2],
    }
    rows = []
    for k, v in rep.items():
        r = {"metric": k}
        for c in ["Overall"] + PHASES:
            r[c] = v.get(c, np.nan)
        if k in published:
            for c, pv in zip(["Overall"] + PHASES, published[k]):
                r[f"published_{c}"] = pv
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "01_reproduction.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))

    # ---- qualifying events per phase x sender --------------------------------
    tab = (named.groupby(["phase", "sender_code"]).size().unstack(0, fill_value=0)
           .reindex(columns=PHASES, fill_value=0))
    tab["Overall"] = tab.sum(axis=1)
    tab = tab.sort_values("Overall", ascending=False)
    tab.loc["Total"] = tab.sum()
    means = named.groupby(["phase", "sender_code"])["latency_minutes"].mean().unstack(0).reindex(columns=PHASES) / 60
    means.columns = [f"mean_h_{c}" for c in means.columns]
    tab = tab.join(means.round(1))
    tab.to_csv(OUT / "01_n_events_by_phase_sender.csv")
    print(tab.to_string())


if __name__ == "__main__":
    main()
