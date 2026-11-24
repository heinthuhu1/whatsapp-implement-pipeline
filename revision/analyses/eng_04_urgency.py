"""Step 6: urgency-flagged messages and urgency-response latency.

Variants of urgency flagging (lexicon from data/Jheeng/config/settings.yaml):
  A original   : substring match on the message field only. This is what the
                 published run did: src/06 reads a translation_en column that
                 messages.csv does not have, and src/05 joins translations on
                 media_filename, which never matches because the anonymiser
                 overwrote the date token in voice-note filenames.
  B +voice     : substring match on the English translation for voice notes
                 (joined via the reconstructed filename) and on the message
                 field otherwise (the behaviour the Methods describe).
  C word       : as A, whole-word match (so 'now' no longer matches 'know').
  D word+voice : as B, whole-word match.

For each flagged message: the next message from a different sender (original
rule), its latency, and whether that reply fell within the same 60-min
conversational session (gap chain < 60 min). No message text is written out.

Outputs: 04_urgency_counts.csv, 04_urgency_events.csv, 04_urgency_keywords.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from eng_common import OUT, PHASES, load_clean, matched_terms, settings

SESSION_T = 60


def word_terms(text, kws):
    if not isinstance(text, str):
        return []
    low = text.lower()
    return [k for k in kws if re.search(r"\b" + re.escape(k.lower()) + r"\b", low)]


def main() -> None:
    pd.set_option("display.width", 250)
    kws = settings()["urgency_keywords"]
    df = load_clean()
    gap = df["timestamp"].diff().dt.total_seconds() / 60
    df["session_id"] = (gap.isna() | (gap >= SESSION_T)).cumsum()
    is_vn = df["message_type"].eq("voice_note")
    text_msg = df["message"].fillna("")
    text_vn = df["translation_en"].where(is_vn & df["translation_en"].notna(), df["message"]).fillna("")

    variants = {
        "A_original_substring_text_only": text_msg.apply(lambda t: matched_terms(t, kws)),
        "B_substring_with_voice_translations": text_vn.apply(lambda t: matched_terms(t, kws)),
        "C_wholeword_text_only": text_msg.apply(lambda t: word_terms(t, kws)),
        "D_wholeword_with_voice_translations": text_vn.apply(lambda t: word_terms(t, kws)),
    }

    # next message from a different sender (original rule), per message
    s = df["sender_code"].to_numpy()
    nxt = np.full(len(df), -1)
    ch = -1
    for i in range(len(df) - 1, -1, -1):
        if i + 1 < len(df) and s[i + 1] != s[i]:
            ch = i + 1
        nxt[i] = ch

    n_by_phase = df.groupby("phase").size()
    count_rows, ev_rows, kw_rows = [], [], []
    for name, terms in variants.items():
        flag = terms.str.len() > 0
        idx = np.where(flag.to_numpy())[0]
        for i in idx:
            j = nxt[i]
            lat = (df.at[j, "timestamp"] - df.at[i, "timestamp"]).total_seconds() / 60 if j >= 0 else np.nan
            ev_rows.append({
                "variant": name, "flag_timestamp": df.at[i, "timestamp"], "phase": df.at[i, "phase"],
                "flagger": df.at[i, "sender_code"], "message_type": df.at[i, "message_type"],
                "keywords_matched": ";".join(terms.iat[i]),
                "responder": df.at[j, "sender_code"] if j >= 0 else None,
                "latency_min": lat,
                "reply_in_same_60min_session": bool(j >= 0 and df.at[j, "session_id"] == df.at[i, "session_id"]),
            })
        for k in kws:
            kw_rows.append({"variant": name, "keyword": k,
                            "n_messages": int(terms.apply(lambda L, k=k: k in L).sum())})
    ev = pd.DataFrame(ev_rows)
    for name in variants:
        e = ev[ev.variant == name]
        for p in ["Overall"] + PHASES:
            ee = e if p == "Overall" else e[e.phase == p]
            n_msgs = int(n_by_phase.sum() if p == "Overall" else n_by_phase[p])
            lat = ee["latency_min"].dropna()
            count_rows.append({
                "variant": name, "phase": p, "n_messages": n_msgs, "n_flagged": len(ee),
                "rate_pct": 100 * len(ee) / n_msgs,
                "n_flagged_voice_notes": int((ee.message_type == "voice_note").sum()),
                "n_distinct_days": ee["flag_timestamp"].dt.normalize().nunique(),
                "n_with_next_other_sender": int(ee["latency_min"].notna().sum()),
                "n_reply_within_session": int(ee["reply_in_same_60min_session"].sum()),
                "median_latency_min": lat.median() if len(lat) else np.nan,
                "mean_latency_min": lat.mean() if len(lat) else np.nan,
                "raw_latencies_min": ";".join(f"{v:.0f}" for v in sorted(lat)),
            })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / "04_urgency_counts.csv", index=False)
    ev.to_csv(OUT / "04_urgency_events.csv", index=False)
    kwt = pd.DataFrame(kw_rows).pivot(index="keyword", columns="variant", values="n_messages")
    kwt.to_csv(OUT / "04_urgency_keywords.csv")
    print(counts.round(2).to_string(index=False))
    print(kwt.to_string())
    print(ev[ev.variant.str.startswith("A")].drop(columns="variant").to_string(index=False))


if __name__ == "__main__":
    main()
