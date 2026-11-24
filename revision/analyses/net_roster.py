"""Roster exposure (A4) and personnel change (B6) reconstructed from WhatsApp system notices.

Membership evidence: 'added', 'joined', 'left', 'removed' notices (entries/exits), security-code
notices (presence at a date), and posting. Members with no add/join notice who post are treated as
founding members present from group creation (2024-09-22). No exit notices exist in the export, so
members are treated as present until the end of the study window.

Outputs: A4_roster_events.csv, A4_roster_members.csv, A4_roster_phase.csv, A4_exposure_by_sender_phase.csv,
         B6_personnel_change.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from net_common import (OUT, PHASE_NAMES, PHASES, STUDY_END, STUDY_START, gini, gini_corrected, load_messages,
                        phase_days, sender_frame)

CODE = re.compile(r"P\d{3}")


def events(df):
    sysrows = df[df["sender_code"].isna() | (df["message_type"] == "system")]
    rows = []
    for ts, txt in zip(sysrows["timestamp"], sysrows["message"].fillna("")):
        low = txt.lower()
        if " added " in f" {low} ":
            kind = "added"
        elif "joined" in low:
            kind = "joined"
        elif " left" in low:
            kind = "left"
        elif "removed" in low:
            kind = "removed"
        elif "security code" in low:
            kind = "security_code_changed"
        elif "created" in low:
            kind = "group_created"
        else:
            kind = "other_notice"
        codes = CODE.findall(txt)
        for c in codes or [""]:
            rows.append({"timestamp": ts, "event": kind, "sender_code": c})
    return pd.DataFrame(rows)


def members(df, ev):
    msgs = sender_frame(df)
    first = msgs.groupby("sender_code")["timestamp"].min()
    last = msgs.groupby("sender_code")["timestamp"].max()
    codes = sorted(set(first.index) | set(ev.loc[ev.sender_code != "", "sender_code"]))
    rows = []
    for c in codes:
        e = ev[ev.sender_code == c]
        add = e.loc[e.event.isin(["added", "joined"]), "timestamp"].min()
        ex = e.loc[e.event.isin(["left", "removed"]), "timestamp"].min()
        entry = add if pd.notna(add) else STUDY_START
        rows.append({"sender_code": c, "entry_basis": "add notice" if pd.notna(add) else "founding (no add notice)",
                     "entry": entry, "exit": ex if pd.notna(ex) else pd.NaT,
                     "first_post": first.get(c, pd.NaT), "last_post": last.get(c, pd.NaT),
                     "days_entry_to_first_post": (first.get(c) - entry).days if c in first else np.nan,
                     "messages_total": int((msgs.sender_code == c).sum()),
                     "security_code_notices": int((e.event == "security_code_changed").sum())})
    return pd.DataFrame(rows)


def exposure(df, mem):
    msgs = sender_frame(df)
    rows = []
    for ph in PHASE_NAMES + ["Overall"]:
        s, e = (STUDY_START, STUDY_END) if ph == "Overall" else PHASES[ph]
        for _, m in mem.iterrows():
            ent = max(m.entry, s)
            ex = min(m.exit if pd.notna(m.exit) else STUDY_END, e)
            days = max(0.0, (ex - ent).total_seconds() / 86400)
            if days <= 0:
                continue
            n = int(((msgs.sender_code == m.sender_code) & (msgs.timestamp >= s) & (msgs.timestamp <= e)).sum())
            rows.append({"phase": ph, "sender_code": m.sender_code, "days_in_group": round(days, 1),
                         "messages": n, "messages_per_30d_exposure": n / days * 30})
    return pd.DataFrame(rows)


def roster_phase(expo, msgs):
    rows = []
    for ph, g in expo.groupby("phase", sort=False):
        act = g[g.messages > 0]
        length = (phase_days(ph) if ph != "Overall" else (STUDY_END.normalize() - STUDY_START).days + 1)
        r = {"phase": ph, "phase_days": length, "roster_lower_bound": len(g), "active_senders": len(act),
             "silent_members": ";".join(sorted(g.loc[g.messages == 0, "sender_code"])),
             "person_days_in_group": round(g.days_in_group.sum(), 0),
             "gini_active_original": gini(act.messages), "gini_active_corrected": gini_corrected(act.messages),
             "gini_roster_incl_silent": gini(g.messages), "gini_roster_corrected": gini_corrected(g.messages),
             "gini_rate_per_exposure_active": gini(act.messages_per_30d_exposure),
             "gini_rate_per_exposure_roster": gini(g.messages_per_30d_exposure),
             # sensitivity: the exporter account ('You') may be a 13th member who never posted
             "gini_roster_plus_one_silent": gini(list(g.messages) + [0]),
             "messages_per_30d": g.messages.sum() / length * 30,
             "messages_per_active_sender": g.messages.sum() / max(len(act), 1),
             "messages_per_30_person_days": g.messages.sum() / g.days_in_group.sum() * 30}
        top = act.sort_values("messages", ascending=False).iloc[0]
        r["top_sender"], r["top_sender_share"] = top.sender_code, top.messages / act.messages.sum()
        rows.append(r)
    return pd.DataFrame(rows)


def personnel(mem, ev):
    rows = []
    for ph in PHASE_NAMES:
        s, e = PHASES[ph]
        inph = lambda t: pd.notna(t) and s <= t <= e
        rows.append({
            "phase": ph,
            "added_or_joined": ";".join(f"{c} ({t:%Y-%m-%d})" for c, t, k in zip(ev.sender_code, ev.timestamp, ev.event)
                                        if k in ("added", "joined") and inph(t)) or "none",
            "n_added_or_joined": int(sum(k in ("added", "joined") and inph(t) for t, k in zip(ev.timestamp, ev.event))),
            "n_left_or_removed": int(sum(k in ("left", "removed") and inph(t) for t, k in zip(ev.timestamp, ev.event))),
            "new_senders_first_post": ";".join(f"{c} ({t:%Y-%m-%d})" for c, t in zip(mem.sender_code, mem.first_post) if inph(t)) or "none",
            "n_new_senders": int(sum(inph(t) for t in mem.first_post)),
            "senders_last_post_in_phase": ";".join(f"{c} ({t:%Y-%m-%d})" for c, t in zip(mem.sender_code, mem.last_post) if inph(t) and t < STUDY_END - pd.Timedelta(days=90)) or "none",
            "security_code_notices": int(sum(k == "security_code_changed" and inph(t) for t, k in zip(ev.timestamp, ev.event))),
        })
    return pd.DataFrame(rows)


def main():
    df = load_messages()
    ev = events(df)
    mem = members(df, ev)
    expo = exposure(df, mem)
    rp = roster_phase(expo, sender_frame(df))
    pc = personnel(mem, ev)
    ev.to_csv(OUT / "A4_roster_events.csv", index=False)
    mem.to_csv(OUT / "A4_roster_members.csv", index=False)
    expo.to_csv(OUT / "A4_exposure_by_sender_phase.csv", index=False)
    rp.to_csv(OUT / "A4_roster_phase.csv", index=False)
    pc.to_csv(OUT / "B6_personnel_change.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(ev.event.value_counts())
    print(mem.to_string())
    print(expo.round(2).to_string())
    print(rp.round(3).T.to_string())
    print(pc.T.to_string())


if __name__ == "__main__":
    main()
