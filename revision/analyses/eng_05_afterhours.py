"""Step 7: after-hours activity (22:00 to 06:00, from the archived config).

Counts and rates per phase, and concentration across senders (coded IDs):
the minimum number of senders that together account for at least 80% of
after-hours messages, the top sender's share, and each sender's share of
after-hours messages relative to their share of all messages.

Outputs: 05_afterhours_by_phase.csv, 05_afterhours_by_sender.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eng_common import OUT, PHASES, flag_case_reports, load_clean, settings


def n_for_share(counts: pd.Series, share: float = 0.8) -> int:
    c = counts.sort_values(ascending=False)
    cum = c.cumsum() / c.sum()
    return int((cum < share).sum() + 1)


def main() -> None:
    pd.set_option("display.width", 250)
    st = settings()
    a, b = st["after_hours_start"], st["after_hours_end"]
    df = load_clean()
    df["is_case_report"] = flag_case_reports(df)
    h = df["timestamp"].dt.hour
    df["after"] = (h >= a) | (h < b)

    rows, srows = [], []
    for p in ["Overall"] + PHASES:
        d = df if p == "Overall" else df[df.phase == p]
        ah = d[d["after"]]
        cnt = ah.groupby("sender_code").size()
        allc = d.groupby("sender_code").size()
        rows.append({
            "phase": p, "n_messages": len(d), "n_after_hours": len(ah),
            "rate_pct": 100 * len(ah) / len(d),
            "n_senders_active": d["sender_code"].nunique(),
            "n_senders_any_after_hours": int((cnt > 0).sum()),
            "n_senders_for_80pct": n_for_share(cnt) if len(cnt) else 0,
            "top_sender": cnt.idxmax() if len(cnt) else None,
            "top_sender_share_after_hours_pct": 100 * cnt.max() / cnt.sum() if len(cnt) else np.nan,
            "top_sender_share_all_msgs_pct": 100 * allc.get(cnt.idxmax(), 0) / allc.sum() if len(cnt) else np.nan,
            "n_senders_for_80pct_of_all_msgs": n_for_share(allc),
            "after_hours_voice_note_pct": 100 * (ah["message_type"] == "voice_note").mean(),
            "after_hours_case_report_pct": 100 * ah["is_case_report"].mean(),
        })
        for s_, n_all in allc.items():
            srows.append({"phase": p, "sender_code": s_, "n_msgs": int(n_all),
                          "n_after_hours": int(cnt.get(s_, 0)),
                          "sender_after_hours_rate_pct": 100 * cnt.get(s_, 0) / n_all,
                          "share_of_after_hours_pct": 100 * cnt.get(s_, 0) / max(len(ah), 1),
                          "share_of_all_msgs_pct": 100 * n_all / len(d)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "05_afterhours_by_phase.csv", index=False)
    so = pd.DataFrame(srows).sort_values(["phase", "n_after_hours"], ascending=[True, False])
    so.to_csv(OUT / "05_afterhours_by_sender.csv", index=False)
    print(out.round(1).to_string(index=False))
    print(so[so.n_after_hours > 0].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
