"""Shared helpers for the engagement-domain re-analysis (revision, R1 #8 and R2 #21).

All inputs are read from the archived Jheeng run under data/Jheeng/ and all
outputs are written to data/Jheeng/revision/engagement/ (gitignored, contains
participant-derived data). Nothing in src/, config/ or data/processed/ is read
or modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
JHEENG = ROOT / "data" / "Jheeng"
MESSAGES = JHEENG / "interim" / "messages.csv"
VOICE_NOTES = JHEENG / "processed" / "voice_notes.csv"
SETTINGS = JHEENG / "config" / "settings.yaml"
OUT = JHEENG / "revision" / "engagement"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = ["2024", "2025", "2026"]
WINDOW_START = pd.Timestamp("2024-09-22")
WINDOW_END = pd.Timestamp("2026-04-11 23:59:59")


def settings() -> dict:
    with open(SETTINGS) as f:
        return yaml.safe_load(f)


def load_raw() -> pd.DataFrame:
    """messages.csv exactly as the original stages read it (system rows dropped
    by message_type only, as in src/03, src/05, src/06)."""
    df = pd.read_csv(MESSAGES, parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"]).copy()
    df["phase"] = df["phase"].astype(str).str.replace(r"\.0+$", "", regex=True)
    return df


def voice_note_lookup() -> pd.DataFrame:
    vn = pd.read_csv(VOICE_NOTES)
    return vn[["filename", "translation_en", "duration_seconds"]]


def reconstruct_vn_filename(df: pd.DataFrame) -> pd.Series:
    """The anonymiser replaced the date token in voice-note filenames with a
    sender code (e.g. 'PTT-[P027]-WA0005.opus'), so media_filename holds only
    '-WA0005.opus'. Rebuild the original name from the message date."""
    stem = df["media_filename"].astype("string")
    date = df["timestamp"].dt.strftime("%Y%m%d")
    return ("PTT-" + date + stem).where(stem.notna() & stem.str.startswith("-WA"))


def load_clean(with_translations: bool = True) -> pd.DataFrame:
    """Revised analysis frame: human-authored messages only (non-empty
    sender_code), sorted, within the fixed study window, with an optional
    voice-note translation joined via the reconstructed filename."""
    df = load_raw()
    df = df[df["sender_code"].notna()].copy()
    df = df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)]
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if with_translations:
        vn = voice_note_lookup()
        df["vn_filename"] = reconstruct_vn_filename(df)
        df = df.merge(vn[["filename", "translation_en"]], how="left",
                      left_on="vn_filename", right_on="filename")
        df = df.drop(columns=["filename"])
    return df


def next_other_sender(df: pd.DataFrame) -> pd.DataFrame:
    """For each message i, find the first later message j whose sender differs
    from i's sender (original src/03 definition). Returns columns:
    j_index, latency_min (NaN if none). df must be sorted and have non-null
    sender_code."""
    s = df["sender_code"].to_numpy()
    t = df["timestamp"].to_numpy()
    n = len(df)
    j_idx = np.full(n, -1)
    # walk backwards: next index where sender changes
    nxt_change = -1
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            if s[i + 1] != s[i]:
                nxt_change = i + 1
            # else keep the run's next change
        j_idx[i] = nxt_change
    lat = np.where(j_idx >= 0,
                   (t[np.maximum(j_idx, 0)] - t).astype("timedelta64[s]").astype(float) / 60.0,
                   np.nan)
    return pd.DataFrame({"j_index": j_idx, "latency_min": lat}, index=df.index)


def match_any(text, keywords) -> bool:
    if not isinstance(text, str):
        return False
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def matched_terms(text, keywords) -> list:
    if not isinstance(text, str):
        return []
    low = text.lower()
    return [k for k in keywords if k.lower() in low]


def describe(x: pd.Series | np.ndarray) -> dict:
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "median": x.median(),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
        "mean": x.mean(),
        "p90": x.quantile(0.90),
        "min": x.min(),
        "max": x.max(),
    }


def fmt_min(m: float) -> str:
    """Human-readable duration from minutes."""
    if m is None or pd.isna(m):
        return "NA"
    if m < 1:
        return f"{m * 60:.0f} s"
    if m < 60:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def flag_case_reports(df: pd.DataFrame) -> pd.Series:
    """True where a message was extracted as a structured case-dispatch report
    (data/Jheeng/interim/case_reports.csv), matched on timestamp, sender and
    message body."""
    cr = pd.read_csv(JHEENG / "interim" / "case_reports.csv", parse_dates=["timestamp"],
                     low_memory=False)
    key_cr = set(zip(cr["timestamp"], cr["sender_code"], cr["raw_message"].astype(str).str.strip()))
    key_ts = set(zip(cr["timestamp"], cr["sender_code"]))
    exact = [ (t, s, str(m).strip()) in key_cr for t, s, m in zip(df["timestamp"], df["sender_code"], df["message"]) ]
    loose = [ (t, s) in key_ts for t, s in zip(df["timestamp"], df["sender_code"]) ]
    flag = pd.Series(exact, index=df.index)
    flag.attrs["n_exact"] = int(sum(exact))
    flag.attrs["n_loose"] = int(sum(loose))
    # fall back to timestamp+sender match when the stored raw_message was normalised
    return pd.Series(np.array(exact) | (np.array(loose) & (df["message_type"] == "text").to_numpy()),
                     index=df.index)


def build_turns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive messages from the same sender into turns. Each turn
    is followed by a turn from a different sender (by construction), so the
    turn-transition interval is one qualifying latency event per turn and a
    burst of N messages is not counted N times (as it is under the original
    message-level rule)."""
    d = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    new = d["sender_code"].ne(d["sender_code"].shift())
    d["turn_id"] = new.cumsum() - 1
    agg = {"sender_code": "first", "phase": "first",
           "timestamp": ["first", "last", "size"]}
    if "is_case_report" in d.columns:
        agg["is_case_report"] = ["all", "any"]
    t = d.groupby("turn_id").agg(agg)
    t.columns = ["_".join(c) if isinstance(c, tuple) else c for c in t.columns]
    t = t.rename(columns={"sender_code_first": "sender_code", "phase_first": "phase",
                          "timestamp_first": "start", "timestamp_last": "end",
                          "timestamp_size": "n_msgs",
                          "is_case_report_all": "all_case_report",
                          "is_case_report_any": "any_case_report"})
    t["next_start"] = t["start"].shift(-1)
    t["next_sender"] = t["sender_code"].shift(-1)
    t["lat_last_min"] = (t["next_start"] - t["end"]).dt.total_seconds() / 60
    t["lat_first_min"] = (t["next_start"] - t["start"]).dt.total_seconds() / 60
    return t.reset_index()


def working_minutes(t0: pd.Timestamp, t1: pd.Timestamp, a: int, b: int) -> float:
    """Minutes of [t0, t1] that fall inside the daily window [a:00, b:00)."""
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return 0.0 if (pd.notna(t0) and pd.notna(t1)) else np.nan
    total = 0.0
    day = t0.normalize()
    while day <= t1:
        ws, we = day + pd.Timedelta(hours=a), day + pd.Timedelta(hours=b)
        lo, hi = max(ws, t0), min(we, t1)
        if hi > lo:
            total += (hi - lo).total_seconds() / 60
        day += pd.Timedelta(days=1)
    return total


def in_window(ts: pd.Series, a: int, b: int) -> pd.Series:
    h = ts.dt.hour
    return (h >= a) & (h < b)


def km(durations, events):
    """Kaplan-Meier estimate. Returns DataFrame(time, surv) at event times."""
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    order = np.argsort(d, kind="mergesort")
    d, e = d[order], e[order]
    times = np.unique(d[e])
    s, out = 1.0, [(0.0, 1.0)]
    for tt in times:
        at_risk = (d >= tt).sum()
        ev = ((d == tt) & e).sum()
        s *= 1 - ev / at_risk
        out.append((tt, s))
    return pd.DataFrame(out, columns=["time", "surv"])


def km_at(curve: pd.DataFrame, t: float) -> float:
    """Survival S(t) (right-continuous step)."""
    return float(curve.loc[curve["time"] <= t, "surv"].iloc[-1])


def km_median(curve: pd.DataFrame) -> float:
    hit = curve.loc[curve["surv"] <= 0.5, "time"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def logrank(durations, events, groups):
    """k-sample log-rank test. Returns (chi2, df, p)."""
    from scipy.stats import chi2
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    g = np.asarray(groups)
    labels = np.unique(g)
    k = len(labels)
    times = np.unique(d[e])
    O = np.zeros(k); E = np.zeros(k); V = np.zeros((k, k))
    for tt in times:
        risk = d >= tt
        n = risk.sum()
        dt = ((d == tt) & e).sum()
        nj = np.array([(risk & (g == L)).sum() for L in labels])
        dj = np.array([((d == tt) & e & (g == L)).sum() for L in labels])
        O += dj
        E += dt * nj / n
        if n > 1:
            f = dt * (n - dt) / (n * n * (n - 1))
            V += f * (np.diag(nj * n) - np.outer(nj, nj))
    U = (O - E)[:-1]
    Vr = V[:-1, :-1]
    stat = float(U @ np.linalg.solve(Vr, U))
    return stat, k - 1, float(chi2.sf(stat, k - 1))
