"""Engagement metrics: volume, velocity, silences, response latency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import load_settings

# Answer-time thresholds (minutes) reported for turn latency
LATENCY_THRESHOLDS = (15, 60, 240)


def weekly_volume(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.set_index("timestamp")
          .resample("W")
          .size()
          .rename("message_count")
          .reset_index()
    )


def phase_volume(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("phase").size().rename("message_count").reset_index()


def velocity_spikes(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    daily = df.set_index("timestamp").resample("D").size().rename("count")
    roll = daily.rolling(window=window_days, min_periods=1)
    mean = roll.mean()
    sd = roll.std().fillna(0)
    spike = daily > (mean + 2 * sd)
    return pd.DataFrame(
        {"date": daily.index, "count": daily.values,
         "rolling_mean": mean.values, "rolling_sd": sd.values, "spike": spike.values}
    )


def silence_periods(df: pd.DataFrame, threshold_hours: int) -> pd.DataFrame:
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    gaps = ts.diff().dt.total_seconds() / 3600.0
    mask = gaps > threshold_hours
    return pd.DataFrame(
        {"gap_start": ts[mask.shift(-1, fill_value=False).values].values
                       if mask.any() else [],
         "gap_end": ts[mask].values if mask.any() else [],
         "gap_hours": gaps[mask].values if mask.any() else []}
    )


def sender_share(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("sender_code").size().rename("count")
    total = counts.sum()
    share = (counts / total * 100).rename("share_pct")
    return pd.concat([counts, share], axis=1).reset_index()


def after_hours_rate(df: pd.DataFrame, start_h: int, end_h: int) -> pd.DataFrame:
    hours = df["timestamp"].dt.hour
    after = (hours >= start_h) | (hours < end_h)
    tmp = df.assign(after_hours=after)
    return (
        tmp.groupby("sender_code")["after_hours"]
           .mean().rename("after_hours_rate").reset_index()
    )


def build_turns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the message stream into turns.

    A turn is a maximal run of consecutive messages by the same sender
    (system rows and rows without a sender are dropped first). Each turn
    carries its sender, phase (of its first message), first and last
    timestamps, and latency_minutes: time from the turn's LAST message to the
    first message of the next turn, which is always by a different sender.
    The final turn has no next turn and gets NaN.
    """
    d = df[df["sender_code"].notna()]
    if "message_type" in d.columns:
        d = d[d["message_type"] != "system"]
    d = d.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["sender_code", "phase", "start", "end",
                                     "n_messages", "latency_minutes"])
    turn_id = d["sender_code"].ne(d["sender_code"].shift()).cumsum()
    turns = d.groupby(turn_id).agg(
        sender_code=("sender_code", "first"), phase=("phase", "first"),
        start=("timestamp", "first"), end=("timestamp", "last"),
        n_messages=("timestamp", "size"),
    ).reset_index(drop=True)
    next_start = turns["start"].shift(-1)
    turns["latency_minutes"] = (next_start - turns["end"]).dt.total_seconds() / 60.0
    return turns


def _latency_summary(lat: pd.Series) -> dict:
    lat = lat.dropna()
    row = {"n_turns": int(len(lat))}
    if lat.empty:
        return row
    row.update({
        "latency_median_min": lat.median(),
        "latency_q1_min": lat.quantile(0.25),
        "latency_q3_min": lat.quantile(0.75),
        "latency_mean_min": lat.mean(),
    })
    for t in LATENCY_THRESHOLDS:
        row[f"share_le_{t}min"] = float((lat <= t).mean())
    return row


def response_latency(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn-based response latency.

    Returns (summary, by_sender): summary has one row per phase plus
    phase="Overall" with n turns that have a next turn, median, quartiles,
    mean and the share answered within each LATENCY_THRESHOLDS value;
    by_sender has the median per sender x phase.
    """
    turns = build_turns(df)
    turns = turns[turns["latency_minutes"].notna()]
    rows = [{"phase": "Overall", **_latency_summary(turns["latency_minutes"])}]
    for phase, sub in turns.groupby("phase", sort=True):
        rows.append({"phase": phase, **_latency_summary(sub["latency_minutes"])})
    summary = pd.DataFrame(rows)
    by_sender = (
        turns.groupby(["sender_code", "phase"])["latency_minutes"]
             .agg(n_turns="size", latency_median_min="median")
             .reset_index()
    )
    return summary, by_sender


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"])
    # stable sort keeps export order for messages sharing a minute
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    weekly = weekly_volume(df).assign(metric="weekly_volume")
    phase = phase_volume(df).assign(metric="phase_volume")
    spikes = velocity_spikes(df, settings["rolling_window_days"]).assign(metric="velocity")
    silences = silence_periods(df, settings["silence_threshold_hours"]).assign(metric="silence")
    share = sender_share(df).assign(metric="sender_share")
    after = after_hours_rate(df, settings["after_hours_start"],
                             settings["after_hours_end"]).assign(metric="after_hours")
    latency, latency_sender = response_latency(df)
    latency = latency.assign(metric="turn_latency")
    latency_sender = latency_sender.assign(metric="turn_latency_by_sender")

    out = Path(settings["paths"]["engagement_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Concatenate vertically with metric column as discriminator
    frames = [weekly, phase, spikes, share, after, latency, latency_sender]
    if not silences.empty:
        frames.append(silences)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out, index=False)
    print(f"[03_engagement] Wrote {out}")


if __name__ == "__main__":
    main()
