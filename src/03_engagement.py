"""Engagement metrics: volume, velocity, silences, response latency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def response_latency(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.sort_values("timestamp").reset_index(drop=True)
    records = []
    for i in range(len(df2) - 1):
        sender = df2.at[i, "sender_code"]
        phase = df2.at[i, "phase"]
        for j in range(i + 1, len(df2)):
            if df2.at[j, "sender_code"] != sender and pd.notna(df2.at[j, "sender_code"]):
                delta = (df2.at[j, "timestamp"] - df2.at[i, "timestamp"]).total_seconds() / 60.0
                records.append({"sender_code": sender, "phase": phase,
                                "latency_minutes": delta})
                break
    lat = pd.DataFrame(records)
    if lat.empty:
        return lat
    return lat.groupby(["sender_code", "phase"])["latency_minutes"].mean().reset_index()


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"])

    weekly = weekly_volume(df).assign(metric="weekly_volume")
    phase = phase_volume(df).assign(metric="phase_volume")
    spikes = velocity_spikes(df, settings["rolling_window_days"]).assign(metric="velocity")
    silences = silence_periods(df, settings["silence_threshold_hours"]).assign(metric="silence")
    share = sender_share(df).assign(metric="sender_share")
    after = after_hours_rate(df, settings["after_hours_start"],
                             settings["after_hours_end"]).assign(metric="after_hours")
    latency = response_latency(df).assign(metric="response_latency")

    out = Path(settings["paths"]["engagement_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Concatenate vertically with metric column as discriminator
    frames = [weekly, phase, spikes, share, after, latency]
    if not silences.empty:
        frames.append(silences)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out, index=False)
    print(f"[03_engagement] Wrote {out}")


if __name__ == "__main__":
    main()
