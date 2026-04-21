"""Theory-of-change coverage, workaround detection, responsiveness."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def match_any(text: str, keywords: list) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def first_match(text: str, mapping: dict) -> list:
    hits = []
    for component, kws in mapping.items():
        if match_any(text, kws):
            hits.append(component)
    return hits


def coverage_by_phase(df: pd.DataFrame, toc: dict) -> pd.DataFrame:
    rows = []
    for phase, sub in df.groupby("phase"):
        total = len(sub)
        if total == 0:
            continue
        for comp, kws in toc.items():
            hits = sub["text_combined"].apply(lambda t: match_any(t, kws)).sum()
            rows.append({"phase": phase, "component": comp,
                         "coverage_pct": hits / total * 100, "hits": int(hits),
                         "kind": "coverage"})
    return pd.DataFrame(rows)


def workarounds(df: pd.DataFrame, toc: dict, workaround_kws: list) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        txt = row["text_combined"]
        if not match_any(txt, workaround_kws):
            continue
        components = first_match(txt, toc)
        if not components:
            components = ["unspecified"]
        for comp in components:
            rows.append({"timestamp": row["timestamp"], "phase": row["phase"],
                         "sender_code": row["sender_code"], "component": comp,
                         "message": txt[:200], "kind": "workaround"})
    return pd.DataFrame(rows)


def responsiveness(df: pd.DataFrame, urgency_kws: list) -> pd.DataFrame:
    df2 = df.sort_values("timestamp").reset_index(drop=True)
    rows = []
    for i, row in df2.iterrows():
        if not match_any(row["text_combined"], urgency_kws):
            continue
        sender = row["sender_code"]
        for j in range(i + 1, len(df2)):
            other = df2.iloc[j]
            if pd.notna(other["sender_code"]) and other["sender_code"] != sender:
                delta = (other["timestamp"] - row["timestamp"]).total_seconds() / 60.0
                rows.append({"flag_timestamp": row["timestamp"],
                             "phase": row["phase"],
                             "flagger": sender,
                             "responder": other["sender_code"],
                             "latency_minutes": delta,
                             "kind": "responsiveness"})
                break
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"]).copy()
    df["text_combined"] = (
        df.get("translation_en", pd.Series([None] * len(df))).fillna(df["message"]).fillna("").astype(str)
    )

    toc = settings["toc_component_keywords"]
    coverage = coverage_by_phase(df, toc)
    wa = workarounds(df, toc, settings["workaround_keywords"])
    resp = responsiveness(df, settings["urgency_keywords"])

    combined = pd.concat([coverage, wa, resp], ignore_index=True, sort=False)
    out = Path(settings["paths"]["fidelity_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"[06_fidelity] Wrote {out}")


if __name__ == "__main__":
    main()
