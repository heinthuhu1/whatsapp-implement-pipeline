"""Theory-of-change coverage, workaround detection, responsiveness."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (content_bearing_text, content_kind, join_voice_translations,
                    load_settings, match_any)


def first_match(text: str, mapping: dict) -> list:
    hits = []
    for component, kws in mapping.items():
        if match_any(text, kws):
            hits.append(component)
    return hits


def coverage_by_phase(df: pd.DataFrame, toc: dict, kind: str = "coverage") -> pd.DataFrame:
    """Share of messages per phase mentioning each ToC component, with counts
    (hits, n_messages) so totals can be pooled across phases."""
    rows = []
    for phase, sub in df.groupby("phase"):
        total = len(sub)
        if total == 0:
            continue
        for comp, kws in toc.items():
            hits = sub["text_combined"].apply(lambda t: match_any(t, kws)).sum()
            rows.append({"phase": phase, "component": comp,
                         "coverage_pct": hits / total * 100, "hits": int(hits),
                         "n_messages": total, "kind": kind})
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
    """Latency from each urgency-flagged content-bearing message to the next
    message by a different sender. df must already be sorted stably by
    timestamp; flags use the content_text column (same denominator as 05)."""
    df2 = df.reset_index(drop=True)
    rows = []
    for i, row in df2.iterrows():
        if not match_any(row["content_text"], urgency_kws):
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
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"])
    # stable sort keeps export order for messages sharing a minute
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Voice-note translations replace the attachment line where available
    df = join_voice_translations(df, settings["paths"].get("voice_notes_output"))
    df["text_combined"] = df["translation_en"].fillna(df["message"]).fillna("").astype(str)

    kinds = content_kind(df)
    df["is_case_report"] = kinds == "case_report"
    df["content_text"] = content_bearing_text(df, kinds)

    toc = settings["toc_component_keywords"]
    coverage = coverage_by_phase(df, toc)
    coverage_excl = coverage_by_phase(df[~df["is_case_report"]], toc,
                                      kind="coverage_excl_case_reports")
    wa = workarounds(df, toc, settings["workaround_keywords"])
    resp = responsiveness(df, settings["urgency_keywords"])

    combined = pd.concat([coverage, coverage_excl, wa, resp], ignore_index=True, sort=False)
    out = Path(settings["paths"]["fidelity_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"[06_fidelity] Wrote {out}")


if __name__ == "__main__":
    main()
