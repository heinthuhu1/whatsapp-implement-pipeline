"""Sentiment, urgency/hedging keyword matching, emotional labour Gini.

Sentiment and lexicon rates use content-bearing messages only: free text and
voice-note English translations. Media placeholders, deleted-message notices,
untranslated voice notes, system notices and structured case-report messages
are not scored (sentiment NaN) and are left out of every denominator.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Block torchvision only if it is already known to be incompatible with the
# installed torch version (import will fail). This avoids masking a valid
# torchvision install on other machines.
try:
    import torchvision  # noqa: F401
except Exception:
    sys.modules.setdefault("torchvision", None)  # type: ignore[assignment]

import numpy as np
import pandas as pd
from transformers import pipeline

from common import (CONTENT_KINDS, content_bearing_text, content_kind,
                    join_voice_translations, load_settings, match_any)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LEXICONS = {
    "urgency": "urgency_keywords",
    "hedging": "hedging_keywords",
    "peer_support": "peer_support_keywords",
}


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def score_to_number(label: str, score: float) -> float:
    label = label.lower()
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def summarise(sub: pd.DataFrame) -> dict:
    """Counts and rates for one phase (or the whole period)."""
    content = sub[sub["content_bearing"]]
    n = len(content)
    row = {
        "n_messages": len(sub),
        "n_content": n,
        "n_text": int((sub["content_kind"] == "text").sum()),
        "n_voice_translated": int((sub["content_kind"] == "voice_translation").sum()),
        "n_voice_untranslated": int((sub["content_kind"] == "voice_untranslated").sum()),
        "n_case_report": int((sub["content_kind"] == "case_report").sum()),
        "n_placeholder": int((sub["content_kind"] == "placeholder").sum()),
        "n_system": int((sub["content_kind"] == "system").sum()),
        "sentiment_mean": content["sentiment_score"].mean(),
        "sentiment_n": int(content["sentiment_score"].notna().sum()),
    }
    for name in LEXICONS:
        k = int(content[name].sum())
        row[f"{name}_n"] = k
        row[f"{name}_rate"] = k / n if n else np.nan
    return row


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Voice-note English translations, joined by media filename
    df = join_voice_translations(df, settings["paths"].get("voice_notes_output"))
    n_vn = int((df["message_type"] == "voice_note").sum())
    n_matched = int(df["translation_en"].notna().sum())
    print(f"[05_sentiment] Voice notes joined to a translation: {n_matched} of {n_vn}")

    df["content_kind"] = content_kind(df)
    df["content_bearing"] = df["content_kind"].isin(CONTENT_KINDS)
    text_col = content_bearing_text(df, df["content_kind"])

    clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
    scores = []
    for txt, keep in zip(text_col, df["content_bearing"]):
        if not keep or not txt.strip():
            scores.append(np.nan)
            continue
        try:
            res = clf(txt[:512])[0]
            scores.append(score_to_number(res["label"], res["score"]))
        except Exception:
            scores.append(np.nan)
    df["sentiment_score"] = scores

    for name, key in LEXICONS.items():
        df[name] = text_col.apply(lambda t, kws=settings[key]: match_any(t, kws))

    daily = df.set_index("timestamp")["sentiment_score"].resample("D").mean()
    roll7 = daily.rolling(7, min_periods=1).mean().rename("sentiment_7d")
    roll14 = daily.rolling(14, min_periods=1).mean().rename("sentiment_14d")
    rolling = pd.concat([daily.rename("sentiment_daily"), roll7, roll14], axis=1).reset_index()
    rolling = rolling.assign(kind="rolling_sentiment")

    phase_summary = pd.DataFrame(
        [{"phase": phase, **summarise(sub)} for phase, sub in df.groupby("phase")]
    ).assign(kind="phase_summary")
    overall = pd.DataFrame([{"phase": "Overall", **summarise(df)}]).assign(kind="overall_summary")

    peer_counts = df[df["peer_support"]].groupby("sender_code").size().values
    peer_gini = pd.DataFrame(
        [{"gini_peer_support": gini(peer_counts), "kind": "peer_support_gini"}]
    )

    combined = pd.concat([rolling, phase_summary, overall, peer_gini],
                         ignore_index=True, sort=False)
    out = Path(settings["paths"]["sentiment_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"[05_sentiment] Wrote {out}")


if __name__ == "__main__":
    main()
