"""Sentiment, urgency/hedging keyword matching, emotional labour Gini."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def match_any(text: str, keywords: list) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def score_to_number(label: str, score: float) -> float:
    label = label.lower()
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def main() -> None:
    settings = load_settings()
    df = pd.read_csv(settings["paths"]["messages_interim"], parse_dates=["timestamp"])
    df = df[df["message_type"] != "system"].dropna(subset=["timestamp"]).copy()

    text_col = df["translation_en"].fillna(df["message"]).fillna("").astype(str)

    clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
    scores = []
    for txt in text_col:
        if not txt.strip():
            scores.append(0.0)
            continue
        try:
            res = clf(txt[:512])[0]
            scores.append(score_to_number(res["label"], res["score"]))
        except Exception:
            scores.append(0.0)
    df["sentiment_score"] = scores

    df["urgency"] = text_col.apply(lambda t: match_any(t, settings["urgency_keywords"]))
    df["hedging"] = text_col.apply(lambda t: match_any(t, settings["hedging_keywords"]))
    df["peer_support"] = text_col.apply(
        lambda t: match_any(t, settings["peer_support_keywords"])
    )

    daily = df.set_index("timestamp")["sentiment_score"].resample("D").mean()
    roll7 = daily.rolling(7, min_periods=1).mean().rename("sentiment_7d")
    roll14 = daily.rolling(14, min_periods=1).mean().rename("sentiment_14d")
    rolling = pd.concat([daily.rename("sentiment_daily"), roll7, roll14], axis=1).reset_index()
    rolling = rolling.assign(kind="rolling_sentiment")

    phase_summary = (
        df.groupby("phase")
          .agg(sentiment_mean=("sentiment_score", "mean"),
               urgency_rate=("urgency", "mean"),
               hedging_rate=("hedging", "mean"),
               peer_support_rate=("peer_support", "mean"))
          .reset_index().assign(kind="phase_summary")
    )

    peer_counts = df[df["peer_support"]].groupby("sender_code").size().values
    peer_gini = pd.DataFrame(
        [{"gini_peer_support": gini(peer_counts), "kind": "peer_support_gini"}]
    )

    combined = pd.concat([rolling, phase_summary, peer_gini], ignore_index=True, sort=False)
    out = Path(settings["paths"]["sentiment_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"[05_sentiment] Wrote {out}")


if __name__ == "__main__":
    main()
