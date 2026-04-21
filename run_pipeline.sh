#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it before running the pipeline." >&2
  exit 1
fi

echo "==> [1/7] Parsing WhatsApp export"
python src/01_parse.py

echo "==> [2/7] Transcribing voice notes"
python src/02_voice_notes.py

echo "==> [3/7] Engagement metrics"
python src/03_engagement.py

echo "==> [4/7] Network metrics"
python src/04_network.py

echo "==> [5/7] Sentiment metrics"
python src/05_sentiment.py

echo "==> [6/7] Fidelity metrics"
python src/06_fidelity.py

echo "==> [7/7] Triangulation"
python src/07_triangulation.py

echo "Pipeline complete."
