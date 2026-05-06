#!/usr/bin/env bash
# Run all pipeline stages in order.
# Usage: bash run_pipeline.sh
set -euo pipefail

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Check for OpenAI key — only needed for stage 02 (voice notes)
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set."
  echo "         Stage 02 (voice note transcription) will be skipped."
  SKIP_VOICE_NOTES=1
else
  SKIP_VOICE_NOTES=0
fi

echo ""
echo "========================================"
echo " WhatsApp Implementation Science Pipeline"
echo "========================================"
echo ""

echo "==> [00/07] Anonymising raw export"
python3 src/00_anonymise.py

echo "==> [01/07] Parsing messages"
python3 src/01_parse.py

if [ "$SKIP_VOICE_NOTES" -eq 0 ]; then
  echo "==> [02/07] Transcribing voice notes"
  python3 src/02_voice_notes.py
else
  echo "==> [02/07] Skipping voice notes (no OPENAI_API_KEY)"
fi

echo "==> [03/07] Engagement metrics"
python3 src/03_engagement.py

echo "==> [04/07] Network metrics"
python3 src/04_network.py

echo "==> [05/07] Sentiment analysis"
python3 src/05_sentiment.py

echo "==> [06/07] Fidelity metrics"
python3 src/06_fidelity.py

echo "==> [07/07] Triangulation"
python3 src/07_triangulation.py

echo "==> [08/09] Visualisations"
python3 src/08_visualise.py

echo "==> [09/09] Table 1 overview"
python3 src/09_table1.py

echo ""
echo "========================================"
echo " Pipeline complete!"
echo " Outputs are in data/processed/"
echo " Figures are in data/processed/figures/"
echo "========================================"
