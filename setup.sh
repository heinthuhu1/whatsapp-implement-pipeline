#!/usr/bin/env bash
# One-time setup script. Run this once before using the pipeline.
set -euo pipefail

echo "==> Checking Python 3.9+"
python3 --version || { echo "ERROR: python3 not found. Install from https://www.python.org/downloads/"; exit 1; }

echo "==> Checking ffmpeg (needed for voice note conversion)"
if ! command -v ffmpeg &>/dev/null; then
  echo ""
  echo "  ffmpeg not found. Install it first:"
  echo "    Mac:   brew install ffmpeg"
  echo "    Linux: sudo apt install ffmpeg"
  echo "    Windows: https://ffmpeg.org/download.html"
  echo ""
  exit 1
fi
echo "    ffmpeg OK: $(ffmpeg -version 2>&1 | head -1)"

echo "==> Creating Python virtual environment"
python3 -m venv venv
source venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "    Dependencies installed."

echo "==> Creating data directories"
mkdir -p data/raw/voice_notes
mkdir -p data/interim
mkdir -p data/processed

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Copy your WhatsApp export into:   data/raw/whatsapp_export.txt"
echo "  2. Copy your .opus voice notes into: data/raw/voice_notes/  (optional)"
echo "  3. Edit config/settings.yaml:        set your phases, milestones, and keywords"
echo "  4. Set your OpenAI API key:          export OPENAI_API_KEY=sk-...  (optional, for voice notes)"
echo "  5. Run the pipeline:                 bash run_pipeline.sh"
echo ""
