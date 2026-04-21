# Maseeha WhatsApp Pipeline

A research pipeline for analysing a Pakistani emergency-ambulance WhatsApp group chat. Turns a raw WhatsApp Android `.txt` export (plus voice-note `.opus` files) into structured engagement, network, sentiment, and fidelity metrics for implementation-science triangulation.

## Folder structure

```
.
├── data/
│   ├── raw/          # WhatsApp export + .opus voice notes (gitignored)
│   ├── interim/      # messages.csv + transcription logs (gitignored)
│   └── processed/    # final metric CSVs + graphml (gitignored)
├── config/
│   ├── settings.yaml       # paths, phases, keyword lists
│   └── role_mapping.csv    # sender_code → role, site (gitignored)
├── src/              # numbered pipeline stages 01–07
├── notebooks/        # exploratory analysis
├── tests/            # pytest unit tests
├── run_pipeline.sh   # runs all stages in sequence
└── requirements.txt
```

All `data/` folders and `config/role_mapping.csv` are gitignored — raw chat data and participant identifiers never leave the researcher's machine.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

1. Place the raw WhatsApp export at `data/raw/whatsapp_export.txt` and the `.opus` voice-note files alongside it.
2. Create `config/role_mapping.csv` with columns `sender_code,role,site` — one row per participant phone-name that appears in the chat.
3. Edit `config/settings.yaml`:
   - Update `phases` with your study's phase date boundaries.
   - Update `milestones` with key programme dates.
   - Tune `urgency_keywords`, `hedging_keywords`, `toc_component_keywords`, `workaround_keywords`, `peer_support_keywords` for your context (English + Urdu romanised are already seeded).

## OPENAI_API_KEY

Stage 02 calls the OpenAI Whisper API for voice-note transcription and translation and falls back to `gpt-4o` for code-switched content. Export your key in the shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
bash run_pipeline.sh
```

Or run individual stages:

```bash
python src/01_parse.py
python src/02_voice_notes.py
python src/03_engagement.py
python src/04_network.py
python src/05_sentiment.py
python src/06_fidelity.py
python src/07_triangulation.py
```

## Tests

```bash
pytest tests/
```
