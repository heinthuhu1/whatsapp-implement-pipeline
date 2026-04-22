# Maseeha WhatsApp Pipeline

A research pipeline that turns a raw WhatsApp group-chat export (plus voice-note `.opus` files) into structured metrics for implementation science: engagement, network analysis, sentiment, and fidelity to a Theory of Change.

Built for the Maseeha emergency ambulance group in Pind Begwal, Pakistan.

---

## What you need before you start

| Requirement | How to get it |
|---|---|
| Python 3.9 or later | https://www.python.org/downloads/ |
| ffmpeg | Mac: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg` · Windows: https://ffmpeg.org/download.html |
| WhatsApp chat export (`.txt`) | In WhatsApp: open the group → ⋮ menu → More → Export chat → Without media |
| Voice note files (`.opus`) | Export with media, then collect the `PTT-*.opus` / `AUD-*.opus` files |
| OpenAI API key | https://platform.openai.com/api-keys — only needed for voice note transcription |

---

## Step-by-step setup (do this once)

**Step 1 — Download the code**
```bash
git clone https://github.com/heinthuhu1/maseeha-whatsapp-pipeline.git
cd maseeha-whatsapp-pipeline
```

**Step 2 — Run the setup script**
```bash
bash setup.sh
```
This installs all Python dependencies and creates the required folders.

**Step 3 — Copy your data files**

Copy your WhatsApp export file into the project:
```bash
cp /path/to/WhatsApp_Chat_with_MASEEHA_AMBULANCE.txt data/raw/
```

Copy your voice notes (optional — only needed for transcription):
```bash
cp /path/to/PTT-*.opus data/raw/voice_notes/
cp /path/to/AUD-*.opus data/raw/voice_notes/
```

**Step 4 — Set your OpenAI API key** *(only needed for voice notes)*
```bash
export OPENAI_API_KEY=sk-...your-key-here...
```

**Step 5 — Run the pipeline**
```bash
bash run_pipeline.sh
```

That's it. Results will appear in `data/processed/`.

---

## Output files

| File | What it contains |
|---|---|
| `data/interim/messages.csv` | Every message parsed and anonymised, one row per message |
| `data/interim/case_reports.csv` | Structured ambulance dispatch records (case no., complaint, location, hospital) |
| `data/interim/voice_notes_transcribed.csv` | Urdu transcriptions + English translations of voice notes |
| `data/processed/engagement_metrics.csv` | Weekly message volume, spikes, after-hours activity, response latency |
| `data/processed/network_metrics.csv` | Per-person centrality scores and co-activity measures, by year |
| `data/processed/network_edges.graphml` | Full network graph (open in Gephi or Cytoscape) |
| `data/processed/sentiment_metrics.csv` | Daily sentiment, urgency/hedging/peer-support rates by year |
| `data/processed/fidelity_metrics.csv` | Theory of Change component coverage by year |
| `data/processed/implementation_summary.csv` | All metrics joined in one summary table for your paper |

---

## Folder structure

```
.
├── data/
│   ├── raw/               # Your WhatsApp export + voice notes (never uploaded to GitHub)
│   ├── interim/           # Intermediate files (never uploaded to GitHub)
│   └── processed/         # Final metric CSVs (never uploaded to GitHub)
├── config/
│   ├── settings.yaml      # Phase dates, keyword lists, file paths
│   ├── sender_lookup.csv  # Real names → coded IDs (never uploaded to GitHub)
│   └── role_mapping.csv   # Coded IDs → roles/sites (you create this)
├── src/                   # Pipeline stages 00–07
├── tests/                 # Automated tests
├── notebooks/             # Analysis notebooks
├── setup.sh               # One-time setup
└── run_pipeline.sh        # Run the full pipeline
```

> **Privacy:** all files under `data/` and `config/sender_lookup.csv` are gitignored — raw chat data and participant identifiers never leave your machine.

---

## Configuration

Open `config/settings.yaml` to customise:

- **`phases`** — date boundaries for your study periods
- **`milestones`** — key programme dates (training start, launch, reviews)
- **`urgency_keywords`** / **`hedging_keywords`** etc. — add Urdu or domain-specific terms

### Role mapping (optional but recommended)

Create `config/role_mapping.csv` to assign roles and sites to each participant:

```csv
sender_code,role,site
P001,paramedic,pind_begwal
P002,supervisor,pind_begwal
P003,paramedic,rawalpindi
```

Check `config/sender_lookup.csv` to see which coded ID corresponds to which person.

---

## Running individual stages

```bash
source venv/bin/activate
python3 src/00_anonymise.py   # anonymise names and phone numbers
python3 src/01_parse.py       # parse messages into CSV
python3 src/02_voice_notes.py # transcribe voice notes (needs OPENAI_API_KEY)
python3 src/03_engagement.py  # engagement metrics
python3 src/04_network.py     # network analysis
python3 src/05_sentiment.py   # sentiment analysis
python3 src/06_fidelity.py    # fidelity metrics
python3 src/07_triangulation.py # combine everything
```

## Running tests

```bash
source venv/bin/activate
pytest tests/ -v
```
