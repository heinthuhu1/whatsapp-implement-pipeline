# WhatsApp Implementation Science Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A general-purpose research pipeline for public health implementers. Turns a WhatsApp group-chat export into structured metrics for implementation science: engagement, social network analysis, sentiment, and fidelity to a Theory of Change (ToC).

Originally developed for a Pakistani emergency ambulance programme. Designed to work with **any** WhatsApp group in **any** language.

---

## What this pipeline produces

| Output | What it tells you |
|---|---|
| **Engagement metrics** | Who is active, when, how often — weekly trends, spikes, after-hours activity, response latency |
| **Network analysis** | Who communicates with whom — centrality, information brokers, network density over time |
| **Sentiment analysis** | Urgency, hedging, peer support rates — are people communicating with confidence? Is the group task-focused or supportive? |
| **Fidelity metrics** | Are your Theory of Change components being discussed? Which components are strong or weak? |
| **Implementation summary** | All metrics in one table, aligned to your programme milestones |
| **Case reports** | If your group logs structured incident/case data, these are automatically extracted |

---

## What you need

| Requirement | How to get it |
|---|---|
| Python 3.9 or later | https://www.python.org/downloads/ |
| ffmpeg | Mac: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg` · Windows: https://ffmpeg.org/download.html |
| WhatsApp chat export (`.txt`) | In WhatsApp: open the group → ⋮ menu → More → Export chat → **Without media** |
| Voice note files (`.opus`) *(optional)* | Export **with media**, then collect the `PTT-*.opus` / `AUD-*.opus` files |
| OpenAI API key *(optional)* | https://platform.openai.com/api-keys — only needed to transcribe voice notes |

---

## Quickstart (5 steps)

**Step 1 — Download the pipeline**
```bash
git clone https://github.com/heinthuhu1/maseeha-whatsapp-pipeline.git
cd maseeha-whatsapp-pipeline
```

**Step 2 — Run setup**
```bash
bash setup.sh
```
This checks your system, installs all Python dependencies, and creates the required folders.

**Step 3 — Copy your data**
```bash
# Rename your WhatsApp export to this filename and copy it in:
cp /path/to/your_export.txt data/raw/whatsapp_export.txt

# Optional: copy voice notes
cp /path/to/PTT-*.opus data/raw/voice_notes/
cp /path/to/AUD-*.opus data/raw/voice_notes/
```

**Step 4 — Configure for your study**

Open `config/settings.yaml` and set:
- Your **phase dates** (baseline, implementation, sustainment — or whatever your study design uses)
- Your **milestone dates** (training start, launch, reviews)
- Your **Theory of Change keywords** — terms in any language that signal each component of your intervention

**Step 5 — Run**
```bash
# Optional: set your OpenAI key if you have voice notes
export OPENAI_API_KEY=sk-...

bash run_pipeline.sh
```

Results appear in `data/processed/`.

---

## Configuration guide

All configuration is in `config/settings.yaml`. The file is fully commented — open it and follow the numbered steps inside.

### Phases
Define as many phases as your study has. Names can be anything:
```yaml
phases:
  - name: baseline
    start: "2023-01-01"
    end: "2023-06-30"
  - name: implementation
    start: "2023-07-01"
    end: "2024-06-30"
```

### Theory of Change keywords
Rename components and add keywords in your group's language:
```yaml
toc_component_keywords:
  community_health_education:
    - education session
    - health talk
    - awareness
  referral_pathway:
    - referral
    - referred
    - sent to clinic
```

### Role mapping *(optional but recommended)*
Create `config/role_mapping.csv` to assign roles and sites to participants.
Check `config/sender_lookup.csv` (generated after first run) to see which coded ID is which person, then fill in roles:
```csv
sender_code,role,site
P001,community health worker,district_a
P002,supervisor,district_a
P003,nurse,district_b
```

---

## Output files

| File | What it contains |
|---|---|
| `data/interim/messages.csv` | Every message: timestamp, anonymised sender, text, type, phase |
| `data/interim/case_reports.csv` | Structured records if your group logs cases/incidents |
| `data/interim/voice_notes_transcribed.csv` | Transcriptions and translations of voice notes |
| `data/processed/engagement_metrics.csv` | Weekly volume, spikes, latency, after-hours rates |
| `data/processed/network_metrics.csv` | Per-person centrality scores by phase |
| `data/processed/network_edges.graphml` | Full network graph (open in Gephi or Cytoscape) |
| `data/processed/sentiment_metrics.csv` | Sentiment, urgency, hedging, peer support by phase |
| `data/processed/fidelity_metrics.csv` | ToC component coverage and responsiveness |
| `data/processed/implementation_summary.csv` | All metrics joined — ready for your paper |

---

## Privacy and data security

- All files under `data/` are **gitignored** — raw chat data never leaves your machine
- `config/sender_lookup.csv` (real names ↔ coded IDs) is gitignored
- The anonymisation stage (`00_anonymise.py`) replaces all sender names with stable coded IDs (P001, P002…) and redacts phone numbers and @mentions before any analysis runs
- Only code and configuration are published to GitHub — no participant data

---

## Folder structure

```
.
├── data/
│   ├── raw/               # Your WhatsApp export + voice notes (gitignored)
│   ├── interim/           # Anonymised export, parsed messages (gitignored)
│   └── processed/         # Final metric CSVs (gitignored)
├── config/
│   ├── settings.yaml      # All configuration — start here
│   ├── sender_lookup.csv  # Auto-generated: real names → coded IDs (gitignored)
│   └── role_mapping.csv   # You create: coded IDs → roles/sites (gitignored)
├── src/
│   ├── 00_anonymise.py    # Replace names, redact phones and mentions
│   ├── 01_parse.py        # Parse messages into structured CSV
│   ├── 02_voice_notes.py  # Transcribe and translate voice notes
│   ├── 03_engagement.py   # Weekly engagement and activity metrics
│   ├── 04_network.py      # Temporal co-activity network analysis
│   ├── 05_sentiment.py    # Sentiment, urgency, hedging, peer support
│   ├── 06_fidelity.py     # Theory of Change fidelity scoring
│   └── 07_triangulation.py# Combine all metrics into summary table
├── tests/                 # Automated tests (pytest)
├── notebooks/             # Analysis notebooks
├── setup.sh               # One-time setup
├── run_pipeline.sh        # Run the full pipeline
└── requirements.txt
```

---

## Running individual stages

```bash
source venv/bin/activate
python3 src/00_anonymise.py    # anonymise names and phone numbers
python3 src/01_parse.py        # parse messages into CSV
python3 src/02_voice_notes.py  # transcribe voice notes (needs OPENAI_API_KEY)
python3 src/03_engagement.py   # engagement metrics
python3 src/04_network.py      # network analysis
python3 src/05_sentiment.py    # sentiment analysis
python3 src/06_fidelity.py     # fidelity metrics
python3 src/07_triangulation.py # combine everything
```

## Running tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Citation

If you use this pipeline in your research, please cite:

> WhatsApp Implementation Science Pipeline (2026). https://github.com/heinthuhu1/maseeha-whatsapp-pipeline
