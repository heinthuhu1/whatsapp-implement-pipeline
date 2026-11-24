# Model and software versions

Recorded on 11 November 2026 from the project's virtual environment (`venv/`) by reading the code and running `venv/bin/pip freeze` and `venv/bin/python --version`. No external API was called to compile this file. Items that cannot be determined from the code or the environment are marked **TO CONFIRM**.

## Pipeline code

| Item | Value |
|---|---|
| Repository commit used for the demonstration run | `abca565` (`abca5652802e0a50b23df5c3e9f5400f888ab271`, 7 May 2026). The archived output files carry the same date. **TO CONFIRM** that no uncommitted changes were present when the run was made. |
| Pipeline stages | `src/00_anonymise.py` to `src/09_table1.py`, run in order by `run_pipeline.sh` |

## Sentiment model (`src/05_sentiment.py`)

| Item | Value |
|---|---|
| Model identifier in code | `distilbert-base-uncased-finetuned-sst-2-english` (constant `MODEL_NAME`; the same identifier is used for the tokenizer). On the Hugging Face Hub this legacy identifier resolves to `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. |
| Revision | The code does not pin a revision, so the Hub's `main` branch is used at download time. The locally cached snapshot is commit `714eb0fa89d2f80546fda750413ed43d93601a13`. Re-scoring the archived demonstration messages with this snapshot reproduced the archived daily sentiment series exactly (maximum absolute difference 1.1e-16 across 418 days), so this is the revision that produced the published scores. To pin it, pass `revision="714eb0fa89d2f80546fda750413ed43d93601a13"` to `pipeline()`. |
| Architecture | DistilBERT (6 layers, 12 heads, hidden size 768, vocabulary 30,522, maximum 512 positions), fine-tuned on SST-2; labels `0 = NEGATIVE`, `1 = POSITIVE` |
| Task call | `transformers.pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)`, default device (CPU unless an accelerator is selected by `transformers`), one message per call, input truncated to the first 512 characters |
| Score mapping | +p for POSITIVE, -p for NEGATIVE (p = probability of the predicted label); 0 for empty text or errors |

## Speech-to-text and translation (`src/02_voice_notes.py`)

| Item | Value |
|---|---|
| Service | OpenAI hosted API through the `openai` Python client (`OpenAI(api_key=...)`) |
| Transcription endpoint | `client.audio.transcriptions.create(model="whisper-1", file=..., language=...)` |
| Translation endpoint | `client.audio.translations.create(model="whisper-1", file=...)` (Whisper translation into English) |
| Model | `whisper-1`. OpenAI does not expose an underlying model version or snapshot date for `whisper-1`. The demonstration audio was processed in April 2026 (the transcription stage was first run on 22 April 2026, per the repository history; later runs skip voice notes already transcribed). |
| Language setting | `transcription_language: "ur"` in the configuration. As written, `main()` does not pass the settings to `process_voice_notes()`, so the language argument is not sent and Whisper auto-detects the language. Whether the archived transcripts were produced with or without `language="ur"` is **TO CONFIRM**. |
| Audio pre-processing | `ffmpeg -i <file> -f ogg -acodec libopus pipe:1`; ffmpeg 8.1 is installed on the analysis machine now; the version used for the demonstration run is **TO CONFIRM** |
| Audio duration | `mutagen` `OggOpus` reader |
| Client library | `openai` 2.32.0 (installed version; see below) |

Stage 02 is optional. A local Whisper model can be substituted by replacing `transcribe()` and `translate()`; no other stage depends on the hosted service.

## Runtime

| Item | Value |
|---|---|
| Python | 3.12.4 (CPython, macOS arm64 build) |
| Operating system of the analysis machine | macOS 26.4, Apple silicon (arm64) |
| Key packages | pandas 3.0.2; numpy 2.4.4; networkx 3.6.1; transformers 5.5.4; tokenizers 0.22.2; torch 2.11.0; huggingface_hub 1.11.0; safetensors 0.7.0; matplotlib 3.10.8; scipy 1.17.1; PyYAML 6.0.3; openai 2.32.0; mutagen 1.47.0; tqdm 4.67.3 |
| Evidence that these are the run versions | The package metadata for pandas, torch and transformers was installed on 22 April 2026, before the demonstration run (7 May 2026), and the sentiment scores reproduce exactly. **TO CONFIRM** that the environment was not changed between those dates. |

`requirements.txt` gives only lower bounds (for example `transformers>=4.40`, `torch>=2.6`, `pandas>=2.0`). To reproduce exactly, install the pinned list below.

## Full `pip freeze` of the analysis environment

```
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
certifi==2026.2.25
click==8.3.2
contourpy==1.3.3
cycler==0.12.1
defusedxml==0.7.1
distro==1.9.0
et_xmlfile==2.0.0
filelock==3.29.0
fonttools==4.62.1
formulaic==1.2.2
fsspec==2026.3.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.11.0
idna==3.12
iniconfig==2.3.0
interface_meta==2.0.1
Jinja2==3.1.6
jiter==0.14.0
kiwisolver==1.5.0
lxml==6.1.1
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mpmath==1.3.0
mutagen==1.47.0
narwhals==2.28.0
networkx==3.6.1
numpy==2.4.4
openai==2.32.0
openpyxl==3.1.5
packaging==26.1
pandas==3.0.2
patsy==1.0.3
pillow==12.2.0
pluggy==1.6.0
pydantic==2.13.3
pydantic_core==2.46.3
Pygments==2.20.0
pyparsing==3.3.2
pytest==9.0.3
python-dateutil==2.9.0.post0
python-docx==1.2.0
PyYAML==6.0.3
regex==2026.4.4
rich==15.0.0
safetensors==0.7.0
scipy==1.17.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
statsmodels==0.15.0
sympy==1.14.0
tokenizers==0.22.2
torch==2.11.0
tqdm==4.67.3
transformers==5.5.4
typer==0.24.1
typing-inspection==0.4.2
typing_extensions==4.15.0
wrapt==2.5.2
```
