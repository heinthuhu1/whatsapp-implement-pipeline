# Processing rules and parameters

This document lists every parameter and processing rule applied by the pipeline, with the file and function where each is implemented. It describes the code in `src/` at commit `abca565` (7 May 2026), the version used for the published demonstration run, as read for this revision. Where the code's behaviour differs from what a reader might expect, this is stated under "Note". Configurable values are set in `config/settings.yaml` (key given in the "Config key" column); all other values are fixed in code.

Stages run in this order (`run_pipeline.sh`): 00 anonymise, 01 parse, 02 voice notes (optional), 03 engagement, 04 network, 05 sentiment, 06 fidelity, 07 triangulation, 08 figures, 09 Table 1. Every stage reads `config/settings.yaml` and resolves all paths relative to the working directory.

## A. Anonymisation (`src/00_anonymise.py`)

| # | Rule | Value | Config key | Where |
|---|---|---|---|---|
| A1 | Message-start detection | Regex `^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)[\s ]?(?:[AaPp][Mm])?\]?[\s ]?-\s` (bracketed or unbracketed date, optional seconds, optional am/pm, optional narrow no-break space) | none | `TIMESTAMP_RE` |
| A2 | Sender coding | Every distinct display name before the first `": "` on a message-start line receives a stable code `P001`, `P002`, ... in order of first appearance. Codes are stored in `config/sender_lookup.csv` and reused on later runs. | `paths.sender_lookup` | `extract_sender`, `next_code`, `anonymise_text`, `save_lookup` |
| A3 | Pakistani mobile numbers | Regex `(?:\+?92[\s\-]?3\d{2}[\s\-]?\d{7}\|03\d{2}[\s\-]?\d{7}\|03\d{9})` replaced by `[PHONE]` | none | `PK_MOBILE_RE`, `redact_phones` |
| A4 | Any other long number | Any run of 7 or more digits replaced by `[NUMBER]` (applied after A3) | none | `GENERIC_NUMBER_RE`, `redact_phones` |
| A5 | @-mentions | `@` followed by text in Unicode isolate marks (U+2068 to U+2069) replaced by `@[MENTION]` | none | `MENTION_RE`, `redact_mentions` |
| A6 | Name tokens | For each looked-up display name: the full name (non-ASCII characters replaced by spaces) and each word of 3 or more characters not in the stop list (`emp, osc, alt, alternate, whatsapp, dr, mam, the, and, mr, mrs, ms`) is replaced, case-insensitively at word boundaries, by the sender's code; a token shared by more than one sender becomes `[REDACTED]`. Longest tokens are replaced first. Names that look like phone numbers are skipped. Applied to message bodies, continuation lines and system notices. | none | `NAME_STOP_TOKENS`, `build_name_patterns`, `redact_names` |
| A7 | Extra aliases | Each `name_aliases` entry is replaced by `[REDACTED]` (whole word, case-insensitive) | `name_aliases` | `build_name_patterns` |
| A8 | Review flags | Each `flag_names` entry found as a whole word is logged (line number and name) to the warnings log; not redacted | `flag_names`, `paths.anonymisation_warnings` | `flag_names` |
| A9 | Order of redaction within a line | phone numbers (A3, A4), then mentions (A5), then names (A6, A7) | none | `anonymise_text` |

Note on A4 and A6: voice-note, image and video file names in the export contain an eight-digit date (`PTT-20241210-WA0037.opus`). Rule A4 replaces this date with `[NUMBER]`, so the date is lost before parsing; rule A6 can further rewrite `[NUMBER]` if any sender's display name contains the word "number". Downstream, `src/01_parse.py` extracts only `-WA0037.opus` as the media file name and the voice-note join in `src/05_sentiment.py` (rule E2) matches nothing. Recommended fix (not applied here): exempt WhatsApp media file names, pattern `(PTT|AUD|IMG|VID|STK|DOC)-\d{8}-WA\d+`, from rule A4, or rebuild the file name from the message date and the `WA` sequence number when joining.

## B. Parsing (`src/01_parse.py`)

| # | Rule | Value | Config key | Where |
|---|---|---|---|---|
| B1 | Message-start detection | As A1, capturing the am/pm marker | none | `TIMESTAMP_RE` |
| B2 | Timestamp formats (tried in order) | With am/pm: `%d/%m/%y %I:%M %p`, `%d/%m/%Y %I:%M %p`, `%m/%d/%y %I:%M %p`, `%m/%d/%Y %I:%M %p`, `%d/%m/%y %I:%M:%S %p`, `%d/%m/%Y %I:%M:%S %p`. Without: `%d/%m/%y %H:%M`, `%d/%m/%Y %H:%M`, `%d/%m/%y %H:%M:%S`, `%d/%m/%Y %H:%M:%S`, `%m/%d/%y %H:%M`, `%m/%d/%Y %H:%M`. Fallback: `pandas.to_datetime(errors="coerce")`. Day-first formats are tried first, so an ambiguous date such as 03/04 is read as 3 April. Timestamps are kept as exported (device local time, no time-zone conversion). | none | `parse_timestamp` |
| B3 | Multi-line messages | A line that does not match B1 is appended (with a newline) to the previous message | none | `parse_whatsapp_export` |
| B4 | Sender and body | Text before the first `": "` after the timestamp is the sender code; the rest is the body | none | `parse_whatsapp_export` |
| B5 | System notices | A line is a system notice if the first word after the timestamp contains no colon and the line matches `added\|removed\|left\|joined\|changed the group\|created group` (case-insensitive) | none | `SYSTEM_PATTERNS`, `is_system_message` |
| B6 | Message type (first match wins) | voice_note: `PTT-.*?\.opus\|\.opus\|audio omitted`; image: `IMG-.*?\.(jpg\|jpeg\|png)\|image omitted`; video: `VID-.*?\.mp4\|video omitted`; document: `\.pdf\|\.docx?\|document omitted`; sticker: `sticker omitted`; otherwise text | none | `MEDIA_PATTERNS`, `classify_message` |
| B7 | Voice-note file name | First match of `([\w\-]+\.opus)` in the body | none | `parse_whatsapp_export` |
| B8 | Phase assignment | A message belongs to the first configured phase with `start <= timestamp < end + 1 day` (end date inclusive); otherwise phase is empty | `phases` | `assign_phase` |
| B9 | Roles and sites | If the role-mapping file exists it is merged on `sender_code`; otherwise `role` and `site` are empty | `paths.role_mapping` | `main` |

Note on B5 and B6: notices without a sender that do not match B5 (for example "Your security code with ... changed", the end-to-end encryption notice, "You created this group") are parsed as `text` with an empty sender. `<Media omitted>` and "This message was deleted" are also typed `text`. Stages 03, 05 and 06 exclude only rows typed `system`, so these rows enter their denominators.

## C. Case-report extraction (`src/01_parse.py`, `parse_case_reports`)

| # | Rule | Value |
|---|---|---|
| C1 | Candidate messages | Body matches `case\s*no\|case\s*\.?\s*\d+\|🚑` (case-insensitive) |
| C2 | Line clean-up | Leading left-to-right marks (U+200E) and whitespace stripped from each line |
| C3 | Case number | First line matching `^\*?[Cc]ase\s*[Nn]o\.?\s*[:\.]?\s*(\d+)\s*$` (for example "Case no 65") or `^\*?[Cc]ase\s*[:\.]?\s*(\d+)\s*$` (for example "Case 87", "Case .103"); otherwise the `case_no` field alias below |
| C4 | Field lines | `^([A-Za-z][A-Za-z\s/\(\)\.]{1,45})[:\.\;][ \t]*(.*)`: a label of up to 46 letters, spaces, slashes, brackets or full stops, then a colon, full stop or semicolon, then the value |
| C5 | Field aliases (label must match the whole regex, case-insensitive) | `case_no`: `case\s*no\.?`; `date`: `date`; `gender`: `gender`; `initial_complaint`: `initial\s*complaint\|complaint`; `location`: `location(\s*transported\s*to)?\|pickup\s*location`; `time_patient_reached`: `time\s*patient\s*reached\|pickup\s*time`; `hospital`: `name\s*of\s*hospital\s*transported\s*to\|hospital\s*transported\s*to\|transported\s*to\|destination`; `time_hospital_reached`: `time\s*(hospital\s*)?(was\s*)?reached\|arrival\s*time`; `first_aid`: `first\s*aid(\s*provided)?`; `emp_personnel`: `(emp\s*person(nel\|al)?\s*(names?)?\|emp\s*personnel)` (`CASE_FIELD_ALIASES`) |
| C6 | Repeated labels | The first value found for each field is kept; an empty value is stored as missing |
| C7 | Retention | A record is kept if it has a case number or a complaint |
| C8 | De-duplication | Records are grouped by the case-number string; within each group the record with the most non-missing values among the nine data fields (date, gender, complaint, location, both times, hospital, first aid, personnel) is kept; ties are broken by sort order. Case numbers are compared as strings, so "07" and "7" are distinct. |
| C9 | No further normalisation | Field values are stored as typed (no spelling, case, date or time normalisation) |

Note on C5: labels that differ from the alias list are not extracted even when a value is present (for example "Time patien:", or "Name of Hospital: X Transported to", where the value sits inside the label). Field completeness in the output therefore reflects both reporting practice and parser recall.

## D. Voice notes (`src/02_voice_notes.py`, optional, hosted service)

| # | Rule | Value | Config key | Where |
|---|---|---|---|---|
| D1 | Input files | All `*.opus` files in the voice-note folder; file names must match `(PTT\|AUD)-(\d{8})-WA(\d+)\.opus` (date and sequence parsed from the name) | `paths.voice_notes_dir` | `FILENAME_RE`, `parse_filename` |
| D2 | Audio conversion | `ffmpeg -i <file> -f ogg -acodec libopus pipe:1` (in memory) | none | `_to_ogg_bytes` |
| D3 | Transcription | OpenAI audio transcription endpoint, model `whisper-1`, optional `language` argument | `transcription_language` | `transcribe` |
| D4 | Translation | OpenAI audio translation endpoint (to English), model `whisper-1`, no language argument | none | `translate` |
| D5 | Duration | `mutagen.oggopus.OggOpus(...).info.length`, rounded to 0.01 s; missing if unreadable | none | `get_duration` |
| D6 | Throughput and resumption | 0.5 s pause after each API call; results appended every 50 files; files already marked `success` in the output are skipped; failures logged with stage and error | `paths.voice_notes_output`, `paths.transcription_failures` | `BATCH_SIZE`, `RATE_LIMIT_DELAY`, `load_existing`, `log_failure` |
| D7 | API key | Read from the `OPENAI_API_KEY` environment variable (`.env` is sourced by `run_pipeline.sh`); the stage is skipped if it is absent | none | `main`, `run_pipeline.sh` |

Note on D3: `main()` calls `process_voice_notes()` without passing the settings, so as written `transcription_language` is not forwarded and Whisper auto-detects the language. `join_to_chat()` is defined but not called by any stage.

## E. Sentiment and communication-style lexicons (`src/05_sentiment.py`)

| # | Rule | Value | Config key |
|---|---|---|---|
| E1 | Messages scored | All rows not typed `system` with a valid timestamp |  |
| E2 | Text scored | English voice-note translation joined on `media_filename == filename`, otherwise the message body (see the note on A4) | `paths.voice_notes_output` |
| E3 | Model | Hugging Face `transformers` pipeline `sentiment-analysis`, model and tokenizer `distilbert-base-uncased-finetuned-sst-2-english` (see `MODEL_VERSIONS.md`) |  |
| E4 | Input length | First 512 characters of the text (character truncation, not token truncation) |  |
| E5 | Score | +p if the predicted label is POSITIVE, -p if NEGATIVE, where p is the model's probability for the predicted label; empty text or a model error scores 0 |  |
| E6 | Lexicon flags | urgency, hedging, peer support: case-insensitive substring match (see `LEXICONS.md`) | `urgency_keywords`, `hedging_keywords`, `peer_support_keywords` |
| E7 | Daily series | Mean score per calendar day (days with no messages are missing) |  |
| E8 | Smoothing | 7-day and 14-day rolling means of the daily series (`min_periods=1`; missing days are skipped within the window) |  |
| E9 | Phase summary | Mean score and the proportion of messages flagged for each lexicon, per phase (denominator: all messages scored in E1) |  |
| E10 | Peer-support concentration | Gini coefficient of per-sender counts of peer-support-flagged messages (senders with none are excluded) |  |

## F. Fidelity and responsiveness (`src/06_fidelity.py`)

| # | Rule | Value | Config key | Where |
|---|---|---|---|---|
| F1 | Messages analysed | Rows not typed `system` with a valid timestamp; text = `translation_en` if that column exists in `messages.csv`, otherwise the message body | | `main` |
| F2 | Topical (ToC) coverage | For each phase and component: percentage of messages in the phase containing at least one of the component's terms (`coverage_pct`), with the count (`hits`) | `toc_component_keywords` | `coverage_by_phase` |
| F3 | Workaround flags | Messages containing a workaround term; each is attributed to every ToC component also mentioned in it, or to `unspecified` | `workaround_keywords` | `workarounds` |
| F4 | Urgency-response latency | For each urgency-flagged message: minutes until the next message from a different, non-empty sender (no upper limit, no working-hours restriction) | `urgency_keywords` | `responsiveness` |

Note on F1: `messages.csv` has no `translation_en` column, so voice notes enter as placeholder text. The fallback series is built with a fresh 0-based index and aligned to the filtered data frame by index label, which leaves the text of a handful of rows (six in the demonstration data) empty. Note on F2: coverage is a per-message proportion, although the colour-bar label in Figure 4 (`src/08_visualise.py`) describes it as a proportion of weeks.

## G. Engagement (`src/03_engagement.py`)

| # | Rule | Value | Config key | Where |
|---|---|---|---|---|
| G1 | Messages analysed | Rows not typed `system` with a valid timestamp | | `main` |
| G2 | Weekly volume | Message counts per calendar week (pandas `resample("W")`, weeks ending Sunday) | | `weekly_volume` |
| G3 | Velocity spike | A day is a spike if its count exceeds the rolling mean plus 2 standard deviations of daily counts over the trailing window that includes the day itself (`min_periods=1`; SD of a single day treated as 0) | `rolling_window_days` (7) | `velocity_spikes` |
| G4 | Silence period | A gap between consecutive messages longer than the threshold | `silence_threshold_hours` (48) | `silence_periods` |
| G5 | After-hours rate | Per sender, proportion of messages with clock hour >= start or < end (local export time) | `after_hours_start` (22), `after_hours_end` (6) | `after_hours_rate` |
| G6 | Response latency | For each message: minutes until the next message from any different, non-empty sender; mean per sender and phase (no censoring, no working-hours restriction) | | `response_latency` |
| G7 | Sender share | Messages per sender and percentage of all messages | | `sender_share` |

## H. Co-activity network (`src/04_network.py`)

| # | Rule | Value | Where |
|---|---|---|---|
| H1 | Messages analysed | Rows not typed `system`, with a timestamp and a sender | `main` |
| H2 | Co-activity window | Two senders are linked when one posts within plus or minus 30 minutes of a message by the other; for each message every distinct other sender in its window adds 1 to the pair; pair totals are halved (integer division) to remove double counting; self-pairs excluded | `WINDOW_MINUTES`, `build_coactivity_edges` |
| H3 | Full-period graph | Undirected weighted graph over all messages, written as GraphML | `main` |
| H4 | Time slices for node metrics | Calendar year of the timestamp (not the configured phases) | `split_by_year` |
| H5 | Node metrics | Degree centrality; betweenness centrality (normalised, on edge distance = 1/weight, so that frequently co-active pairs are close); eigenvector centrality (weighted, max 1,000 iterations, missing if not converged); message count; number of co-activity partners | `compute_node_metrics` |
| H6 | Concentration | Gini coefficient of per-sender message counts within each slice | `gini` |

Note on H4: Figure 2 (`src/08_visualise.py`, `fig_network`) slices by configured phase dates instead. With calendar-year phases, as in the demonstration, the two coincide.

## I. Triangulation, figures and Table 1

| # | Rule | Where |
|---|---|---|
| I1 | Phase labels normalised to strings ("2024.0" to "2024"); every numeric column of each domain output averaged by phase and merged into one row per phase; configured milestones appended as columns | `src/07_triangulation.py`, `summarize_by_phase` |
| I2 | Figures 1 to 5 written to `data/processed/figures/` (paths fixed in code): weekly volume with a 4-week centred rolling mean; per-phase co-activity networks (spring layout, seed 42); sentiment (7- and 14-day rolling) and urgency and peer-support rates; ToC coverage heat map; case-report profile | `src/08_visualise.py` |
| I3 | Table 1: corpus structure and domain metrics, overall and by phase, written as CSV and fixed-width text. Voice notes dated outside the configured phases are excluded from voice-note rows. The row "Text messages and system notices (n)" counts every parsed row, including voice notes and media. | `src/09_table1.py` |

## J. Phases used in the demonstration

| Phase | Start | End (inclusive) |
|---|---|---|
| 2024 | 2024-09-22 | 2024-12-31 |
| 2025 | 2025-01-01 | 2025-12-31 |
| 2026 | 2026-01-01 | 2026-04-11 |

`milestones` was empty. The synthetic worked example uses phases 2024 (2024-11-04 to 2024-12-31) and 2025 (2025-01-01 to 2025-02-28).
