# Keyword lexicons

These are the complete keyword lexicons used in the demonstration analysis, copied verbatim from the configuration file used for that run (`settings.yaml`, keys shown below). The same lexicons are reproduced in `synthetic_settings.yaml` in this folder. All terms are in English. No Urdu-script or Roman Urdu terms were configured, so text messages written in Urdu (in either script) can only be flagged through English words embedded in them; voice notes can be flagged only if their English translation reaches the scoring stages (see "Known limitation" below).

## Matching rule (applies to every lexicon)

Implemented by `match_any()` in `src/05_sentiment.py` and `src/06_fidelity.py`:

* the message text and each term are lower-cased;
* a message is flagged if **any** term occurs as a **substring** anywhere in the text (no word boundaries, no stemming, no negation handling);
* multi-word terms (for example `not sure`) must occur as that exact character sequence, with a single space;
* the text matched is the English voice-note translation where one has been joined to the message, otherwise the message body as exported.

Because matching is by substring, short terms also match inside longer words. For example `now` matches "know" and "snow", `call` matches "recall", `kit` matches "kitchen", `help` matches "helpful", `monitor` matches "monitoring", and `review` matches "preview". Users adapting the lexicons should check high-frequency short terms against a sample of flagged messages.

## Lexicons

| Lexicon | Config key | Used in | Terms (n) | Language | Terms |
|---|---|---|---|---|---|
| Urgency | `urgency_keywords` | `src/05_sentiment.py` (urgency rate); `src/06_fidelity.py` (urgency-response latency) | 7 | English | urgent; emergency; immediately; asap; critical; help; now |
| Hedging | `hedging_keywords` | `src/05_sentiment.py` | 6 | English | maybe; perhaps; might; possibly; not sure; i think |
| Peer support | `peer_support_keywords` | `src/05_sentiment.py` (peer-support rate and Gini) | 7 | English | well done; good job; thank you; proud; appreciate; thanks; great work |
| Workaround | `workaround_keywords` | `src/06_fidelity.py` | 6 | English | workaround; instead; bypass; not working; broken; alternative |
| ToC: training | `toc_component_keywords.training` | `src/06_fidelity.py` | 5 | English | training; workshop; refresher; drill; practice |
| ToC: coordination | `toc_component_keywords.coordination` | `src/06_fidelity.py` | 5 | English | dispatch; call; rescue; pickup; respond |
| ToC: clinical protocol | `toc_component_keywords.clinical_protocol` | `src/06_fidelity.py` | 5 | English | protocol; triage; vitals; sop; checklist |
| ToC: equipment | `toc_component_keywords.equipment` | `src/06_fidelity.py` | 5 | English | equipment; kit; stretcher; oxygen; defibrillator |
| ToC: supervision | `toc_component_keywords.supervision` | `src/06_fidelity.py` | 5 | English | supervisor; incharge; review; monitor; oversight |

Total: 51 terms across 9 lexicons.

Notes:

* `peer_support_keywords` contains both `thank you` and `thanks`; because matching is by substring these overlap only partly ("thank you" does not contain "thanks").
* `i think` is matched after lower-casing, so it also matches "I think".
* The ToC component names are free-form: each top-level key under `toc_component_keywords` becomes one component in the fidelity output, so components can be renamed, added or removed in the configuration without code changes.

## Anonymisation word lists

Two further lists in the configuration are used by `src/00_anonymise.py`, not by the content analyses. Both were empty in the demonstration run.

| Config key | Effect |
|---|---|
| `name_aliases` | Each entry is replaced by `[REDACTED]` wherever it appears as a whole word (case-insensitive). |
| `flag_names` | Each entry found as a whole word is written to the anonymisation warnings log for manual review; it is **not** redacted. |

`synthetic_settings.yaml` sets `flag_names: [Sparrow]` (a fictional non-member mentioned in one synthetic message) to demonstrate the warning log. Sender names are replaced by codes before this check, so `flag_names` is only useful for names of people who are not group members.

## Known limitation affecting all lexicon rates

`src/00_anonymise.py` replaces any run of seven or more digits with `[NUMBER]`. WhatsApp voice-note file names contain an eight-digit date (for example `PTT-20241210-WA0037.opus`), so the date is removed from every voice-note reference before parsing. `src/01_parse.py` then stores only the tail of the file name (`-WA0037.opus`), and the join on file name in `src/05_sentiment.py` matches no translations. As released, therefore, voice notes enter the lexicon and sentiment stages as their placeholder text (`PTT-[NUMBER]-WA0037.opus (file attached)`), which contains no lexicon terms. The synthetic worked example reproduces this behaviour. See `PROCESSING_RULES.md` (rule A6) for the details and the recommended fix.
