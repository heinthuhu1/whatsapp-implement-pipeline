"""Helpers shared by several pipeline stages.

Stages run as scripts with src/ on sys.path, so `from common import ...` works.
"""
from __future__ import annotations

import importlib.util
import os
import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = "config/settings.yaml"


def config_path() -> str:
    """Settings file to use: $PIPELINE_CONFIG if set, else config/settings.yaml."""
    return os.environ.get("PIPELINE_CONFIG") or DEFAULT_CONFIG


def load_settings(path: str | None = None) -> dict:
    with open(path or config_path()) as f:
        return yaml.safe_load(f)


# ── keyword matching ────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    alternatives = "|".join(re.escape(k.lower()) for k in keywords)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})", re.IGNORECASE)


def match_any(text: str, keywords: list) -> bool:
    """True if any keyword occurs at the start of a word in text.

    Matching is case-insensitive and anchored at a word start only, so
    "monitor" matches "monitoring" but "now" does not match "know".
    Multi-word keywords ("not sure") match as written.
    """
    if not isinstance(text, str) or not keywords:
        return False
    return bool(_keyword_pattern(tuple(str(k) for k in keywords)).search(text))


# ── placeholders and content-bearing text ───────────────────────────────────
# Whole lines that carry no free text: media placeholders, attachment lines,
# deleted-message notices.
PLACEHOLDER_LINE_RE = re.compile(
    r"^(?:<media omitted>"
    r"|(?:image|video|audio|sticker|document|gif|contact card) omitted"
    r"|this message was deleted|you deleted this message|null"
    r"|\S+\.(?:opus|jpe?g|png|webp|gif|mp4|3gp|pdf|docx?|vcf)\s*\(file attached\))$",
    re.IGNORECASE,
)
EDITED_SUFFIX_RE = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)


def strip_placeholders(text) -> str:
    """Return the free text of a message with placeholder lines removed."""
    if not isinstance(text, str):
        return ""
    kept = []
    for line in text.splitlines():
        line = EDITED_SUFFIX_RE.sub("", line.replace("‎", "")).strip()
        if line and not PLACEHOLDER_LINE_RE.match(line):
            kept.append(line)
    return "\n".join(kept)


def is_placeholder(text) -> bool:
    """True when a message has no free text of its own (media placeholder,
    attachment line, deleted-message notice, or empty)."""
    return strip_placeholders(text) == ""


@lru_cache(maxsize=1)
def _stage01():
    """Load src/01_parse.py (its filename starts with a digit)."""
    path = Path(__file__).resolve().parent / "01_parse.py"
    spec = importlib.util.spec_from_file_location("stage01_parse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_case_report(text) -> bool:
    """True when 01_parse would extract this message as a structured case
    report (same trigger and field rule as parse_case_reports)."""
    if not isinstance(text, str):
        return False
    return _stage01().extract_case_fields(text) is not None


def content_kind(df: pd.DataFrame) -> pd.Series:
    """Label each row: text, voice_translation, voice_untranslated,
    case_report, placeholder or system (system notices and rows without a
    sender). df needs message, message_type, sender_code and translation_en."""
    kinds = pd.Series("text", index=df.index, dtype=object)
    for idx, row in df.iterrows():
        if pd.isna(row["sender_code"]) or row["message_type"] == "system":
            kinds[idx] = "system"
        elif row["message_type"] == "voice_note":
            t = row["translation_en"]
            kinds[idx] = ("voice_translation" if isinstance(t, str) and t.strip()
                          else "voice_untranslated")
        elif is_case_report(row["message"]):
            kinds[idx] = "case_report"
        elif is_placeholder(row["message"]):
            kinds[idx] = "placeholder"
    return kinds


CONTENT_KINDS = ("text", "voice_translation")


def content_bearing_text(df: pd.DataFrame, kinds: pd.Series | None = None) -> pd.Series:
    """Text to analyse for each row, or "" where the row is not content-bearing.

    Content-bearing rows are free text (placeholder lines removed) and voice
    notes with an English translation. Media placeholders, deleted-message
    notices, untranslated voice notes, system notices and structured
    case-report messages get "".
    """
    if kinds is None:
        kinds = content_kind(df)
    out = pd.Series("", index=df.index, dtype=object)
    vt = kinds == "voice_translation"
    out[vt] = df.loc[vt, "translation_en"].astype(str).str.strip()
    tx = kinds == "text"
    out[tx] = df.loc[tx, "message"].map(strip_placeholders)
    return out


# ── voice-note translation join ─────────────────────────────────────────────
VN_FILENAME_RE = re.compile(r"^(?:(PTT|AUD)-)?(\d{8})?-WA(\d+)\.opus$", re.IGNORECASE)


def _candidate_filenames(media_filename: str, msg_date) -> list[str]:
    """Filenames to look up for one chat media reference.

    A full name (PTT-20241210-WA0037.opus) is used as is. When the date token
    is missing (exports anonymised before the redaction fix kept only
    '-WA0037.opus'), rebuild it from the message date, trying both PTT and AUD
    prefixes, then the day before and after (file date and chat date can differ
    by one day around midnight).
    """
    m = VN_FILENAME_RE.match(media_filename.strip())
    if not m:
        return [media_filename]
    prefix, date_tok, seq = m.group(1), m.group(2), m.group(3)
    if date_tok:
        return [media_filename]
    if pd.isna(msg_date):
        return []
    prefixes = [prefix.upper()] if prefix else ["PTT", "AUD"]
    names = []
    for shift in (0, -1, 1):
        d = (msg_date + timedelta(days=shift)).strftime("%Y%m%d")
        names.extend(f"{p}-{d}-WA{seq}.opus" for p in prefixes)
    return names


def join_voice_translations(df: pd.DataFrame, voice_notes_csv) -> pd.DataFrame:
    """Return a copy of df with vn_filename and translation_en added.

    Voice-note rows are matched to successful transcripts in voice_notes_csv
    (column filename) on media_filename, with the fallback described in
    _candidate_filenames. The index of df is preserved.
    """
    out = df.copy()
    out["vn_filename"] = pd.Series(pd.NA, index=out.index, dtype=object)
    out["translation_en"] = pd.Series(pd.NA, index=out.index, dtype=object)
    path = Path(voice_notes_csv) if voice_notes_csv else None
    if path is None or not path.exists() or "media_filename" not in out.columns:
        return out

    vn = pd.read_csv(path)
    if "status" in vn.columns:
        vn = vn[vn["status"] == "success"]
    vn = vn.dropna(subset=["filename"]).drop_duplicates(subset="filename", keep="first")
    lookup = dict(zip(vn["filename"].astype(str), vn["translation_en"]))

    rows = out["media_filename"].notna()
    if "message_type" in out.columns:
        rows &= out["message_type"] == "voice_note"
    for idx in out.index[rows]:
        ts = out.at[idx, "timestamp"]
        msg_date = ts.date() if pd.notna(ts) else ts
        for name in _candidate_filenames(str(out.at[idx, "media_filename"]), msg_date):
            if name in lookup:
                out.at[idx, "vn_filename"] = name
                out.at[idx, "translation_en"] = lookup[name]
                break
    return out
