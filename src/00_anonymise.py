"""Anonymise a raw WhatsApp export before parsing.

Replaces sender names with stable coded IDs (P001, P002, ...), redacts phone
numbers, and flags Pakistani/Urdu given names that appear inside message
bodies for manual review (no auto-redaction).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

TIMESTAMP_RE = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)\s?(?:AM|PM)?\]?\s?-\s"
)

PK_MOBILE_RE = re.compile(
    r"(?:\+?92[\s\-]?3\d{2}[\s\-]?\d{7}|03\d{2}[\s\-]?\d{7}|03\d{9})"
)
GENERIC_NUMBER_RE = re.compile(r"\d{7,}")

SETTINGS_PATH = Path("config/settings.yaml")
LOOKUP_PATH = Path("config/sender_lookup.csv")
OUTPUT_PATH = Path("data/interim/anonymised_export.txt")
WARNINGS_PATH = Path("data/interim/anonymisation_warnings.log")


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_lookup(path: Path = LOOKUP_PATH) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["real_name"].astype(str), df["coded_id"].astype(str)))


def save_lookup(lookup: dict, path: Path = LOOKUP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"real_name": k, "coded_id": v} for k, v in lookup.items()]
    ).sort_values("coded_id")
    df.to_csv(path, index=False)


def next_code(lookup: dict) -> str:
    used = set(lookup.values())
    i = 1
    while f"P{i:03d}" in used:
        i += 1
    return f"P{i:03d}"


def extract_sender(line: str) -> str | None:
    m = TIMESTAMP_RE.match(line)
    if not m:
        return None
    rest = line[m.end():]
    if ": " not in rest:
        return None
    sender = rest.split(": ", 1)[0].strip()
    if not sender:
        return None
    return sender


def redact_phones(text: str) -> tuple[str, int]:
    text, n1 = PK_MOBILE_RE.subn("[PHONE]", text)
    text, n2 = GENERIC_NUMBER_RE.subn("[NUMBER]", text)
    return text, n1 + n2


def flag_names(text: str, flag_list: list) -> list:
    hits = []
    for name in flag_list:
        pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        if pat.search(text):
            hits.append(name)
    return hits


def anonymise_text(
    raw: str, lookup: dict, flag_list: list
) -> tuple[str, dict, int, list]:
    lines = raw.splitlines()
    # First pass: discover all unique senders, assign codes
    for line in lines:
        sender = extract_sender(line)
        if sender and sender not in lookup:
            lookup[sender] = next_code(lookup)

    out_lines = []
    phones_redacted = 0
    warnings = []
    for i, line in enumerate(lines, start=1):
        m = TIMESTAMP_RE.match(line)
        if m:
            rest = line[m.end():]
            prefix = line[: m.end()]
            if ": " in rest:
                sender, body = rest.split(": ", 1)
                sender = sender.strip()
                coded = lookup.get(sender, sender)
                body, n = redact_phones(body)
                phones_redacted += n
                hits = flag_names(body, flag_list)
                for h in hits:
                    warnings.append(f"line {i}\tname_flag\t{h}")
                new_line = f"{prefix}{coded}: {body}"
            else:
                # system message — still redact phones, no sender to map
                new_rest, n = redact_phones(rest)
                phones_redacted += n
                new_line = f"{prefix}{new_rest}"
        else:
            # continuation line
            new_line, n = redact_phones(line)
            phones_redacted += n
            hits = flag_names(new_line, flag_list)
            for h in hits:
                warnings.append(f"line {i}\tname_flag\t{h}")
        out_lines.append(new_line)

    return "\n".join(out_lines) + ("\n" if raw.endswith("\n") else ""), lookup, phones_redacted, warnings


def main() -> None:
    settings = load_settings()
    input_path = Path(settings["paths"]["raw_input_file"])
    flag_list = settings.get("flag_names", [])
    raw = input_path.read_text(encoding="utf-8")

    lookup = load_lookup()
    n_before = len(lookup)

    anonymised, lookup, phones_redacted, warnings = anonymise_text(
        raw, lookup, flag_list
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(anonymised, encoding="utf-8")

    WARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WARNINGS_PATH, "w") as f:
        f.write("\n".join(warnings))

    save_lookup(lookup)

    print(f"[00_anonymise] Lines processed:    {len(raw.splitlines())}")
    print(f"[00_anonymise] Unique senders:     {len(lookup)} ({len(lookup) - n_before} new)")
    print(f"[00_anonymise] Numbers redacted:   {phones_redacted}")
    print(f"[00_anonymise] Name flags raised:  {len(warnings)}")
    print(f"[00_anonymise] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
