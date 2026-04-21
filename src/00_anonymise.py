"""Anonymise a raw WhatsApp export before parsing.

Replaces sender names with stable coded IDs (P001, P002, ...), redacts phone
numbers, @mentions, and all known name tokens (first names, surnames, full
names) wherever they appear in message bodies and system messages.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

TIMESTAMP_RE = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)[\s ]?(?:[AaPp][Mm])?\]?[\s ]?-\s"
)

PK_MOBILE_RE = re.compile(
    r"(?:\+?92[\s\-]?3\d{2}[\s\-]?\d{7}|03\d{2}[\s\-]?\d{7}|03\d{9})"
)
GENERIC_NUMBER_RE = re.compile(r"\d{7,}")
MENTION_RE = re.compile(r"@⁨[^⁩]+⁩")

# Words to exclude when tokenising names from the lookup
NAME_STOP_TOKENS = {
    "emp", "osc", "alt", "alternate", "whatsapp", "dr", "mam",
    "the", "and", "mr", "mrs", "ms",
}

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
    rows = [{"real_name": k, "coded_id": v} for k, v in lookup.items()]
    df = pd.DataFrame(rows, columns=["real_name", "coded_id"])
    if not df.empty:
        df = df.sort_values("coded_id")
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
    return sender if sender else None


def build_name_patterns(lookup: dict, aliases: list[str] | None = None) -> list[tuple[re.Pattern, str]]:
    """Build (pattern, replacement) pairs from all name tokens in the lookup.

    Extracts full names and individual word tokens. Where a token maps to
    multiple codes (ambiguous), replacement is [REDACTED]. Sorted longest-
    first so multi-word names are replaced before their constituent words.
    """
    token_to_codes: dict[str, set[str]] = {}

    for real_name, code in lookup.items():
        # Skip bare phone-number entries
        if re.match(r"^[\+0][\d\s\-]+$", real_name.strip()):
            continue

        # Normalise unicode decorative letters to plain ASCII for matching
        plain_name = "".join(c if c.isascii() else " " for c in real_name).strip()

        candidates = set()
        candidates.add(plain_name)

        for word in re.split(r"[\s_]+", plain_name):
            word = word.strip()
            if len(word) >= 3 and word.lower() not in NAME_STOP_TOKENS:
                candidates.add(word)

        for token in candidates:
            if not token:
                continue
            token_to_codes.setdefault(token, set()).add(code)

    # Add freetext aliases (surnames etc. not in lookup) as [REDACTED]
    for alias in (aliases or []):
        alias = alias.strip()
        if alias and alias.lower() not in NAME_STOP_TOKENS:
            token_to_codes.setdefault(alias, set()).add("[REDACTED]")

    patterns = []
    for token, codes in sorted(token_to_codes.items(), key=lambda x: -len(x[0])):
        replacement = list(codes)[0] if len(codes) == 1 else "[REDACTED]"
        try:
            pat = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
            patterns.append((pat, replacement))
        except re.error:
            pass

    return patterns


def redact_phones(text: str) -> tuple[str, int]:
    text, n1 = PK_MOBILE_RE.subn("[PHONE]", text)
    text, n2 = GENERIC_NUMBER_RE.subn("[NUMBER]", text)
    return text, n1 + n2


def redact_mentions(text: str) -> tuple[str, int]:
    result, n = MENTION_RE.subn("@[MENTION]", text)
    return result, n


def redact_names(text: str, patterns: list[tuple[re.Pattern, str]]) -> tuple[str, int]:
    total = 0
    for pat, replacement in patterns:
        text, n = pat.subn(replacement, text)
        total += n
    return text, total


def flag_names(text: str, flag_list: list) -> list:
    hits = []
    for name in flag_list:
        pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        if pat.search(text):
            hits.append(name)
    return hits


def anonymise_text(
    raw: str, lookup: dict, flag_list: list, aliases: list[str] | None = None
) -> tuple[str, dict, int, int, int, list]:
    lines = raw.splitlines()

    # First pass: discover all unique senders and assign codes
    for line in lines:
        sender = extract_sender(line)
        if sender and sender not in lookup:
            lookup[sender] = next_code(lookup)

    # Build name patterns now that all senders are known
    name_patterns = build_name_patterns(lookup, aliases)

    out_lines = []
    phones_redacted = 0
    mentions_redacted = 0
    names_redacted = 0
    warnings = []

    for i, line in enumerate(lines, start=1):
        m = TIMESTAMP_RE.match(line)
        if m:
            rest = line[m.end():]
            prefix = line[: m.end()]
            if ": " in rest:
                sender, body = rest.split(": ", 1)
                coded = lookup.get(sender.strip(), sender.strip())
                body, n = redact_phones(body)
                phones_redacted += n
                body, nm = redact_mentions(body)
                mentions_redacted += nm
                body, nn = redact_names(body, name_patterns)
                names_redacted += nn
                hits = flag_names(body, flag_list)
                for h in hits:
                    warnings.append(f"line {i}\tname_flag\t{h}")
                new_line = f"{prefix}{coded}: {body}"
            else:
                # System message line — redact names here too
                rest, n = redact_phones(rest)
                phones_redacted += n
                rest, nm = redact_mentions(rest)
                mentions_redacted += nm
                rest, nn = redact_names(rest, name_patterns)
                names_redacted += nn
                new_line = f"{prefix}{rest}"
        else:
            # Continuation / standalone body line
            new_line, n = redact_phones(line)
            phones_redacted += n
            new_line, nm = redact_mentions(new_line)
            mentions_redacted += nm
            new_line, nn = redact_names(new_line, name_patterns)
            names_redacted += nn
            hits = flag_names(new_line, flag_list)
            for h in hits:
                warnings.append(f"line {i}\tname_flag\t{h}")
        out_lines.append(new_line)

    return (
        "\n".join(out_lines) + ("\n" if raw.endswith("\n") else ""),
        lookup,
        phones_redacted,
        mentions_redacted,
        names_redacted,
        warnings,
    )


def main() -> None:
    settings = load_settings()
    input_path = Path(settings["paths"]["raw_input_file"])
    flag_list = settings.get("flag_names", [])
    aliases = settings.get("name_aliases", [])
    raw = input_path.read_text(encoding="utf-8")

    lookup = load_lookup()
    n_before = len(lookup)

    anonymised, lookup, phones_redacted, mentions_redacted, names_redacted, warnings = anonymise_text(
        raw, lookup, flag_list, aliases
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
    print(f"[00_anonymise] Mentions redacted:  {mentions_redacted}")
    print(f"[00_anonymise] Names redacted:     {names_redacted}")
    print(f"[00_anonymise] Name flags raised:  {len(warnings)}")
    print(f"[00_anonymise] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
