"""Parse a WhatsApp Android .txt export into a structured messages CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

TIMESTAMP_RE = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)[\s ]?([AaPp][Mm])?\]?[\s ]?-\s"
)

MEDIA_PATTERNS = {
    "voice_note": re.compile(r"(PTT-.*?\.opus|\.opus|audio omitted)", re.IGNORECASE),
    "image": re.compile(r"(IMG-.*?\.(jpg|jpeg|png)|image omitted)", re.IGNORECASE),
    "video": re.compile(r"(VID-.*?\.mp4|video omitted)", re.IGNORECASE),
    "document": re.compile(r"(\.pdf|\.docx?|document omitted)", re.IGNORECASE),
    "sticker": re.compile(r"sticker omitted", re.IGNORECASE),
}

SYSTEM_PATTERNS = [
    re.compile(r"added|removed|left|joined|changed the group|created group", re.IGNORECASE),
]


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_timestamp(date_str: str, time_str: str, ampm: str | None = None) -> pd.Timestamp:
    dt_str = f"{date_str} {time_str} {ampm.upper()}" if ampm else f"{date_str} {time_str}"
    if ampm:
        fmts = ("%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
                "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
                "%d/%m/%y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p")
    else:
        fmts = ("%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
                "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
                "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M")
    for fmt in fmts:
        try:
            return pd.to_datetime(dt_str, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(dt_str, errors="coerce")


def classify_message(text: str) -> str:
    for mtype, pat in MEDIA_PATTERNS.items():
        if pat.search(text):
            return mtype
    return "text"


def is_system_message(line_after_ts: str) -> bool:
    if ":" in line_after_ts.split(" ", 1)[0]:
        return False
    for pat in SYSTEM_PATTERNS:
        if pat.search(line_after_ts):
            return True
    return False


def parse_whatsapp_export(text: str) -> pd.DataFrame:
    rows = []
    current = None
    for raw_line in text.splitlines():
        m = TIMESTAMP_RE.match(raw_line)
        if m:
            if current:
                rows.append(current)
            date_str, time_str, ampm = m.group(1), m.group(2), m.group(3)
            rest = raw_line[m.end():]
            ts = parse_timestamp(date_str, time_str, ampm)
            if is_system_message(rest):
                current = {
                    "timestamp": ts, "sender_code": None, "message": rest.strip(),
                    "message_type": "system", "media_filename": None,
                }
            else:
                if ": " in rest:
                    sender, msg = rest.split(": ", 1)
                else:
                    sender, msg = None, rest
                mtype = classify_message(msg)
                media_fn = None
                if mtype == "voice_note":
                    fn_match = re.search(r"([\w\-]+\.opus)", msg)
                    media_fn = fn_match.group(1) if fn_match else None
                current = {
                    "timestamp": ts, "sender_code": sender, "message": msg.strip(),
                    "message_type": mtype, "media_filename": media_fn,
                }
        else:
            if current is not None:
                current["message"] = (current["message"] + "\n" + raw_line).strip()
    if current:
        rows.append(current)
    return pd.DataFrame(rows)


def assign_phase(ts: pd.Timestamp, phases: list) -> str | None:
    if pd.isna(ts):
        return None
    for p in phases:
        start = pd.to_datetime(p["start"])
        end = pd.to_datetime(p["end"]) + pd.Timedelta(days=1)
        if start <= ts < end:
            return p["name"]
    return None


def main() -> None:
    settings = load_settings()
    input_path = Path(settings["paths"]["input_file"])
    text = input_path.read_text(encoding="utf-8")

    df = parse_whatsapp_export(text)

    role_map_path = Path(settings["paths"]["role_mapping"])
    if role_map_path.exists():
        roles = pd.read_csv(role_map_path)
        df = df.merge(roles, on="sender_code", how="left")
    else:
        df["role"] = None
        df["site"] = None

    df["phase"] = df["timestamp"].apply(lambda t: assign_phase(t, settings["phases"]))

    out_path = Path(settings["paths"]["messages_interim"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[01_parse] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
