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


# Canonical field names and their regex aliases (all lowercase, stripped)
CASE_FIELD_ALIASES = {
    "case_no": re.compile(r"^case\s*no\.?$", re.IGNORECASE),
    "date": re.compile(r"^date$", re.IGNORECASE),
    "gender": re.compile(r"^gender$", re.IGNORECASE),
    "initial_complaint": re.compile(r"^initial\s*complaint$", re.IGNORECASE),
    "location": re.compile(r"^location(\s*transported\s*to)?$", re.IGNORECASE),
    "time_patient_reached": re.compile(r"^time\s*patient\s*reached$", re.IGNORECASE),
    "hospital": re.compile(
        r"^(name\s*of\s*hospital\s*transported\s*to|hospital\s*transported\s*to|transported\s*to)$",
        re.IGNORECASE,
    ),
    "time_hospital_reached": re.compile(
        r"^time\s*(hospital\s*)?(was\s*)?reached$", re.IGNORECASE
    ),
    "first_aid": re.compile(r"^first\s*aid\s*provided$", re.IGNORECASE),
    "emp_personnel": re.compile(r"^emp\s*person(nel|al)?\s*(names?)?$", re.IGNORECASE),
}

FIELD_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z\s/\(\)\.]{1,45}):\s*(.*)")


def _canonical_field(label: str) -> str | None:
    label = label.strip()
    for canon, pat in CASE_FIELD_ALIASES.items():
        if pat.match(label):
            return canon
    return None


def parse_case_reports(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured case report fields from multi-line ambulance dispatch messages."""
    records = []
    case_trigger = re.compile(r"case\s*no|🚑", re.IGNORECASE)

    for _, row in messages_df.iterrows():
        msg = str(row.get("message", ""))
        if not case_trigger.search(msg):
            continue

        fields: dict = {
            "timestamp": row["timestamp"],
            "sender_code": row["sender_code"],
            "phase": row.get("phase"),
            "case_no": None,
            "date": None,
            "gender": None,
            "initial_complaint": None,
            "location": None,
            "time_patient_reached": None,
            "hospital": None,
            "time_hospital_reached": None,
            "first_aid": None,
            "emp_personnel": None,
            "raw_message": msg,
        }

        for line in msg.splitlines():
            line = line.strip()
            m = FIELD_LINE_RE.match(line)
            if not m:
                continue
            label, value = m.group(1).strip(), m.group(2).strip()
            canon = _canonical_field(label)
            if canon and fields.get(canon) is None:
                fields[canon] = value if value else None

        # Only keep rows where we extracted at least a case number or complaint
        if fields["case_no"] or fields["initial_complaint"]:
            records.append(fields)

    return pd.DataFrame(records)


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

    case_df = parse_case_reports(df)
    case_path = Path(settings["paths"]["case_reports"])
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(case_path, index=False)
    print(f"[01_parse] Wrote {len(case_df)} case reports to {case_path}")


if __name__ == "__main__":
    main()
