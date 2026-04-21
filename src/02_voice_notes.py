"""Transcribe and translate .opus voice notes as a parallel dataset via OpenAI Whisper."""
from __future__ import annotations

import io
import os
import re
import subprocess
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

FILENAME_RE = re.compile(
    r"(?:PTT|AUD)-(\d{4})(\d{2})(\d{2})-WA(\d+)\.opus$", re.IGNORECASE
)

BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.5


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def log_failure(log_path: Path, filename: str, stage: str, err: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{filename}\t{stage}\t{err}\n")


def parse_filename(filename: str) -> dict | None:
    """Extract date and sequence from a PTT/AUD filename. Returns None if no match."""
    m = FILENAME_RE.search(filename)
    if not m:
        return None
    year, month, day, seq = m.group(1), m.group(2), m.group(3), m.group(4)
    return {
        "filename": filename,
        "date": date(int(year), int(month), int(day)).isoformat(),
        "sequence": int(seq),
    }


def get_duration(audio_path: Path) -> float | None:
    """Return duration in seconds using mutagen if available, else None."""
    try:
        from mutagen.oggopus import OggOpus
        info = OggOpus(str(audio_path))
        return round(info.info.length, 2)
    except Exception:
        return None


def _to_ogg_bytes(audio_path: Path) -> io.BytesIO:
    """Convert any audio file to ogg/opus in-memory via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-f", "ogg", "-acodec", "libopus", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    buf = io.BytesIO(result.stdout)
    buf.name = audio_path.stem + ".ogg"
    return buf


def transcribe(client: OpenAI, audio_path: Path) -> str:
    buf = _to_ogg_bytes(audio_path)
    resp = client.audio.transcriptions.create(
        model="whisper-1", file=buf, language="ur"
    )
    return resp.text


def translate(client: OpenAI, audio_path: Path) -> str:
    buf = _to_ogg_bytes(audio_path)
    resp = client.audio.translations.create(model="whisper-1", file=buf)
    return resp.text


def load_existing(output_path: Path) -> set[str]:
    """Return set of already-processed filenames from an existing output CSV."""
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    done = df.loc[df["status"] == "success", "filename"]
    return set(done.tolist())


def process_voice_notes(
    voice_dir: Path,
    output_path: Path,
    log_path: Path,
    client: OpenAI,
) -> pd.DataFrame:
    opus_files = sorted(voice_dir.glob("*.opus"))
    if not opus_files:
        print(f"[02_voice_notes] No .opus files found in {voice_dir}")
        return pd.DataFrame()

    already_done = load_existing(output_path)
    pending = [f for f in opus_files if f.name not in already_done]
    print(f"[02_voice_notes] {len(opus_files)} total | {len(already_done)} skipped | {len(pending)} to process")

    rows: list[dict] = []

    for i, audio_path in enumerate(tqdm(pending, desc="transcribing")):
        meta = parse_filename(audio_path.name)
        if meta is None:
            log_failure(log_path, audio_path.name, "parse_filename", ValueError("Filename did not match expected pattern"))
            rows.append({
                "filename": audio_path.name, "date": None, "sequence": None,
                "duration_seconds": None, "transcription_ur": None,
                "translation_en": None, "status": "failed",
            })
            continue

        row = {**meta, "duration_seconds": get_duration(audio_path),
               "transcription_ur": None, "translation_en": None, "status": "failed"}

        try:
            row["transcription_ur"] = transcribe(client, audio_path)
            time.sleep(RATE_LIMIT_DELAY)
            row["translation_en"] = translate(client, audio_path)
            time.sleep(RATE_LIMIT_DELAY)
            row["status"] = "success"
        except Exception as e:
            stage = "translate" if row["transcription_ur"] else "transcribe"
            log_failure(log_path, audio_path.name, stage, e)
            row["status"] = "failed"

        rows.append(row)

        if (i + 1) % BATCH_SIZE == 0:
            _append_rows(output_path, rows)
            rows = []

    if rows:
        _append_rows(output_path, rows)

    return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()


def _append_rows(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        new_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(output_path, index=False)


def join_to_chat(messages_df: pd.DataFrame, voice_notes_df: pd.DataFrame) -> pd.DataFrame:
    """Loose join: merge both datasets by date, returning a combined daily summary."""
    if voice_notes_df.empty:
        return messages_df.copy()

    vn = voice_notes_df[voice_notes_df["status"] == "success"].copy()
    vn["date"] = pd.to_datetime(vn["date"]).dt.date

    if "timestamp" in messages_df.columns:
        msgs = messages_df.copy()
        msgs["date"] = pd.to_datetime(msgs["timestamp"], errors="coerce").dt.date
    else:
        return messages_df.copy()

    daily_vn = (
        vn.groupby("date")
        .agg(
            voice_note_count=("filename", "count"),
            transcriptions_ur=("transcription_ur", lambda x: " | ".join(x.dropna())),
            translations_en=("translation_en", lambda x: " | ".join(x.dropna())),
        )
        .reset_index()
    )

    merged = msgs.merge(daily_vn, on="date", how="left")
    return merged


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    settings = load_settings()
    client = OpenAI(api_key=api_key)

    voice_dir = Path(settings["paths"]["voice_notes_dir"])
    output_path = Path(settings["paths"]["voice_notes_output"])
    log_path = Path(settings["paths"]["transcription_failures"])

    process_voice_notes(voice_dir, output_path, log_path, client)
    print(f"[02_voice_notes] Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
