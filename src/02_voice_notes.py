"""Transcribe and translate WhatsApp voice notes via OpenAI Whisper + GPT fallback."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a translator for a Pakistani emergency ambulance WhatsApp group. "
    "Translate Urdu or mixed Urdu/English accurately to English, "
    "preserving meaning and tone."
)


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def log_failure(log_path: Path, filename: str, stage: str, err: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{filename}\t{stage}\t{err}\n")


def transcribe(client: OpenAI, audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="ur"
        )
    return resp.text


def translate(client: OpenAI, audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.translations.create(model="whisper-1", file=f)
    return resp.text


def gpt_fallback(client: OpenAI, transcription: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
    )
    return resp.choices[0].message.content


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    settings = load_settings()
    client = OpenAI(api_key=api_key)

    msgs_path = Path(settings["paths"]["messages_interim"])
    voice_dir = Path(settings["paths"]["voice_notes_dir"])
    log_path = Path(settings["paths"]["transcription_failures"])

    df = pd.read_csv(msgs_path)
    if "transcription_ur" not in df.columns:
        df["transcription_ur"] = None
    if "translation_en" not in df.columns:
        df["translation_en"] = None

    vn_mask = df["message_type"] == "voice_note"
    for idx in tqdm(df[vn_mask].index, desc="voice_notes"):
        filename = df.at[idx, "media_filename"]
        if not isinstance(filename, str):
            continue
        audio_path = voice_dir / filename
        if not audio_path.exists():
            log_failure(log_path, filename, "missing_file", FileNotFoundError(audio_path))
            continue
        try:
            df.at[idx, "transcription_ur"] = transcribe(client, audio_path)
        except Exception as e:
            log_failure(log_path, filename, "transcribe", e)
            continue
        try:
            df.at[idx, "translation_en"] = translate(client, audio_path)
        except Exception as e:
            log_failure(log_path, filename, "translate", e)
            try:
                df.at[idx, "translation_en"] = gpt_fallback(
                    client, df.at[idx, "transcription_ur"] or ""
                )
            except Exception as e2:
                log_failure(log_path, filename, "gpt_fallback", e2)

    df.to_csv(msgs_path, index=False)
    print(f"[02_voice_notes] Updated {msgs_path}")


if __name__ == "__main__":
    main()
