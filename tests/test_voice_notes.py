"""Tests for src/02_voice_notes.py — all API calls mocked."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "02_voice_notes.py"
spec = importlib.util.spec_from_file_location("voice_notes", SRC)
vn = importlib.util.module_from_spec(spec)
sys.modules["voice_notes"] = vn
spec.loader.exec_module(vn)


# ---------------------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------------------

class TestParseFilename:
    def test_ptt_prefix(self):
        result = vn.parse_filename("PTT-20240315-WA0042.opus")
        assert result == {"filename": "PTT-20240315-WA0042.opus", "date": "2024-03-15", "sequence": 42}

    def test_aud_prefix(self):
        result = vn.parse_filename("AUD-20241201-WA0001.opus")
        assert result == {"filename": "AUD-20241201-WA0001.opus", "date": "2024-12-01", "sequence": 1}

    def test_case_insensitive_prefix(self):
        result = vn.parse_filename("ptt-20240101-WA0099.opus")
        assert result is not None
        assert result["sequence"] == 99

    def test_no_match_returns_none(self):
        assert vn.parse_filename("random_audio.opus") is None

    def test_no_match_mp3(self):
        assert vn.parse_filename("PTT-20240315-WA0001.mp3") is None

    def test_date_parsed_correctly(self):
        result = vn.parse_filename("PTT-20230708-WA0010.opus")
        assert result["date"] == "2023-07-08"
        assert result["sequence"] == 10


# ---------------------------------------------------------------------------
# load_existing / resume logic
# ---------------------------------------------------------------------------

class TestLoadExisting:
    def test_returns_empty_set_when_no_file(self, tmp_path):
        result = vn.load_existing(tmp_path / "missing.csv")
        assert result == set()

    def test_returns_only_success_filenames(self, tmp_path):
        csv = tmp_path / "vn.csv"
        df = pd.DataFrame([
            {"filename": "PTT-20240101-WA0001.opus", "status": "success"},
            {"filename": "PTT-20240101-WA0002.opus", "status": "failed"},
            {"filename": "PTT-20240101-WA0003.opus", "status": "success"},
        ])
        df.to_csv(csv, index=False)
        result = vn.load_existing(csv)
        assert result == {"PTT-20240101-WA0001.opus", "PTT-20240101-WA0003.opus"}

    def test_skips_already_processed_in_process(self, tmp_path):
        voice_dir = tmp_path / "voice_notes"
        voice_dir.mkdir()
        output_path = tmp_path / "out.csv"
        log_path = tmp_path / "failures.log"

        f1 = voice_dir / "PTT-20240101-WA0001.opus"
        f2 = voice_dir / "PTT-20240101-WA0002.opus"
        f1.write_bytes(b"fake")
        f2.write_bytes(b"fake")

        # f1 already successfully processed
        pd.DataFrame([{
            "filename": "PTT-20240101-WA0001.opus", "date": "2024-01-01",
            "sequence": 1, "duration_seconds": None,
            "transcription_ur": "test", "translation_en": "test", "status": "success",
        }]).to_csv(output_path, index=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(text="transcription")
        mock_client.audio.translations.create.return_value = MagicMock(text="translation")

        import unittest.mock as um
        with um.patch.object(vn, "get_duration", return_value=5.0):
            vn.process_voice_notes(voice_dir, output_path, log_path, mock_client)

        # Only f2 should have been sent to the API
        assert mock_client.audio.transcriptions.create.call_count == 1


# ---------------------------------------------------------------------------
# join_to_chat
# ---------------------------------------------------------------------------

class TestJoinToChat:
    def _messages(self):
        return pd.DataFrame([
            {"timestamp": "2024-03-15 09:00:00", "sender_code": "P01", "message": "hello"},
            {"timestamp": "2024-03-15 10:00:00", "sender_code": "P02", "message": "ok"},
            {"timestamp": "2024-03-16 08:00:00", "sender_code": "P01", "message": "test"},
        ])

    def _voice_notes(self):
        return pd.DataFrame([
            {"filename": "PTT-20240315-WA0001.opus", "date": "2024-03-15",
             "sequence": 1, "transcription_ur": "بہت اچھا", "translation_en": "Very good", "status": "success"},
            {"filename": "PTT-20240315-WA0002.opus", "date": "2024-03-15",
             "sequence": 2, "transcription_ur": "ٹھیک ہے", "translation_en": "Okay", "status": "success"},
            {"filename": "PTT-20240316-WA0001.opus", "date": "2024-03-16",
             "sequence": 1, "transcription_ur": "سلام", "translation_en": "Hello", "status": "failed"},
        ])

    def test_merge_adds_voice_note_columns(self):
        result = vn.join_to_chat(self._messages(), self._voice_notes())
        assert "voice_note_count" in result.columns
        assert "translations_en" in result.columns

    def test_correct_count_on_matched_date(self):
        result = vn.join_to_chat(self._messages(), self._voice_notes())
        march15 = result[result["timestamp"].str.startswith("2024-03-15")]
        assert (march15["voice_note_count"] == 2).all()

    def test_failed_voice_notes_excluded_from_join(self):
        result = vn.join_to_chat(self._messages(), self._voice_notes())
        march16 = result[result["timestamp"].str.startswith("2024-03-16")]
        count = march16["voice_note_count"]
        assert count.isna().all() or (count == 0).all()

    def test_empty_voice_notes_returns_messages_unchanged(self):
        msgs = self._messages()
        result = vn.join_to_chat(msgs, pd.DataFrame())
        assert list(result.columns) == list(msgs.columns)
        assert len(result) == len(msgs)
