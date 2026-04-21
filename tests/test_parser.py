"""Unit tests for the WhatsApp parser — synthetic data only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "01_parse.py"
spec = importlib.util.spec_from_file_location("parse_module", SRC)
parse_module = importlib.util.module_from_spec(spec)
sys.modules["parse_module"] = parse_module
spec.loader.exec_module(parse_module)


def test_multiline_message_is_joined():
    text = (
        "12/04/2024, 10:15 AM - A01: First line\n"
        "continues here\n"
        "and here\n"
        "12/04/2024, 10:16 AM - A02: Second message\n"
    )
    df = parse_module.parse_whatsapp_export(text)
    assert len(df) == 2
    assert "continues here" in df.iloc[0]["message"]
    assert "and here" in df.iloc[0]["message"]
    assert df.iloc[1]["message"] == "Second message"


def test_voice_note_detection():
    text = "12/04/2024, 10:15 AM - A01: PTT-20240412-WA0001.opus (file attached)\n"
    df = parse_module.parse_whatsapp_export(text)
    assert df.iloc[0]["message_type"] == "voice_note"
    assert df.iloc[0]["media_filename"] == "PTT-20240412-WA0001.opus"


def test_system_message_parsing():
    text = (
        "12/04/2024, 10:15 AM - A01 added A02\n"
        "12/04/2024, 10:16 AM - A02 left\n"
    )
    df = parse_module.parse_whatsapp_export(text)
    assert df.iloc[0]["message_type"] == "system"
    assert df.iloc[0]["sender_code"] is None
    assert df.iloc[1]["message_type"] == "system"


def test_role_mapping_join(tmp_path):
    text = (
        "12/04/2024, 10:15 AM - A01: hello\n"
        "12/04/2024, 10:16 AM - B02: hi\n"
    )
    df = parse_module.parse_whatsapp_export(text)
    roles = pd.DataFrame({
        "sender_code": ["A01", "B02"],
        "role": ["medic", "supervisor"],
        "site": ["karachi", "lahore"],
    })
    merged = df.merge(roles, on="sender_code", how="left")
    assert merged.loc[merged["sender_code"] == "A01", "role"].iloc[0] == "medic"
    assert merged.loc[merged["sender_code"] == "B02", "site"].iloc[0] == "lahore"


def test_phase_assignment():
    phases = [
        {"name": "baseline", "start": "2024-01-01", "end": "2024-03-31"},
        {"name": "pilot", "start": "2024-04-01", "end": "2024-06-30"},
    ]
    ts = pd.Timestamp("2024-02-15 10:00")
    assert parse_module.assign_phase(ts, phases) == "baseline"
    ts2 = pd.Timestamp("2024-05-10 10:00")
    assert parse_module.assign_phase(ts2, phases) == "pilot"
    ts3 = pd.Timestamp("2023-12-31 23:59")
    assert parse_module.assign_phase(ts3, phases) is None


def test_classify_message_text_default():
    assert parse_module.classify_message("just plain text") == "text"


def test_classify_image():
    assert parse_module.classify_message("IMG-20240412-WA0001.jpg (file attached)") == "image"
