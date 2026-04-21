"""Unit tests for the anonymisation stage — synthetic data only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src" / "00_anonymise.py"
spec = importlib.util.spec_from_file_location("anon_module", SRC)
anon = importlib.util.module_from_spec(spec)
sys.modules["anon_module"] = anon
spec.loader.exec_module(anon)


def test_phone_redaction_dashed():
    text = "Call me at 0321-1234567 please"
    out, n = anon.redact_phones(text)
    assert "[PHONE]" in out
    assert "0321-1234567" not in out
    assert n == 1


def test_phone_redaction_no_dash():
    text = "Number is 03211234567 thanks"
    out, n = anon.redact_phones(text)
    assert "[PHONE]" in out
    assert "03211234567" not in out
    assert n == 1


def test_phone_redaction_international():
    text = "Try +923211234567 now"
    out, n = anon.redact_phones(text)
    assert "[PHONE]" in out
    assert "+923211234567" not in out
    assert n == 1


def test_generic_long_number_redacted():
    text = "Ref code 98765432 confirmed"
    out, n = anon.redact_phones(text)
    assert "[NUMBER]" in out
    assert "98765432" not in out


def test_short_numbers_not_redacted():
    text = "Bed 5 at 10am"
    out, n = anon.redact_phones(text)
    assert out == text
    assert n == 0


def test_sender_name_replacement():
    raw = (
        "12/04/2024, 10:15 AM - John Medic: first\n"
        "12/04/2024, 10:16 AM - Sara Supervisor: second\n"
        "12/04/2024, 10:17 AM - John Medic: third\n"
    )
    out, lookup, _, _ = anon.anonymise_text(raw, {}, [])
    assert "John Medic" not in out
    assert "Sara Supervisor" not in out
    assert lookup["John Medic"] == "P001"
    assert lookup["Sara Supervisor"] == "P002"
    # same sender reused → same code, and appears twice
    assert out.count("P001:") == 2
    assert out.count("P002:") == 1


def test_lookup_persistence_across_runs(tmp_path):
    lookup_path = tmp_path / "sender_lookup.csv"
    lookup = {"Alice": "P001", "Bob": "P002"}
    anon.save_lookup(lookup, lookup_path)

    reloaded = anon.load_lookup(lookup_path)
    assert reloaded == lookup

    raw = (
        "12/04/2024, 10:15 AM - Alice: hi\n"
        "12/04/2024, 10:16 AM - Charlie: new person\n"
    )
    out, extended, _, _ = anon.anonymise_text(raw, reloaded, [])
    assert extended["Alice"] == "P001"  # stable
    assert extended["Bob"] == "P002"    # preserved even if absent from new run
    assert extended["Charlie"] == "P003"  # next free code
    assert "P001: hi" in out
    assert "P003: new person" in out


def test_name_flag_standalone_word():
    hits = anon.flag_names("Ahmed came to help", ["Ahmed", "Ali"])
    assert "Ahmed" in hits


def test_name_flag_case_insensitive():
    hits = anon.flag_names("ahmed was here", ["Ahmed"])
    assert hits == ["Ahmed"]


def test_name_flag_not_substring():
    # "Ali" should not match inside "Alice" or "Pakistani"
    hits = anon.flag_names("Alice from Pakistani team", ["Ali"])
    assert hits == []


def test_name_flag_not_auto_redacted():
    raw = "12/04/2024, 10:15 AM - P001: Ahmed came to help\n"
    out, _, _, warnings = anon.anonymise_text(raw, {"P001": "P001"}, ["Ahmed"])
    assert "Ahmed" in out  # not redacted, just flagged
    assert any("Ahmed" in w for w in warnings)


def test_end_to_end_anonymisation():
    raw = (
        "12/04/2024, 10:15 AM - John Medic: Call 0321-1234567 about Ahmed\n"
        "12/04/2024, 10:16 AM - Sara: ok\n"
    )
    out, lookup, phones, warnings = anon.anonymise_text(raw, {}, ["Ahmed"])
    assert "John Medic" not in out
    assert "0321-1234567" not in out
    assert "[PHONE]" in out
    assert phones == 1
    assert any("Ahmed" in w for w in warnings)
    assert len(lookup) == 2
