"""Tests for src/common.py and the turn latency in src/03_engagement.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import common

SRC = Path(__file__).resolve().parents[1] / "src"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eng = _load("engagement_module", "03_engagement.py")
net = _load("network_module_common", "04_network.py")


# ---------------------------------------------------------------------------
# load_settings / PIPELINE_CONFIG
# ---------------------------------------------------------------------------

def test_config_path_default(monkeypatch):
    monkeypatch.delenv("PIPELINE_CONFIG", raising=False)
    assert common.config_path() == "config/settings.yaml"


def test_config_path_env(monkeypatch, tmp_path):
    cfg = tmp_path / "s.yaml"
    cfg.write_text("paths: {a: b}\n")
    monkeypatch.setenv("PIPELINE_CONFIG", str(cfg))
    assert common.load_settings() == {"paths": {"a": "b"}}


# ---------------------------------------------------------------------------
# match_any (word-start boundary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,keywords,expected", [
    ("I know the way", ["now"], False),
    ("come now please", ["now"], True),
    ("Now!", ["now"], True),
    ("monitoring the patient", ["monitor"], True),
    ("we will help", ["help"], True),
    ("whelp", ["help"], False),
    ("I am not sure yet", ["not sure"], True),
    ("Thank you all", ["thank you"], True),
    ("recall the drill", ["call"], False),
    ("called the ambulance", ["call"], True),
    ("URGENT case", ["urgent"], True),
    ("asap.", ["asap"], True),
    ("", ["now"], False),
    (None, ["now"], False),
    ("anything", [], False),
])
def test_match_any(text, keywords, expected):
    assert common.match_any(text, keywords) is expected


def test_match_any_escapes_regex_characters():
    assert common.match_any("dose 1+1 given", ["1+1"])
    assert not common.match_any("dose 11 given", ["1+1"])


# ---------------------------------------------------------------------------
# placeholders and case reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "<Media omitted>", "image omitted", "This message was deleted",
    "You deleted this message", "PTT-20240412-WA0001.opus (file attached)",
    "IMG-20240412-WA0001.jpg (file attached)", "", None,
])
def test_is_placeholder(text):
    assert common.is_placeholder(text)


def test_caption_after_placeholder_is_content():
    assert not common.is_placeholder("<Media omitted>\nPatient shifted to CMH")
    assert common.strip_placeholders("<Media omitted>\nPatient shifted") == "Patient shifted"


def test_edited_suffix_removed():
    assert common.strip_placeholders("on my way <This message was edited>") == "on my way"


def test_is_case_report():
    report = ("🚑🏥\nDate: 25-03-2026\nCase No: 621\nGender: male\n"
              "Initial Complaint: fever\nName of Hospital: CMH Transported to: ambulance")
    assert common.is_case_report(report)
    assert common.is_case_report("<Media omitted>\n" + report)
    assert not common.is_case_report("in case no one answers, call me")
    assert not common.is_case_report("hello")


def test_content_kind_and_text():
    df = pd.DataFrame({
        "sender_code": ["P1", "P1", "P2", "P2", None, "P3"],
        "message_type": ["text", "text", "voice_note", "voice_note", "text", "text"],
        "message": ["hello team", "<Media omitted>", "PTT-x.opus (file attached)",
                    "PTT-y.opus (file attached)", "Your security code changed",
                    "Case No: 4\nInitial Complaint: fever"],
        "translation_en": [None, None, "We reached", None, None, None],
    }, index=[10, 11, 12, 13, 14, 15])
    kinds = common.content_kind(df)
    assert kinds.tolist() == ["text", "placeholder", "voice_translation",
                              "voice_untranslated", "system", "case_report"]
    text = common.content_bearing_text(df, kinds)
    assert text.tolist() == ["hello team", "", "We reached", "", "", ""]
    assert list(text.index) == [10, 11, 12, 13, 14, 15]


# ---------------------------------------------------------------------------
# voice-note join
# ---------------------------------------------------------------------------

def test_join_voice_translations(tmp_path):
    vn = pd.DataFrame({
        "filename": ["PTT-20241210-WA0037.opus", "AUD-20241210-WA0040.opus",
                     "PTT-20241211-WA0002.opus", "PTT-20241210-WA0050.opus"],
        "translation_en": ["full name", "aud prefix", "next day", "failed"],
        "status": ["success", "success", "success", "failed"],
    })
    path = tmp_path / "voice_notes.csv"
    vn.to_csv(path, index=False)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-12-10 09:00"] * 5 + ["2024-12-10 23:59"]),
        "message_type": ["voice_note"] * 5 + ["text"],
        "media_filename": ["PTT-20241210-WA0037.opus",  # exact
                           "-WA0040.opus",              # redacted date, AUD prefix
                           "-WA0002.opus",              # redacted date, file dated next day
                           "PTT-20241210-WA0050.opus",  # transcript failed
                           "PTT-20241210-WA0099.opus",  # no transcript
                           None],
    }, index=[5, 7, 9, 11, 13, 15])  # gapped index must be preserved
    out = common.join_voice_translations(df, path)
    assert list(out.index) == [5, 7, 9, 11, 13, 15]
    assert out["translation_en"].tolist()[:3] == ["full name", "aud prefix", "next day"]
    assert out["translation_en"].iloc[3:].isna().all()
    assert out.loc[7, "vn_filename"] == "AUD-20241210-WA0040.opus"


def test_join_voice_translations_missing_file(tmp_path):
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]),
                       "message_type": ["voice_note"],
                       "media_filename": ["PTT-20240101-WA0001.opus"]})
    out = common.join_voice_translations(df, tmp_path / "missing.csv")
    assert out["translation_en"].isna().all()


# ---------------------------------------------------------------------------
# turn latency (03_engagement)
# ---------------------------------------------------------------------------

def _msgs(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "sender_code", "phase"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["message_type"] = "text"
    return df


def test_build_turns_collapses_bursts():
    df = _msgs([
        ("2024-01-01 10:00", "A", "p1"),
        ("2024-01-01 10:05", "A", "p1"),   # burst: one turn 10:00-10:05
        ("2024-01-01 10:20", "B", "p1"),   # reply 15 min after A's LAST message
        ("2024-01-01 11:20", "A", "p1"),   # 60 min
        ("2024-01-01 11:21", None, "p1"),  # no sender: ignored
        ("2024-01-01 15:20", "C", "p1"),   # 240 min
    ])
    turns = eng.build_turns(df)
    assert turns["sender_code"].tolist() == ["A", "B", "A", "C"]
    assert turns["n_messages"].tolist() == [2, 1, 1, 1]
    assert turns["latency_minutes"].tolist()[:3] == [15.0, 60.0, 240.0]
    assert np.isnan(turns["latency_minutes"].iloc[-1])


def test_build_turns_stable_within_minute():
    # same-minute messages keep export order, so A,A,B is two turns not three
    df = _msgs([
        ("2024-01-01 10:00", "A", "p1"),
        ("2024-01-01 10:00", "A", "p1"),
        ("2024-01-01 10:00", "B", "p1"),
    ] * 1)
    assert eng.build_turns(df)["sender_code"].tolist() == ["A", "B"]


def test_response_latency_summary():
    df = _msgs([
        ("2024-01-01 10:00", "A", "p1"),
        ("2024-01-01 10:10", "B", "p1"),   # 10
        ("2024-01-01 12:10", "A", "p1"),   # 120
        ("2025-01-01 10:00", "B", "p2"),   # last p1 turn: long gap
        ("2025-01-01 10:30", "A", "p2"),   # 30
    ])
    summary, by_sender = eng.response_latency(df)
    s = summary.set_index("phase")
    assert s.loc["Overall", "n_turns"] == 4
    assert s.loc["p1", "n_turns"] == 3
    assert s.loc["p1", "latency_median_min"] == 120.0
    assert s.loc["p2", "latency_median_min"] == 30.0
    assert s.loc["p2", "share_le_60min"] == 1.0
    assert s.loc["p1", "share_le_15min"] == pytest.approx(1 / 3)
    assert set(by_sender.columns) >= {"sender_code", "phase", "n_turns", "latency_median_min"}


# ---------------------------------------------------------------------------
# network: betweenness uses distance = 1/weight; corrected Gini
# ---------------------------------------------------------------------------

def test_betweenness_prefers_strong_ties():
    # A-B and B-C are strong, A-C weak: shortest A-C path by distance runs
    # through B, so B is the broker.
    edges = pd.DataFrame({"source": ["A", "B", "A"], "target": ["B", "C", "C"],
                          "weight": [10, 10, 1]})
    msgs = pd.DataFrame({"sender_code": ["A", "B", "C"]})
    rows = {r["sender_code"]: r for r in net.compute_node_metrics(msgs, edges, "p")}
    assert rows["B"]["betweenness_centrality"] == pytest.approx(1.0)
    assert rows["A"]["betweenness_centrality"] == 0.0


def test_gini_corrected_and_top_share():
    msgs = pd.DataFrame({"sender_code": ["A"] * 6 + ["B"] * 2})
    edges = pd.DataFrame({"source": ["A"], "target": ["B"], "weight": [1]})
    row = net.compute_node_metrics(msgs, edges, "p")[0]
    g = net.gini(np.array([6.0, 2.0]))
    assert row["gini_coefficient"] == pytest.approx(g, abs=1e-6)
    assert row["gini_corrected"] == pytest.approx(g * 2, abs=1e-6)
    assert row["top_sender_share"] == pytest.approx(0.75)
