"""Tests for src/04_network.py — synthetic data only, no file I/O."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "04_network.py"
spec = importlib.util.spec_from_file_location("network_module", SRC)
net = importlib.util.module_from_spec(spec)
sys.modules["network_module"] = net
spec.loader.exec_module(net)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_messages(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "sender_code", "message_type"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "message_type" not in df.columns:
        df["message_type"] = "text"
    return df


# ---------------------------------------------------------------------------
# build_coactivity_edges
# ---------------------------------------------------------------------------

class TestBuildCoactivityEdges:
    def test_two_senders_in_window_creates_edge(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P002"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        assert len(edges) == 1
        assert set(edges.iloc[0][["source", "target"]]) == {"P001", "P002"}
        assert edges.iloc[0]["weight"] > 0

    def test_senders_outside_window_no_edge(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 11:00", "sender_code": "P002"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        assert len(edges) == 0

    def test_self_loops_excluded(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P001"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        assert len(edges) == 0

    def test_weight_increases_with_more_co_occurrences(self):
        # P001 and P002 co-occur in multiple windows
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": "P002"},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:15", "sender_code": "P002"},
            {"timestamp": "2024-01-01 14:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 14:05", "sender_code": "P002"},
        ])
        edges_long = net.build_coactivity_edges(df, window_minutes=30)
        edges_short = net.build_coactivity_edges(df, window_minutes=5)
        # Long window catches more co-occurrences
        assert edges_long.iloc[0]["weight"] >= edges_short.iloc[0]["weight"]

    def test_three_senders_creates_three_edges(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": "P002"},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P003"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        pairs = {tuple(sorted([r["source"], r["target"]])) for _, r in edges.iterrows()}
        assert pairs == {("P001", "P002"), ("P001", "P003"), ("P002", "P003")}

    def test_nan_senders_skipped(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": None},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P002"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        for _, e in edges.iterrows():
            assert e["source"] != "nan" and e["target"] != "nan"

    def test_empty_dataframe_returns_empty_edges(self):
        df = make_messages([])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# gini
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_equality_is_zero(self):
        assert net.gini(np.array([10, 10, 10, 10])) == pytest.approx(0.0, abs=1e-6)

    def test_perfect_inequality_approaches_one(self):
        # One person sends everything
        result = net.gini(np.array([0, 0, 0, 100]))
        assert result > 0.7

    def test_empty_array_returns_zero(self):
        assert net.gini(np.array([])) == 0.0

    def test_all_zeros_returns_zero(self):
        assert net.gini(np.array([0, 0, 0])) == 0.0

    def test_two_equal_values(self):
        assert net.gini(np.array([5, 5])) == pytest.approx(0.0, abs=1e-6)

    def test_value_between_zero_and_one(self):
        result = net.gini(np.array([1, 2, 3, 10, 20]))
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# split_by_year
# ---------------------------------------------------------------------------

class TestSplitByYear:
    def test_splits_into_correct_years(self):
        df = make_messages([
            {"timestamp": "2023-06-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-03-01 10:00", "sender_code": "P002"},
            {"timestamp": "2024-11-01 10:00", "sender_code": "P003"},
        ])
        splits = net.split_by_year(df)
        assert set(splits.keys()) == {"2023", "2024"}
        assert len(splits["2023"]) == 1
        assert len(splits["2024"]) == 2

    def test_year_column_not_in_output(self):
        df = make_messages([
            {"timestamp": "2023-01-01 10:00", "sender_code": "P001"},
        ])
        splits = net.split_by_year(df)
        assert "_year" not in splits["2023"].columns


# ---------------------------------------------------------------------------
# compute_node_metrics
# ---------------------------------------------------------------------------

class TestComputeNodeMetrics:
    def test_returns_row_per_node(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": "P002"},
            {"timestamp": "2024-01-01 10:10", "sender_code": "P003"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        rows = net.compute_node_metrics(df, edges, label="2024")
        assert len(rows) == 3
        codes = {r["sender_code"] for r in rows}
        assert codes == {"P001", "P002", "P003"}

    def test_gini_consistent_across_nodes_in_same_slice(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
            {"timestamp": "2024-01-01 10:05", "sender_code": "P002"},
        ])
        edges = net.build_coactivity_edges(df, window_minutes=30)
        rows = net.compute_node_metrics(df, edges, label="2024")
        ginis = {r["gini_coefficient"] for r in rows}
        assert len(ginis) == 1  # same value for all nodes in a slice

    def test_empty_edges_returns_empty(self):
        df = make_messages([
            {"timestamp": "2024-01-01 10:00", "sender_code": "P001"},
        ])
        edges = pd.DataFrame(columns=["source", "target", "weight"])
        rows = net.compute_node_metrics(df, edges, label="2024")
        assert rows == []
