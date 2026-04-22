"""Triangulate engagement, network, sentiment, and fidelity metrics by phase."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalise_phase(s: pd.Series) -> pd.Series:
    """Coerce phase to clean string: '2023.0' → '2023', 'full_period' → 'full_period'."""
    def _clean(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v)
    return s.map(_clean)


def summarize_by_phase(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "phase" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["phase"] = _normalise_phase(df["phase"])
    numeric = [c for c in df.select_dtypes("number").columns if c != "phase"]
    if len(numeric) == 0:
        return pd.DataFrame({"phase": df["phase"].dropna().unique()})
    g = df.groupby("phase")[numeric].mean().reset_index()
    g.columns = ["phase"] + [f"{prefix}_{c}" for c in g.columns[1:]]
    return g


def main() -> None:
    settings = load_settings()
    paths = settings["paths"]

    eng = pd.read_csv(paths["engagement_metrics"])
    net = pd.read_csv(paths["network_metrics"])
    sen = pd.read_csv(paths["sentiment_metrics"])
    fid = pd.read_csv(paths["fidelity_metrics"])

    # Collect all normalised phase values across datasets
    all_phases: set = set()
    for df in (eng, net, sen, fid):
        if "phase" in df.columns:
            all_phases.update(_normalise_phase(df["phase"].dropna()).unique())

    merged = pd.DataFrame({"phase": sorted(all_phases)})
    for df, prefix in [(eng, "eng"), (net, "net"), (sen, "sen"), (fid, "fid")]:
        s = summarize_by_phase(df, prefix)
        if not s.empty:
            merged = merged.merge(s, on="phase", how="left")

    milestones = settings.get("milestones", {})
    for name, date in milestones.items():
        merged[f"milestone_{name}"] = date

    out = Path(paths["implementation_summary"])
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[07_triangulation] Wrote {out}")


if __name__ == "__main__":
    main()
