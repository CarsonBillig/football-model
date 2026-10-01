"""Line snapshots: every run appends each upcoming game's current lines, prices and model numbers with a timestamp.

Over a season this builds our own record of how lines move from early in the week to kickoff, at the times we actually
run the model. That is the data needed to test (and later train) the "bet early" edge honestly; historical
"opening line" feeds are often look-ahead lines posted before the previous week's games.

NFL: output/line_snapshots.csv    College: output/cfb/line_snapshots.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = ["snapshot_at", "season", "week", "game_id", "kickoff", "away_team", "home_team", "spread", "home_spread_odds",
        "away_spread_odds", "total_line", "over_odds", "under_odds", "home_ml", "away_ml", "open_spread", "open_total",
        "fund_margin", "fund_total", "fair_spread", "fair_total_line", "mo_fair_spread", "mo_fair_total_line", "p_home_win"]


def save(df: pd.DataFrame, path: Path, now: pd.Timestamp) -> int:
    """Append one row per game that hasn't kicked off yet. Returns rows written."""
    kick = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    up = df[kick > now].copy()
    if up.empty:
        return 0
    up["snapshot_at"] = now.isoformat(timespec="seconds")
    up = up.reindex(columns=COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    up.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(up)
