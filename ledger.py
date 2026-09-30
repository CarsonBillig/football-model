"""Pick history (NFL: output/ledger.csv, college: output/cfb/ledger.csv): every game's picks are logged BEFORE kickoff, then graded after the final whistle.

One row per game. Re-running predict.py before kickoff refreshes a game's row with the latest lines; once a game
has kicked off its row is locked, so the record only ever reflects picks you could actually have made.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEDGER = Path("output/ledger.csv")
COLUMNS = [
    # when / what
    "logged_at", "season", "week", "game_id", "kickoff", "away_team", "home_team",
    # projections
    "proj_away", "proj_home", "fair_margin", "fair_total", "fair_spread", "fair_total_line", "fund_margin", "fund_total", "mkt_margin", "mkt_total",
    "p_home_win", "p_home_win_raw", "mkt_p_home_win",
    # lines at the time of logging (home-favored-positive spread) and at the open
    "open_spread", "spread", "home_spread_odds", "away_spread_odds", "open_total", "total_line", "over_odds",
    "under_odds", "home_ml", "away_ml",
    # picks
    "su_pick", "su_prob", "ats_pick", "ats_line", "ats_odds", "ats_prob", "ats_prob_np", "ats_ev", "ats_stake", "value_ats",
    "tot_pick", "tot_prob", "tot_prob_np", "tot_odds", "tot_ev", "tot_stake", "value_tot",
    "ml_pick", "ml_odds", "ml_prob", "ml_ev", "ml_stake", "value_ml",
    # model-only leans (pure model, no market input)
    "mo_proj_home", "mo_proj_away", "mo_fair_spread", "mo_fair_total_line", "mo_p_home_win", "mo_su_pick", "mo_su_prob",
    "mo_ats_pick", "mo_ats_line", "mo_ats_prob_np", "mo_tot_pick", "mo_tot_prob_np",
    "mo_su_hit", "mo_ats_hit", "mo_tot_hit",
    # grading
    "home_score", "away_score", "su_out", "ats_out", "tot_out", "ml_out", "mo_su_out", "mo_ats_out", "mo_tot_out",
    "ats_profit", "tot_profit", "ml_profit", "close_spread", "close_total", "ats_clv", "tot_clv",
]


CFB_LEDGER = Path("output/cfb/ledger.csv")


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or LEDGER
    if path.exists():
        return pd.read_csv(path).reindex(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame, path: Path | None = None):
    path = path or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).sort_values(["season", "week", "kickoff", "game_id"]).to_csv(path, index=False)


def record(new: pd.DataFrame, now: pd.Timestamp, path: Path | None = None):
    """Insert/refresh picks for games that have not kicked off. Locked rows (kicked off or graded) never change."""
    old = load(path)
    kick = pd.to_datetime(old["kickoff"], utc=True, errors="coerce")
    locked = old["su_out"].notna() | (kick <= now)
    keep = old[locked | ~old["game_id"].isin(new["game_id"])]
    fresh = new[~new["game_id"].isin(keep["game_id"])]
    save(pd.concat([keep, fresh.reindex(columns=COLUMNS)], ignore_index=True), path)
    return len(fresh)
