"""Pick history (NFL: output/ledger.csv, college: output/cfb/ledger.csv): every game's picks are logged BEFORE kickoff, then graded after the final whistle.

One row per game. Re-running predict.py before kickoff refreshes a game's row with the latest lines; once a game
has kicked off its row is locked, so the record only ever reflects picks you could actually have made.

The FIRST spread and total pick posted for each game (early_* columns) is kept as well and never refreshed: our edge
in testing came from betting early, so the early picks and their closing-line value are graded separately.
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
    # first posted picks (kept from the first run that logged the game)
    "early_logged_at", "early_spread", "early_ats_pick", "early_total", "early_tot_pick",
    "early_ats_out", "early_tot_out", "early_ats_clv", "early_tot_clv",
]
EARLY = {"early_logged_at": "logged_at", "early_spread": "spread", "early_ats_pick": "ats_pick",
         "early_total": "total_line", "early_tot_pick": "tot_pick"}


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
    fresh = new[~new["game_id"].isin(keep["game_id"])].reindex(columns=COLUMNS)
    # first posted picks: carried over from the earlier row if there is one, otherwise this run's picks
    prev = old[~locked].drop_duplicates("game_id", keep="last").set_index("game_id")
    for early, now_col in EARLY.items():
        first = fresh["game_id"].map(prev[early]) if len(prev) else pd.Series(index=fresh.index, dtype=object)
        legacy = fresh["game_id"].map(prev[now_col]) if len(prev) else first      # rows logged before early_* existed
        fresh[early] = first.where(first.notna(), legacy).where(lambda s: s.notna(), fresh[now_col]).astype(object)
    save(pd.concat([keep, fresh], ignore_index=True), path)
    return len(fresh)
