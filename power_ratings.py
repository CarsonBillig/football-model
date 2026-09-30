"""Pre-game ridge power ratings (leakage-safe).

For every week, a small ridge regression is fit on games from EARLIER weeks only:

    target  ≈  HFA * (not neutral)  +  R[home]  -  R[away]

with recency weights (half-life in weeks) and an extra discount on previous seasons, so ratings regress toward
average over the offseason instead of carrying 2023 form straight into 2024.

Two ratings are produced:
  results  - target = final home margin (what teams actually did)
  market   - target = closing spread_line of past games (what the market thought of them, ex current week).
             The CURRENT week's line is never used, so this is a legitimate pre-game input.
Totals get the same treatment with target = total points / total_line and an additive model
(total ≈ league level + O[home] + O[away]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _week_index(season, week):
    return np.asarray(season) * 25 + np.asarray(week)      # 25 > max week incl. playoffs


def _fit_week(past, teams, target, additive, hl_weeks, season_carry, alpha, now_idx, now_season):
    ti = {t: i for i, t in enumerate(teams)}
    n = len(past)
    age = now_idx - _week_index(past["season"], past["week"])
    w = 0.5 ** (age / hl_weeks) * season_carry ** (now_season - past["season"].values)
    X = np.zeros((n, len(teams)))
    X[np.arange(n), past["home_team"].map(ti).values] = 1.0
    X[np.arange(n), past["away_team"].map(ti).values] = 1.0 if additive else -1.0
    if additive:
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X, past[target], sample_weight=w)
        return m.coef_, float(m.intercept_), ti
    hfa_col = (past["location"] != "Neutral").astype(float).values
    m = Ridge(alpha=alpha, fit_intercept=False).fit(np.column_stack([X, hfa_col]), past[target], sample_weight=w)
    return m.coef_[:-1], float(m.coef_[-1]), ti


def pregame_rating(games: pd.DataFrame, target: str, additive: bool = False, hl_weeks: float = 6.0,
                   season_carry: float = 0.5, alpha: float = 2.0, lookback_seasons: int = 2,
                   min_games: int = 100) -> pd.Series:
    """Predicted value of `target` for each game using only games from earlier weeks.

    additive=False -> home-minus-away model (margins/spreads); additive=True -> home-plus-away (totals)."""
    g = games[games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB", "POST"])]      # POST = college postseason
    g = g.assign(_idx=_week_index(g["season"], g["week"]).astype(int))
    out = pd.Series(np.nan, index=games.index)
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    for (season, idx), cur in g.groupby(["season", "_idx"], sort=True):
        past = g[(g["_idx"] < idx) & g[target].notna() & (g["season"] >= season - lookback_seasons)]
        if len(past) < min_games:
            continue
        r, level, ti = _fit_week(past, teams, target, additive, hl_weeks, season_carry, alpha, idx, season)
        h, a = r[cur["home_team"].map(ti).values], r[cur["away_team"].map(ti).values]
        if additive:
            out.loc[cur.index] = level + h + a
        else:
            out.loc[cur.index] = h - a + level * (cur["location"] != "Neutral").values
    return out


def add_power_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Adds res_prior / mkt_prior (margins) and res_total_prior / mkt_total_prior (totals)."""
    g = games.copy()
    g["res_prior"] = pregame_rating(g, "result", hl_weeks=10, season_carry=0.5, alpha=5.0)
    g["mkt_prior"] = pregame_rating(g, "spread_line", hl_weeks=4, season_carry=0.6, alpha=1.0)
    g["res_total_prior"] = pregame_rating(g, "total", additive=True, hl_weeks=10, season_carry=0.5, alpha=5.0)
    g["mkt_total_prior"] = pregame_rating(g, "total_line", additive=True, hl_weeks=4, season_carry=0.6, alpha=1.0)
    return g
