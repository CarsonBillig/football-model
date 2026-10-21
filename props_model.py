"""NFL player prop projections (passing yards, rushing yards, receiving yards, receptions) and their backtest.

A player's projection comes from his recent volume and efficiency (attempts / carries / targets and yards per
each, recent games weighted more), his team's recent play volume, what the opponent has been allowing, and the
game's expected scoring from the spread and total (teams expected to trail throw more). Everything is pre-game:
each game's own stats never feed its own projection. One model per market, trained only on earlier seasons.

The chance of going over a line comes from how actual results have spread around past projections in that
market (the ratio actual / projection), so the long right tail of yardage is handled without assuming a bell curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HALFLIFE = 4.0                      # games
MARKETS = {  # market -> (stat column, volume column, position groups, pre-game role: minimum recent volume per game)
    "pass_yds": ("passing_yards", "attempts", {"QB"}, 20),
    "rush_yds": ("rushing_yards", "carries", {"QB", "RB"}, 4),
    "rec_yds": ("receiving_yards", "targets", {"WR", "TE", "RB"}, 3),
    "receptions": ("receptions", "targets", {"WR", "TE", "RB"}, 3),
}


def _ewm_prior(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    """Recency-weighted average of each column over a group's EARLIER games (the current game is excluded)."""
    g = df.groupby(by, sort=False)
    return pd.DataFrame({f"e_{c}": g[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=1).mean())
                         for c in cols}, index=df.index)


def player_games(seasons, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player-game with pre-game features and the actual stats.

    upcoming: games not played yet (player_id, player_display_name, position, team, opponent_team, game_id, season,
    week). They get the same pre-game features from all earlier games; their stats stay empty (`played` is False)."""
    import nflreadpy as nfl
    ps = nfl.load_player_stats(list(seasons)).to_pandas()
    ps = ps[ps["season_type"] == "REG"].copy()
    ps["played"] = True
    if upcoming is not None and len(upcoming):
        ps = pd.concat([ps, upcoming.assign(season_type="REG", played=False)], ignore_index=True)
    sched = nfl.load_schedules(list(seasons)).to_pandas()[["game_id", "home_team", "away_team", "spread_line", "total_line", "gameday"]]
    stats = ["attempts", "completions", "passing_yards", "carries", "rushing_yards", "targets", "receptions", "receiving_yards"]
    ps[stats] = ps[stats].astype(float).fillna(0)

    # team volume per game and what each defense allowed (sums over the players who recorded stats)
    team = ps.groupby(["game_id", "team", "opponent_team"], as_index=False)[stats].sum()
    team = team.merge(sched[["game_id", "gameday"]], on="game_id").sort_values(["gameday", "game_id"])
    tv = team[["game_id", "team"]].join(_ewm_prior(team, "team", ["attempts", "carries", "targets"]).add_prefix("team_"))
    allowed = team.rename(columns={"team": "off", "opponent_team": "defense"})
    al = allowed[["game_id", "defense"]].join(_ewm_prior(allowed, "defense", ["passing_yards", "rushing_yards", "receptions", "attempts", "carries"]).add_prefix("opp_"))

    ps = ps.merge(sched, on="game_id", how="left").sort_values(["gameday", "game_id"])
    ps = ps.join(_ewm_prior(ps, "player_id", stats))
    ps["games_before"] = ps.groupby("player_id").cumcount()
    ps = ps.merge(tv, on=["game_id", "team"], how="left").merge(al.rename(columns={"defense": "opponent_team"}), on=["game_id", "opponent_team"], how="left")
    home = ps["team"] == ps["home_team"]
    spread = np.where(home, ps["spread_line"], -ps["spread_line"])              # + = this team favored
    ps["team_spread"] = spread
    ps["implied_pts"] = (ps["total_line"] + spread) / 2
    ps["is_home"] = home.astype(int)
    for vol in ("attempts", "carries", "targets"):
        ps[f"share_{vol}"] = ps[f"e_{vol}"] / ps[f"team_e_{vol}"].replace(0, np.nan)
    return ps


FEATURES = ["e_attempts", "e_completions", "e_passing_yards", "e_carries", "e_rushing_yards", "e_targets", "e_receptions",
            "e_receiving_yards", "share_attempts", "share_carries", "share_targets", "team_e_attempts", "team_e_carries",
            "team_e_targets", "opp_e_passing_yards", "opp_e_rushing_yards", "opp_e_receptions", "opp_e_attempts",
            "opp_e_carries", "team_spread", "implied_pts", "is_home", "games_before"]


def fit(train: pd.DataFrame, stat: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=7).fit(train[FEATURES], train[stat])


def role_players(pg: pd.DataFrame, market: str) -> pd.DataFrame:
    """Players with a real role going into the game (recent volume, known before kickoff), like the ones books post
    lines for."""
    stat, vol, positions, role = MARKETS[market]
    return pg[pg["position"].isin(positions) & (pg["games_before"] >= 2) & (pg[f"e_{vol}"] >= role)]


def walk_forward(pg: pd.DataFrame, market: str, first_test: int) -> pd.DataFrame:
    """Projections for every season from first_test on, each from a model trained on earlier seasons only.
    Adds `proj` and the training-residual ratios used for over/under chances (column `ratios`, one array per season)."""
    stat, vol, _, _ = MARKETS[market]
    d = role_players(pg, market).copy()
    out = []
    for s in sorted(d["season"].unique()):
        if s < first_test:
            continue
        tr = d[(d["season"] < s) & (d[vol] > 0)]                     # train on games the player actually featured in
        te = d[d["season"] == s].copy()
        m = fit(tr, stat)
        te["proj"] = np.clip(m.predict(te[FEATURES]), 0.5, None)
        fit = np.clip(m.predict(tr[FEATURES]), 0.5, None)
        te.attrs["ratios"] = (tr[stat].to_numpy() / fit)
        te["market"] = market
        out.append((s, te, np.sort(tr[stat].to_numpy() / fit)))
    return out


def p_over(proj, line, ratios_sorted) -> np.ndarray:
    """Chance actual > line, from the empirical distribution of actual / projection in training."""
    thr = np.asarray(line, float) / np.asarray(proj, float)
    return 1 - np.searchsorted(ratios_sorted, thr, side="right") / len(ratios_sorted)
