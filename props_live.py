"""Weekly NFL player-prop picks: an EXPERIMENT, tracked as a paper trade.

Only rushing-yards unders. In the 2024-26 backtest our player projections did not beat the book's lines, but
rushing-yards unders made money on 2025-26 (561 bets, 55.8%), mostly because rushing unders hit more often
market-wide (50.7% in 2024, 54.7% in 2025, 58.3% in 2026). This tracks whether that holds up live.

Each run: grade finished picks, then list every game's top 3 rushing unders for the coming week from DraftKings'
lines (ESPN feed), ranked by our calibrated chance of the under, which is shown as the confidence. A "pick" is an under
where our projection is below the line; a "lean" is riskier: our projection is at or above it. Backtest, 2025-26:
picks 57.3% (309 bets), leans 55.0% (320 bets); both about 51% in 2024. Picks lock at kickoff.
Ledger: output/props_ledger.csv. DraftKings' prices aren't in the feed: picks assume -110.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import props_backtest as pb
import props_data
import props_model as pm

MARKET = "rush_yds"
PER_GAME = 3
LEDGER = Path("output/props_ledger.csv")
COLS = ["logged_at", "season", "week", "game_id", "kickoff", "away", "home", "player_id", "player", "team", "market",
        "side", "tier", "line", "open_line", "proj", "p_side", "edge", "actual", "result"]


def load() -> pd.DataFrame:
    return pd.read_csv(LEDGER).reindex(columns=COLS) if LEDGER.exists() else pd.DataFrame(columns=COLS)


def calibration():
    """Logistic fit of 'went over' on (projection vs line, book's price), on every graded rushing-yards line so far."""
    from sklearn.linear_model import LogisticRegression
    g = pd.read_parquet("cache/props_graded.parquet")
    g = g[(g["market"] == MARKET) & (g["when"] == "now") & (g["actual"] != g["line"])]
    return LogisticRegression(C=1e6).fit(pb.calib_features(g), (g["actual"] > g["line"]).astype(int))


def week_lines(season: int, week: int) -> pd.DataFrame:
    """DraftKings' rushing-yards lines for the week's games that haven't started."""
    rows = []
    for ev in props_data._get(props_data.SCOREBOARD, dates=season, seasontype=2, week=week).get("events", []):
        if ev.get("status", {}).get("type", {}).get("state") != "pre":
            continue
        d = props_data.event_props(ev["id"])
        if d.empty:
            continue
        d = d[(d["market"] == MARKET) & (d["side"] == "under")].drop_duplicates(["espn_athlete_id"])
        rows.append(d.assign(kickoff=ev.get("date")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_picks(season: int, week: int) -> pd.DataFrame:
    import nflreadpy as nfl
    lines = week_lines(season, week)
    if lines.empty:
        return pd.DataFrame(columns=COLS)
    players = nfl.load_players().to_pandas().dropna(subset=["espn_id"])
    players["espn_athlete_id"] = players["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    lines = lines.merge(players[["espn_athlete_id", "gsis_id", "display_name", "position", "latest_team"]], on="espn_athlete_id")
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[(sched["week"] == week) & (sched["game_type"] == "REG")]
    games = pd.concat([sched.assign(team=sched["home_team"], opponent_team=sched["away_team"]),
                       sched.assign(team=sched["away_team"], opponent_team=sched["home_team"])])
    up = lines.merge(games[["team", "opponent_team", "game_id", "home_team", "away_team"]],
                     left_on="latest_team", right_on="team")
    upcoming = up.rename(columns={"gsis_id": "player_id", "display_name": "player_display_name"})[
        ["player_id", "player_display_name", "position", "team", "opponent_team", "game_id"]].assign(season=season, week=week)
    pg = pm.player_games(range(2018, season + 1), upcoming=upcoming)
    stat, vol = pm.MARKETS[MARKET][:2]
    role = pm.role_players(pg, MARKET)
    model = pm.fit(role[role["played"] & (role[vol] > 0)], stat)
    x = role[~role["played"]].merge(up[["gsis_id", "line", "open_line", "kickoff"]],
                                    left_on="player_id", right_on="gsis_id")
    if x.empty:
        return pd.DataFrame(columns=COLS)
    x["proj"] = np.clip(model.predict(x[pm.FEATURES]), 0.5, None)
    x = x[(x["proj"] / x["line"] - 1).abs() <= pb.MAX_GAP].assign(over_odds=-110.0, under_odds=-110.0)
    x["p_side"] = 1 - calibration().predict_proba(pb.calib_features(x))[:, 1]
    x["edge"] = x["p_side"] - 0.5                       # an even -110 / -110 line: no-margin chance 50%
    x["tier"] = np.where(x["proj"] < x["line"], "pick", "lean")
    x = x.sort_values("edge", ascending=False).groupby("game_id").head(PER_GAME)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return pd.DataFrame({"logged_at": now, "season": season, "week": week, "game_id": x["game_id"], "kickoff": x["kickoff"],
                         "away": x["away_team"], "home": x["home_team"], "player_id": x["player_id"],
                         "player": x["player_display_name"], "team": x["team"], "market": MARKET, "side": "under",
                         "tier": x["tier"],
                         "line": x["line"], "open_line": x["open_line"], "proj": x["proj"].round(1),
                         "p_side": x["p_side"].round(3), "edge": x["edge"].round(3)}).reindex(columns=COLS)


def record(new: pd.DataFrame):
    """Replace picks for games that haven't kicked off; games that have started stay locked."""
    old = load()
    now = pd.Timestamp.now(tz="UTC")
    locked = old["result"].notna() | (pd.to_datetime(old["kickoff"], utc=True) <= now)
    keep = old[locked | ~old["game_id"].isin(new["game_id"])]
    LEDGER.parent.mkdir(exist_ok=True)
    pd.concat([keep, new], ignore_index=True).sort_values(["season", "week", "kickoff", "game_id"]).to_csv(LEDGER, index=False)


def grade():
    """Fill in rushing yards for finished games: under wins below the line; a player who didn't play is void."""
    import nflreadpy as nfl
    led = load()
    todo = led["result"].isna() & (pd.to_datetime(led["kickoff"], utc=True) < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5))
    if not todo.any():
        return led
    seasons = sorted(int(s) for s in led.loc[todo, "season"].unique())
    ps = nfl.load_player_stats(seasons).to_pandas().set_index(["player_id", "season", "week"])["rushing_yards"]
    done = set(nfl.load_schedules(seasons).to_pandas().dropna(subset=["result"])["game_id"])
    led["result"] = led["result"].astype(object)
    for i in led.index[todo]:
        key = (led.at[i, "player_id"], int(led.at[i, "season"]), int(led.at[i, "week"]))
        if key in ps.index:
            a = float(ps.loc[key]); led.at[i, "actual"] = a
            led.at[i, "result"] = "Win" if a < led.at[i, "line"] else "Loss" if a > led.at[i, "line"] else "Push"
        elif led.at[i, "game_id"] in done:
            led.at[i, "result"] = "Void"
    led.to_csv(LEDGER, index=False)
    return led


def run(season: int, week: int):
    led = grade()
    g = led[led["result"].isin(["Win", "Loss"])]
    for tier in ("pick", "lean"):
        t = g[g["tier"] == tier]
        print(f"Prop experiment, rushing-yard unders ({tier}s, paper trade): {int((t.result == 'Win').sum())}-{int((t.result == 'Loss').sum())}")
    picks = make_picks(season, week)
    record(picks)
    print(f"{len(picks)} prop picks for {season} week {week} logged to {LEDGER}")
    return picks
