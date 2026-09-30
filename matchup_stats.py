"""Side-by-side team stats for each game, grouped (Offense / Defense / Advanced), with which team has the edge in each.

Each game gets a JSON list in the column `matchup_stats`:
    [{"group": "Offense", "label": "Rushing yards / game", "away": "121.3", "home": "98.0", "edge": "away"}, ...]
An edge is only called when the gap is at least EDGE_SD of that stat's spread across teams; smaller gaps are "even".
Stats with no better direction (e.g. pace) get edge "".

Box-score stats are season to date (games before this week). If a team hasn't played yet this season, last season is used.
NFL: nflverse weekly team stats + scores.  College: CollegeFootballData season team stats (1 API call per run, cached).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_SD = 0.25          # a gap under a quarter of a standard deviation is called even

# (group, label, key, higher is better (True/False/None), format)
BOX = [
    ("Offense", "Points / game", "ppg", True, ".1f"),
    ("Offense", "Total yards / game", "ypg", True, ".1f"),
    ("Offense", "Yards / play", "ypp", True, ".2f"),
    ("Offense", "Passing yards / game", "pass_ypg", True, ".1f"),
    ("Offense", "Yards / pass attempt", "ypa", True, ".2f"),
    ("Offense", "Completion %", "cmp_pct", True, "pct"),
    ("Offense", "Rushing yards / game", "rush_ypg", True, ".1f"),
    ("Offense", "Yards / carry", "ypc", True, ".2f"),
    ("Offense", "Third-down conversion %", "third_pct", True, "pct"),
    ("Offense", "Turnovers / game", "to_pg", False, ".2f"),
    ("Offense", "Sacks allowed / game", "sacked_pg", False, ".2f"),
    ("Defense", "Points allowed / game", "ppg_a", False, ".1f"),
    ("Defense", "Yards allowed / game", "ypg_a", False, ".1f"),
    ("Defense", "Yards / play allowed", "ypp_a", False, ".2f"),
    ("Defense", "Passing yards allowed / game", "pass_ypg_a", False, ".1f"),
    ("Defense", "Yards / pass attempt allowed", "ypa_a", False, ".2f"),
    ("Defense", "Rushing yards allowed / game", "rush_ypg_a", False, ".1f"),
    ("Defense", "Yards / carry allowed", "ypc_a", False, ".2f"),
    ("Defense", "Third-down % allowed", "third_pct_a", False, "pct"),
    ("Defense", "Takeaways / game", "take_pg", True, ".2f"),
    ("Defense", "Sacks / game", "sacks_pg", True, ".2f"),
]

NFL_ADV = [
    ("Advanced", "Offense EPA / play", "r_off_epa", True, "+.3f"),
    ("Advanced", "Pass offense EPA / play", "r_off_epa_pass", True, "+.3f"),
    ("Advanced", "Rush offense EPA / play", "r_off_epa_rush", True, "+.3f"),
    ("Advanced", "Offense success rate", "r_off_sr", True, "pct"),
    ("Advanced", "Defense EPA / play allowed", "r_def_epa", False, "+.3f"),
    ("Advanced", "Pass defense EPA allowed", "r_def_epa_pass", False, "+.3f"),
    ("Advanced", "Rush defense EPA allowed", "r_def_epa_rush", False, "+.3f"),
    ("Advanced", "Defense success rate allowed", "r_def_sr", False, "pct"),
    ("Advanced", "Explosive plays / game", "r_explosive_plays", True, ".1f"),
    ("Advanced", "QB rating (EPA / dropback)", "qb_rating", True, "+.3f"),
    ("Advanced", "Days of rest", "rest", True, ".0f"),
]

CFB_ADV = [
    ("Advanced", "Offense PPA / play", "off_ppa", True, "+.3f"),
    ("Advanced", "Defense PPA / play allowed", "def_ppa", False, "+.3f"),
    ("Advanced", "Offense success rate", "off_sr", True, "pct"),
    ("Advanced", "Defense success rate allowed", "def_sr", False, "pct"),
    ("Advanced", "Offense explosiveness", "off_expl", True, ".2f"),
    ("Advanced", "Explosiveness allowed", "def_expl", False, ".2f"),
    ("Advanced", "Plays / game (pace)", "plays", None, ".1f"),
    ("Advanced", "Roster talent (247 composite)", "talent", True, ".0f"),
    ("Advanced", "Returning production", "returning", True, "pct"),
]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if f == "pct":
        return f"{v * 100:.1f}%"
    return format(v, f).replace("-", "−")


def _percentile(v, values, higher):
    """0-100 league rank of a value, where 100 is best (for lower-is-better stats the scale is flipped)."""
    if v is None or values is None or len(values) == 0:
        return None
    p = float(((values < v).mean() + 0.5 * (values == v).mean()) * 100)
    return round(100 - p) if higher is False else round(p)


def matchup_json(away: dict, home: dict, specs, sd: dict, dist: dict | None = None) -> str:
    rows = []
    dist = dist or {}
    for group, label, key, higher, f in specs:
        a, h = away.get(key), home.get(key)
        a = None if a is None or pd.isna(a) else float(a)
        h = None if h is None or pd.isna(h) else float(h)
        edge = ""
        if higher is not None and a is not None and h is not None:
            gap = h - a if higher else a - h            # > 0 means home is better
            edge = "even" if abs(gap) <= EDGE_SD * sd.get(key, 0.0) else ("home" if gap > 0 else "away")
        if a is None and h is None:
            continue                                  # stat not available for this sport/week
        vals = dist.get(key)
        rows.append({"group": group, "label": label, "away": _fmt(a, f), "home": _fmt(h, f), "edge": edge,
                     "ap": _percentile(a, vals, higher), "hp": _percentile(h, vals, higher), "dir": higher})
    return json.dumps(rows)


def _derive(t: pd.DataFrame) -> pd.DataFrame:
    """Per-game and per-play stats from season totals (one row per team)."""
    g = t["games"].replace(0, np.nan)
    d = pd.DataFrame(index=t.index)
    d["games"] = t["games"]
    d["ppg"], d["ppg_a"] = t["pts"] / g, t["pts_a"] / g
    for sfx in ("", "_a"):
        yds = t[f"pass_yds{sfx}"] + t[f"rush_yds{sfx}"]
        plays = t[f"dropbacks{sfx}"] + t[f"rush_att{sfx}"]
        d[f"ypg{sfx}"] = yds / g
        d[f"ypp{sfx}"] = yds / plays.replace(0, np.nan)
        d[f"pass_ypg{sfx}"] = t[f"pass_yds{sfx}"] / g
        d[f"ypa{sfx}"] = t[f"pass_yds{sfx}"] / t[f"dropbacks{sfx}"].replace(0, np.nan)
        d[f"rush_ypg{sfx}"] = t[f"rush_yds{sfx}"] / g
        d[f"ypc{sfx}"] = t[f"rush_yds{sfx}"] / t[f"rush_att{sfx}"].replace(0, np.nan)
        if f"third_att{sfx}" in t:
            d[f"third_pct{sfx}"] = t[f"third_conv{sfx}"] / t[f"third_att{sfx}"].replace(0, np.nan)
    d["cmp_pct"] = t["cmp"] / t["att"].replace(0, np.nan)
    d["to_pg"], d["take_pg"] = t["turnovers"] / g, t["takeaways"] / g
    d["sacked_pg"], d["sacks_pg"] = t["sacked"] / g, t["sacks"] / g
    return d


def _dist(table: pd.DataFrame, specs) -> dict:
    return {k: table[k].dropna().to_numpy(float) for _, _, k, _, _ in specs if k in table}


def _sd(table: pd.DataFrame, specs) -> dict:
    return {k: float(table[k].std()) if k in table and table[k].notna().sum() > 5 else 0.0 for _, _, k, _, _ in specs}


# ------------------------------------------------------------------------------------------------ NFL
def nfl_box(season: int, week: int) -> pd.DataFrame:
    import nflreadpy as nfl
    ts = nfl.load_team_stats([season - 1, season], summary_level="week").to_pandas()
    ts = ts[ts["season_type"] == "REG"]
    sched = nfl.load_schedules([season - 1, season]).to_pandas()
    pts = pd.concat([sched[["game_id", "home_team", "home_score", "away_score"]].set_axis(["game_id", "team", "pts", "pts_a"], axis=1),
                     sched[["game_id", "away_team", "away_score", "home_score"]].set_axis(["game_id", "team", "pts", "pts_a"], axis=1)])
    x = pd.DataFrame({
        "season": ts["season"], "week": ts["week"], "game_id": ts["game_id"], "team": ts["team"], "opp": ts["opponent_team"],
        "pass_yds": ts["passing_yards"] - ts["sack_yards_lost"], "dropbacks": ts["attempts"] + ts["sacks_suffered"],
        "cmp": ts["completions"], "att": ts["attempts"], "rush_yds": ts["rushing_yards"], "rush_att": ts["carries"],
        "sacked": ts["sacks_suffered"], "sacks": ts["def_sacks"],
        "turnovers": ts["passing_interceptions"] + ts["rushing_fumbles_lost"] + ts["receiving_fumbles_lost"] + ts["sack_fumbles_lost"],
        "takeaways": ts["def_interceptions"] + ts["fumble_recovery_opp"]})
    opp = x[["game_id", "team", "pass_yds", "dropbacks", "rush_yds", "rush_att"]].rename(
        columns={"team": "opp", "pass_yds": "pass_yds_a", "dropbacks": "dropbacks_a", "rush_yds": "rush_yds_a", "rush_att": "rush_att_a"})
    x = x.merge(opp, on=["game_id", "opp"], how="left").merge(pts, on=["game_id", "team"], how="left")
    cur = x[(x["season"] == season) & (x["week"] < week)]
    prev = x[x["season"] == season - 1]
    agg = lambda d: d.groupby("team").agg(games=("game_id", "nunique"), **{c: (c, "sum") for c in
          ["pts", "pts_a", "pass_yds", "dropbacks", "cmp", "att", "rush_yds", "rush_att", "sacked", "sacks", "turnovers",
           "takeaways", "pass_yds_a", "dropbacks_a", "rush_yds_a", "rush_att_a"]})
    t = agg(cur)
    last = agg(prev)
    t = pd.concat([t, last[~last.index.isin(t.index)]])       # teams with no games yet: last season
    return _derive(t)


def add_nfl(wk: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    season, week = int(wk["season"].iloc[0]), int(wk["week"].iloc[0])
    try:
        box = nfl_box(season, week)
    except Exception as e:                      # stats download failed: show advanced stats only
        print(f"  (box-score stats unavailable: {e})")
        box = pd.DataFrame()
    cur = games[games["season"] == season]
    adv_vals = {k: pd.concat([cur.get(f"h_{k}", pd.Series(dtype=float)), cur.get(f"a_{k}", pd.Series(dtype=float))]).dropna()
                for _, _, k, _, _ in NFL_ADV}
    adv_sd = {k: float(v.std()) for k, v in adv_vals.items()}
    sd = {**_sd(box, BOX), **adv_sd}
    dist = {**(_dist(box, BOX) if not box.empty else {}), **{k: v.to_numpy(float) for k, v in adv_vals.items()}}
    specs = (BOX if not box.empty else []) + NFL_ADV
    out = wk.copy()
    rows = []
    for _, r in out.iterrows():
        side = {}
        for s in ("away", "home"):
            t = r[f"{s}_team"]
            vals = box.loc[t].to_dict() if t in box.index else {}
            vals.update({k: r.get(f"{'a' if s == 'away' else 'h'}_{k}") for _, _, k, _, _ in NFL_ADV})
            side[s] = vals
        rows.append(matchup_json(side["away"], side["home"], specs, sd, dist))
    out["matchup_stats"] = rows
    return out


# ------------------------------------------------------------------------------------------------ College
CFB_MAP = {"games": "games", "netPassingYards": "pass_yds", "passAttempts": "att", "passCompletions": "cmp",
           "rushingYards": "rush_yds", "rushingAttempts": "rush_att", "turnovers": "turnovers", "turnoversOpponent": "takeaways",
           "sacks": "sacks", "sacksOpponent": "sacked", "thirdDowns": "third_att", "thirdDownConversions": "third_conv",
           "thirdDownsOpponent": "third_att_a", "thirdDownConversionsOpponent": "third_conv_a",
           "netPassingYardsOpponent": "pass_yds_a", "passAttemptsOpponent": "att_a", "rushingYardsOpponent": "rush_yds_a",
           "rushingAttemptsOpponent": "rush_att_a"}


def cfb_box(season: int, week: int, games: pd.DataFrame) -> pd.DataFrame:
    """College season-to-date totals through last week (CFBD, one cached call per season/week)."""
    import cfb_client
    cache = Path("cache/cfb") / f"season_stats_{season}_wk{week:02d}.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = pd.DataFrame(cfb_client.get("/stats/season", year=season, endWeek=max(week - 1, 1)))
        if not raw.empty:
            raw.to_parquet(cache)
    t = raw[raw["statName"].isin(CFB_MAP)].pivot_table(index="team", columns="statName", values="statValue", aggfunc="first")
    t = t.rename(columns=CFB_MAP).astype(float)
    # "sacks" in CFBD = sacks made by the team's defense; "sacksOpponent" = sacks the offense allowed
    t["dropbacks"], t["dropbacks_a"] = t["att"], t["att_a"]
    g = games[(games["season"] == season) & games["played"] & (games["week"] < week)]
    pts = pd.concat([g[["home_team", "home_score", "away_score"]].set_axis(["team", "pts", "pts_a"], axis=1),
                     g[["away_team", "away_score", "home_score"]].set_axis(["team", "pts", "pts_a"], axis=1)]).groupby("team").sum()
    t = t.join(pts, how="left")
    return _derive(t.fillna({"pts": 0, "pts_a": 0}))


def cfb_adv_table(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    g = games[(games["season"] == season) & games["played"] & (games["week"] < week)]
    rows = [pd.DataFrame({"team": g[f"{s}_team"], "off_ppa": g.get(f"{s}_off_ppa"), "def_ppa": g.get(f"{s}_def_ppa"),
                          "off_sr": g.get(f"{s}_off_sr"), "def_sr": g.get(f"{s}_def_sr"), "off_expl": g.get(f"{s}_off_expl"),
                          "def_expl": g.get(f"{s}_def_expl"), "plays": g.get(f"{s}_off_plays")}) for s in ("home", "away")]
    t = pd.concat(rows).groupby("team").mean(numeric_only=True)
    cur = games[games["season"] == season]
    pre = pd.concat([cur[[f"{s}_team", f"{s}_talent", f"{s}_returning"]].set_axis(["team", "talent", "returning"], axis=1)
                     for s in ("home", "away") if f"{s}_talent" in cur]).groupby("team").first()
    return t.join(pre, how="outer")


def add_cfb(wk: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    season, week = int(wk["season"].iloc[0]), int(wk["week"].iloc[0])
    try:
        box = cfb_box(season, week, games) if week > 1 else pd.DataFrame()
    except Exception as e:
        print(f"  (college box-score stats unavailable: {e})")
        box = pd.DataFrame()
    adv = cfb_adv_table(games, season, week)
    fbs = set(games.loc[games["home_fbs"] == 1, "home_team"]) | set(games.loc[games["away_fbs"] == 1, "away_team"])
    table = box.join(adv, how="outer") if not box.empty else adv
    sd = _sd(table[table.index.isin(fbs)], (BOX if not box.empty else []) + CFB_ADV)
    specs = (BOX if not box.empty else []) + CFB_ADV
    dist = _dist(table[table.index.isin(fbs)], specs)
    get = lambda team: (table.loc[team].to_dict() if team in table.index else {})
    out = wk.copy()
    out["matchup_stats"] = [matchup_json(get(r["away_team"]), get(r["home_team"]), specs, sd, dist) for _, r in out.iterrows()]
    return out
