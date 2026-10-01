"""NFL injury features: how many regular starters each team is missing, from official injury reports.

A "regular starter" played at least 60% of his unit's snaps on average over his previous four games (pre-game only,
earlier seasons count early in the year). Each one listed on that week's injury report counts as:
Out 1.0, Doubtful 0.9, Questionable 0.25 (most questionable players end up playing).
QBs are left out here: the QB rating feature already uses the listed starter.

Data: nflverse injury reports (2009+) and snap counts (2012+).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.9, "Questionable": 0.25}
SKIP_POS = {"QB", "K", "P", "LS"}
STARTER_PCT = 0.6


def _norm(name: str) -> str:
    n = re.sub(r"[^a-z ]", "", str(name).lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()


def _starters(seasons) -> pd.DataFrame:
    """Regular starters entering each (season, week): player key, team, unit."""
    import nflreadpy as nfl
    sc = nfl.load_snap_counts(list(seasons)).to_pandas()
    sc = sc[~sc["position"].isin(SKIP_POS)].copy()
    sc["key"] = sc["player"].map(_norm)
    sc["pct"] = sc[["offense_pct", "defense_pct"]].max(axis=1)
    sc["unit"] = np.where(sc["offense_pct"] >= sc["defense_pct"], "off", "def")
    sc = sc.sort_values(["key", "team", "season", "week"])
    # average share over the previous four games, shifted so a game never counts toward its own week
    sc["prev_pct"] = sc.groupby(["key", "team"])["pct"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    return sc[["season", "week", "team", "key", "unit", "prev_pct"]]


def team_week_injuries(seasons) -> pd.DataFrame:
    """One row per (season, week, team): off_out, def_out (weighted counts of regular starters on the report)."""
    import nflreadpy as nfl
    seasons = [int(s) for s in seasons if int(s) >= 2012]
    inj = nfl.load_injuries(seasons).to_pandas()
    inj = inj[inj["report_status"].isin(STATUS_WEIGHT) & ~inj["position"].isin(SKIP_POS)].copy()
    inj["key"] = inj["full_name"].map(_norm)
    inj["w"] = inj["report_status"].map(STATUS_WEIGHT)
    st = _starters(seasons)
    # a player's status for week w is matched to his starter share entering week w (his latest game up to that week,
    # last season's games count early in the year)
    st["t"] = (st["season"] * 100 + st["week"]).astype("int64")
    inj["t"] = (inj["season"] * 100 + inj["week"]).astype("int64")
    st = st.sort_values("t")
    inj = inj.sort_values("t")
    m = pd.merge_asof(inj, st.drop(columns=["season", "week"]), on="t", by=["team", "key"],
                      allow_exact_matches=True, direction="backward")
    m = m[m["prev_pct"] >= STARTER_PCT]
    out = m.pivot_table(index=["season", "week", "team"], columns="unit", values="w", aggfunc="sum", fill_value=0)
    out = out.rename(columns={"off": "off_out", "def": "def_out"}).reset_index()
    for c in ("off_out", "def_out"):
        if c not in out:
            out[c] = 0.0
    return out[["season", "week", "team", "off_out", "def_out"]]


def add_injury_features(g: pd.DataFrame) -> pd.DataFrame:
    """Adds h_/a_ off_out and def_out plus margin (diff) and total (sum) features. Missing data = 0 (nobody out)."""
    try:
        t = team_week_injuries(sorted(int(s) for s in g["season"].dropna().unique()))
    except Exception as e:                       # data unavailable: neutral features, model still runs
        print(f"  (injury reports unavailable: {e})")
        t = pd.DataFrame(columns=["season", "week", "team", "off_out", "def_out"])
    out = g.copy()
    t = t.astype({"season": "float64", "week": "float64"})          # match the schedule's key types
    out["season"], out["week"] = out["season"].astype("float64"), out["week"].astype("float64")
    for side, pre in (("home", "h"), ("away", "a")):
        x = t.rename(columns={"team": f"{side}_team", "off_out": f"{pre}_off_out", "def_out": f"{pre}_def_out"})
        out = out.merge(x, on=["season", "week", f"{side}_team"], how="left")
        out[[f"{pre}_off_out", f"{pre}_def_out"]] = out[[f"{pre}_off_out", f"{pre}_def_out"]].fillna(0.0)
    out["inj_off_diff"] = out["h_off_out"] - out["a_off_out"]
    out["inj_def_diff"] = out["h_def_out"] - out["a_def_out"]
    out["inj_off_sum"] = out["h_off_out"] + out["a_off_out"]
    out["inj_def_sum"] = out["h_def_out"] + out["a_def_out"]
    return out
