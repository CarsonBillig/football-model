"""ESPN public JSON feeds: this week's board (open + current lines, TV, venue, records, logos) and historical odds.

Only unauthenticated public endpoints are used. Historical odds are cached per season in cache/espn_odds_<yr>.parquet.
Opening lines exist in ESPN's feed from 2024 on; earlier seasons come back empty.

Sign convention returned here matches nflverse: spread values are HOME-favored-positive
(ESPN's own "home -3.5" becomes +3.5).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
SESSION = requests.Session()


def _get(url, **params):
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def _team(abbr: str) -> str:
    return TO_NFLVERSE.get(abbr, abbr)


def _num(x):
    """'+2.5' / '-110' / 'EVEN' / None -> float."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().upper()
    if x in ("EVEN", "EV", "PK", "PICK"):
        return 100.0 if x.startswith("EV") else 0.0
    try:
        return float(x.replace("O", "").replace("U", ""))
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------------------- this week's board
def week_board(season: int, week: int, seasontype: int = 2) -> pd.DataFrame:
    """One row per game: kickoff, TV, venue, records, logos, OPEN and CURRENT lines (DraftKings via ESPN)."""
    js = _get(f"{SITE}/scoreboard", seasontype=seasontype, week=week, dates=season)
    rows = []
    for ev in js.get("events", []):
        c = ev["competitions"][0]
        teams = {t["homeAway"]: t for t in c["competitors"]}
        h, a = teams["home"], teams["away"]
        row = {
            "espn_id": ev["id"], "kickoff": pd.to_datetime(ev["date"], utc=True),
            "home_team": _team(h["team"]["abbreviation"]), "away_team": _team(a["team"]["abbreviation"]),
            "home_name": h["team"].get("displayName"), "away_name": a["team"].get("displayName"),
            "home_logo": h["team"].get("logo"), "away_logo": a["team"].get("logo"),
            "home_color": h["team"].get("color"), "away_color": a["team"].get("color"),
            "home_alt_color": h["team"].get("alternateColor"), "away_alt_color": a["team"].get("alternateColor"),
            "home_record": (h.get("records") or [{}])[0].get("summary"),
            "away_record": (a.get("records") or [{}])[0].get("summary"),
            "venue": c.get("venue", {}).get("fullName"),
            "city": ", ".join(filter(None, [c.get("venue", {}).get("address", {}).get(k) for k in ("city", "state")])),
            "tv": " / ".join(n for b in c.get("broadcasts", []) for n in b.get("names", [])),
            "status": ev.get("status", {}).get("type", {}).get("name"),
            "home_score_live": _num(h.get("score")), "away_score_live": _num(a.get("score")),
        }
        o = (c.get("odds") or [{}])[0]
        row["book"] = o.get("provider", {}).get("name")
        for when, key in (("open", "open"), ("cur", "close")):
            ps, ml, tot = o.get("pointSpread", {}), o.get("moneyline", {}), o.get("total", {})
            row[f"{when}_spread"] = -_num(ps.get("home", {}).get(key, {}).get("line"))
            row[f"{when}_home_spread_odds"] = _num(ps.get("home", {}).get(key, {}).get("odds"))
            row[f"{when}_away_spread_odds"] = _num(ps.get("away", {}).get(key, {}).get("odds"))
            row[f"{when}_home_ml"] = _num(ml.get("home", {}).get(key, {}).get("odds"))
            row[f"{when}_away_ml"] = _num(ml.get("away", {}).get(key, {}).get("odds"))
            row[f"{when}_total"] = _num((tot.get("over", {}).get(key, {}).get("line") or "").lstrip("oOuU") or None)
            row[f"{when}_over_odds"] = _num(tot.get("over", {}).get(key, {}).get("odds"))
            row[f"{when}_under_odds"] = _num(tot.get("under", {}).get(key, {}).get("odds"))
        if np.isnan(row["cur_spread"]) and o.get("spread") is not None:      # older payload shape
            row["cur_spread"] = -float(o["spread"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- historical odds
def _event_odds(eid: str) -> dict:
    js = _get(f"{CORE}/events/{eid}/competitions/{eid}/odds")
    items = [i for i in js.get("items", []) if "Live" not in i.get("provider", {}).get("name", "")]
    if not items:
        return {}
    it = items[0]
    out = {"book": it.get("provider", {}).get("name")}
    for when in ("open", "close"):
        h, a = it.get("homeTeamOdds", {}).get(when, {}), it.get("awayTeamOdds", {}).get(when, {})
        out[f"{when}_spread"] = -_num(h.get("pointSpread", {}).get("american"))
        out[f"{when}_home_spread_odds"] = _num(h.get("spread", {}).get("american"))
        out[f"{when}_away_spread_odds"] = _num(a.get("spread", {}).get("american"))
        out[f"{when}_home_ml"] = _num(h.get("moneyLine", {}).get("american"))
        out[f"{when}_away_ml"] = _num(a.get("moneyLine", {}).get("american"))
        tot = it.get(when, {}).get("total", {})
        out[f"{when}_total"] = _num((tot.get("american") or "").lstrip("oOuU") or None)
    return out


def season_odds(season: int, refresh: bool = False, max_week: int = 18) -> pd.DataFrame:
    """Open/close lines for every regular-season game of a season (cached; the current season is re-fetched)."""
    f = CACHE / f"espn_odds_{season}.parquet"
    if f.exists() and not refresh:
        cached = pd.read_parquet(f)
        if cached["final"].all() and len(cached) >= 256:
            return cached
    rows = []
    for wk in range(1, max_week + 1):
        js = _get(f"{SITE}/scoreboard", seasontype=2, week=wk, dates=season)
        for ev in js.get("events", []):
            c = ev["competitions"][0]
            t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
            final = ev.get("status", {}).get("type", {}).get("completed", False)
            rows.append({"season": season, "week": wk, "espn_id": ev["id"], "home_team": _team(t["home"]),
                         "away_team": _team(t["away"]), "final": final, **_event_odds(ev["id"])})
            time.sleep(0.15)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(f)
    return df


def historical_odds(seasons, refresh: bool = False) -> pd.DataFrame:
    frames = [season_odds(s, refresh) for s in seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    import sys
    for yr in map(int, sys.argv[1:] or [2024, 2025, 2026]):
        d = season_odds(yr, refresh=True)
        print(yr, len(d), "games,", int(d["open_spread"].notna().sum()), "with opening spread", flush=True)
