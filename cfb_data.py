"""College football data from CollegeFootballData.com, cached per season in cache/cfb/.

Completed seasons are downloaded once (about 6 API calls each) and never re-fetched; the current season is refreshed
when `refresh=True` (weekly update: ~4 calls). The free tier allows 1,000 calls a month.

Output follows the NFL pipeline's schema so the same model, pricing and backtest code can be reused:
    game_id, season, week, game_type (REG/POST), gameday, home_team, away_team, home_score, away_score, result, total,
    played, spread_line (HOME-favored-positive), total_line, open_spread, open_total, home_moneyline, away_moneyline,
    location ('Neutral'/'Home'), home_fbs, away_fbs, conference_game
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import cfb_client as api

CACHE = Path(__file__).parent / "cache" / "cfb"
CACHE.mkdir(parents=True, exist_ok=True)
BOOK_PRIORITY = ["DraftKings", "ESPN Bet", "Bovada", "William Hill (New Jersey)", "consensus"]


def current_season() -> int:
    today = pd.Timestamp.now(tz="UTC")
    return today.year if today.month >= 7 else today.year - 1


def _cached(name: str, year: int, fetch, refresh: bool) -> pd.DataFrame:
    f = CACHE / f"{name}_{year}.parquet"
    if f.exists() and not (refresh and year >= current_season()):
        return pd.read_parquet(f)
    df = fetch()
    if not df.empty:
        df.to_parquet(f)
    return df


# ------------------------------------------------------------------------------------------------ raw pulls
def games(year: int, refresh=False) -> pd.DataFrame:
    def fetch():
        rows = api.get("/games", year=year, seasonType="regular") + api.get("/games", year=year, seasonType="postseason")
        return pd.json_normalize(rows)
    return _cached("games", year, fetch, refresh)


def lines(year: int, refresh=False) -> pd.DataFrame:
    """One row per game: median closing/opening spread and total across books, moneyline from the preferred book."""
    def fetch():
        out = []
        for g in api.get("/lines", year=year):
            ls = g.get("lines") or []
            if not ls:
                continue
            med = lambda k: float(np.nanmedian([np.nan if l.get(k) is None else l[k] for l in ls])) if any(l.get(k) is not None for l in ls) else np.nan
            ml = next((l for b in BOOK_PRIORITY for l in ls if l.get("provider") == b and l.get("homeMoneyline") is not None),
                      next((l for l in ls if l.get("homeMoneyline") is not None), {}))

            def book(k):
                """A real book's number (DraftKings first) so lines are ones you could bet; median if none."""
                l = next((l for b in BOOK_PRIORITY for l in ls if l.get("provider") == b and l.get(k) is not None), None)
                return float(l[k]) if l else med(k)
            out.append({"game_id": g["id"], "spread_line": -book("spread"), "open_spread": -book("spreadOpen"),
                        "total_line": book("overUnder"), "open_total": book("overUnderOpen"),
                        "home_moneyline": ml.get("homeMoneyline"), "away_moneyline": ml.get("awayMoneyline"),
                        "books": len(ls)})
        return pd.DataFrame(out)
    return _cached("lines", year, fetch, refresh)


def advanced(year: int, refresh=False) -> pd.DataFrame:
    """Per team-game efficiency (garbage time excluded): offensive/defensive PPA, success rate, explosiveness, plays."""
    def fetch():
        rows = api.get("/stats/game/advanced", year=year, excludeGarbageTime="true")
        df = pd.json_normalize(rows)
        keep = {"gameId": "game_id", "team": "team", "offense.ppa": "off_ppa", "offense.successRate": "off_sr",
                "offense.explosiveness": "off_expl", "offense.plays": "off_plays", "defense.ppa": "def_ppa",
                "defense.successRate": "def_sr", "defense.explosiveness": "def_expl"}
        return df[[c for c in keep if c in df]].rename(columns=keep)
    return _cached("advanced", year, fetch, refresh)


def talent(year: int) -> pd.DataFrame:
    def fetch():
        return pd.DataFrame(api.get("/talent", year=year)).rename(columns={"team": "team", "talent": "talent"})[["team", "talent"]]
    try:
        return _cached("talent", year, fetch, False)
    except Exception:
        return pd.DataFrame(columns=["team", "talent"])


def returning(year: int) -> pd.DataFrame:
    """Share of last season's production that returns: overall and passing (a proxy for QB continuity)."""
    def fetch():
        return pd.DataFrame(api.get("/player/returning", year=year))[["team", "percentPPA", "percentPassingPPA"]].rename(
            columns={"percentPPA": "returning", "percentPassingPPA": "ret_pass"})
    try:
        return _cached("returning2", year, fetch, False)
    except Exception:
        return pd.DataFrame(columns=["team", "returning", "ret_pass"])


def portal(year: int) -> pd.DataFrame:
    """Transfer portal per team: net recruiting rating of transfers in minus out, and the best incoming QB's rating."""
    def fetch():
        p = pd.DataFrame(api.get("/player/portal", year=year))
        if p.empty:
            return pd.DataFrame(columns=["team", "portal_net", "qb_in"])
        p["rating"] = pd.to_numeric(p["rating"], errors="coerce").fillna(0.75)       # unrated transfers ~ low 3-star
        inn = p.dropna(subset=["destination"]).groupby("destination")["rating"].sum()
        out = p.groupby("origin")["rating"].sum()
        qb = p[(p["position"] == "QB")].dropna(subset=["destination"]).groupby("destination")["rating"].max()
        t = pd.DataFrame({"portal_in": inn, "portal_out": out, "qb_in": qb}).fillna(0)
        t["portal_net"] = t["portal_in"] - t["portal_out"]
        return t.reset_index().rename(columns={"index": "team"})[["team", "portal_net", "qb_in"]]
    if year < 2021:                                    # the portal era: earlier years have little data
        return pd.DataFrame(columns=["team", "portal_net", "qb_in"])
    try:
        return _cached("portal", year, fetch, False)
    except Exception:
        return pd.DataFrame(columns=["team", "portal_net", "qb_in"])


def teams(year: int) -> pd.DataFrame:
    """FBS team info for the page: abbreviation, colors, logos."""
    def fetch():
        rows = api.get("/teams/fbs", year=year)
        return pd.DataFrame([{"team": t["school"], "abbr": t.get("abbreviation") or t["school"][:4].upper(),
                              "mascot": t.get("mascot"), "color": t.get("color"), "alt_color": t.get("alternateColor"),
                              "logo": (t.get("logos") or [None])[0], "conference": t.get("conference")} for t in rows])
    return _cached("teams", year, fetch, False)


# ------------------------------------------------------------------------------------------------ assembled
def load_games(seasons, refresh_current=False) -> pd.DataFrame:
    """All games involving at least one FBS team, in the NFL-pipeline schema, with lines and per-game efficiency."""
    frames = []
    for yr in seasons:
        fresh = refresh_current and yr >= current_season()
        g = games(yr, fresh)
        if g.empty:
            continue
        g = g[(g["homeClassification"] == "fbs") | (g["awayClassification"] == "fbs")]
        d = pd.DataFrame({
            "game_id": g["id"].astype(int), "season": g["season"].astype(int), "week": g["week"].astype(int),
            "game_type": np.where(g["seasonType"] == "postseason", "POST", "REG"),
            "kickoff": pd.to_datetime(g["startDate"], utc=True),
            "home_team": g["homeTeam"], "away_team": g["awayTeam"],
            "home_score": pd.to_numeric(g["homePoints"], errors="coerce"),
            "away_score": pd.to_numeric(g["awayPoints"], errors="coerce"),
            "location": np.where(g["neutralSite"].fillna(False), "Neutral", "Home"),
            "home_fbs": (g["homeClassification"] == "fbs").astype(int), "away_fbs": (g["awayClassification"] == "fbs").astype(int),
            "conference_game": g["conferenceGame"].fillna(False).astype(int), "venue": g.get("venue"),
            "home_conf": g.get("homeConference"), "away_conf": g.get("awayConference"),
        })
        # postseason weeks restart at 1; push them after the regular season so ordering stays chronological
        d.loc[d["game_type"] == "POST", "week"] += 20
        ln = lines(yr, fresh)
        if not ln.empty:
            d = d.merge(ln, on="game_id", how="left")
        adv = advanced(yr, fresh)
        if not adv.empty:
            for side in ("home", "away"):
                a = adv.rename(columns={c: f"{side}_{c}" for c in adv.columns if c not in ("game_id", "team")})
                d = d.merge(a.rename(columns={"team": f"{side}_team"}), on=["game_id", f"{side}_team"], how="left")
        pre = [talent(yr), returning(yr), portal(yr)]
        for side in ("home", "away"):
            for df in pre:
                if not df.empty:
                    d = d.merge(df.rename(columns={c: (f"{side}_team" if c == "team" else f"{side}_{c}") for c in df.columns}),
                                on=f"{side}_team", how="left")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    for c in ("spread_line", "open_spread", "total_line", "open_total", "home_moneyline", "away_moneyline"):
        if c not in out:
            out[c] = np.nan
    out["gameday"] = out["kickoff"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    out["result"] = out["home_score"] - out["away_score"]
    out["total"] = out["home_score"] + out["away_score"]
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        out[c] = np.nan                                  # CFBD has no spread/total juice: priced at -110
    return out.sort_values(["season", "week", "kickoff", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    yrs = range(int(sys.argv[1]) if len(sys.argv) > 1 else 2014, current_season() + 1)
    df = load_games(yrs, refresh_current=True)
    print(df.groupby("season").agg(games=("game_id", "size"), lines=("spread_line", lambda s: s.notna().sum()),
                                   opens=("open_spread", lambda s: s.notna().sum()), ppa=("home_off_ppa", lambda s: s.notna().sum())))
    print(api.get("/info")["remainingCalls"], "API calls left this month")
