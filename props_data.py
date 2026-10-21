"""NFL player prop lines from ESPN's free odds feed (ESPN BET in 2024-25, DraftKings from 2026), cached per game.

    python props_data.py 2024 2026        # download every regular-season game's props, politely, once

Each row is one side of a book's MAIN line (the line priced on both sides; alternate lines without prices are
skipped): season, week, event_id, home, away, provider, espn_athlete_id, market, side (over/under), the line and
American odds now (near the close for finished games) and at the open. DraftKings' records in the feed (2026)
carry lines but no prices, so their odds are blank.
Cache: cache/props/<season>/<event_id>.parquet. Finished games are fetched once.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache" / "props"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{eid}/competitions/{eid}/odds"
MARKETS = {  # ESPN market name prefix -> our name
    "Total Passing Yards": "pass_yds", "Total Rushing Yards": "rush_yds", "Total Receiving Yards": "rec_yds",
    "Total Receptions": "receptions", "Total Passing Attempts": "pass_att", "Total Pass Completions": "completions",
    "Total Carries": "carries", "Total Passing Touchdowns": "pass_td",
}
PACE = 0.7                          # seconds between requests
_last = [0.0]


def _get(url, **params):
    for attempt in range(5):
        time.sleep(max(0.0, _last[0] + PACE - time.time()))
        _last[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after") or 30) + 2)
                continue
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (attempt + 1))
    return {}


def _market(name: str):
    if "1st Half" in name or "Plus" in name or "Longest" in name:
        return None
    return next((v for k, v in MARKETS.items() if name.startswith(k)), None)


def _odds(side):
    a = (side or {}).get("american")
    try:
        return float(str(a).replace("EVEN", "100"))
    except (TypeError, ValueError):
        return None


def event_props(eid: str) -> pd.DataFrame:
    rows = []
    for prov in _get(CORE.format(eid=eid)).get("items", []):
        p = prov.get("provider") or {}
        if "live" in str(p.get("name", "")).lower():
            continue
        items, page, pages = [], 1, 1
        while page <= pages:
            js = _get(f"{CORE.format(eid=eid)}/{p.get('id')}/propBets", limit=1000, page=page)
            pages = js.get("pageCount", 1) or 1
            items += js.get("items", [])
            page += 1
        # A book that prices its props: keep only the main line (priced on both sides; unpriced entries are alternate
        # lines). A book with no prices at all (DraftKings in ESPN's feed from 2026): it posts one line per player,
        # so keep it with the price left blank.
        priced = any(_odds((it.get("current") or {}).get(s)) is not None for it in items for s in ("over", "under"))
        for it in items:
            market = _market(it.get("type", {}).get("name", ""))
            ref = (it.get("athlete") or {}).get("$ref", "")
            m = re.search(r"athletes/(\d+)", ref)
            if not market or not m:
                continue
            cur, opn = it.get("current") or {}, it.get("open") or {}
            for side in ("over", "under"):
                if priced and _odds(cur.get(side)) is None:
                    continue
                rows.append({"event_id": eid, "provider": p.get("name"), "espn_athlete_id": m.group(1),
                             "market": market, "side": side,
                             "line": (cur.get("target") or {}).get("value"), "odds": _odds(cur.get(side)),
                             "open_line": (opn.get("target") or {}).get("value"), "open_odds": _odds(opn.get(side)),
                             "updated": it.get("lastUpdated")})
    return pd.DataFrame(rows)


def season(year: int, weeks=range(1, 19)) -> pd.DataFrame:
    out_dir = CACHE / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for wk in weeks:
        for ev in _get(SCOREBOARD, dates=year, seasontype=2, week=wk).get("events", []):
            eid = ev["id"]
            f = out_dir / f"{eid}.parquet"
            done = ev.get("status", {}).get("type", {}).get("completed", False)
            if f.exists() and done:
                frames.append(pd.read_parquet(f))
                continue
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
            df = event_props(eid)
            if df.empty:
                continue
            df = df.assign(season=year, week=wk, home=teams.get("home"), away=teams.get("away"),
                           kickoff=ev.get("date"), completed=done)
            if done:
                df.to_parquet(f)
            frames.append(df)
        print(f"  {year} week {wk}: {sum(len(f) for f in frames)} prop rows so far", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    a = list(map(int, sys.argv[1:])) or [2024, 2026]
    for y in range(a[0], (a[1] if len(a) > 1 else a[0]) + 1):
        d = season(y)
        print(y, len(d), "rows,", d["event_id"].nunique() if len(d) else 0, "games", flush=True)
