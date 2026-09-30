"""Predict the upcoming NFL week, log the picks before kickoff, and rebuild the website.

    python predict.py                        # next week with unplayed games
    python predict.py --season 2026 --week 5
    python predict.py --no-site              # skip the HTML rebuild

Lines come from ESPN's live board (DraftKings, open + current); nflverse lines are the fallback.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import espn
import ledger
from features import build_game_features
from margin_dist import ScoreDist
from model import predict_models
from nfl_data import current_season
from pricing import add_model_only, add_tested_hit_rates, fair_view, picks

OUT = Path("output")
LINE_MAP = {  # unified column <- (ESPN current, nflverse fallback)
    "spread": ("cur_spread", "spread_line"), "home_spread_odds": ("cur_home_spread_odds", "home_spread_odds"),
    "away_spread_odds": ("cur_away_spread_odds", "away_spread_odds"), "total_line": ("cur_total", "total_line"),
    "over_odds": ("cur_over_odds", "over_odds"), "under_odds": ("cur_under_odds", "under_odds"),
    "home_ml": ("cur_home_ml", "home_moneyline"), "away_ml": ("cur_away_ml", "away_moneyline"),
}


def next_week(games: pd.DataFrame) -> tuple[int, int]:
    up = games[(games["game_type"] == "REG") & ~games["played"]].sort_values(["gameday", "game_id"])
    if up.empty:
        raise SystemExit("No upcoming regular-season games found.")
    return int(up.iloc[0]["season"]), int(up.iloc[0]["week"])


def attach_lines(wk: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    try:
        board = espn.week_board(season, week)
    except Exception as e:                       # ESPN down: fall back to nflverse lines only
        print(f"  (ESPN board unavailable: {e}; using nflverse lines)")
        board = pd.DataFrame()
    if not board.empty:
        wk = wk.merge(board, on=["home_team", "away_team"], how="left")
    for col, (live, fallback) in LINE_MAP.items():
        base = wk[live] if live in wk else pd.Series(np.nan, index=wk.index)
        wk[f"L_{col}"] = base.astype(float).fillna(wk[fallback].astype(float))
    for c in ("open_spread", "open_total", "kickoff", "tv", "venue", "city", "home_logo", "away_logo",
              "home_color", "away_color", "home_alt_color", "away_alt_color",
              "home_name", "away_name", "home_record", "away_record", "book"):
        if c not in wk:
            wk[c] = np.nan
    fallback_kick = pd.to_datetime(wk["gameday"].astype(str) + " " + wk.get("gametime", pd.Series("13:00", index=wk.index)).fillna("13:00"),
                                   errors="coerce").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    wk["kickoff"] = pd.to_datetime(wk["kickoff"], utc=True).fillna(fallback_kick)
    return wk


def run(season=None, week=None, build_site=True):
    bundle = joblib.load(OUT / "models.joblib")
    meta = json.loads((OUT / "model_meta.json").read_text())
    mdist, tdist = ScoreDist.from_dict(meta["margin_dist"]), ScoreDist.from_dict(meta["total_dist"])

    games, features = build_game_features(range(2014, current_season() + 1))
    if features != bundle["features"]:
        raise SystemExit("Feature list changed since training - run `python train_final.py`.")
    if season is None or week is None:
        s, w = next_week(games)
        season, week = season or s, week or w
    wk = games[(games["game_type"] == "REG") & (games["season"] == season) & (games["week"] == week)].copy()
    if wk.empty:
        raise SystemExit(f"No games for {season} week {week}.")
    wk = attach_lines(wk.reset_index(drop=True), season, week)
    wk = pd.concat([wk, predict_models(bundle["models"], wk, features)], axis=1)

    L = {k: f"L_{k}" for k in LINE_MAP}
    wk = fair_view(wk, mdist, tdist, meta["beta_margin"], meta["beta_total"], spread=L["spread"],
                   home_odds=L["home_spread_odds"], away_odds=L["away_spread_odds"], total=L["total_line"],
                   over_odds=L["over_odds"], under_odds=L["under_odds"], win_calib=tuple(meta["win_calib"]))
    wk = picks(wk, spread=L["spread"], home_odds=L["home_spread_odds"], away_odds=L["away_spread_odds"],
               total=L["total_line"], over_odds=L["over_odds"], under_odds=L["under_odds"],
               home_ml=L["home_ml"], away_ml=L["away_ml"])
    wk = add_model_only(wk, mdist, tdist, win_calib=tuple(meta["win_calib"]), spread=L["spread"],
                        home_odds=L["home_spread_odds"], away_odds=L["away_spread_odds"], total=L["total_line"],
                        over_odds=L["over_odds"], under_odds=L["under_odds"], home_ml=L["home_ml"],
                        away_ml=L["away_ml"]).sort_values(["kickoff", "game_id"])
    wk = add_tested_hit_rates(wk, meta.get("mo_hit_rates", {}), L["home_ml"], L["away_ml"])
    import matchup_stats
    wk = matchup_stats.add_nfl(wk, games)

    now = pd.Timestamp.now(tz="UTC")
    log = wk.rename(columns={L["spread"]: "spread", L["home_spread_odds"]: "home_spread_odds",
                             L["away_spread_odds"]: "away_spread_odds", L["total_line"]: "total_line",
                             L["over_odds"]: "over_odds", L["under_odds"]: "under_odds",
                             L["home_ml"]: "home_ml", L["away_ml"]: "away_ml"}, errors="ignore")
    log = log.loc[:, ~log.columns.duplicated(keep="last")]
    log["logged_at"] = now.isoformat(timespec="seconds")
    log["kickoff"] = log["kickoff"].astype(str)
    n = ledger.record(log[log["played"].eq(False) & (pd.to_datetime(log["kickoff"], utc=True) > now)], now)

    OUT.mkdir(exist_ok=True)
    keep = [c for c in ledger.COLUMNS if c in log] + ["tv", "venue", "city", "home_logo", "away_logo", "home_name",
                                                       "away_name", "home_record", "away_record", "book", "played",
                                                       "home_color", "away_color", "home_alt_color", "away_alt_color",
                                                       "matchup_stats"]
    log[keep].to_csv(OUT / "current_predictions.csv", index=False)
    (OUT / f"week_{season}_{week:02d}.json").write_text(log[keep].to_json(orient="records", indent=1))
    (OUT / "current.json").write_text(json.dumps({"season": int(season), "week": int(week)}))
    print_report(wk, season, week, L, meta)
    print(f"\n{n} games logged to output/ledger.csv (games already kicked off are locked).")
    if build_site:
        import site_builder
        path = site_builder.build(season, week)
        print(f"website: {path.resolve()}")
    return wk


def print_report(wk, season, week, L, meta):
    print(f"\n{'=' * 96}\nNFL {season} WEEK {week}   (model trained through {meta['trained_through']})\n{'=' * 96}")
    print(f"{'matchup':14s} {'proj score':>14s} {'line':>7s} {'fair':>6s}  {'winner':>11s}  {'ATS pick':>12s} "
          f"{'cover':>6s}  {'total':>12s}  value")
    for _, r in wk.iterrows():
        line = "" if pd.isna(r[L["spread"]]) else f"{-r[L['spread']]:+.1f}"
        ats = "" if r["ats_pick"] is None else f"{r['ats_pick']} {r['ats_line']:+.1f}"
        tot = "" if r["tot_pick"] is None else f"{r['tot_pick'][0]} {r[L['total_line']]:.1f}"
        flags = [f"{m.upper()} EV {r[f'{m}_ev']:+.1%}" for m in ("ats", "tot", "ml") if r[f"value_{m}"]]
        print(f"{r.away_team:>3s} @ {r.home_team:<8s} {r.proj_away:5.1f}-{r.proj_home:<5.1f}   {line:>7s} {-r.fair_spread:+6.1f}  "
              f"{r.su_pick:>4s} {r.su_prob:5.1%}  {ats:>12s} {r.ats_prob_np:6.1%}  {tot:>12s}  {', '.join(flags)}")
    print("line/fair are the HOME team's spread (negative = home favored).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-site", action="store_true")
    a = ap.parse_args()
    run(a.season, a.week, not a.no_site)
