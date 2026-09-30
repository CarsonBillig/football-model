"""College weekly pipeline: grade -> train -> predict -> log picks. Called by update_model.py.

    python cfb_pipeline.py                 # full weekly run for college (then rebuild the site with update_model.py)
    python cfb_pipeline.py --week 6

API use per run: about 6 CollegeFootballData calls (games, lines, efficiency, TV, rankings).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import cfb_client
import cfb_data
import cfb_model
import grade
import ledger
from margin_dist import ScoreDist
from model import fit_models, predict_models, usable
from pricing import add_model_only, add_tested_hit_rates, estimate_beta, fair_view, fit_hit_rates, fit_platt, picks

OUT = Path("output/cfb")


def schedule_for_grading(games: pd.DataFrame) -> pd.DataFrame:
    return games[["game_id", "home_score", "away_score", "spread_line", "total_line", "played"]]


def evidence(games_oos: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "result", "total", "fund_margin", "fund_total", "mkt_margin", "mkt_total", "p_home_win_raw"]
    ev = games_oos[[c for c in cols if c in games_oos]]
    live = ledger.load(ledger.CFB_LEDGER)
    live = live[live["su_out"].notna()]
    if not live.empty:
        live = live.assign(result=live["home_score"] - live["away_score"], total=live["home_score"] + live["away_score"])
        ev = pd.concat([ev[~ev["game_id"].isin(live["game_id"])], live.reindex(columns=cols)], ignore_index=True)
    return ev, int(len(live))


def train(games: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    oos_file = OUT / "oos_predictions.csv"
    if not oos_file.exists():
        raise SystemExit("output/cfb/oos_predictions.csv missing - run `python cfb_model.py` (the college backtest) first.")
    tr = usable(games)
    last = tr.iloc[-1]
    print(f"College: training on {len(tr)} games through {int(last.season)} week {int(last.week)} ...", flush=True)
    models = fit_models(tr, cfb_model.FEATURES)
    lined = tr.dropna(subset=["spread_line", "total_line"])
    mdist = ScoreDist("margin").fit(lined["result"], lined["spread_line"])
    tdist = ScoreDist("total").fit(lined["total"], lined["total_line"])
    ev, n_live = evidence(pd.read_csv(oos_file))
    bm = estimate_beta(ev["result"], ev["mkt_margin"], ev["fund_margin"])
    btt = estimate_beta(ev["total"], ev["mkt_total"], ev["fund_total"])
    dec = ev[(ev["result"] != 0) & ev["p_home_win_raw"].notna()]
    calib = fit_platt(dec["p_home_win_raw"], dec["result"] > 0)
    summary = json.loads((OUT / "backtest_summary.json").read_text())
    meta = {"margin_dist": mdist.to_dict(), "total_dist": tdist.to_dict(), "beta_margin": bm["beta"], "beta_total": btt["beta"],
            "beta_detail": {"margin": bm, "total": btt}, "win_calib": calib, "live_games_in_evidence": n_live,
            "mo_hit_rates": fit_hit_rates(pd.read_csv(oos_file)),
            "training_games": int(len(tr)), "trained_through": f"{int(last.season)} week {int(last.week)}",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "verdict": summary.get("verdict", ""),
            "backtest_records": summary.get("records", {}),
            "backtest_span": f"{summary.get('test_start', '')}-{summary.get('test_end', '')}"}
    joblib.dump({"models": models, "features": cfb_model.FEATURES}, OUT / "models.joblib")
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"College: beta margin {bm['beta']:.3f} (t={bm['t']:.1f}), total {btt['beta']:.3f} (t={btt['t']:.1f}), "
          f"{n_live} graded live games used")
    return meta


def _records(games: pd.DataFrame, season: int) -> dict:
    g = games[(games["season"] == season) & games["played"]]
    rec = {}
    for side, opp in (("home", "away"), ("away", "home")):
        for t, won in zip(g[f"{side}_team"], g[f"{side}_score"] > g[f"{opp}_score"]):
            w, l = rec.get(t, (0, 0))
            rec[t] = (w + int(won), l + int(not won))
    return {t: f"{w}-{l}" for t, (w, l) in rec.items()}


def _extras(season: int, week: int) -> tuple[dict, dict]:
    """TV networks by game id and AP ranks by team (2 API calls; failures just leave them blank)."""
    tv, ranks = {}, {}
    try:
        for m in cfb_client.get("/games/media", year=season, week=week, seasonType="regular"):
            tv.setdefault(m["id"], []).append(m.get("outlet"))
    except Exception:
        pass
    try:
        polls = cfb_client.get("/rankings", year=season, week=week, seasonType="regular") or \
                cfb_client.get("/rankings", year=season, week=week - 1, seasonType="regular")
        for wk in polls[-1:]:
            for p in wk.get("polls", []):
                if p.get("poll") in ("AP Top 25", "Playoff Committee Rankings"):
                    ranks = {r["school"]: r["rank"] for r in p.get("ranks", [])}
    except Exception:
        pass
    return {k: " / ".join(dict.fromkeys(filter(None, v))) for k, v in tv.items()}, ranks


def predict(games: pd.DataFrame, season=None, week=None) -> pd.DataFrame:
    bundle = joblib.load(OUT / "models.joblib")
    meta = json.loads((OUT / "model_meta.json").read_text())
    mdist, tdist = ScoreDist.from_dict(meta["margin_dist"]), ScoreDist.from_dict(meta["total_dist"])
    if season is None or week is None:
        recent = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=4)        # cancelled old games never get scores
        up = games[(games["game_type"] == "REG") & ~games["played"] & (games["kickoff"] > recent)].sort_values("kickoff")
        if up.empty:
            raise SystemExit("No upcoming college regular-season games.")
        season, week = season or int(up.iloc[0]["season"]), week or int(up.iloc[0]["week"])
    wk = games[(games["season"] == season) & (games["week"] == week) & (games["game_type"] == "REG")].copy()
    wk = pd.concat([wk, predict_models(bundle["models"], wk, bundle["features"])], axis=1)
    wk = fair_view(wk, mdist, tdist, meta["beta_margin"], meta["beta_total"], win_calib=tuple(meta["win_calib"]))
    wk = picks(wk)
    wk = add_model_only(wk, mdist, tdist, win_calib=tuple(meta["win_calib"]))
    wk = add_tested_hit_rates(wk, meta.get("mo_hit_rates", {}))

    info = cfb_data.teams(season).set_index("team")
    tv, ranks = _extras(season, week)
    recs = _records(games, season)
    for side in ("home", "away"):
        t = wk[f"{side}_team"]
        wk[f"{side}_name"] = t
        wk[f"{side}_abbr"] = t.map(info["abbr"]).fillna(t.str[:4].str.upper())
        wk[f"{side}_logo"] = t.map(info["logo"])
        wk[f"{side}_color"] = t.map(info["color"]).str.lstrip("#")
        wk[f"{side}_alt_color"] = t.map(info["alt_color"]).str.lstrip("#")
        wk[f"{side}_record"] = t.map(recs).fillna("0-0")
        wk[f"{side}_rank"] = t.map(ranks)
    wk["tv"] = wk["game_id"].map(tv)
    wk["city"] = np.nan
    wk = wk.rename(columns={"spread_line": "spread", "home_moneyline": "home_ml", "away_moneyline": "away_ml"})
    wk["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wk["kickoff"] = wk["kickoff"].astype(str)
    wk = wk.sort_values(["kickoff", "game_id"])

    now = pd.Timestamp.now(tz="UTC")
    todo = wk[wk["spread"].notna() & ~wk["played"] & (pd.to_datetime(wk["kickoff"], utc=True) > now)]
    n = ledger.record(todo, now, ledger.CFB_LEDGER)
    extra = ["tv", "venue", "city", "played", "home_name", "away_name", "home_abbr", "away_abbr", "home_logo", "away_logo",
             "home_color", "away_color", "home_alt_color", "away_alt_color", "home_record", "away_record",
             "home_rank", "away_rank", "home_conf", "away_conf"]
    keep = [c for c in ledger.COLUMNS if c in wk] + [c for c in extra if c in wk]
    (OUT / f"week_{season}_{week:02d}.json").write_text(wk[keep].to_json(orient="records", indent=1))
    (OUT / "current.json").write_text(json.dumps({"season": int(season), "week": int(week)}))
    lined = wk["spread"].notna().sum()
    print(f"College {season} week {week}: {len(wk)} games ({lined} with lines), {n} picks logged to output/cfb/ledger.csv")
    return wk


def run(season=None, week=None):
    games = cfb_model.load(refresh_current=True)
    grade.main(ledger.CFB_LEDGER, schedule_for_grading(games), "College")
    train(games)
    return predict(games, season, week)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    a = ap.parse_args()
    run(a.season, a.week)
