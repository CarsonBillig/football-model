"""Fit the production model on every completed game and calibrate it from out-of-sample evidence.

Learns from history in three ways:
  1. the fundamentals model is refit on all completed games (including last week's)
  2. key-number distributions are refit on all completed games with lines
  3. the market-vs-model blend weight (beta) and the win-probability calibration are estimated from the backtest's
     out-of-sample predictions PLUS every graded pick in output/ledger.csv. If the model's live disagreements with
     the line keep losing, beta shrinks toward 0 (trust the market); if they keep winning, it grows.

Run backtest.py at least once first.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

import ledger
from features import build_game_features
from margin_dist import ScoreDist
from model import fit_models, usable
from nfl_data import current_season
from pricing import estimate_beta, fit_platt

OUT = Path("output")


def evidence() -> pd.DataFrame:
    """Out-of-sample rows (backtest) + graded live picks, with the columns needed to estimate beta/calibration."""
    cols = ["game_id", "result", "total", "fund_margin", "fund_total", "mkt_margin", "mkt_total", "p_home_win_raw"]
    oos = pd.read_csv(OUT / "oos_predictions.csv", usecols=lambda c: c in cols)
    live = ledger.load()
    live = live[live["su_out"].notna()].rename(columns={"home_score": "hs", "away_score": "as_"})
    if not live.empty:
        live = live.assign(result=live["hs"] - live["as_"], total=live["hs"] + live["as_"])
        live = live.reindex(columns=cols)
        oos = pd.concat([oos[~oos["game_id"].isin(live["game_id"])], live], ignore_index=True)
    return oos, int(len(live))


def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT / "oos_predictions.csv").exists():
        raise FileNotFoundError("output/oos_predictions.csv not found - run `python backtest.py` first.")

    games, features = build_game_features(range(2014, current_season() + 1))
    train = usable(games)
    last = train.iloc[-1]
    print(f"Training on {len(train)} games through {int(last.season)} week {int(last.week)} ...", flush=True)
    models = fit_models(train, features)

    lined = train.dropna(subset=["spread_line", "total_line"])
    mdist = ScoreDist("margin").fit(lined["result"], lined["spread_line"])
    tdist = ScoreDist("total").fit(lined["total"], lined["total_line"])

    ev, n_live = evidence()
    bm = estimate_beta(ev["result"], ev["mkt_margin"], ev["fund_margin"])
    bt = estimate_beta(ev["total"], ev["mkt_total"], ev["fund_total"])
    decided = ev[(ev["result"] != 0) & ev["p_home_win_raw"].notna()]
    calib = fit_platt(decided["p_home_win_raw"], decided["result"] > 0)

    joblib.dump({"models": models, "features": features}, OUT / "models.joblib")
    summary = json.loads((OUT / "backtest_summary.json").read_text()) if (OUT / "backtest_summary.json").exists() else {}
    meta = {
        "margin_dist": mdist.to_dict(), "total_dist": tdist.to_dict(),
        "beta_margin": bm["beta"], "beta_total": bt["beta"], "beta_detail": {"margin": bm, "total": bt},
        "win_calib": calib, "live_games_in_evidence": n_live,
        "training_games": int(len(train)), "trained_through": f"{int(last.season)} week {int(last.week)}",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": summary.get("verdict", ""), "backtest_records": summary.get("records", {}),
        "backtest_span": f"{summary.get('test_start', '')}-{summary.get('test_end', '')}",
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"saved output/models.joblib + model_meta.json | beta margin {bm['beta']:.3f} (t={bm['t']:.1f}), "
          f"total {bt['beta']:.3f} (t={bt['t']:.1f}), {n_live} graded live games used")


if __name__ == "__main__":
    main()
