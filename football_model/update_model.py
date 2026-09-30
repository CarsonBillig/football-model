"""One command, every week, for both sports:

    python update_model.py              # NFL + college: grade last week -> retrain -> predict next week -> rebuild docs/index.html
    python update_model.py --backtest   # also rerun both walk-forward backtests first (a few minutes; do this monthly)
    python update_model.py --nfl-only   # skip college (e.g. if the CollegeFootballData key is missing)

Run it any time before kickoff; re-running refreshes picks for games that haven't started and locks the rest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import backtest
import grade
import predict
import site_builder
import train_final


def main(run_backtest=False, season=None, week=None, college=True):
    if run_backtest or not Path("output/oos_predictions.csv").exists():
        print("\n### NFL backtest"); backtest.main()
    print("\n### NFL: grading pick history"); grade.main()
    print("\n### NFL: training (all completed games + graded picks)"); train_final.main()
    print("\n### NFL: predicting"); predict.run(season, week, build_site=False)
    if college:
        import cfb_model
        import cfb_pipeline
        if run_backtest or not Path("output/cfb/oos_predictions.csv").exists():
            print("\n### College backtest"); cfb_model.main()
        print("\n### College: grade, train, predict"); cfb_pipeline.run()
    print("\n### website"); print(f"site: {site_builder.build().resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--nfl-only", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    a = ap.parse_args()
    main(a.backtest, a.season, a.week, not a.nfl_only)
