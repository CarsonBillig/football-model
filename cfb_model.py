"""College football: features and honest walk-forward backtest.

Reuses the NFL machinery (model.py ridge + walk-forward, pricing.py, backtest.py sections) on college data shaped to
the same schema by cfb_data.py. College-specific pieces:
  * all FCS opponents share one rating ("FCS"): each FCS team plays too few FBS games to rate on its own
  * ridge power ratings from final margins, past closing lines, and per-play efficiency (PPA, success rate)
  * preseason priors (roster talent, returning production) that fade as the season goes on

    python cfb_model.py        # backtest 2016+  -> output/cfb/backtest_report.txt, backtest_summary.json, oos_predictions.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import cfb_data
from betting import BREAKEVEN_110
from margin_dist import ScoreDist
from model import usable, walk_forward
from power_ratings import pregame_rating
from pricing import estimate_beta, fair_view, picks

OUT = Path("output/cfb")
FIRST_SEASON, FIRST_TEST_SEASON = 2014, 2016

FEATURES = {
    "margin": ["res_prior", "mkt_prior", "ppa_prior", "sr_prior", "talent_diff", "returning_diff",
               "talent_early", "returning_early", "fcs_diff", "is_neutral",
               "ret_pass_early", "portal_early", "qb_in_early", "coach_early"],
    "total": ["res_total_prior", "mkt_total_prior", "ppa_total_prior", "plays_prior", "is_neutral"],
}


def build_features(g: pd.DataFrame) -> pd.DataFrame:
    """Pre-game features for every game. Ratings only ever use games from earlier weeks."""
    g = g.copy()
    r = g.assign(home_team=np.where(g["home_fbs"] == 1, g["home_team"], "FCS"),
                 away_team=np.where(g["away_fbs"] == 1, g["away_team"], "FCS"))
    r["net_ppa"] = g["home_off_ppa"] - g["away_off_ppa"]
    r["net_sr"] = g["home_off_sr"] - g["away_off_sr"]
    r["ppa_sum"] = g["home_off_ppa"] + g["away_off_ppa"]
    r["plays_sum"] = g["home_off_plays"] + g["away_off_plays"]
    kw = dict(hl_weeks=8, season_carry=0.45, alpha=3.0)
    g["res_prior"] = pregame_rating(r, "result", **kw)
    g["mkt_prior"] = pregame_rating(r, "spread_line", hl_weeks=4, season_carry=0.5, alpha=1.0)
    g["ppa_prior"] = pregame_rating(r, "net_ppa", **kw)
    g["sr_prior"] = pregame_rating(r, "net_sr", **kw)
    g["res_total_prior"] = pregame_rating(r, "total", additive=True, **kw)
    g["mkt_total_prior"] = pregame_rating(r, "total_line", additive=True, hl_weeks=4, season_carry=0.5, alpha=1.0)
    g["ppa_total_prior"] = pregame_rating(r, "ppa_sum", additive=True, **kw)
    g["plays_prior"] = pregame_rating(r, "plays_sum", additive=True, **kw)

    early = np.clip((7 - g["week"]) / 6, 0, 1).where(g["game_type"] == "REG", 0.0)
    fcs_talent = np.nanpercentile(pd.concat([g["home_talent"], g["away_talent"]]).dropna(), 2) if g["home_talent"].notna().any() else 0
    ht, at = g["home_talent"].fillna(fcs_talent), g["away_talent"].fillna(fcs_talent)
    g["talent_diff"] = ht - at
    g["returning_diff"] = g["home_returning"].fillna(0.5) - g["away_returning"].fillna(0.5)
    g["talent_early"] = g["talent_diff"] * early
    g["returning_early"] = g["returning_diff"] * early
    g["fcs_diff"] = g["home_fbs"] - g["away_fbs"]
    # QB continuity and transfer portal: preseason information that fades as games are played
    for c in ("ret_pass", "portal_net", "qb_in"):
        for side in ("home", "away"):
            if f"{side}_{c}" not in g:
                g[f"{side}_{c}"] = np.nan
    rp = lambda s: s.clip(0, 1.2).fillna(0.5)          # share of passing value returning (outliers capped)
    g["ret_pass_early"] = (rp(g["home_ret_pass"]) - rp(g["away_ret_pass"])) * early
    g["portal_early"] = (g["home_portal_net"].fillna(0) - g["away_portal_net"].fillna(0)) * early
    g["qb_in_early"] = (g["home_qb_in"].fillna(0) - g["away_qb_in"].fillna(0)) * early
    # a new head coach (first season at the school): new systems take a few weeks to settle
    for side in ("home", "away"):
        if f"{side}_new_coach" not in g:
            g[f"{side}_new_coach"] = np.nan
    g["coach_early"] = (g["home_new_coach"].fillna(0) - g["away_new_coach"].fillna(0)) * early
    g["is_neutral"] = (g["location"] == "Neutral").astype(int)
    return g


def load(refresh_current=False) -> pd.DataFrame:
    return build_features(cfb_data.load_games(range(FIRST_SEASON, cfb_data.current_season() + 1), refresh_current))


# ------------------------------------------------------------------------------------------------ opening lines
def opening_line_section(d: pd.DataFrame, games: pd.DataFrame) -> dict:
    bt.header("G. OPENING LINES (2021+) - does the model predict line moves, and do picks at the OPEN win?")
    x = d.dropna(subset=["open_spread", "spread_line"])
    if len(x) < 300:
        bt.say(f"  only {len(x)} games with opening lines - skipped")
        return {}
    move = estimate_beta(x.spread_line, x.open_spread, x.fund_margin)
    bt.say(f"  games: {len(x)}. Share of (model - open) the line moves by kickoff: {move['slope']:.3f} (t={move['t']:.1f})")
    # holdout: move share from 2021-2023, then price 2024+ at the OPEN exactly as the live pipeline would
    train, test = x[x.season <= 2023], x[x.season >= 2024].copy()
    bm = estimate_beta(train.spread_line, train.open_spread, train.fund_margin)["beta"]
    hist = usable(games).dropna(subset=["spread_line", "total_line"])
    prior = hist[hist.season < 2024]
    md = ScoreDist("margin").fit(prior.result, prior.spread_line)
    td = ScoreDist("total").fit(prior.total, prior.total_line)
    test = test.assign(o_total=test["open_total"].fillna(test["total_line"]))
    p = picks(fair_view(test, md, td, bm, 0.0, spread="open_spread", total="o_total"),
              spread="open_spread", total="o_total")
    side = np.where(p.ats_pick == p.home_team, 1, -1)
    ats = np.sign((p.result - p.open_spread) * side)
    clv = (p.spread_line - p.open_spread) * side
    rows = []
    for label, sel in (("every game", np.ones(len(p), bool)), ("value bets (EV>=2%)", p.value_ats.to_numpy()),
                       ("model 3+ pts off the open", (p.fund_margin - p.open_spread).abs().to_numpy() >= 3)):
        r = bt.record(pd.Series(ats[sel]), BREAKEVEN_110)
        c = pd.Series(clv[sel])
        rows.append({"picks at the open": label, "n": int(sel.sum()), "W": r["W"], "L": r["L"], "P": r["P"],
                     "win_pct": r["win_pct"], "avg_CLV": c.mean(), "beat_close": (c > 0).mean(), "p_better": r["p_better"]})
    bt.say(f"\n  HOLDOUT 2024+ (move share fit on 2021-2023: {bm:.3f})")
    bt.show(pd.DataFrame(rows))
    return {"n": int(len(x)), "move_share": move["slope"], "move_t": move["t"], "beta_open": move["beta"],
            "holdout": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bt.LOG.clear()
    games = load()
    bt.say(f"college games: {len(games)}; margin features {len(FEATURES['margin'])}, total {len(FEATURES['total'])}")
    oos = walk_forward(games, FEATURES, first_test_season=FIRST_TEST_SEASON).dropna(subset=["spread_line", "total_line"])
    priced, betas = bt.price_walk_forward(games, oos)
    d = bt.grade(priced)
    bt.say(f"out-of-sample games: {len(d)} ({d.season.min()}-{d.season.max()})")
    bt.accuracy_section(d)
    info = bt.info_section(d)
    bt.winprob_section(d)
    bt.records_section(d)
    bt.value_section(d)
    bt.calibration_section(d)
    openl = opening_line_section(d, games)
    recs = {"su": bt.record(d.su_out), "ats": bt.record(d.ats_out, BREAKEVEN_110), "tot": bt.record(d.tot_out, BREAKEVEN_110),
            "mo_su": bt.record(d.mo_su_out), "mo_ats": bt.record(d.mo_ats_out, BREAKEVEN_110),
            "mo_tot": bt.record(d.mo_tot_out, BREAKEVEN_110)}
    ats = recs["ats"]
    verdict = (f"College backtest {d.season.min()}-{d.season.max()}: winners {recs['su']['W']}-{recs['su']['L']} "
               f"({recs['su']['win_pct']:.1%}), ATS {ats['W']}-{ats['L']}-{ats['P']} ({ats['win_pct']:.1%}), "
               f"model-only ATS {recs['mo_ats']['win_pct']:.1%}. Model-vs-closing-line information t={info['margin']['t']:.1f}.")
    bt.say("\n" + verdict)
    summary = {"test_start": int(d.season.min()), "test_end": int(d.season.max()), "games": int(len(d)),
               "records": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()} for k, v in recs.items()},
               "info_margin": info["margin"], "info_total": info["total"], "opening_line_test": openl, "verdict": verdict}
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    d.to_csv(OUT / "oos_predictions.csv", index=False)
    (OUT / "backtest_report.txt").write_text("\n".join(bt.LOG), encoding="utf-8")
    bt.say("saved output/cfb/backtest_summary.json, oos_predictions.csv, backtest_report.txt")


if __name__ == "__main__":
    main()
