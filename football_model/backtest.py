"""Honest walk-forward backtest of the full pipeline (model -> market blend -> picks), 2018 to date.

Every number is out of sample:
  * the fundamentals model is retrained each week on earlier weeks only
  * the key-number distributions and the blend weight (beta) for season S use seasons < S only
  * bets are settled at the recorded closing prices

Outputs (used by train_final.py / the site):
  output/backtest_summary.json   headline records, beta, info tests, verdict
  output/oos_predictions.csv     every test game with projections, probabilities and picks
  output/backtest_report.txt     the full report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import espn
from betting import BREAKEVEN_110, american_to_decimal, bootstrap_roi, devig_two_way
from features import build_game_features
from margin_dist import ScoreDist
from model import usable, walk_forward
from nfl_data import current_season
from pricing import EV_THRESHOLD, add_model_only, estimate_beta, fair_view, fit_platt, picks

FIRST_TEST_SEASON = 2018
OUT = Path("output")
LOG: list[str] = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def header(t):
    say("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def show(df):
    say(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def record(outcome: pd.Series, breakeven=0.5) -> dict:
    """W-L-P, win% (pushes excluded) and one-sided p-value that the true rate beats `breakeven`."""
    o = pd.Series(outcome).dropna()
    w, l, p = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
    pv = binomtest(w, w + l, breakeven, alternative="greater").pvalue if w + l else np.nan
    return {"W": w, "L": l, "P": p, "win_pct": w / (w + l) if w + l else np.nan, "p_better": pv}


# ------------------------------------------------------------------------------------------------ pipeline
def price_walk_forward(games: pd.DataFrame, oos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Season by season: fit distributions + beta on earlier seasons, price this season."""
    hist = usable(games).dropna(subset=["spread_line", "total_line"])
    parts, betas = [], {}
    for season in sorted(oos["season"].unique()):
        prior = hist[hist["season"] < season]
        md = ScoreDist("margin").fit(prior["result"], prior["spread_line"])
        td = ScoreDist("total").fit(prior["total"], prior["total_line"])
        cur = oos[oos["season"] == season].copy()
        earlier = pd.concat(parts) if parts else pd.DataFrame()
        bm = estimate_beta(earlier["result"], earlier["mkt_margin"], earlier["fund_margin"]) if len(earlier) else {"beta": 0.0}
        bt = estimate_beta(earlier["total"], earlier["mkt_total"], earlier["fund_total"]) if len(earlier) else {"beta": 0.0}
        betas[int(season)] = (bm["beta"], bt["beta"])
        calib = None
        if len(earlier) >= 500:
            e = earlier[earlier["result"] != 0]
            calib = fit_platt(e["p_home_win_raw"], e["result"] > 0)
        priced = picks(fair_view(cur, md, td, bm["beta"], bt["beta"], win_calib=calib))
        priced = add_model_only(priced, md, td, win_calib=calib)
        parts.append(priced.assign(beta_margin=bm["beta"], beta_total=bt["beta"]))
    return pd.concat(parts, ignore_index=True), betas


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """+1 win / -1 loss / 0 push for every pick; profit at the listed price."""
    d = df.copy()
    margin, total = d["result"], d["total"]
    home_su = np.where(d["su_pick"] == d["home_team"], 1, -1)
    d["su_out"] = np.where(margin == 0, 0, np.sign(margin * home_su))
    home_ats = np.where(d["ats_pick"] == d["home_team"], 1, -1)
    d["ats_out"] = np.where(d["ats_pick"].isna(), np.nan, np.sign((margin - d["spread_line"]) * home_ats))
    over = np.where(d["tot_pick"] == "OVER", 1, -1)
    d["tot_out"] = np.where(d["tot_pick"].isna(), np.nan, np.sign((total - d["total_line"]) * over))
    home_ml = np.where(d["ml_pick"] == d["home_team"], 1, -1)
    d["ml_out"] = np.where(d["ml_pick"].isna(), np.nan, np.where(margin == 0, 0, np.sign(margin * home_ml)))
    # model-only leans (pure model, no market input)
    d["mo_su_out"] = np.where(margin == 0, 0, np.sign(margin * np.where(d["mo_su_pick"] == d["home_team"], 1, -1)))
    d["mo_ats_out"] = np.sign((margin - d["spread_line"]) * np.where(d["mo_ats_pick"] == d["home_team"], 1, -1))
    d["mo_tot_out"] = np.sign((total - d["total_line"]) * np.where(d["mo_tot_pick"] == "OVER", 1, -1))
    for m in ("ats", "tot", "ml"):
        dec = american_to_decimal(np.where(pd.isna(d[f"{m}_odds"]), -110.0, d[f"{m}_odds"].astype(float)))
        d[f"{m}_profit"] = np.where(d[f"{m}_out"] > 0, dec - 1, np.where(d[f"{m}_out"] < 0, -1.0, 0.0))
    return d


# ------------------------------------------------------------------------------------------------ report
def accuracy_section(d):
    header("A. PROJECTION ACCURACY vs THE CLOSING LINE   (paired t > 0 = worse than market; |t| > 2 significant)")
    rows = []
    for name, col, tgt, mk in [("MARKET closing spread", "spread_line", "result", None),
                               ("fundamentals model", "fund_margin", "result", "spread_line"),
                               ("fair blend (published)", "fair_margin", "result", "spread_line"),
                               ("MARKET closing total", "total_line", "total", None),
                               ("fundamentals model (total)", "fund_total", "total", "total_line"),
                               ("fair blend (total)", "fair_total", "total", "total_line")]:
        e = d[tgt] - d[col]
        r = {"projection": name, "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(), "t_vs_market": np.nan}
        if mk:
            dd = e ** 2 - (d[tgt] - d[mk]) ** 2
            r["t_vs_market"] = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        rows.append(r)
    show(pd.DataFrame(rows))


def info_section(d):
    header("B. DOES THE MODEL KNOW SOMETHING THE LINE DOESN'T?   slope of (result - market) on (model - market)")
    say("slope 0 = no information beyond the line, 1 = the model is right and the line is wrong. t > 2 is evidence.")
    rows = []
    for name, tgt, mk, mod in [("margin", "result", "mkt_margin", "fund_margin"), ("total", "total", "mkt_total", "fund_total")]:
        for per, sub in [("2018-2022", d[d.season <= 2022]), ("2023+", d[d.season >= 2023]), ("all", d)]:
            b = estimate_beta(sub[tgt], sub[mk], sub[mod])
            rows.append({"target": name, "period": per, "n": b["n"], "slope": b["slope"], "t": b["t"]})
    show(pd.DataFrame(rows))
    return {r["target"]: r for r in rows if r["period"] == "all"}


def winprob_section(d):
    header("C. WIN PROBABILITY QUALITY   (lower log loss / Brier is better)")
    x = d[(d.result != 0) & d.home_moneyline.notna() & d.away_moneyline.notna()]
    y = (x.result > 0).astype(float).to_numpy()
    mh, _ = devig_two_way(x.home_moneyline, x.away_moneyline)

    def sc(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return {"accuracy": ((p >= .5) == (y == 1)).mean(), "brier": ((p - y) ** 2).mean(),
                "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()}
    show(pd.DataFrame([{"source": "MARKET moneyline (de-vigged)", **sc(mh)},
                       {"source": "fair blend (published)", **sc(x.p_home_win)}]))


def records_section(d):
    header("D. PICK RECORD - one pick per game per market, the way the site shows it (closing lines)")
    rows = []
    for label, col, be in [("straight up (winner)", "su_out", 0.5), ("against the spread", "ats_out", BREAKEVEN_110),
                           ("totals (over/under)", "tot_out", BREAKEVEN_110)]:
        for per, sub in [("2018-2023", d[d.season <= 2023]), ("2024+", d[d.season >= 2024]), ("all", d)]:
            rows.append({"market": label, "period": per, **record(sub[col], be)})
    for label, col, be in [("MODEL-ONLY winner", "mo_su_out", 0.5), ("MODEL-ONLY spread", "mo_ats_out", BREAKEVEN_110),
                           ("MODEL-ONLY totals", "mo_tot_out", BREAKEVEN_110)]:
        for per, sub in [("2018-2023", d[d.season <= 2023]), ("2024+", d[d.season >= 2024]), ("all", d)]:
            rows.append({"market": label, "period": per, **record(sub[col], be)})
    show(pd.DataFrame(rows))
    say("p_better = one-sided p-value that the true hit rate beats break-even (50% SU, 52.4% at -110).")
    say("\nBy season:")
    show(d.groupby("season").apply(lambda s: pd.Series({
        "SU": "{W}-{L}".format(**record(s.su_out)), "ATS": "{W}-{L}-{P}".format(**record(s.ats_out)),
        "ATS%": record(s.ats_out)["win_pct"], "O/U": "{W}-{L}-{P}".format(**record(s.tot_out)),
        "beta_m": s.beta_margin.iloc[0], "beta_t": s.beta_total.iloc[0]}), include_groups=False).reset_index())


def value_section(d):
    header(f"E. VALUE BETS - only picks with EV >= {EV_THRESHOLD:.0%} at the closing price")
    rows = []
    for label, m in [("spread", "ats"), ("total", "tot"), ("moneyline", "ml")]:
        for per, sub in [("2018-2023", d[d.season <= 2023]), ("2024+", d[d.season >= 2024]), ("all", d)]:
            b = sub[sub[f"value_{m}"]]
            lo, hi = bootstrap_roi(b[f"{m}_profit"])
            n = len(b)
            t = b[f"{m}_profit"].mean() / b[f"{m}_profit"].std() * np.sqrt(n) if n > 2 else np.nan
            rows.append({"market": label, "period": per, "bets": n, **{k: v for k, v in record(b[f"{m}_out"]).items() if k in "WLP"},
                         "roi": b[f"{m}_profit"].mean() if n else np.nan, "roi_lo": lo, "roi_hi": hi, "t": t})
    show(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def calibration_section(d):
    header("F. CALIBRATION - published win probability vs what happened")
    x = d.assign(y=(d.result > 0).astype(float))
    x["bin"] = pd.qcut(x.p_home_win, 8, duplicates="drop")
    show(x.groupby("bin", observed=True).agg(games=("y", "size"), predicted=("p_home_win", "mean"),
                                             actual=("y", "mean")).reset_index(drop=True))


OPEN_COLS = ["open_spread", "open_home_spread_odds", "open_away_spread_odds", "open_home_ml", "open_away_ml",
             "open_total", "close_spread", "close_total"]


def opening_line_section(d, games):
    header("G. OPENING LINES (ESPN, 2024+) - does the model predict where the line MOVES, and can you bet it?")
    say("Closing line value (CLV) is the professional yardstick: far less noisy than wins and losses.")
    try:
        odds = espn.historical_odds(range(2024, current_season() + 1))
    except Exception as e:                                   # network trouble should not kill the backtest
        say(f"  skipped: could not load ESPN odds ({e})")
        return {}
    if odds.empty:
        say("  skipped: no ESPN odds cached - run `python espn.py`")
        return {}
    x = d.drop(columns=[c for c in OPEN_COLS if c in d]).merge(
        odds[["season", "week", "home_team", "away_team"] + OPEN_COLS],
        on=["season", "week", "home_team", "away_team"], how="inner").dropna(subset=["open_spread", "close_spread"])
    if len(x) < 100:
        say(f"  only {len(x)} games matched - skipped")
        return {}
    move = estimate_beta(x.close_spread, x.open_spread, x.fund_margin)
    tx = x.dropna(subset=["open_total", "close_total"])
    tmove = estimate_beta(tx.close_total, tx.open_total, tx.fund_total)
    say(f"  games: {len(x)}.  Share of (model - open) that the line moves by kickoff: "
        f"spread {move['slope']:.3f} (t={move['t']:.1f}), total {tmove['slope']:.3f} (t={tmove['t']:.1f})")

    # holdout: estimate the move share on 2024 only, then price 2025+ at the OPEN exactly as predict.py would
    train, test = x[x.season == 2024], x[x.season >= 2025].copy()
    bm = estimate_beta(train.close_spread, train.open_spread, train.fund_margin)["beta"]
    bt = estimate_beta(train.close_total, train.open_total, train.fund_total)["beta"]
    hist = usable(games).dropna(subset=["spread_line", "total_line"])
    prior = hist[hist.season < 2025]
    md = ScoreDist("margin").fit(prior.result, prior.spread_line)
    td = ScoreDist("total").fit(prior.total, prior.total_line)
    test = test.assign(o_over=np.nan, o_under=np.nan)
    # with current == open, the anchored fair line is simply open + share * (model - open)
    p2 = fair_view(test, md, td, bm, bt, spread="open_spread", home_odds="open_home_spread_odds",
                   away_odds="open_away_spread_odds", total="open_total", over_odds="o_over", under_odds="o_under")
    p = picks(p2, spread="open_spread", home_odds="open_home_spread_odds", away_odds="open_away_spread_odds",
              total="open_total", over_odds="o_over", under_odds="o_under", home_ml="open_home_ml", away_ml="open_away_ml")
    side = np.where(p.ats_pick == p.home_team, 1, -1)
    ats = np.sign((p.result - p.open_spread) * side)
    clv = (p.close_spread - p.open_spread) * side
    tside = np.where(p.tot_pick == "OVER", 1, -1)
    tot = np.sign((p.total - p.open_total) * tside)
    tclv = (p.close_total - p.open_total) * tside
    say(f"\n  HOLDOUT 2025+ (move share fit on 2024: spread {bm:.3f}, total {bt:.3f}), picks made at the OPEN:")
    rows = []
    for label, o, c, sel in [("ATS every game", ats, clv, np.ones(len(p), bool)),
                             ("ATS value bets (EV>=2%)", ats, clv, p.value_ats.to_numpy()),
                             ("O/U every game", tot, tclv, np.ones(len(p), bool)),
                             ("O/U value bets (EV>=2%)", tot, tclv, p.value_tot.to_numpy())]:
        r = record(pd.Series(o[sel]), BREAKEVEN_110)
        cc = pd.Series(c[sel]).dropna()
        rows.append({"picks": label, "n": int(sel.sum()), "W": r["W"], "L": r["L"], "P": r["P"], "win_pct": r["win_pct"],
                     "avg_CLV_pts": cc.mean(), "CLV_t": cc.mean() / cc.std() * np.sqrt(len(cc)) if len(cc) > 2 else np.nan,
                     "beat_close_pct": (cc > 0).mean(), "p_better": r["p_better"]})
    show(pd.DataFrame(rows))
    say("  beat_close_pct = share of picks where the closing number moved your way (sharp bettors run ~55%+).")
    say("  CAUTION: ESPN's 'open' is often the look-ahead line posted BEFORE the previous week's games, and line moves")
    say("  correlate with last week's results. Part of this CLV is the model already knowing last week's box scores -")
    say("  not something you could have bet. Live CLV in output/ledger.csv (line at the time you run it) is the real test.")
    return {"n": int(len(x)), "move_share_spread": move["slope"], "move_t_spread": move["t"],
            "move_share_total": tmove["slope"], "move_t_total": tmove["t"],
            "beta_open_margin": move["beta"], "beta_open_total": tmove["beta"],
            "holdout_ats": rows[0], "holdout_ats_value": rows[1], "holdout_tot": rows[2]}


def verdict_text(info, recs, clv) -> str:
    t = info["margin"]["t"]
    ats = recs["ats"]
    parts = [f"Backtest {FIRST_TEST_SEASON}-{recs['last']}: straight up {recs['su']['W']}-{recs['su']['L']} "
             f"({recs['su']['win_pct']:.1%}), ATS {ats['W']}-{ats['L']}-{ats['P']} ({ats['win_pct']:.1%})."]
    if ats["win_pct"] > BREAKEVEN_110 and ats["p_better"] < 0.05:
        parts.append("The ATS record beats the -110 break-even with statistical significance.")
    elif ats["win_pct"] > BREAKEVEN_110:
        parts.append(f"ATS is above break-even but not significant (p={ats['p_better']:.2f}).")
    else:
        parts.append("ATS is below the 52.4% needed to profit at -110.")
    parts.append(f"Model-vs-closing-line information: t={t:.1f} "
                 f"({'real signal' if t > 2 else 'no reliable signal'}).")
    if clv:
        h = clv["holdout_ats"]
        parts.append(f"Lines move toward the model ({clv['move_share_spread']:.0%} of its gap with ESPN's opener, "
                     f"t={clv['move_t_spread']:.1f}), but holdout picks at the open went {h['W']}-{h['L']} ATS, so that "
                     f"is not yet a proven, bettable edge. Picks stay market-anchored; track live CLV.")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    LOG.clear()
    cur = current_season()
    games, features = build_game_features(range(2014, cur + 1))
    say(f"features: margin {len(features['margin'])}, total {len(features['total'])}")
    oos = walk_forward(games, features, first_test_season=FIRST_TEST_SEASON)
    oos = oos.dropna(subset=["spread_line", "total_line"])
    priced, betas = price_walk_forward(games, oos)
    d = grade(priced)
    say(f"out-of-sample games: {len(d)} ({d.season.min()}-{d.season.max()})")

    accuracy_section(d)
    info = info_section(d)
    winprob_section(d)
    records_section(d)
    value_section(d)
    calibration_section(d)
    clv = opening_line_section(d, games)

    recs = {"su": record(d.su_out), "ats": record(d.ats_out, BREAKEVEN_110), "tot": record(d.tot_out, BREAKEVEN_110),
            "mo_su": record(d.mo_su_out), "mo_ats": record(d.mo_ats_out, BREAKEVEN_110),
            "mo_tot": record(d.mo_tot_out, BREAKEVEN_110),
            "last": int(d.season.max())}
    verdict = verdict_text(info, recs, clv)
    say("\n" + verdict)

    summary = {
        "test_start": FIRST_TEST_SEASON, "test_end": recs["last"], "games": int(len(d)),
        "records": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()} for k, v in recs.items() if k != "last"},
        "info_margin": info["margin"], "info_total": info["total"], "opening_line_test": clv,
        "verdict": verdict,
    }
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    d.to_csv(OUT / "oos_predictions.csv", index=False)
    (OUT / "backtest_report.txt").write_text("\n".join(LOG), encoding="utf-8")
    say("\nsaved: output/backtest_summary.json, output/oos_predictions.csv, output/backtest_report.txt")


if __name__ == "__main__":
    main()
