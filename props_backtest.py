"""Backtest NFL player prop picks against the book's real lines and prices (ESPN feed, 2024+).

Rule set before running: pick each market's edge threshold on 2024 only; a market passes if its picks then make
money at the actual prices in 2025-26 over at least 300 bets. Also reports the product we'd actually show:
the top 2-3 picks per game.

    python props_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import props_model as pm

THRESHOLDS = (0.02, 0.04, 0.06, 0.08, 0.10)
MIN_TUNE_BETS = 150
MAX_GAP = 0.25          # projection more than 25% away from the line = the model is missing something: no pick


def dec(odds):
    o = np.asarray(odds, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / -o)


def calib_features(x: pd.DataFrame) -> np.ndarray:
    """How far our projection is from the line, and the book's own (no-margin) chance of the over. The book's price
    carries information (a -140 over usually means something), so it goes in too."""
    io, iu = 1 / dec(x["over_odds"]), 1 / dec(x["under_odds"])
    fair = np.clip(io / (io + iu), 0.02, 0.98)
    return np.column_stack([(x["proj"] / x["line"] - 1).clip(-MAX_GAP, MAX_GAP), np.log(fair / (1 - fair))])


def load_lines() -> pd.DataFrame:
    """One row per game, player, market and book: the main line with both prices, now and at the open."""
    files = sorted(Path("cache/props").glob("*/*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # DraftKings' 2026 lines come without prices: grade them at the standard -110 on both sides
    raw[["odds", "open_odds"]] = raw[["odds", "open_odds"]].astype(float).fillna(-110.0)
    keys = ["season", "week", "event_id", "provider", "espn_athlete_id", "market"]
    out = []
    for when, line, odds in (("now", "line", "odds"), ("open", "open_line", "open_odds")):
        w = raw.pivot_table(index=keys + [line], columns="side", values=odds, aggfunc="first").reset_index()
        w = w.dropna(subset=["over", "under"]).rename(columns={line: "line", "over": "over_odds", "under": "under_odds"})
        # main line = the one priced closest to even (a few players also have an alternate line priced both ways)
        w["_tilt"] = np.abs(1 / dec(w["over_odds"]) - 1 / dec(w["under_odds"]))
        w = w.sort_values("_tilt").drop_duplicates(keys).drop(columns="_tilt")
        out.append(w.assign(when=when))
    return pd.concat(out, ignore_index=True)


def grade(lines: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    import nflreadpy as nfl
    players = nfl.load_players().to_pandas()[["gsis_id", "espn_id"]].dropna()
    players["espn_id"] = players["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    lines = lines.merge(players.rename(columns={"espn_id": "espn_athlete_id", "gsis_id": "player_id"}), on="espn_athlete_id")
    matched = []
    for market, (stat, vol, _, _) in pm.MARKETS.items():
        for season, te, ratios in pm.walk_forward(pg, market, 2024):
            x = lines[(lines["market"] == market) & (lines["season"] == season)].merge(
                te[["player_id", "season", "week", "proj", stat, vol]], on=["player_id", "season", "week"])
            if len(x):
                matched.append((market, stat, x.assign(actual=x[stat]), ratios))
    # Chance of going over, calibrated directly against the book's lines on 2024 (the tuning season): a logistic fit
    # of "went over" on how far our projection sat from the line. Lines sit near the median outcome (yardage has a
    # long right tail), and the book's line is a bit more accurate than our projection; this handles both.
    from sklearn.linear_model import LogisticRegression
    gap = calib_features
    calib = {}
    for market in pm.MARKETS:
        for when in ("now", "open"):
            t = pd.concat([x for m, _, x, _ in matched if m == market and x["season"].iloc[0] == 2024])
            t = t[(t["when"] == when) & (t["actual"] != t["line"])]
            calib[(market, when)] = LogisticRegression(C=1e6).fit(gap(t), (t["actual"] > t["line"]).astype(int))
    print("calibration on 2024 (intercept, gap slope, book-price slope):", {f"{m}/{w}": (round(float(c.intercept_[0]), 2), *(round(float(v), 2) for v in c.coef_[0]))
                                                     for (m, w), c in calib.items()})
    frames = []
    for market, stat, x, ratios in matched:
        x = x[(x["proj"] / x["line"] - 1).abs() <= MAX_GAP].copy()
        if x.empty:
            continue
        x["p_over"] = np.nan
        for when, idx in x.groupby("when").groups.items():
            x.loc[idx, "p_over"] = calib[(market, when)].predict_proba(gap(x.loc[idx]))[:, 1]
        io, iu = 1 / dec(x["over_odds"]), 1 / dec(x["under_odds"])
        x["fair_over"] = io / (io + iu)
        x["edge_over"] = x["p_over"] - x["fair_over"]
        x["edge_under"] = (1 - x["p_over"]) - (1 - x["fair_over"])
        x["side"] = np.where(x["edge_over"] >= x["edge_under"], "over", "under")
        x["edge"] = np.maximum(x["edge_over"], x["edge_under"])
        x["price"] = np.where(x["side"] == "over", x["over_odds"], x["under_odds"])
        diff = (x[stat] - x["line"]) * np.where(x["side"] == "over", 1, -1)
        x["result"] = np.sign(diff)
        x["units"] = np.where(x["result"] > 0, dec(x["price"]) - 1, np.where(x["result"] < 0, -1.0, 0.0))
        frames.append(x)
    return pd.concat(frames, ignore_index=True)


def summarize(b: pd.DataFrame, label: str) -> str:
    n = len(b)
    if not n:
        return f"{label}: no bets"
    w, l = int((b.result > 0).sum()), int((b.result < 0).sum())
    return f"{label}: {n} bets, {w}-{l} ({w / max(w + l, 1):.1%}), {b.units.sum():+.1f} units, ROI {b.units.sum() / n:+.1%}"


def main():
    pg = pm.player_games(range(2018, 2027))
    g = grade(load_lines(), pg)
    g.to_parquet("cache/props_graded.parquet")
    for when in ("open", "now"):
        print(f"\n===== lines at the {'OPEN' if when == 'open' else 'later snapshot (closer to kickoff)'} =====")
        gw = g[g["when"] == when]
        for market in pm.MARKETS:
            gm = gw[gw["market"] == market]
            tune, hold = gm[gm.season == 2024], gm[gm.season >= 2025]
            best = max((t for t in THRESHOLDS if (tune.edge >= t).sum() >= MIN_TUNE_BETS),
                       key=lambda t: tune[tune.edge >= t].units.mean(), default=None)
            if best is None:
                print(f"  {market}: too few 2024 bets to tune")
                continue
            h = hold[hold.edge >= best]
            verdict = "PASS" if len(h) >= 300 and h.units.sum() > 0 else "fail"
            print(f"  {market:10s} threshold {best:.0%} (set on 2024) | {summarize(tune[tune.edge >= best], '2024')}")
            print(f"  {'':10s} {verdict:>15s} | {summarize(h, 'holdout 2025-26')}")
        # the product: top 2-3 picks per game (largest edges), holdout seasons only
        top = gw[gw.season >= 2025].sort_values("edge", ascending=False).groupby("event_id").head(3)
        print("  " + summarize(top, "top 3 per game, holdout 2025-26"))


if __name__ == "__main__":
    main()
