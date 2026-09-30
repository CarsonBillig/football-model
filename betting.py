"""Odds conversion, de-vigging, probabilities, and bet grading."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest

BREAKEVEN_110 = 110 / 210
WIN_PAYOUT_110 = 100 / 110


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / np.abs(odds))


def american_to_prob(odds):
    return 1 / american_to_decimal(odds)


def devig_two_way(odds_a, odds_b):
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s


def margin_to_win_prob(margin, sigma):
    return norm.cdf(np.asarray(margin) / np.asarray(sigma))


def cover_prob(pred_margin, spread_line, sigma):
    """
    nflverse spread_line is positive when the home team is favored.

    Example:
      spread_line = +3.5 means home is favored by 3.5.
      Home covers when actual_margin > spread_line.

    Therefore:
      P(home cover) = Phi((pred_margin - spread_line) / sigma)
    """
    return norm.cdf(
        (np.asarray(pred_margin) - np.asarray(spread_line))
        / np.asarray(sigma)
    )


def grade_spread(actual_margin, spread_line):
    # +1 home cover, -1 home fails, 0 push.
    # nflverse spread_line is positive when home is favored.
    value = np.asarray(actual_margin) - np.asarray(spread_line)
    return np.sign(value)


def grade_moneyline(actual_margin):
    return np.sign(np.asarray(actual_margin))


def american_profit(outcome, odds=-110):
    """Profit in units when 1 unit is risked."""
    outcome = np.asarray(outcome)
    dec = american_to_decimal(odds)
    return np.where(outcome > 0, dec - 1, np.where(outcome < 0, -1.0, 0.0))


def summarize_bets(outcomes, profit=None, label=""):
    outcomes = pd.Series(outcomes).astype(float)
    w = int((outcomes > 0).sum())
    l = int((outcomes < 0).sum())
    p = int((outcomes == 0).sum())
    if profit is None:
        profit = pd.Series(
            outcomes.map({1: WIN_PAYOUT_110, -1: -1.0, 0: 0.0}),
            index=outcomes.index,
        )
    profit = pd.Series(profit, index=outcomes.index)
    n = w + l
    pval = binomtest(w, n, BREAKEVEN_110).pvalue if n else np.nan
    return {
        "label": label,
        "bets": w + l + p,
        "W": w,
        "L": l,
        "P": p,
        "win_pct": w / n if n else np.nan,
        "roi": profit.sum() / (w + l + p) if (w + l + p) else np.nan,
        "units": profit.sum(),
        "p_vs_breakeven": pval,
    }


# ============================================================================
# v2 additions: vectorized bet settlement, bootstrap intervals, line formatting
# ============================================================================
DEFAULT_ODDS = -110.0


def spread_bets(df: pd.DataFrame, edge: pd.Series, thr: float) -> pd.DataFrame:
    """Bet HOME when edge >= thr, AWAY when edge <= -thr.

    edge = model_margin - spread_line (nflverse: spread_line > 0 means HOME favored).
    Settles at the real spread odds (falls back to -110 if a price is missing)."""
    sel = edge.abs() >= thr
    b = df.loc[sel].copy()
    side = np.where(edge[sel] > 0, 1, -1)
    margin = b["home_score"] - b["away_score"]
    outcome = np.sign((margin - b["spread_line"]) * side)
    odds = np.where(side > 0, b.get("home_spread_odds"), b.get("away_spread_odds")).astype(float)
    odds = np.where(np.isnan(odds), DEFAULT_ODDS, odds)
    return b.assign(side=side, outcome=outcome, odds=odds, profit=american_profit(outcome, odds))


def total_bets(df: pd.DataFrame, edge: pd.Series, thr: float) -> pd.DataFrame:
    """Bet OVER when edge >= thr, UNDER when edge <= -thr. edge = model_total - total_line."""
    sel = edge.abs() >= thr
    b = df.loc[sel].copy()
    side = np.where(edge[sel] > 0, 1, -1)
    total = b["home_score"] + b["away_score"]
    outcome = np.sign((total - b["total_line"]) * side)
    odds = np.where(side > 0, b.get("over_odds"), b.get("under_odds")).astype(float)
    odds = np.where(np.isnan(odds), DEFAULT_ODDS, odds)
    return b.assign(side=side, outcome=outcome, odds=odds, profit=american_profit(outcome, odds))


def moneyline_bets(df: pd.DataFrame, p_home: pd.Series, thr: float) -> pd.DataFrame:
    """Bet the side whose model win prob beats the de-vigged market prob by >= thr."""
    mh, ma = devig_two_way(df["home_moneyline"], df["away_moneyline"])
    e_home, e_away = p_home.values - mh, (1 - p_home.values) - ma
    take_home = (e_home >= thr) & (e_home >= e_away)
    take_away = (e_away >= thr) & ~take_home
    sel = take_home | take_away
    b = df.loc[sel].copy()
    side = np.where(take_home[sel], 1, -1)
    margin = b["home_score"] - b["away_score"]
    outcome = np.sign(margin * side)
    odds = np.where(side > 0, b["home_moneyline"], b["away_moneyline"]).astype(float)
    return b.assign(side=side, outcome=outcome, odds=odds, profit=american_profit(outcome, odds))


def bootstrap_roi(profit, n_boot=4000, seed=7):
    """95% bootstrap interval for ROI per bet. Wide intervals = you can't tell yet."""
    p = np.asarray(profit, dtype=float)
    if len(p) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(p, size=(n_boot, len(p)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def bet_row(b: pd.DataFrame, label) -> dict:
    """One results-table row for a set of settled bets."""
    o, pr = pd.Series(b["outcome"].values), pd.Series(b["profit"].values)
    r = summarize_bets(o, pr, label=label)
    lo, hi = bootstrap_roi(pr)
    r.update(roi_lo=lo, roi_hi=hi)
    return r


def fmt_line(team: str, line: float) -> str:
    """Format a team's own betting line: negative = favorite. fmt_line('GB', -4.5) -> 'GB -4.5'."""
    if line is None or pd.isna(line):
        return f"{team} N/A"
    if abs(line) < 1e-9:
        return f"{team} PK"
    return f"{team} {line:+.1f}"


def home_line(spread_line: float) -> float:
    """nflverse spread_line is HOME-favored-positive; a sportsbook line is the opposite sign."""
    return -spread_line
