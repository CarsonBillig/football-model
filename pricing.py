"""Turn model projections + market lines into fair probabilities, picks and bet values.

Fair centre of the outcome distribution:

    mu_fair = mu_market + beta * (model - mu_market)

* mu_market  - where the market's price says the margin/total is centred (spread + juice, key-number aware)
* model      - the independent fundamentals projection
* beta       - how much of the model's disagreement has actually shown up in results, estimated out of sample
               (backtest + graded live picks). beta = 0 means "trust the market", 1 means "trust the model".

Probabilities then come from the key-number-aware distributions in margin_dist.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from betting import american_to_decimal, devig_two_way
from margin_dist import ScoreDist

EV_THRESHOLD = 0.02          # flag a bet only when expected value is at least +2% of stake
KELLY_FRACTION = 0.25        # quarter Kelly: full Kelly is far too aggressive for noisy edge estimates
MAX_STAKE = 0.03             # never more than 3% of bankroll on one bet


# ------------------------------------------------------------------------------------------- beta
def estimate_beta(result, market_mu, model) -> dict:
    """Slope of (result - market) on (model - market), clipped to [0, 1], with its t-stat.

    This is the honest "does the model know something the line doesn't?" test. slope ~ 0 -> no."""
    y = np.asarray(result, float) - np.asarray(market_mu, float)
    x = np.asarray(model, float) - np.asarray(market_mu, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 50 or np.var(x) == 0:
        return {"beta": 0.0, "slope": np.nan, "t": np.nan, "n": int(len(x))}
    slope = float((x @ y) / (x @ x))                       # through the origin: no model gap -> no adjustment
    resid = y - slope * x
    se = float(np.sqrt((resid @ resid) / (len(x) - 1) / (x @ x)))
    return {"beta": float(np.clip(slope, 0.0, 1.0)), "slope": slope, "t": slope / se, "n": int(len(x))}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(p, calib):
    """Apply logit(p') = a * logit(p) + b; identity when calib is None."""
    if calib is None:
        return np.asarray(p, float)
    a, b = calib
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def fit_platt(p, won) -> tuple:
    """Fit the (a, b) Platt correction on out-of-sample probabilities and 0/1 outcomes."""
    from sklearn.linear_model import LogisticRegression
    x, y = _logit(p), np.asarray(won, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    m = LogisticRegression(C=1e4).fit(x[ok, None], y[ok])
    return float(m.coef_[0, 0]), float(m.intercept_[0])


# ------------------------------------------------------------------------------------------- fair view
def fair_view(df: pd.DataFrame, mdist: ScoreDist, tdist: ScoreDist, beta_margin: float, beta_total: float,
              spread="spread_line", home_odds="home_spread_odds", away_odds="away_spread_odds",
              total="total_line", over_odds="over_odds", under_odds="under_odds",
              open_move: tuple | None = None, win_calib: tuple | None = None) -> pd.DataFrame:
    """Adds mkt_mu / fair_margin / fair_total and win-cover-over probabilities.

    Games with no line yet fall back to the pure model (beta effectively 1).
    open_move = (b_margin, b_total): share of (model - opening line) that historically shows up as line movement.
    win_calib = (a, b) Platt correction for the win probability, fitted out of sample."""
    out = df.copy()
    n = len(out)
    fm, ft = out["fund_margin"].to_numpy(float), out["fund_total"].to_numpy(float)

    s = out[spread].to_numpy(float) if spread in out else np.full(n, np.nan)
    has_s = ~np.isnan(s)
    mkt_m = np.full(n, np.nan)
    if has_s.any():
        mkt_m[has_s] = mdist.solve_mu(s[has_s], out.loc[has_s, home_odds].to_numpy(float),
                                      out.loc[has_s, away_odds].to_numpy(float))
    out["mkt_margin"] = mkt_m
    out["fair_margin"] = np.where(has_s, mkt_m + beta_margin * (fm - mkt_m), fm)

    t = out[total].to_numpy(float) if total in out else np.full(n, np.nan)
    has_t = ~np.isnan(t)
    mkt_t = np.full(n, np.nan)
    if has_t.any():
        mkt_t[has_t] = tdist.solve_mu(t[has_t], out.loc[has_t, over_odds].to_numpy(float),
                                      out.loc[has_t, under_odds].to_numpy(float))
    out["mkt_total"] = mkt_t
    out["fair_total"] = np.where(has_t, mkt_t + beta_total * (ft - mkt_t), ft)

    # Opening-line anchoring: the model predicts where lines CLOSE (backtest section G). The pick leans toward the
    # part of that expected move which has not happened yet; if the line already got there (or moved the other
    # way on news the model can't see), nothing extra is added.
    if open_move is not None:
        b_m, b_t = open_move
        for kind, fair_col, mkt, fund, dist, b, cols in (
                ("margin", "fair_margin", mkt_m, fm, mdist, b_m, ("open_spread", "open_home_spread_odds", "open_away_spread_odds")),
                ("total", "fair_total", mkt_t, ft, tdist, b_t, ("open_total", "open_over_odds", "open_under_odds"))):
            if cols[0] not in out or not b:
                continue
            ol = out[cols[0]].to_numpy(float)
            ok = ~(np.isnan(ol) | np.isnan(mkt))
            if not ok.any():
                continue
            open_mu = np.full(n, np.nan)
            open_mu[ok] = dist.solve_mu(ol[ok], out.loc[ok, cols[1]].to_numpy(float) if cols[1] in out else None,
                                        out.loc[ok, cols[2]].to_numpy(float) if cols[2] in out else None)
            expected = b * (fund - open_mu)
            left = expected - (mkt - open_mu)
            remaining = np.sign(expected) * np.clip(np.sign(expected) * left, 0, np.abs(expected))
            out[f"{kind}_expected_close"] = open_mu + expected
            out[fair_col] = np.where(ok, mkt + np.nan_to_num(remaining), out[fair_col])

    out["fair_spread"] = mdist.fair_line(out["fair_margin"].to_numpy(float))
    out["fair_total_line"] = tdist.fair_line(out["fair_total"].to_numpy(float))
    wp = mdist.win_probs(out["fair_margin"].to_numpy(float))
    out["p_home_win_raw"] = wp["p_home_win_ex_tie"]
    out["p_home_win"] = platt(wp["p_home_win_ex_tie"], win_calib)
    out["proj_home"] = (out["fair_total"] + out["fair_margin"]) / 2
    out["proj_away"] = (out["fair_total"] - out["fair_margin"]) / 2

    if has_s.any():
        pc, pp = mdist.prob_over(out["fair_margin"].to_numpy(float), np.nan_to_num(s))
        out["p_home_cover"] = np.where(has_s, pc, np.nan)
        out["p_spread_push"] = np.where(has_s, pp, np.nan)
    else:
        out["p_home_cover"] = out["p_spread_push"] = np.nan
    if has_t.any():
        po, pq = tdist.prob_over(out["fair_total"].to_numpy(float), np.nan_to_num(t))
        out["p_over"] = np.where(has_t, po, np.nan)
        out["p_total_push"] = np.where(has_t, pq, np.nan)
    else:
        out["p_over"] = out["p_total_push"] = np.nan
    return out


# ------------------------------------------------------------------------------------------- bet maths
def ev_and_kelly(p_win, p_push, odds):
    """Expected profit per unit staked and the (fractional, capped) Kelly stake as a share of bankroll."""
    p_win, p_push = np.asarray(p_win, float), np.asarray(p_push, float)
    b = american_to_decimal(np.where(np.isnan(odds), -110.0, odds)) - 1
    p_lose = 1 - p_win - p_push
    ev = p_win * b - p_lose
    with np.errstate(divide="ignore", invalid="ignore"):
        full = (b * p_win - p_lose) / (b * (p_win + p_lose))
    return ev, np.clip(KELLY_FRACTION * np.nan_to_num(full), 0, MAX_STAKE)


def picks(df: pd.DataFrame, spread="spread_line", home_odds="home_spread_odds", away_odds="away_spread_odds",
          total="total_line", over_odds="over_odds", under_odds="under_odds",
          home_ml="home_moneyline", away_ml="away_moneyline") -> pd.DataFrame:
    """One pick per market per game (the more likely side), with probability, EV and suggested stake.

    `value_*` is True only when the pick is +EV by at least EV_THRESHOLD at the listed price."""
    o = df.copy()
    home, away = o["home_team"], o["away_team"]

    # straight-up winner
    o["su_pick"] = np.where(o["p_home_win"] >= 0.5, home, away)
    o["su_prob"] = np.maximum(o["p_home_win"], 1 - o["p_home_win"])

    # spread: side with the higher fair cover probability at the listed number
    ph, pp = o["p_home_cover"], o["p_spread_push"]
    pa = 1 - ph - pp
    home_side = ph >= pa
    o["ats_pick"] = np.where(o[spread].isna(), None, np.where(home_side, home, away))
    o["ats_line"] = np.where(home_side, -o[spread], o[spread])            # the picked team's own line
    o["ats_prob"] = np.where(home_side, ph, pa)
    o["ats_odds"] = np.where(home_side, o[home_odds], o[away_odds])
    o["ats_ev"], o["ats_stake"] = ev_and_kelly(o["ats_prob"], pp, o["ats_odds"])
    o["ats_prob_np"] = o["ats_prob"] / (ph + pa)                     # ignoring pushes, the way bettors quote it

    # total
    po, pq = o["p_over"], o["p_total_push"]
    pu = 1 - po - pq
    over = po >= pu
    o["tot_pick"] = np.where(o[total].isna(), None, np.where(over, "OVER", "UNDER"))
    o["tot_prob"] = np.where(over, po, pu)
    o["tot_odds"] = np.where(over, o[over_odds], o[under_odds])
    o["tot_ev"], o["tot_stake"] = ev_and_kelly(o["tot_prob"], pq, o["tot_odds"])
    o["tot_prob_np"] = o["tot_prob"] / (po + pu)

    # moneyline: whichever side has the better EV at its price. The moneyline market's own de-vigged price is
    # averaged in (logit scale): the backtest showed spread-derived probabilities alone flag moneyline "value"
    # that loses money, so a flag now needs the model to disagree with BOTH markets.
    hml, aml = o[home_ml].to_numpy(float), o[away_ml].to_numpy(float)
    has_ml = ~(np.isnan(hml) | np.isnan(aml))
    p_ml = o["p_home_win"].to_numpy(float).copy()
    if has_ml.any():
        mh, _ = devig_two_way(hml[has_ml], aml[has_ml])
        p_ml[has_ml] = 1 / (1 + np.exp(-(_logit(p_ml[has_ml]) + _logit(mh)) / 2))
    o["p_home_win_ml"] = p_ml
    ev_h, k_h = ev_and_kelly(p_ml, np.zeros(len(o)), hml)
    ev_a, k_a = ev_and_kelly(1 - p_ml, np.zeros(len(o)), aml)
    take_h = ev_h >= ev_a
    o["ml_pick"] = np.where(has_ml, np.where(take_h, home, away), None)
    o["ml_odds"] = np.where(take_h, hml, aml)
    o["ml_prob"] = np.where(take_h, p_ml, 1 - p_ml)
    o["ml_ev"], o["ml_stake"] = np.where(take_h, ev_h, ev_a), np.where(take_h, k_h, k_a)
    if has_ml.any():
        mh, _ = devig_two_way(hml, aml)
        o["mkt_p_home_win"] = mh

    for m in ("ats", "tot", "ml"):
        o[f"value_{m}"] = (o[f"{m}_ev"] >= EV_THRESHOLD) & o[f"{m}_pick"].notna()
        o.loc[~o[f"value_{m}"], f"{m}_stake"] = 0.0
    return o


# ------------------------------------------------------------------------------------------- model-only view
MODEL_ONLY_COLS = ["proj_home", "proj_away", "fair_spread", "fair_total_line", "p_home_win", "su_pick", "su_prob",
                   "ats_pick", "ats_line", "ats_prob_np", "tot_pick", "tot_prob_np"]


def add_model_only(df: pd.DataFrame, mdist: ScoreDist, tdist: ScoreDist, win_calib=None, **cols) -> pd.DataFrame:
    """Adds mo_* columns: projections and picks from the pure model, ignoring the market (beta = 1).

    These are LEANS. In the backtest this view lost against closing lines; it is tracked separately so the
    pick history shows whether it earns its keep."""
    view_keys = ("spread", "home_odds", "away_odds", "total", "over_odds", "under_odds")
    fv = fair_view(df, mdist, tdist, 1.0, 1.0, win_calib=win_calib, **{k: v for k, v in cols.items() if k in view_keys})
    mo = picks(fv, **cols)
    out = df.copy()
    for c in MODEL_ONLY_COLS:
        out[f"mo_{c}"] = mo[c].values
    return out


# ------------------------------------------------------------------------------------------- tested hit rates
HIT_MARKETS = {"su": ("mo_su_prob", "mo_su_out"), "ats": ("mo_ats_prob_np", "mo_ats_out"), "tot": ("mo_tot_prob_np", "mo_tot_out")}


def fit_hit_rates(oos: pd.DataFrame) -> dict:
    """For each model-only market, map 'confidence the model stated' -> 'how often picks like that actually hit'
    (isotonic fit on out-of-sample games, pushes excluded). Stored in model_meta.json."""
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for m, (pc, oc) in HIT_MARKETS.items():
        if pc not in oos or oc not in oos:
            continue
        x = oos[[pc, oc]].dropna()
        x = x[x[oc] != 0]
        if len(x) < 200:
            continue
        # bins of >= ~250 picks, each shrunk toward the overall hit rate (a lucky handful can't create confidence),
        # then forced to rise with stated confidence
        hit = (x[oc] > 0).astype(float)
        overall = hit.mean()
        n_bins = int(np.clip(len(x) // 250, 2, 12))
        b = pd.qcut(x[pc], n_bins, duplicates="drop")
        g = pd.DataFrame({"p": x[pc], "hit": hit, "b": b}).groupby("b", observed=True).agg(p=("p", "mean"), hits=("hit", "sum"), n=("hit", "size"))
        k = 200.0
        # spreads/totals: no evidence of skill -> shrink toward the overall hit rate (about 50%).
        # moneyline: the stated chance was already close to right -> shrink toward the stated chance itself.
        prior = g["p"] if m == "su" else overall
        g["rate"] = (g["hits"] + k * prior) / (g["n"] + k)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(g["p"].to_numpy(float), g["rate"].to_numpy(float), sample_weight=g["n"].to_numpy(float))
        out[m] = {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
    # Moneyline when a price exists: learn the hit rate from BOTH the model-only chance and the sportsbook's chance.
    # In testing, when the two disagreed, results followed the sportsbook, so this keeps the model from crying wolf.
    need = {"mo_su_pick", "mo_su_prob", "home_team", "home_moneyline", "away_moneyline", "result"}
    if need <= set(oos.columns):
        x = oos.dropna(subset=list(need))
        x = x[x["result"] != 0]
        if len(x) >= 300:
            from sklearn.linear_model import LogisticRegression
            ph, _ = devig_two_way(x["home_moneyline"].to_numpy(float), x["away_moneyline"].to_numpy(float))
            home = (x["mo_su_pick"] == x["home_team"]).to_numpy()
            mkt = np.where(home, ph, 1 - ph)
            won = np.where(home, x["result"] > 0, x["result"] < 0).astype(int)
            Z = np.column_stack([_logit(mkt), _logit(x["mo_su_prob"].to_numpy(float))])
            lr = LogisticRegression(C=1.0).fit(Z, won)
            out["su_vs_market"] = {"coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0])}
    return out


def tested_hit_rate(table: dict, market: str, p) -> np.ndarray:
    """Apply fit_hit_rates' mapping; returns NaN where no mapping exists."""
    p = np.asarray(p, float)
    t = (table or {}).get(market)
    if not t:
        return np.full(p.shape, np.nan)
    return np.where(np.isnan(p), np.nan, np.interp(p, t["x"], t["y"]))


def add_tested_hit_rates(df: pd.DataFrame, table: dict, home_ml: str = "home_moneyline",
                         away_ml: str = "away_moneyline") -> pd.DataFrame:
    out = df.copy()
    for m, (pc, _) in HIT_MARKETS.items():
        if pc in out:
            out[f"mo_{m}_hit"] = tested_hit_rate(table, m, out[pc].to_numpy(float))
    sv = (table or {}).get("su_vs_market")
    if sv and {home_ml, away_ml, "mo_su_pick", "mo_su_prob"} <= set(out.columns):
        h, a = out[home_ml].to_numpy(float), out[away_ml].to_numpy(float)
        ok = ~(np.isnan(h) | np.isnan(a))
        if ok.any():
            ph = np.full(len(out), np.nan)
            ph[ok], _ = devig_two_way(h[ok], a[ok])
            home = (out["mo_su_pick"] == out["home_team"]).to_numpy()
            mkt = np.where(home, ph, 1 - ph)
            z = sv["coef"][0] * _logit(mkt) + sv["coef"][1] * _logit(out["mo_su_prob"].to_numpy(float)) + sv["intercept"]
            out["mo_su_hit"] = np.where(ok, 1 / (1 + np.exp(-z)), out["mo_su_hit"])
    return out
