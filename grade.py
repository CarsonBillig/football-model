"""Grade the pick history against final scores; report records, ROI and closing line value (CLV).

CLV = how much better the number you logged was than the closing number. Positive average CLV over many picks is
the earliest reliable sign of a real edge - much less noisy than wins and losses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ledger
from betting import BREAKEVEN_110, american_to_decimal
from nfl_data import load_schedule


def _profit(out, odds):
    dec = american_to_decimal(np.where(pd.isna(odds), -110.0, odds.astype(float)))
    return np.where(out > 0, dec - 1, np.where(out < 0, -1.0, 0.0))


def grade_ledger(df: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """schedule: game_id, home_score, away_score, spread_line, total_line, played. Defaults to the NFL schedule."""
    if df.empty:
        return df
    if schedule is None:
        schedule = load_schedule(sorted(df["season"].dropna().astype(int).unique()))
    s = schedule.drop_duplicates("game_id").set_index("game_id")
    todo = df["su_out"].isna() & df["game_id"].isin(s.index[s["played"]])
    if not todo.any():
        return df
    d = df.loc[todo].copy()
    g = s.loc[d["game_id"]]
    hs, as_ = g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)
    margin, total = hs - as_, hs + as_
    home = d["home_team"].to_numpy()
    d["home_score"], d["away_score"] = hs, as_
    d["su_out"] = np.where(margin == 0, 0, np.sign(margin * np.where(d["su_pick"] == home, 1, -1)))
    ats_side = np.where(d["ats_pick"] == home, 1, -1)
    d["ats_out"] = np.where(d["ats_pick"].isna(), np.nan, np.sign((margin - d["spread"].astype(float)) * ats_side))
    tot_side = np.where(d["tot_pick"] == "OVER", 1, -1)
    d["tot_out"] = np.where(d["tot_pick"].isna(), np.nan, np.sign((total - d["total_line"].astype(float)) * tot_side))
    d["ml_out"] = np.where(d["ml_pick"].isna(), np.nan,
                           np.where(margin == 0, 0, np.sign(margin * np.where(d["ml_pick"] == home, 1, -1))))
    if "mo_su_pick" in d:
        d["mo_su_out"] = np.where(d["mo_su_pick"].isna(), np.nan,
                                  np.where(margin == 0, 0, np.sign(margin * np.where(d["mo_su_pick"] == home, 1, -1))))
        d["mo_ats_out"] = np.where(d["mo_ats_pick"].isna(), np.nan, np.sign(
            (margin - d["spread"].astype(float)) * np.where(d["mo_ats_pick"] == home, 1, -1)))
        d["mo_tot_out"] = np.where(d["mo_tot_pick"].isna(), np.nan, np.sign(
            (total - d["total_line"].astype(float)) * np.where(d["mo_tot_pick"] == "OVER", 1, -1)))
    for m in ("ats", "tot", "ml"):
        d[f"{m}_profit"] = _profit(d[f"{m}_out"].to_numpy(float), d[f"{m}_odds"])
    d["close_spread"], d["close_total"] = g["spread_line"].to_numpy(float), g["total_line"].to_numpy(float)
    # home-favored-positive: a home pick gains when the close is bigger than the number you got
    d["ats_clv"] = (d["close_spread"] - d["spread"].astype(float)) * ats_side
    d["tot_clv"] = (d["close_total"] - d["total_line"].astype(float)) * tot_side
    # first posted picks (what you would have bet early in the week)
    if "early_ats_pick" in d:
        e_ats = np.where(d["early_ats_pick"] == home, 1, -1)
        e_spread = pd.to_numeric(d["early_spread"], errors="coerce").to_numpy(float)
        d["early_ats_out"] = np.where(d["early_ats_pick"].isna(), np.nan, np.sign((margin - e_spread) * e_ats))
        d["early_ats_clv"] = np.where(d["early_ats_pick"].isna(), np.nan, (d["close_spread"] - e_spread) * e_ats)
        e_tot = np.where(d["early_tot_pick"] == "OVER", 1, -1)
        e_total = pd.to_numeric(d["early_total"], errors="coerce").to_numpy(float)
        d["early_tot_out"] = np.where(d["early_tot_pick"].isna(), np.nan, np.sign((total - e_total) * e_tot))
        d["early_tot_clv"] = np.where(d["early_tot_pick"].isna(), np.nan, (d["close_total"] - e_total) * e_tot)
    for c in d.columns:                             # the ledger can load all-NaN columns as float; allow mixed types
        if df[c].dtype != d[c].dtype:
            df[c] = df[c].astype(object)
    df.loc[todo, d.columns] = d
    return df


def summary(df: pd.DataFrame) -> dict:
    """Records for the site and the console."""
    g = df[df["su_out"].notna()]
    out = {}
    for m, be in (("su", 0.5), ("ats", BREAKEVEN_110), ("tot", BREAKEVEN_110), ("ml", None),
                  ("mo_su", 0.5), ("mo_ats", BREAKEVEN_110), ("mo_tot", BREAKEVEN_110)):
        o = g[f"{m}_out"].dropna().astype(float)
        w, l, p = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
        out[m] = {"W": w, "L": l, "P": p, "pct": w / (w + l) if w + l else None}
    for m in ("ats", "tot", "ml"):
        v = g[g[f"value_{m}"].astype(str).str.lower() == "true"]
        o = v[f"{m}_out"].dropna().astype(float)
        out[f"value_{m}"] = {"W": int((o > 0).sum()), "L": int((o < 0).sum()), "P": int((o == 0).sum()),
                             "units": float(v[f"{m}_profit"].astype(float).sum()) if len(v) else 0.0}
    out["ats_clv"] = float(g["ats_clv"].astype(float).mean()) if len(g) else None
    for m in ("ats", "tot"):
        o = pd.to_numeric(g.get(f"early_{m}_out", pd.Series(dtype=float)), errors="coerce").dropna()
        c = pd.to_numeric(g.get(f"early_{m}_clv", pd.Series(dtype=float)), errors="coerce").dropna()
        out[f"early_{m}"] = {"W": int((o > 0).sum()), "L": int((o < 0).sum()), "P": int((o == 0).sum()),
                             "clv": float(c.mean()) if len(c) else None, "beat": float((c > 0).mean()) if len(c) else None,
                             "n_clv": int(len(c))}
    out["graded"] = int(len(g))
    return out


def main(path=None, schedule=None, label="NFL"):
    df = ledger.load(path)
    if df.empty:
        print(f"{label} pick history is empty - run the prediction step first.")
        return {}
    df = grade_ledger(df, schedule)
    ledger.save(df, path)
    s = summary(df)
    print(f"\n{label} pick history: {len(df)} games logged, {s['graded']} graded")
    names = {"su": "Straight up", "ats": "Against spread", "tot": "Over/under", "ml": "Moneyline side",
             "mo_su": "Model-only SU", "mo_ats": "Model-only ATS", "mo_tot": "Model-only O/U"}
    for m, name in names.items():
        r = s[m]
        pct = f"{r['pct']:.1%}" if r["pct"] is not None else "-"
        print(f"  {name:15s} {r['W']}-{r['L']}-{r['P']}  ({pct})")
    for m in ("ats", "tot", "ml"):
        r = s[f"value_{m}"]
        print(f"  value {m:4s} bets   {r['W']}-{r['L']}-{r['P']}  {r['units']:+.2f} units")
    if s["ats_clv"] is not None and not np.isnan(s["ats_clv"]):
        print(f"  avg spread CLV   {s['ats_clv']:+.2f} pts (positive = you beat the closing number)")
    for m, name in (("ats", "spread"), ("tot", "total")):
        r = s.get(f"early_{m}")
        if r and r["W"] + r["L"] + r["P"]:
            clv = f", CLV {r['clv']:+.2f} pts, beat the close {r['beat']:.0%}" if r["clv"] is not None else ""
            print(f"  first-posted {name} picks {r['W']}-{r['L']}-{r['P']}{clv}")
    return s


if __name__ == "__main__":
    main()
