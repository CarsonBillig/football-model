"""Publish the pick history as CSV files for Google Sheets (docs/history/, served by GitHub Pages).

One file per sport per week, in the same layout as a classic model-tracking sheet:
    homeTeam, awayTeam, predicted, side, sideLine, homeScore, awayScore, margin, atsWin,
followed by the winner pick, the total pick and the model-only lean with their results, and the confidence % for
the moneyline, spread and total picks (market-adjusted; spread/total ignore pushes).
predicted = projected home margin (positive = home wins). sideLine = the picked team's own spread (-3.5 = laying 3.5).

Also writes summary.csv (record per week and season) and index.json (list of weekly files), which
google_sheets_sync.gs reads to build one tab per week in your Google Sheet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ledger

HISTORY = Path("docs/history")
SPORTS = {"nfl": ("NFL", ledger.LEDGER), "cfb": ("College", ledger.CFB_LEDGER)}
RESULT = {1.0: "Win", -1.0: "Loss", 0.0: "Push"}


def _nfl_names() -> dict:
    try:
        import nflreadpy as nfl
        t = nfl.load_teams().to_pandas()
        return dict(zip(t["team_abbr"], t["team_name"]))
    except Exception:                       # offline: keep abbreviations
        return {}


def _res(x):
    return RESULT.get(float(np.sign(x)), "") if pd.notna(x) else ""


def week_table(wk: pd.DataFrame, names: dict) -> pd.DataFrame:
    n = lambda t: names.get(t, t) if isinstance(t, str) else ""
    num = lambda c: pd.to_numeric(wk[c], errors="coerce")
    out = pd.DataFrame({
        "homeTeam": wk["home_team"].map(n),
        "awayTeam": wk["away_team"].map(n),
        "predicted": (num("proj_home") - num("proj_away")).round(2),
        "side": wk["ats_pick"].map(n),
        "sideLine": num("ats_line"),
        "homeScore": num("home_score"),
        "awayScore": num("away_score"),
        "margin": num("home_score") - num("away_score"),
        "atsWin": wk["ats_out"].map(_res),
        "valueBet": np.where(wk["value_ats"].astype(str).str.lower() == "true", "Yes", ""),
        "winnerPick": wk["su_pick"].map(n),
        "winnerProb": (num("su_prob") * 100).round(1),
        "winnerResult": wk["su_out"].map(_res),
        "totalPick": wk["tot_pick"].fillna("").str.title() + " " + num("total_line").map(lambda v: "" if pd.isna(v) else f"{v:g}"),
        "totalResult": wk["tot_out"].map(_res),
        "modelOnlyPredicted": (num("mo_proj_home") - num("mo_proj_away")).round(2),
        "modelOnlySide": wk["mo_ats_pick"].map(n),
        "modelOnlyLine": num("mo_ats_line"),
        "modelOnlyAtsWin": wk["mo_ats_out"].map(_res),
        "mlConfidence": (num("su_prob") * 100).round(1),
        "spreadConfidence": (num("ats_prob_np") * 100).round(1),
        "totalConfidence": (num("tot_prob_np") * 100).round(1),
        "modelOnlyMlTested": (num("mo_su_hit") * 100).round(1),
        "modelOnlySpreadTested": (num("mo_ats_hit") * 100).round(1),
        "modelOnlyTotalTested": (num("mo_tot_hit") * 100).round(1),
        "closingLineValue": num("ats_clv").round(1),
        "kickoff": wk["kickoff"],
    })
    out["totalPick"] = out["totalPick"].str.strip()
    return out


def _record(col: pd.Series) -> tuple[int, int, int]:
    return int((col == "Win").sum()), int((col == "Loss").sum()), int((col == "Push").sum())


def export() -> Path:
    HISTORY.mkdir(parents=True, exist_ok=True)
    names = {"nfl": _nfl_names(), "cfb": {}}
    index, summary = [], []
    for sport, (label, path) in SPORTS.items():
        led = ledger.load(path)
        if led.empty:
            continue
        for (season, week), wk in led.groupby(["season", "week"]):
            season, week = int(season), int(week)
            tab = week_table(wk.sort_values(["kickoff", "game_id"]), names[sport])
            fname = f"{sport}_{season}_week_{week:02d}.csv"
            tab.to_csv(HISTORY / fname, index=False)
            index.append({"sport": label, "season": season, "week": week, "file": fname,
                          "tab": f"{label} {season} Wk {week}", "games": len(tab), "graded": int((tab["atsWin"] != "").sum())})
            row = {"sport": label, "season": season, "week": week, "games": len(tab)}
            for key, col in (("ats", "atsWin"), ("winner", "winnerResult"), ("total", "totalResult"), ("modelOnlyAts", "modelOnlyAtsWin")):
                w, l, p = _record(tab[col])
                row.update({f"{key}W": w, f"{key}L": l, f"{key}P": p})
            vb = tab[tab["valueBet"] == "Yes"]
            row["valueW"], row["valueL"], row["valueP"] = _record(vb["atsWin"])
            row["avgCLV"] = round(float(tab["closingLineValue"].mean()), 2) if tab["closingLineValue"].notna().any() else None
            summary.append(row)
    s = pd.DataFrame(summary)
    if not s.empty:
        s = s.sort_values(["sport", "season", "week"])
        tot = s.groupby(["sport", "season"], as_index=False).sum(numeric_only=True).assign(week="Season")
        tot["avgCLV"] = None
        s = pd.concat([s, tot], ignore_index=True)
        for key in ("ats", "winner", "total", "modelOnlyAts", "value"):
            w, l = s[f"{key}W"], s[f"{key}L"]
            s[f"{key}Pct"] = np.where(w + l > 0, (w / (w + l).replace(0, np.nan) * 100).round(1), None)
    s.to_csv(HISTORY / "summary.csv", index=False)
    (HISTORY / "index.json").write_text(json.dumps(sorted(index, key=lambda x: (x["sport"], x["season"], x["week"])), indent=1))
    return HISTORY


if __name__ == "__main__":
    print(f"wrote {export()}")
