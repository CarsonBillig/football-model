"""Build the weekly predictions website: docs/index.html (one self-contained file, no server needed).

Called automatically by predict.py. Layout:
  * masthead: week, live countdown to the next kickoff, market-adjusted / model-only switch, record ribbon
  * spotlight: the week's most confident winner and biggest spread / total leans
  * board: one row per game (team-colour win bar, market -> model line, total, picks); rows expand for details
  * tabs: all games / value bets / pick history / key (what everything means + how the model works)
"""
from __future__ import annotations

import colorsys
import json
import os
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import grade
import ledger

OUT = Path("output")
SITE = Path("docs")                  # GitHub Pages publishes this folder
ET = ZoneInfo("America/New_York")
OUTCOMES = ("home_score", "away_score", "su_out", "ats_out", "tot_out", "ml_out", "mo_su_out", "mo_ats_out", "mo_tot_out")


def F(fmt: str) -> str:
    """strftime without zero padding on every OS ('%-d' on Linux/macOS, '%#d' on Windows)."""
    return fmt.replace("%-", "%#") if os.name == "nt" else fmt


# ------------------------------------------------------------------------------------------ formatting helpers
def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _has(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and str(x) not in ("", "nan", "None")


def fav_line(home, away, s) -> str:
    """Home-favored-positive spread -> 'KC −3.5' (favorite first), like a sportsbook board."""
    s = _num(s)
    if s is None:
        return "—"
    if abs(s) < 0.05:
        return "Pick'em"
    return f"{home if s > 0 else away} −{abs(s):.1f}"


def team_line(team, line) -> str:
    line = _num(line)
    if line is None:
        return str(team)
    return f"{team} {'PK' if abs(line) < 0.05 else f'{line:+.1f}'.replace('-', '−')}"


def odds(o) -> str:
    o = _num(o)
    return "" if o is None else f"{o:+.0f}".replace("-", "−")


def pct(p, digits=0) -> str:
    p = _num(p)
    return "—" if p is None else f"{p * 100:.{digits}f}%"


def rec(r) -> str:
    return f"{r['W']}–{r['L']}" + (f"–{r['P']}" if r.get("P") else "")


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def mark(out) -> str:
    o = _num(out)
    if o is None:
        return ""
    return {1: '<span class="mk win" title="won">✓</span>', -1: '<span class="mk loss" title="lost">✗</span>'}.get(
        int(np.sign(o)), '<span class="mk push" title="push (tie with the line)">P</span>')


# ------------------------------------------------------------------------------------------ team colours
def _rgb(hexs):
    hexs = str(hexs or "").strip().lstrip("#")
    if len(hexs) != 6:
        return None
    try:
        return tuple(int(hexs[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None


def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _hex(c):
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def _pick(primary, alt, dark: bool):
    """A team colour that stays visible on the page background (light or dark)."""
    p, a = _rgb(primary), _rgb(alt)
    if p is None:
        return None
    bad = (lambda c: _lum(c) < 0.12) if dark else (lambda c: _lum(c) > 0.82)
    if bad(p) and a is not None and not bad(a):
        return a
    if bad(p):                                   # no usable alternate: nudge lightness
        h, l, s = colorsys.rgb_to_hls(*p)
        return colorsys.hls_to_rgb(h, 0.62 if dark else 0.35, s)
    return p


def team_colors(r) -> dict:
    """CSS custom properties for the away/home bar segments in light and dark mode; clashing pairs use the alt colour."""
    out = {}
    for dark, suffix in ((False, ""), (True, "-d")):
        a = _pick(r.get("away_color"), r.get("away_alt_color"), dark)
        h = _pick(r.get("home_color"), r.get("home_alt_color"), dark)
        if a is not None and h is not None and sum((x - y) ** 2 for x, y in zip(a, h)) < 0.06:
            alt = _pick(r.get("home_alt_color"), None, dark)
            if alt is not None:
                h = alt
        out[f"--ca{suffix}"] = _hex(a) if a else "var(--ink)"
        out[f"--ch{suffix}"] = _hex(h) if h else "var(--faint)"
    return out


# ------------------------------------------------------------------------------------------ data
def load_week(season: int, week: int, out: Path = OUT, ledger_path: Path | None = None) -> pd.DataFrame:
    wk = pd.read_json(out / f"week_{season}_{week:02d}.json")
    led = ledger.load(ledger_path)
    led = led[(led["season"] == season) & (led["week"] == week)].set_index("game_id")
    wk["logged"] = wk["game_id"].isin(led.index)
    # a game that has kicked off shows the picks that were LOGGED before kickoff, not a re-computed version
    pick_cols = [c for c in ledger.COLUMNS if c != "game_id" and c in wk.columns]
    wk = wk.set_index("game_id")
    now = pd.Timestamp.now(tz="UTC")
    for gid in wk.index.intersection(led.index):
        if pd.notna(led.at[gid, "su_out"]) or pd.to_datetime(led.at[gid, "kickoff"], utc=True) <= now:
            for c in pick_cols:
                wk.at[gid, c] = led.at[gid, c]
    for c in OUTCOMES:
        wk[c] = led[c].reindex(wk.index) if c in led else np.nan
    wk = wk.reset_index()
    wk["kick_et"] = pd.to_datetime(wk["kickoff"], utc=True).dt.tz_convert(ET)
    return wk.sort_values(["kick_et", "game_id"])


# ------------------------------------------------------------------------------------------ html pieces
def pill(text, prob=None, out=None, value=False, note=None, kind="") -> str:
    o = _num(out)
    state = "" if o is None else {1: " win", -1: " loss"}.get(int(np.sign(o)), " push")
    p = f"<i>{escape(note)}</i>" if note else (f"<i>{pct(prob)}</i>" if _num(prob) is not None else "")
    label = f'<small>{kind}</small>' if kind else ""
    return f'<span class="pill{state}{" value" if value else ""}">{label}{escape(str(text))}{p}{mark(out)}</span>'


def both(mkt_html: str, mod_html: str) -> str:
    """Market-adjusted and model-only versions side by side; the page switch shows one of them."""
    return f'<span class="vw v-mkt">{mkt_html}</span><span class="vw v-mod">{mod_html}</span>'


def view(r, pre: str, final: bool) -> dict:
    """Everything that differs between the two views. pre = '' (market-adjusted) or 'mo_' (model only)."""
    home, away = r["home_team"], r["away_team"]
    g = lambda c: r.get(f"{pre}{c}")
    lean = pre == "mo_"
    ph = _num(g("p_home_win")) or 0.5
    fair = _num(g("fair_spread")) if _num(g("fair_spread")) is not None else (None if lean else _num(r.get("fair_margin")))
    ft = _num(g("fair_total_line")) if _num(g("fair_total_line")) is not None else (None if lean else _num(r.get("fair_total")))
    spread, tl = _num(r.get("spread")), _num(r.get("total_line"))
    scores = {}
    for s in ("away", "home"):
        proj, score = _num(g(f"proj_{s}")), _num(r.get(f"{s}_score"))
        proj_s = "—" if proj is None else f"{proj:.1f}"
        scores[s] = (f'<b class="sc">{score:.0f}</b><span class="was">proj {proj_s}</span>' if final
                     else f'<b class="sc">{proj_s}</b>')
    # model-only cover probabilities are overconfident (that view hit ~51% ATS in testing): show the gap in points
    gap_s = abs(fair - spread) if fair is not None and spread is not None else None
    gap_t = abs(ft - tl) if ft is not None and tl is not None else None
    pills = []
    if _has(g("su_pick")):
        pills.append(pill(g("su_pick"), g("su_prob"), r.get(f"{pre}su_out"), kind="Win"))
    if _has(g("ats_pick")):
        pills.append(pill(team_line(g("ats_pick"), g("ats_line")), g("ats_prob_np"), r.get(f"{pre}ats_out"),
                          not lean and truthy(r.get("value_ats")), f"{gap_s:.1f} pts" if lean and gap_s is not None else None,
                          kind="ATS"))
    if _has(g("tot_pick")) and tl is not None:
        pills.append(pill(f"{'O' if g('tot_pick') == 'OVER' else 'U'} {tl:.1f}", g("tot_prob_np"), r.get(f"{pre}tot_out"),
                          not lean and truthy(r.get("value_tot")), f"{gap_t:.1f} pts" if lean and gap_t is not None else None,
                          kind="O/U"))
    if not lean and truthy(r.get("value_ml")):
        pills.append(pill(f"{r['ml_pick']} {odds(r.get('ml_odds'))}", r.get("ml_prob"), r.get("ml_out"), True, kind="ML"))
    return {"ph": ph, "scores": scores, "su_pick": g("su_pick"), "fair": fair, "ft": ft, "gap_s": gap_s, "gap_t": gap_t,
            "spread_txt": f"→ {fav_line(home, away, fair)}", "total_txt": f"→ {'—' if ft is None else f'{ft:.1f}'}",
            "pills": "".join(pills) or '<span class="pill muted">no line yet</span>'}


def short(r, team):
    """Short team code for tight cells (college rows carry home_abbr/away_abbr; NFL teams already are codes)."""
    for side in ("home", "away"):
        if team == r.get(f"{side}_team") and _has(r.get(f"{side}_abbr")):
            return str(r[f"{side}_abbr"])
    return team


def ml_text(r) -> str:
    """Moneyline, favorite first: 'PIT −148 · CLE +124'."""
    h, a = _num(r.get("home_ml")), _num(r.get("away_ml"))
    if h is None or a is None:
        return "—"
    H, A = short(r, r["home_team"]), short(r, r["away_team"])
    first, second = ((H, h), (A, a)) if h <= a else ((A, a), (H, h))
    return f"{first[0]} {odds(first[1])} · {second[0]} {odds(second[1])}"


def cell(label, value, cls="") -> str:
    return f'<div class="cell {cls}"><span class="lbl">{label}</span><span class="val">{value}</span></div>'


def game_row(r) -> str:
    home, away = r["home_team"], r["away_team"]
    H, A = short(r, home), short(r, away)
    ranked = _has(r.get("home_rank")) or _has(r.get("away_rank"))
    final = _num(r.get("home_score")) is not None
    mk, md = view(r, "", final), view(r, "mo_", final)
    k = r["kick_et"]
    tv = escape(str(r.get("tv"))) if _has(r.get("tv")) else ""
    venue = " · ".join(escape(str(x)) for x in (r.get("venue"), r.get("city")) if _has(x))
    colors = ";".join(f"{k_}:{v}" for k_, v in team_colors(r).items())

    def side(s, abbr):
        name = r.get(f"{s}_name")
        name = escape(str(name)) if _has(name) else abbr
        logo = r.get(f"{s}_logo")
        img = f'<img src="{escape(str(logo))}" alt="" loading="lazy">' if _has(logo) else '<span class="nologo"></span>'
        recd = escape(str(r.get(f"{s}_record"))) if _has(r.get(f"{s}_record")) else ""
        fav = (" fav-mkt" if mk["su_pick"] == abbr else "") + (" fav-mod" if md["su_pick"] == abbr else "")
        rk = f'<span class="rk">{int(_num(r.get(f"{s}_rank")))}</span>' if _num(r.get(f"{s}_rank")) is not None else ""
        return (f'<div class="side{fav}">{img}<div class="tn"><span class="name">{rk}{name}</span><span class="rec">{recd}</span></div>'
                f'<div class="scw"><span class="lbl">{"Final" if final else "Projected"}</span>{both(mk["scores"][s], md["scores"][s])}</div></div>')

    def bar(v):
        pa = 1 - v["ph"]
        return (f'<div class="bar"><span class="a" style="flex:{pa:.4f}"></span><span class="h" style="flex:{v["ph"]:.4f}"></span></div>'
                f'<div class="barlbl"><span>{A} {pct(pa)}</span><span>win chance</span><span>{pct(v["ph"])} {H}</span></div>')

    def winner(v, pre):
        sp = v["su_pick"]
        return "—" if not _has(sp) else f"{short(r, sp)} <i>{pct(max(v['ph'], 1 - v['ph']))}</i> {mark(r.get(f'{pre}su_out'))}"

    def pick(pre, kind):
        lean = pre == "mo_"
        v = md if lean else mk
        g = lambda c: r.get(f"{pre}{c}")
        tl_ = _num(r.get("total_line"))
        if kind == "ats":
            if not _has(g("ats_pick")):
                return "—"
            txt = team_line(g("ats_pick"), g("ats_line"))
            note = f"{v['gap_s']:.1f} pts" if lean and v["gap_s"] is not None else pct(g("ats_prob_np"))
            out, val = r.get(f"{pre}ats_out"), not lean and truthy(r.get("value_ats"))
        else:
            if not _has(g("tot_pick")) or tl_ is None:
                return "—"
            txt = f"{'Over' if g('tot_pick') == 'OVER' else 'Under'} {tl_:.1f}"
            note = f"{v['gap_t']:.1f} pts" if lean and v["gap_t"] is not None else pct(g("tot_prob_np"))
            out, val = r.get(f"{pre}tot_out"), not lean and truthy(r.get("value_tot"))
        tag = '<span class="tag">Value</span>' if val else ""
        return f"{escape(txt)} <i>{note}</i> {mark(out)}{tag}"

    spread, tl = _num(r.get("spread")), _num(r.get("total_line"))
    has_value = any(truthy(r.get(f"value_{m}")) for m in ("ats", "tot", "ml"))
    ml_value = ""
    if truthy(r.get("value_ml")):
        ml_value = (f'<div class="pickrow v-mktonly"><span class="lbl">Moneyline value</span><span class="val">'
                    f'{r["ml_pick"]} {odds(r.get("ml_odds"))} <i>{pct(r.get("ml_prob"))}</i> {mark(r.get("ml_out"))}'
                    f'<span class="tag">Value</span></span></div>')

    # expanded detail
    mkt = _num(r.get("mkt_p_home_win"))
    bets = []
    for m, label, what in (("ats", "Spread", team_line(r.get("ats_pick") or "", r.get("ats_line"))),
                           ("tot", "Total", f"{str(r.get('tot_pick') or '').title()} {tl:.1f}" if tl is not None else ""),
                           ("ml", "Moneyline", f"{r.get('ml_pick') or ''} {odds(r.get('ml_odds'))}")):
        ev = _num(r.get(f"{m}_ev"))
        if ev is None or not what.strip():
            continue
        stake = _num(r.get(f"{m}_stake")) or 0
        o = odds(r.get(f"{m}_odds")) or "−110"
        bets.append(f'<tr{" class=val" if truthy(r.get(f"value_{m}")) else ""}><td>{label}</td><td>{escape(what)}'
                    f'{"" if m == "ml" else f" ({o})"}</td><td>{ev * 100:+.1f}%</td>'
                    f'<td>{f"{stake * 100:.1f}%" if stake else "pass"}</td></tr>')
    open_total = _num(r.get("open_total"))
    detail = f"""
    <details class="more"><summary>Details</summary><div class="more-in">
      <dl>
        <div><dt>Opening line</dt><dd>{fav_line(H, A, r.get('open_spread'))} · {'—' if open_total is None else f"O/U {open_total:.1f}"}</dd></div>
        <div><dt>Sportsbook win chance</dt><dd>{'—' if mkt is None else f'{H} {pct(mkt, 1)}'}</dd></div>
        <div><dt>Market-adjusted win chance</dt><dd>{H} {pct(mk['ph'], 1)}</dd></div>
        <div><dt>Model-only win chance</dt><dd>{H} {pct(md['ph'], 1)}</dd></div>
      </dl>
      <table class="ev"><thead><tr><th>Bet</th><th>Side</th><th>Exp. value</th><th>Stake</th></tr></thead><tbody>{''.join(bets)}</tbody></table>
      <p class="evnote">Value and stakes use the market-adjusted numbers, the only ones that held up in testing.</p>
    </div></details>"""

    mk_ft = "—" if mk["ft"] is None else f"{mk['ft']:.1f}"
    md_ft = "—" if md["ft"] is None else f"{md['ft']:.1f}"
    return f"""
<article class="game{' is-final' if final else ''}" data-value="{int(has_value)}" data-kick="{k.isoformat()}" data-final="{int(final)}" data-top="{int(ranked)}" style="{colors}">
  <header><span class="when"><b>{k.strftime(F('%-I:%M %p'))} ET</b>{' · ' + tv if tv else ''}</span><span class="chip" data-chip></span></header>
  <div class="venue">{venue}</div>
  <div class="teams">{side('away', away)}{side('home', home)}{both(bar(mk), bar(md))}</div>
  <div class="grid3">
    {cell('Moneyline', ml_text(r), 'mkt')}
    {cell('Market line', fav_line(H, A, spread), 'mkt')}
    {cell('Market total', '—' if tl is None else f'{tl:.1f}', 'mkt')}
    {cell('Projected winner', both(winner(mk, ''), winner(md, 'mo_')), 'mod')}
    {cell('Model line', both(fav_line(H, A, mk['fair']), fav_line(H, A, md['fair'])), 'mod')}
    {cell('Model total', both(mk_ft, md_ft), 'mod')}
  </div>
  <div class="picksbox">
    <div class="pickrow"><span class="lbl">Model pick</span><span class="val">{both(pick('', 'ats'), pick('mo_', 'ats'))}</span></div>
    <div class="pickrow"><span class="lbl">Total pick</span><span class="val">{both(pick('', 'tot'), pick('mo_', 'tot'))}</span></div>
    {ml_value}
  </div>
  {detail}
</article>"""


def spotlight(wk: pd.DataFrame) -> str:
    """Three cards per view: most confident winner, biggest spread lean, biggest total lean."""
    def cards(pre):
        rows = [(r, view(r, pre, _num(r.get("home_score")) is not None)) for _, r in wk.iterrows()]
        rows = [x for x in rows if _has(x[1]["su_pick"])]
        if not rows:
            return ""
        out = []
        r, v = max(rows, key=lambda x: max(x[1]["ph"], 1 - x[1]["ph"]))
        conf = max(v["ph"], 1 - v["ph"])
        out.append(("Surest winner", v["su_pick"], f"{pct(conf)} to win", f"{r['away_team']} @ {r['home_team']}"))
        sp = [x for x in rows if x[1]["gap_s"] is not None and _has(x[0].get(f"{pre}ats_pick"))]
        if sp:
            r, v = max(sp, key=lambda x: x[1]["gap_s"])
            out.append(("Biggest spread lean", team_line(r[f"{pre}ats_pick"], r[f"{pre}ats_line"]),
                        f"model {v['gap_s']:.1f} pts off the line", f"{r['away_team']} @ {r['home_team']}"))
        tt = [x for x in rows if x[1]["gap_t"] is not None and _has(x[0].get(f"{pre}tot_pick"))]
        if tt:
            r, v = max(tt, key=lambda x: x[1]["gap_t"])
            out.append(("Biggest total lean", f"{str(r[f'{pre}tot_pick']).title()} {_num(r['total_line']):.1f}",
                        f"model {v['gap_t']:.1f} pts off the line", f"{r['away_team']} @ {r['home_team']}"))
        return "".join(f'<div class="spot"><span class="k">{a}</span><b>{escape(str(b))}</b><span>{c}</span><em>{d}</em></div>'
                       for a, b, c, d in out)
    return f'<section class="spots">{both(cards(""), cards("mo_"))}</section>'


def history_html(led: pd.DataFrame) -> str:
    g = led[led["su_out"].notna()].copy()
    if g.empty:
        return ('<div class="empty"><b>No graded picks yet.</b><p>Every game is logged before kickoff. After the games, run the '
                'update again and each pick shows up here marked ✓ (won), ✗ (lost) or P (push).</p></div>')
    out = []
    for (season, week), wk in sorted(g.groupby(["season", "week"]), key=lambda x: x[0], reverse=True):
        s = grade.summary(wk)
        rows = []
        for _, r in wk.sort_values("kickoff").iterrows():
            tl = _num(r.get("total_line"))

            def cells(pre):
                tp, ap, sp = r.get(f"{pre}tot_pick"), r.get(f"{pre}ats_pick"), r.get(f"{pre}su_pick")
                tot = f"{'O' if tp == 'OVER' else 'U'} {tl:.1f}" if _has(tp) and tl is not None else "—"
                ats = escape(team_line(ap, r.get(f"{pre}ats_line"))) if _has(ap) else "—"
                val = ' <span class="v">value</span>' if not pre and truthy(r.get("value_ats")) else ""
                return (f"{sp if _has(sp) else '—'} {mark(r.get(f'{pre}su_out'))}",
                        f"{ats} {mark(r.get(f'{pre}ats_out'))}{val}", f"{tot} {mark(r.get(f'{pre}tot_out'))}")
            a, b = cells(""), cells("mo_")
            clv = _num(r.get("ats_clv"))
            rows.append(f"<tr><td>{r.away_team} <span class=at>@</span> {r.home_team}</td>"
                        f"<td class=n>{_num(r.away_score):.0f}–{_num(r.home_score):.0f}</td>"
                        + "".join(f"<td>{both(x, y)}</td>" for x, y in zip(a, b))
                        + f"<td class=n>{'' if clv is None else f'{clv + 0.0:+.1f}'.replace('-0.0', '+0.0')}</td></tr>")
        summ = both(f"Winners {rec(s['su'])} · Spread {rec(s['ats'])} · Totals {rec(s['tot'])}",
                    f"Winners {rec(s['mo_su'])} · Spread {rec(s['mo_ats'])} · Totals {rec(s['mo_tot'])}")
        out.append(f"""
<section class="wk">
  <header><h3>Week {int(week)} <span>{int(season)}</span></h3><p>{summ}</p></header>
  <div class="scroll"><table>
    <thead><tr><th>Game</th><th>Final</th><th>Winner</th><th>Spread</th><th>Total</th><th title="closing line value, points">CLV</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>""")
    return "".join(out)


def ribbon_item(label, r) -> str:
    n = r["W"] + r["L"]
    p = r.get("pct")
    return (f'<div class="ri"><span>{label}</span><b>{rec(r) if n else "0–0"}</b>'
            f'<em>{pct(p, 1) if p is not None and n else "—"}</em></div>')


def ribbons(live: dict, bt: dict, meta: dict) -> str:
    conv = lambda r: {"W": int(r["W"]), "L": int(r["L"]), "P": int(r.get("P") or 0), "pct": r.get("win_pct")}
    span = escape(meta.get("backtest_span", ""))
    out = []
    for pre, name in (("", "mkt"), ("mo_", "mod")):
        if live["graded"] == 0 and bt.get(f"{pre}su"):
            label = f"Backtest {span}"
            items = "".join(ribbon_item(l, conv(bt[f"{pre}{k}"])) for l, k in (("Winners", "su"), ("Spread", "ats"), ("Totals", "tot")))
            note = "Out-of-sample test on every game at closing lines. Live tracking starts with this week's picks."
        else:
            label = "Live record"
            items = "".join(ribbon_item(l, live[f"{pre}{k}"]) for l, k in (("Winners", "su"), ("Spread", "ats"), ("Totals", "tot")))
            note = f"{live['graded']} games graded, all picks logged before kickoff"
            if not pre:
                vrec = {k: sum(live[f"value_{m}"][k] for m in ("ats", "tot", "ml")) for k in ("W", "L", "P")}
                vrec["pct"] = vrec["W"] / (vrec["W"] + vrec["L"]) if vrec["W"] + vrec["L"] else None
                vu = sum(live[f"value_{m}"]["units"] for m in ("ats", "tot", "ml"))
                items += ribbon_item(f"Value bets ({vu:+.1f}u)", vrec)
                clv = live["ats_clv"]
                note += "" if clv is None or np.isnan(clv) else f" · average closing line value {clv:+.2f} pts"
            bts = bt.get(f"{pre}ats") or {}
            note += f" · backtest spread record {int(bts['W'])}–{int(bts['L'])}" if bts else ""
        if pre:
            note = "Pure model, no sportsbook input: leans, not bets. " + note
        out.append(f'<div class="ribbon v-{name}"><span class="lbl">{label}</span>{items}</div><p class="note v-{name}">{note}</p>')
    return "".join(out)


# ------------------------------------------------------------------------------------------ page
SPORTS = {
    "nfl": {"label": "NFL", "out": Path("output"), "ledger": None,
            "model": "A ridge regression on opponent-adjusted EPA, success rate, QB rating and rest, plus two power ratings "
                     "(one from final scores, one from past betting lines), retrained every week on every completed game.",
            "keynum": "NFL games end by 3 or 7 far more often than other margins, so a half point around 3 or 7 is worth far more than elsewhere.",
            "market": "NFL closing lines are very efficient"},
    "cfb": {"label": "College", "out": Path("output/cfb"), "ledger": Path("output/cfb/ledger.csv"),
            "model": "A ridge regression on power ratings built from final scores, past betting lines and per-play efficiency "
                     "(PPA and success rate), plus roster talent and returning production that matter most early in the season. "
                     "All FCS opponents share one rating. Retrained every week on every completed game.",
            "keynum": "Games end by 3 or 7 more often than other margins (less sharply in college than the NFL), so half points near them matter more.",
            "market": "College closing lines are efficient, though less than the NFL's"},
}


def section(sport: str, cfg: dict) -> str | None:
    cur = cfg["out"] / "current.json"
    if not cur.exists():
        return None
    c = json.loads(cur.read_text())
    season, week = c["season"], c["week"]
    wk = load_week(season, week, cfg["out"], cfg["ledger"])
    led = ledger.load(cfg["ledger"])
    live = grade.summary(led if not led.empty else pd.DataFrame(columns=ledger.COLUMNS))
    meta = json.loads((cfg["out"] / "model_meta.json").read_text())
    bt = meta.get("backtest_records", {})
    first, last = wk["kick_et"].iloc[0], wk["kick_et"].iloc[-1]
    span = first.strftime(F("%b %-d")) + ("" if first.date() == last.date() else " – " + last.strftime(F("%b %-d")))
    rows = []
    for _, grp in wk.groupby(wk["kick_et"].dt.date, sort=True):
        rows.append(f'<h2 class="day"><span>{grp["kick_et"].iloc[0].strftime(F("%A"))}</span> '
                    f'{grp["kick_et"].iloc[0].strftime(F("%B %-d"))}<em>{len(grp)} game{"s" if len(grp) > 1 else ""}</em></h2>'
                    + '<div class="cards">' + "".join(game_row(r) for _, r in grp.iterrows()) + "</div>")
    n_value = int(sum(any(truthy(r.get(f"value_{m}")) for m in ("ats", "tot", "ml")) for _, r in wk.iterrows()))
    n_top = int(sum(_has(r.get("home_rank")) or _has(r.get("away_rank")) for _, r in wk.iterrows()))
    top_tab = f'<button data-t="top" aria-pressed="false">Top 25<sup>{n_top}</sup></button>' if n_top else ""
    return SECTION.substitute(
        sport=sport, label=cfg["label"], season=season, week=week, span=span, n_games=len(wk), n_value=n_value, top_tab=top_tab,
        updated=datetime.now(ET).strftime(F("%a %b %-d, %-I:%M %p ET")), trained=escape(meta.get("trained_through", "")),
        ribbon=ribbons(live, bt, meta), spotlight=spotlight(wk), rows="".join(rows), history=history_html(led),
        n_hist=int(led["su_out"].notna().sum()) if not led.empty else 0, beta=f"{meta.get('beta_margin', 0.0):.0%}",
        bt_ats=(lambda r: f"{r['win_pct'] * 100:.1f}%" if r else "—")(bt.get("ats")),
        bt_mo_ats=(lambda r: f"{r['win_pct'] * 100:.1f}%" if r else "—")(bt.get("mo_ats")),
        model_text=cfg["model"], keynum_text=cfg["keynum"], market_text=cfg["market"])


def build(*_ignored) -> Path:
    """Rebuild docs/index.html from every sport that has a current week (output/current.json, output/cfb/current.json)."""
    SITE.mkdir(exist_ok=True)
    parts = [(k, section(k, cfg)) for k, cfg in SPORTS.items()]
    parts = [(k, html) for k, html in parts if html]
    if not parts:
        raise SystemExit("No predictions to show yet - run update_model.py.")
    switch = ""
    if len(parts) > 1:
        switch = ('<div class="seg" id="sportsw" role="group" aria-label="Sport">' + "".join(
            f'<button data-s="{k}" aria-pressed="{str(n == 0).lower()}">{SPORTS[k]["label"]}</button>' for n, (k, _) in enumerate(parts)) + "</div>")
    html = SHELL.substitute(sections="".join(parts_html for _, parts_html in parts), sport_switch=switch,
                            first=parts[0][0], brand="Football model" if len(parts) > 1 else f"{SPORTS[parts[0][0]]['label']} model")
    path = SITE / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


SECTION = Template(r"""<div class="sport" data-sport="$sport">
<section class="mast">
  <h1><small class="lg-label">$label</small>Week <em>$week</em></h1>
  <div class="kick"><span>$season season · $span · $n_games games</span>
    <span class="count">—</span><span class="countlbl">until next kickoff</span>
    <span>Updated $updated · trained through $trained</span></div>
</section>
$ribbon
$spotlight
<nav class="tabs">
  <button data-t="all" aria-pressed="true">All games<sup>$n_games</sup></button>
  $top_tab
  <button data-t="value" aria-pressed="false">Value bets<sup>$n_value</sup></button>
  <button data-t="history" aria-pressed="false">Pick history<sup>$n_hist</sup></button>
  <button data-t="key" aria-pressed="false">Key</button>
</nav>
<main>
<div class="board">$rows
  <div class="empty hidden novalue"><b>No value bets this week.</b><p>A bet only gets flagged when the model expects at least +2% profit at the listed price.
  Most weeks the sportsbook line is close to right, so passing is the normal outcome. Each game's leans are still under All games.</p></div>
</div>
<div class="history hidden">$history</div>
<div class="key hidden">
  <h2>Reading the board</h2>
  <p class="lede">Each game card shows the sportsbook's numbers on the top row and the model's on the highlighted row beneath, then the model's picks. The switch at the top changes every model number on the page between the two views.</p>
  <div class="legend">
    <div class="lg"><div class="ex"><span class="lbl">Projected</span><span class="sc">24.5</span></div>
      <div><h4>Projected score</h4><p>Points the model expects each team to score. After the game it switches to the real final score, with the projection in small grey type underneath.</p></div></div>
    <div class="lg"><div class="ex exbar"><div class="bar"><span class="a" style="flex:.62"></span><span class="h" style="flex:.38"></span></div><div class="barlbl"><span>62%</span><span>38%</span></div></div>
      <div><h4>Win-chance bar</h4><p>Each team's chance to win, in team colors. The team name in <b style="color:var(--accent)">orange</b> is the projected winner.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Moneyline</span><span class="mono">KC −190 · LV +160</span></div>
      <div><h4>Moneyline</h4><p>The sportsbook's price to bet each team to win outright, favorite first. −190 means risk $$190 to win $$100; +160 means risk $$100 to win $$160.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Market line</span><span class="mono">KC −4.5</span></div>
      <div><h4>Market line</h4><p>The sportsbook's current point spread, favorite shown with a minus. KC −4.5 means KC must win by 5 or more to cover.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Market total</span><span class="mono">47.5</span></div>
      <div><h4>Market total</h4><p>The sportsbook's over/under: the combined points it expects both teams to score.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Projected winner</span><span class="mono" style="color:var(--accent)">KC 69%</span></div>
      <div><h4>Projected winner</h4><p>The team the model expects to win outright, and its chance of winning.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Model line</span><span class="mono" style="color:var(--accent)">KC −5.6</span></div>
      <div><h4>Model line</h4><p>The model's fair spread, where each side covers half the time. Compare it with the market line: a bigger number than the book means the model likes the favorite; a smaller one, the underdog.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Model total</span><span class="mono" style="color:var(--accent)">45.9</span></div>
      <div><h4>Model total</h4><p>The model's fair combined score. Lower than the market total means it leans under; higher means over.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Model pick</span><span class="mono"><b>KC −4.5</b> <i style="color:var(--muted);font-style:normal">53%</i></span></div>
      <div><h4>Model pick and total pick</h4><p>The side to take against the spread and on the total, with the model's chance of winning the bet (pushes ignored). In Model only view the gray number is how many points the model disagrees with the book instead, because its raw percentages ran too high in testing.</p></div></div>
    <div class="lg"><div class="ex"><span class="mono"><b>DET −2.5</b><span class="tag">Value</span></span></div>
      <div><h4>Value bet</h4><p>Only shown when a pick is expected to earn at least +2% at the listed price. Open <b>Details</b> on a game for the exact value and a suggested stake. Without the tag, a pick is a lean and the stake says pass.</p></div></div>
    <div class="lg"><div class="ex"><span><span class="mk win">✓</span> won &nbsp;<span class="mk loss">✗</span> lost &nbsp;<span class="mk push">P</span> push</span></div>
      <div><h4>Results</h4><p>After games are graded, each pick is marked. A push means the final margin landed exactly on the line, so the bet is refunded.</p></div></div>
    <div class="lg"><div class="ex"><span class="chip">in 2h 10m</span><span class="chip locked">Locked</span><span class="chip final">Final</span></div>
      <div><h4>Game status</h4><p>Before kickoff the chip counts down, and re-running the update can still refresh the pick. At kickoff the pick is locked into the history. Final means graded.</p></div></div>
    <div class="lg"><div class="ex"><span class="seg"><button aria-pressed="true" tabindex="-1">Market-adjusted</button><button aria-pressed="false" tabindex="-1">Model only</button></span></div>
      <div><h4>Two views</h4><p><b>Market-adjusted</b> starts from the sportsbook line and moves only as far as the model has earned. <b>Model only</b> is the pure model, the way most public models show picks. Both are tracked in the pick history.</p></div></div>
    <div class="lg"><div class="ex"><span class="lbl">Details +</span></div>
      <div><h4>Details</h4><p>Opens the opening line, the sportsbook's own win chance next to both model views, and the expected value and suggested stake for each bet.</p></div></div>
  </div>

  <h2>Glossary</h2>
  <dl class="gloss">
    <div><dt>Spread</dt><dd>Points the favorite must win by. KC −3.5 means KC must win by 4 or more; the other side covers if it loses by 3 or less, or wins.</dd></div>
    <div><dt>Moneyline</dt><dd>A bet on who wins outright. −150 means risk $$150 to win $$100; +130 means risk $$100 to win $$130.</dd></div>
    <div><dt>Total (over/under)</dt><dd>A bet on whether both teams combined score more or fewer points than the line.</dd></div>
    <div><dt>Juice / vig</dt><dd>The sportsbook's cut. At the standard −110 you risk $$110 to win $$100.</dd></div>
    <div><dt>Break-even 52.4%</dt><dd>Because of the juice, you must win 52.4% of −110 bets just to break even.</dd></div>
    <div><dt>Expected value (EV)</dt><dd>Average profit per $$1 bet if the model's probability is right. +2% means 2 cents per dollar over the long run.</dd></div>
    <div><dt>Stake</dt><dd>Suggested bet size as a share of your bankroll: a quarter of the Kelly formula, capped at 3%.</dd></div>
    <div><dt>Opening / current / closing line</dt><dd>The first number posted, the number now, and the final number at kickoff. Lines move as money and news come in.</dd></div>
    <div><dt>CLV (closing line value)</dt><dd>How much better your number was than the closing line. Consistently positive CLV is the best early sign of a real edge.</dd></div>
    <div><dt>Push</dt><dd>The result lands exactly on the line, such as a 3-point win at −3. The bet is refunded.</dd></div>
    <div><dt>Key numbers</dt><dd>$keynum_text</dd></div>
    <div><dt>Backtest</dt><dd>Replaying past seasons week by week, using only information available before each game, to see how the model would have done.</dd></div>
  </dl>

  <h2>How the model works</h2>
  <div class="how">
    <p><b>Projection.</b> $model_text</p>
    <p><b>Respecting the market.</b> $market_text, so in Market-adjusted view the model moves off the sportsbook's number only by the
    share its disagreements have earned in testing (currently $beta for spreads). That share is re-estimated every week from the backtest and every graded
    pick, so the model gains or loses trust depending on how it actually does.</p>
    <p><b>Probabilities</b> come from a score distribution that knows about key numbers, calibrated so that 70% picks win about 70% of the time.</p>
    <p><b>Honest scorekeeping.</b> In testing, market-adjusted spread picks hit $bt_ats and model-only picks $bt_mo_ats; 52.4% is needed to profit.
    Picks are logged before kickoff and never changed afterwards.</p>
  </div>
</div>
</main>
</div>
""")


SHELL = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Picks</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f5f1;--card:#fdfcfa;--ink:#141417;--muted:#74747a;--faint:#b8b8bc;--rule:#e5e3dd;
--accent:#e8590c;--accent-soft:#fff0e6;--win:#2b8a52;--loss:#c9382a;--shadow:0 1px 2px rgba(20,20,23,.04),0 8px 24px -12px rgba(20,20,23,.12)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e0f11;--card:#15161a;--ink:#eeede9;--muted:#8f8f95;--faint:#505056;--rule:#24252a;
--accent:#ff7b3d;--accent-soft:#2b1a10;--win:#62c98c;--loss:#f2806f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6)}}
:root[data-theme="dark"]{--bg:#0e0f11;--card:#15161a;--ink:#eeede9;--muted:#8f8f95;--faint:#505056;--rule:#24252a;
--accent:#ff7b3d;--accent-soft:#2b1a10;--win:#62c98c;--loss:#f2806f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6)}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
.num,.pill,.when,.sc,.barlbl,table,.ri b,.count{font-family:"IBM Plex Mono",ui-monospace,monospace}
button{font:inherit;color:inherit}
.top{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--rule)}
.top .wrap{display:flex;align-items:center;gap:14px;height:56px}
.brand{font-weight:600;letter-spacing:-.01em;display:flex;align-items:center;gap:8px}
.brand i{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.top .sp{flex:1}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:999px;padding:3px;background:var(--card)}
.seg button{background:none;border:0;border-radius:999px;padding:5px 12px;font-size:13px;color:var(--muted);cursor:pointer;transition:background .2s,color .2s}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.icon{background:none;border:1px solid var(--rule);border-radius:999px;width:34px;height:34px;cursor:pointer;color:var(--muted)}
.mast{padding:56px 0 30px;display:grid;grid-template-columns:auto 1fr;gap:32px;align-items:end}
.mast h1{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:clamp(72px,13vw,140px);line-height:.82;margin:0;letter-spacing:-.025em}
.mast h1 em{color:var(--accent)}
.lg-label{display:block;font-family:Inter,sans-serif;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.kick{justify-self:end;text-align:right;display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:13px}
.kick .count{font-size:28px;color:var(--ink);letter-spacing:-.01em}
.kick .count small{font-size:13px;color:var(--muted);margin:0 6px 0 2px}
.ribbon{display:flex;flex-wrap:wrap;align-items:center;gap:10px 30px;padding:16px 0 8px;border-top:1px solid var(--rule)}
.ribbon .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:600}
.ri{display:flex;gap:8px;align-items:baseline} .ri span{color:var(--muted);font-size:13px} .ri b{font-weight:500;font-size:18px} .ri em{font-style:normal;color:var(--muted);font-size:12.5px}
.note{color:var(--muted);font-size:12.5px;margin:0 0 6px}
.spots{margin:24px 0 8px}
.spots .v-mkt,.spots .v-mod{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
body:not([data-view="mod"]) .spots .v-mod,body[data-view="mod"] .spots .v-mkt{display:none!important}
.spot{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:16px 18px;display:flex;flex-direction:column;gap:2px;box-shadow:var(--shadow);position:relative;overflow:hidden;animation:rise .5s both}
.spot:nth-child(2){animation-delay:.06s} .spot:nth-child(3){animation-delay:.12s}
.spot::after{content:"";position:absolute;inset:auto 0 0 0;height:3px;background:linear-gradient(90deg,var(--accent),transparent)}
.spot .k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.spot b{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:30px;line-height:1.1;margin-top:4px}
.spot span:not(.k){font-size:13px} .spot em{font-style:normal;font-size:12px;color:var(--muted)}
.tabs{display:flex;gap:6px;margin:34px 0 0;overflow-x:auto;overflow-y:hidden;scrollbar-width:none;border-bottom:1px solid var(--rule)}
.tabs button{background:none;border:0;padding:10px 12px;font-size:14px;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tabs button[aria-pressed="true"]{color:var(--ink);border-color:var(--accent)}
.tabs sup{font-family:"IBM Plex Mono",monospace;font-size:10px;margin-left:3px;color:var(--muted)}
.day{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:30px;margin:36px 0 10px;display:flex;align-items:baseline;gap:10px}
.day span{color:var(--accent);font-style:italic} .day em{font-family:Inter,sans-serif;font-style:normal;font-size:12px;color:var(--muted);margin-left:auto}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.game{background:var(--card);border:1px solid var(--rule);border-radius:16px;box-shadow:var(--shadow);transition:transform .18s ease,border-color .18s;display:flex;flex-direction:column;overflow:hidden;animation:rise .4s both}
.game:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--accent) 35%,var(--rule))}
.game>header{display:flex;justify-content:space-between;align-items:center;padding:14px 18px 0}
.when{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)} .when b{color:var(--ink);font-weight:500}
.venue{padding:2px 18px 0;font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chip{font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-family:"IBM Plex Mono",monospace}
.chip:empty{display:none} .chip.locked{background:var(--rule);color:var(--muted)} .chip.final{background:var(--ink);color:var(--bg)}
.teams{padding:12px 18px 14px}
.side{display:grid;grid-template-columns:34px 1fr auto;gap:12px;align-items:center;padding:6px 0}
.side img,.nologo{width:34px;height:34px;object-fit:contain}
.tn{display:flex;flex-direction:column;line-height:1.25;min-width:0} .tn .name{font-size:17px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tn .rec{color:var(--muted);font-size:12px;font-family:"IBM Plex Mono",monospace}
.scw{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1} .scw .lbl{font-size:9.5px}
.sc{font-size:28px;font-weight:500;letter-spacing:-.02em}
.was{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--muted)}
body:not([data-view="mod"]) .side.fav-mkt .name,body[data-view="mod"] .side.fav-mod .name{color:var(--accent);font-weight:600}
.bar{display:flex;gap:3px;height:6px;margin-top:10px}
.bar span{border-radius:3px;min-width:4px}
.bar .a{background:var(--ca)} .bar .h{background:var(--ch)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .bar .a{background:var(--ca-d)}:root:not([data-theme="light"]) .bar .h{background:var(--ch-d)}}
:root[data-theme="dark"] .bar .a{background:var(--ca-d)} :root[data-theme="dark"] .bar .h{background:var(--ch-d)}
.barlbl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);margin-top:4px}
.lbl{font-family:Inter,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-top:1px solid var(--rule)}
.cell{padding:10px 18px;display:flex;flex-direction:column;gap:3px;border-right:1px solid var(--rule);min-width:0}
.cell:nth-child(3n){border-right:0} .cell:nth-child(n+4){border-top:1px solid var(--rule)}
.cell .val{font-family:"IBM Plex Mono",monospace;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell.mod{background:color-mix(in srgb,var(--accent-soft) 45%,transparent)} .cell.mod .val{color:var(--accent);font-weight:500}
body[data-view="mod"] .cell.mod{background:color-mix(in srgb,var(--ink) 5%,transparent)} body[data-view="mod"] .cell.mod .val{color:var(--ink)}
.cell i,.pickrow i{font-style:normal;color:var(--muted);font-weight:400}
.picksbox{border-top:1px solid var(--rule);padding:10px 18px;display:flex;flex-direction:column;gap:6px}
.pickrow{display:flex;justify-content:space-between;align-items:center;gap:10px}
.pickrow .val{font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:600;text-align:right}
.tag{font-family:Inter,sans-serif;font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;background:var(--accent);color:#fff;padding:2px 6px;border-radius:4px;margin-left:8px;vertical-align:2px}
body[data-view="mod"] .v-mktonly{display:none}
.mk{font-weight:600} .mk.win{color:var(--win)} .mk.loss{color:var(--loss)} .mk.push{color:var(--muted)}
.more{border-top:1px solid var(--rule);margin-top:auto}
.more>summary{list-style:none;cursor:pointer;padding:10px 18px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between}
.more>summary::-webkit-details-marker{display:none} .more>summary::after{content:"+";font-family:"IBM Plex Mono",monospace}
.more[open]>summary::after{content:"−"} .more>summary:hover{color:var(--ink)}
.pill{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--rule);border-radius:999px;padding:4px 10px;font-size:12px;white-space:nowrap;background:var(--bg)}
.pill small{font-family:Inter,sans-serif;font-size:9.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.pill i{font-style:normal;color:var(--muted)}
.pill.value{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}
.more-in{padding:0 18px 16px;animation:rise .25s ease}
.more dl{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:10px 22px}
.more dt{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em} .more dd{margin:2px 0 0;font-size:13.5px}
.more table{margin-top:14px} .evnote{color:var(--muted);font-size:11.5px;margin:8px 0 0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-family:Inter,sans-serif;font-weight:500;font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule)}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
table.ev tr.val td{color:var(--accent);font-weight:500}
.vw{display:contents}
body:not([data-view="mod"]) .v-mod,body[data-view="mod"] .v-mkt{display:none!important}
.hidden{display:none!important}
.wk{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:6px 18px 10px;margin:18px 0;box-shadow:var(--shadow)}
.wk header{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;padding:10px 0}
.wk h3{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:28px;margin:0} .wk h3 span{color:var(--muted);font-size:16px}
.wk p{margin:0;color:var(--muted);font-size:13px} .scroll{overflow-x:auto} .at{color:var(--faint)} td.n{text-align:right}
.v{font-size:9.5px;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
.empty{background:var(--card);border:1px dashed var(--rule);border-radius:14px;padding:26px;margin:24px 0;color:var(--muted);max-width:620px}
.empty b{color:var(--ink)} .empty p{margin:6px 0 0}
.key{padding:10px 0}
.key h2{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:34px;margin:30px 0 6px}
.key .lede{color:var(--muted);max-width:640px;margin:0 0 18px}
.legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.lg{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:16px 18px;display:grid;grid-template-columns:150px 1fr;gap:16px;align-items:center}
.lg .ex{display:flex;flex-direction:column;gap:6px;align-items:flex-start;font-size:13px}
.lg h4{margin:0 0 3px;font-size:14px;font-weight:600} .lg p{margin:0;color:var(--muted);font-size:13px;line-height:1.55}
.lg .seg button{padding:4px 8px;font-size:11px;cursor:default}
.gloss{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 28px;margin:0}
.gloss div{padding:12px 0;border-bottom:1px solid var(--rule)} .gloss dt{font-weight:600;font-size:14px} .gloss dd{margin:3px 0 0;color:var(--muted);font-size:13px;line-height:1.55}
.how{max-width:720px;color:var(--muted);line-height:1.7} .how b{color:var(--ink);font-weight:500}
.exbar{width:150px;max-width:100%;align-items:stretch!important} .exbar .bar{margin-top:0;width:100%} .exbar .bar .a{background:#1f4e9c} .exbar .bar .h{background:#c9382a}
footer{margin:70px 0 0;padding:22px 0 44px;border-top:1px solid var(--rule);color:var(--muted);font-size:12.5px}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:460px){.cell{padding:9px 10px}.cell .val{font-size:12px;white-space:normal}.tn .name{font-size:15px}.sc{font-size:24px}}
@media (max-width:860px){
  .mast{grid-template-columns:1fr;gap:18px;padding-top:36px} .kick{justify-self:start;text-align:left}
  .spots .v-mkt,.spots .v-mod{grid-template-columns:1fr!important}
  .cards{grid-template-columns:1fr}
  .legend,.gloss{grid-template-columns:1fr} .lg{grid-template-columns:1fr}
  .top .brand span{display:none} #sportsw button{padding:5px 9px}
}
.rk{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);margin-right:6px;vertical-align:1px}
.sport.off{display:none}
</style></head>
<body>
<header class="top"><div class="wrap">
  <div class="brand"><i></i><span>$brand</span></div>$sport_switch<span class="sp"></span>
  <div class="seg" id="viewsw" role="group" aria-label="Projection view">
    <button data-v="mkt" aria-pressed="true" title="Starts from the sportsbook line; moves only as far as the model has earned">Market-adjusted</button>
    <button data-v="mod" aria-pressed="false" title="Pure model, no sportsbook input">Model only</button>
  </div>
  <button class="icon" id="theme" aria-label="Toggle dark mode">◐</button>
</div></header>
<div class="wrap">
$sections
<footer>Research and entertainment only. No model guarantees profit; bet responsibly. Data: ESPN / DraftKings, nflverse, CollegeFootballData.com.</footer>
</div>
<script>
(function(){
  var root=document.documentElement,body=document.body;
  function store(k,v){try{localStorage.setItem(k,v)}catch(e){}}
  function load(k){try{return localStorage.getItem(k)}catch(e){return null}}
  function fmt(ms){var m=Math.floor(ms/6e4),d=Math.floor(m/1440),h=Math.floor(m%1440/60),mm=m%60;
    return d?d+'d '+h+'h':h?h+'h '+mm+'m':mm+'m'}
  var sections=[].slice.call(document.querySelectorAll('.sport'));
  sections.forEach(function(sec){
    var sp=sec.dataset.sport,tabs=sec.querySelectorAll('.tabs button'),board=sec.querySelector('.board'),
        hist=sec.querySelector('.history'),key=sec.querySelector('.key'),none=sec.querySelector('.novalue');
    function tab(t){
      if(![].some.call(tabs,function(b){return b.dataset.t===t}))t='all';
      tabs.forEach(function(b){b.setAttribute('aria-pressed',b.dataset.t===t?'true':'false')});
      board.classList.toggle('hidden',!(t==='all'||t==='value'||t==='top'));
      hist.classList.toggle('hidden',t!=='history'); key.classList.toggle('hidden',t!=='key');
      var shown=0;
      board.querySelectorAll('.game').forEach(function(g){
        var h=(t==='value'&&g.dataset.value!=='1')||(t==='top'&&g.dataset.top!=='1');g.classList.toggle('hidden',h);if(!h)shown++});
      board.querySelectorAll('.day').forEach(function(d){
        var grid=d.nextElementSibling,any=[].some.call(grid.children,function(g){return !g.classList.contains('hidden')});
        d.classList.toggle('hidden',!any); grid.classList.toggle('hidden',!any);
      });
      none.classList.toggle('hidden',!(t==='value'&&shown===0));
      store('fm-tab-'+sp,t);
    }
    tabs.forEach(function(b){b.addEventListener('click',function(){tab(b.dataset.t)})});
    tab(load('fm-tab-'+sp)||'all');
    var games=[].slice.call(sec.querySelectorAll('.game')),el=sec.querySelector('.count'),lbl=sec.querySelector('.countlbl');
    function tick(){
      var now=Date.now(),next=null;
      games.forEach(function(g){
        var t=Date.parse(g.dataset.kick),c=g.querySelector('[data-chip]');
        if(g.dataset.final==='1'){c.textContent='Final';c.className='chip final'}
        else if(t<=now){c.textContent='Locked';c.className='chip locked'}
        else{c.textContent='in '+fmt(t-now);c.className='chip';if(next===null||t<next)next=t}
      });
      if(next===null){el.textContent='All games locked';lbl.textContent=''}
      else{var m=Math.floor((next-now)/6e4),d=Math.floor(m/1440),h=Math.floor(m%1440/60),mm=m%60;
        el.innerHTML=(d?d+'<small>d</small>':'')+h+'<small>h</small>'+(mm<10?'0':'')+mm+'<small>m</small>'}
    }
    tick(); setInterval(tick,30000);
  });
  var sb=document.querySelectorAll('#sportsw button');
  function sport(s){
    if(!sections.some(function(x){return x.dataset.sport===s}))s=sections[0].dataset.sport;
    sections.forEach(function(x){x.classList.toggle('off',x.dataset.sport!==s)});
    sb.forEach(function(b){b.setAttribute('aria-pressed',b.dataset.s===s?'true':'false')});
    store('fm-sport',s);
  }
  sb.forEach(function(b){b.addEventListener('click',function(){sport(b.dataset.s);window.scrollTo(0,0)})});
  sport(load('fm-sport')||'$first');
  var vb=document.querySelectorAll('#viewsw button');
  function view(v){body.dataset.view=v;vb.forEach(function(b){b.setAttribute('aria-pressed',b.dataset.v===v?'true':'false')});store('fm-view',v)}
  vb.forEach(function(b){b.addEventListener('click',function(){view(b.dataset.v)})});
  view(load('fm-view')||'mkt');
  var th=load('fm-theme'); if(th) root.dataset.theme=th;
  document.getElementById('theme').addEventListener('click',function(){
    var dark=root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme=dark?'light':'dark'; store('fm-theme',root.dataset.theme);
  });
})();
</script>
</body></html>
""")
