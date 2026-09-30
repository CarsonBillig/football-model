# Football model (NFL + college)

A weekly NFL and college football prediction pipeline that projects every game, prices winners/spreads/totals/moneylines against the
market, logs every pick **before kickoff**, grades them afterwards, and publishes it all as a single web page.

## Running it yourself (no Claude needed)

**First time on a computer** (skip if the `.venv` folder already exists):
1. Install Python 3.11+ from python.org and tick **"Add python.exe to PATH"**.
2. Double-click **`setup.bat`**. It installs everything into a local `.venv` folder (a few minutes).
3. Paste your CollegeFootballData.com key into **`cfbd_key.txt`** and save.

**Every week:** double-click **`run_weekly.bat`**. It grades last week, retrains, predicts this week for NFL and college,
opens the updated page, and (once connected) publishes it to GitHub Pages. Run it early in the week, and again any time
before kickoff to refresh lines. From a terminal you can add options: `run_weekly.bat --backtest` (monthly refresh of
the backtests) or `run_weekly.bat --nfl-only`.

## Every week: one command

```
.venv\Scripts\python update_model.py
```

1. **grades** last week's logged picks against final scores (✓ / ✗, units, closing line value)
2. **retrains** on every completed game, and re-estimates how much to trust the model vs the market using the
   backtest **plus your graded pick history** (this is how the model learns from its own record)
3. **predicts** the next week, pulling live lines (NFL: ESPN/DraftKings; college: CollegeFootballData.com)
4. **logs** each game's picks to `output/ledger.csv` (NFL) and `output/cfb/ledger.csv` (college); games that have kicked off are locked
5. **rebuilds** `docs/index.html` with an NFL / College switch: open it in a browser, or upload that one file anywhere (GitHub Pages, Netlify drop)

College needs the free CollegeFootballData.com key in `cfbd_key.txt` (it never changes; the 1,000 calls/month allowance
resets automatically, and a weekly run uses about 6). Use `--nfl-only` to skip college.

Run it any time during the week; re-running refreshes picks for games that haven't started.
Add `--backtest` about once a month to refresh the full walk-forward test (~2 minutes).

| file | what it does |
|---|---|
| `features.py`, `power_ratings.py` | leakage-safe pre-game features: opponent-adjusted EPA/success rate, QB rating, rest, ridge power ratings from scores and from past lines |
| `model.py` | ridge models for margin and total, walk-forward validation |
| `margin_dist.py` | key-number-aware score distributions (3 and 7 matter), market anchoring incl. juice |
| `pricing.py` | market + model blend, win/cover/over probabilities, EV, quarter-Kelly stakes |
| `backtest.py` | honest walk-forward test, 2018 to date → `output/backtest_report.txt` |
| `espn.py` | ESPN board (open/current lines, TV, venue, logos) + historical opening lines 2024+ |
| `ledger.py`, `grade.py` | pick history and grading |
| `site_builder.py` | the web page: labeled game cards, market-adjusted / model-only switch, pick history, Key tab |
| `cfb_client.py`, `cfb_data.py` | CollegeFootballData.com access and per-season cache (`cache/cfb/`); key in `cfbd_key.txt`, keep it private |
| `cfb_model.py` | college features (FCS opponents share one rating, talent + returning production early in the season) and backtest → `output/cfb/` |
| `cfb_pipeline.py` | college weekly grade / train / predict |

## What the backtest says (2,175 out-of-sample games, 2018 – week 3 2026)

| | record | note |
|---|---|---|
| Straight-up winners | 1443–724 (66.6%) | same as simply taking the market favorite |
| Against the spread, every game | 1092–1028–55 (51.5%) | 52.4% needed to profit at −110 |
| Over/under, every game | 1091–1064–20 (50.6%) | |
| Value spread bets (EV ≥ 2%) | 240–202 (+5.5% ROI) | not statistically significant (t = 1.2) |

* **Projection accuracy improved.** Margin RMSE is 13.05 vs 13.29 for v2's model, and the published blend matches the closing line (12.78).
* **The closing line is extremely hard to beat.** Nothing tested predicts *result minus closing line*: not the model, power ratings, ATS form, rest, QB changes or divisional games.
  The model's weight against the closing line is therefore small (about 6%), and it re-earns that weight every week.
* **Lines move toward the model.** Between ESPN's opening line and the close, spreads move about 22% of the way toward the model's number (t = 6.6), and totals do the same (t = 5.8).
  That kind of closing line value (CLV) is what professional bettors aim for. **Caveat:** ESPN's "open" is often a look-ahead line posted before the previous week's games.
  Betting the model's side at those openers went 140–174 in the 2025+ holdout, so this is **not yet proven to be bettable**.
  The live ledger records the line you actually saw when you ran the model, so its CLV column will answer the question honestly within a season.
* Win probabilities are well calibrated: every probability bucket lands within about 3 points of actual.

**Bottom line:** trust the winner probabilities and projected scores. Treat spread and total picks as leans unless
a value flag appears, and judge the model by live CLV over hundreds of picks, not by one week's record.

## College backtest (8,297 games, 2016 – week 4 2026)

| | record | note |
|---|---|---|
| Straight-up winners | 6337–1960 (76.4%) | |
| Against the spread at the close, every game | 4090–4054–153 (50.2%) | no edge at the closing line |
| Model info beyond the closing total | t = 3.5 | a real signal on totals |
| Lines move toward the model, open → close | t = 12.6 (4,456 games, 2021+) | |
| Picks at the **opening** line, 2024+ holdout | 1055–954 (52.5%); 586–504 (53.8%) when 3+ pts off the open | promising, not yet significant |

College lines are softer than the NFL's: the model's number shows up in where lines move, and picks made at the opening
number beat those made at the close. The live pick history (logged at the line you saw) is the test that settles it.

## Changes from v2

* **Bugs:** the verdict said "statistically significant edge" when the model was significantly *losing* (171–194). The test is now one-sided and checks direction.
  188 games (a third of 2022) were silently dropped for missing weather data; they're now imputed.
* **Model:** 98 noisy columns replaced with 14 margin and 10 total features, plus ridge power ratings with offseason regression.
  XGBoost and the fixed-weight ensemble were dropped because they tested worse.
* **Probabilities:** a normal curve was replaced with key-number-aware distributions that also read the juice (−3 at −120 is not −3 at +100), then Platt-calibrated.
* **Moneyline:** the moneyline market's own price is blended in. Without it, "value" moneyline bets lost 5%; with it they're around breakeven (+2.9%).
* **Picks:** decided by expected value at the actual price, not by a 0.36-point edge.
* **Live data:** ESPN open and current lines, TV, venue and records.
* **Pick history + website.**

v2's files are in `_v2_backup/`.

## Conventions

nflverse `spread_line > 0` means the **home** team is favored. The site and pick text use sportsbook style (favorite shown as −).
Weather is not used: history holds actual game-day weather, but live predictions would only have forecasts.
