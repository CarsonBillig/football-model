# CLAUDE.md: NFL, college football and NHL betting models

## The goal (read this first)

**Beat the betting market as well as we can, and measure ourselves honestly against the market and other public models.**
Every change, test and answer is judged by that.

- **Against the market:** the record at the line available when the bet is placed, with break-even at 52.4% for -110 bets,
  plus closing-line value (CLV): did our number beat the closing line? Win-loss needs hundreds of bets to mean
  anything; CLV shows up sooner.
- **Against other models:** use the CollegeFootballData Model Pick'em contest metrics (predictions.collegefootballdata.com),
  where straight-up, against-the-spread and mean absolute error are all graded against the Vegas closing line.
  Reference points: David Sasser's all-time record is 52.7% ATS over 2,014 games, the 2025 consensus of all models was
  about 52.5%, and most models finish at 50-51%.
- **Be honest:** give the sample size with every record and never present a hot streak as an edge. Say plainly when
  we are behind the benchmarks, and make that gap the next thing to work on.

## How we test (success criteria)

- **Walk-forward only:** nothing from a game itself, or from after kickoff, feeds its own prediction.
- **Tune, then hold out:** tune on earlier seasons and confirm on later seasons the model never saw (NFL: tune 2018-2022,
  hold out 2023+; college: hold out 2024+; NHL: test 2022+).
- **Ship only real gains:** a change goes in only if it beats the current model on the holdout, on the same games.
  Otherwise revert it.
- **Decide the metric and threshold before running the test.** Never pick a threshold after seeing the results.
- **Keep a record of what failed**, so it isn't tried again by accident (see git history and backtest reports).

## Project rules

- **NHL data:** only the NHL's official API and our own expected-goals model. Never use or download MoneyPuck data.
  Don't scrape sites whose terms forbid it.
- **Websites:** keep the layout as it is; picks update automatically. Ask before adding tags or sections.
- **GitHub:** publish from the CarsonBillig account (repo-local git config).
- **API key:** the CollegeFootballData key lives in `cfbd_key.txt`, which is gitignored. Never print, paste or commit it.
  Don't install the `cfbd` pip package (it breaks nflreadpy). The free tier allows about 1,000 calls a month, so cache
  everything.
- **Free APIs:** be polite. The NHL API rate-limits (HTTP 429), so make one request at a time and honour Retry-After.
- **Picks:** they are model output, not financial advice. Say so when giving them.

## Where things are

- **This folder (`nfl_model_v2`):** NFL and college. Weekly run: `run_weekly.bat` -> `update_model.py`. Backtests:
  `backtest.py` (NFL) and `cfb_model.py` (college). Pick history: `output/ledger.csv` and `output/cfb/ledger.csv`.
- **`../nhl_model`:** NHL. Daily run: `run_daily.bat` -> `nhl_predict.py`. Backtest: `nhl_backtest.py`.

## Coding guidelines

From [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) (MIT), based on Andrej Karpathy's
observations about common LLM coding mistakes. They favour caution over speed; use judgment on trivial tasks.

1. **Think before coding.** State assumptions. If something has more than one reading, present them rather than
   picking silently. Say so when a simpler approach exists. When something is unclear, stop and ask.
2. **Simplicity first.** Write the minimum code that solves the problem: no unrequested features, no one-use
   abstractions, no handling for impossible cases. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what the task needs and match the existing style. Don't refactor working code.
   Remove only the leftovers your own change created; mention other dead code rather than deleting it.
4. **Goal-driven execution.** Turn each task into a check that can be verified (for these models, usually a holdout
   test), state a short plan with a check for each step, and loop until the check passes.
