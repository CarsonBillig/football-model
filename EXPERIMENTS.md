# Model experiments (football)

Every idea tested, with the result and the decision, so nothing failed gets tried again by accident (see CLAUDE.md).
"Holdout" means seasons the model never trained on. Records are against the spread unless noted.

| Date | Idea | Result | Decision |
|---|---|---|---|
| 2026-10-01 | NFL: injury report (regular starters out, by unit) | every-game spread picks 51.5% -> 52.5%; value spread bets 54.3% -> 56.3% (139-108) | shipped |
| 2026-10-01 | NFL: wind and cold for totals | model-only totals error 13.434 -> 13.399; market-adjusted totals unchanged | shipped |
| 2026-10-01 | College: returning QB production, transfer portal talent, incoming QB | picks 3+ pts off the open 53.8% -> 54.7% | shipped |
| 2026-10-01 | NFL: retune rating half-lives and season carry-over | about 0.004 pts better on 2018-22, totals worse on 2023+ | not shipped |
| 2026-10-05 | NFL totals: run/pass mix, turnovers | holdout totals error 13.193 -> 13.199 / 13.194 | not shipped |
| 2026-10-05 | College: new head coach (early season) | margin error 16.642 -> 16.628; picks 3+ off the open 54.7% -> 54.8% | shipped |
| 2026-10-06 | NFL: success rate split into passing and rushing | holdout error 12.970 -> 12.970 / 12.973 | not shipped |
| 2026-10-13 | College: rate each FCS team individually (all teams, all games) | FCS error better, FBS games slightly worse | not shipped |
| 2026-10-13 | College: individual FCS ratings, only for games involving an FCS team | holdout FCS error vs Vegas +2.53 -> +1.68; FBS unchanged; ATS vs close 50.1% -> 51.0% | shipped |
| 2026-10-13 | College: incoming transfers' production at their previous school (early season) | holdout weeks 1-6 error vs Vegas +1.48 -> +1.28; holdout ATS 51.1% -> 51.8%; 2021-23 slightly worse | shipped |
| 2026-10-13 | Market bias: fade the team that beat last week's line by 14+ more | college 49.7% / 51.5%; NFL 50.4% (2016-23) / 55.1% (2024-26) | no edge (fails the discovery years) |
| 2026-10-13 | Market bias: bet against brand-name teams | college 47.2% / 52.3%; NFL 46.1% / 49.0% | no edge |
| 2026-10-21 | NFL: inexperienced QBs start from backup level (ramp over 500 dropbacks) | holdout QB-change error 14.188 -> 14.093, but holdout ATS 49.8% -> 49.3% and tuning-years error slightly worse | not shipped |
| 2026-10-21 | NFL player props (pass/rush/rec yards, receptions) vs ESPN BET / DraftKings lines 2024-26 | book lines more accurate than our projections in every market; passing yards looked good on 2024 but failed 2025-26 (open +1.3% on 217 bets, later -8.5% on 179); top 3 per game about break-even (858 bets, -1.1% / +0.8%); rushing unders passed (561 bets, 55.8%) but all unders, tracking a market-wide trend (unders hit 50.7% / 54.7% / 58.3% in 2024/25/26); requiring our projection below the line: 57.1% on 317 bets in 2025-26 | rushing unders shipped as a labeled paper trade only |
| 2026-10-21 | Props: every game's top 3 rushing unders, labeled pick (projection below line) or lean (at/above) | 2025-26: picks 57.3% on 309 bets (+7.9%), leans 55.0% on 320 (+4.0%); both about 51% in 2024; covers every game with usable lines | shipped as the paper trade, confidence shown per prop |
| 2026-11-04 | Line moved 2+ pts against our side since the open (skip those picks) | NFL: those picks 47.2% (2024, 36 games) and 43.7% (2025-26, 71); dropping them 48.6% -> 50.0% on 2025-26. College: 44.5% (2021-23) but 51.0% (2024-26) | NFL: use as guidance when giving picks (small sample, no site flag); college: no edge |
| 2026-11-04 | College: do picks do better in low-attention games (weeknight non-power-conference, FCS)? | low-attention 50.4% vs 50.5% (2016-23), 48.4% vs 52.6% (2024-26) | no edge |
| 2026-11-04 | College: rest days, bye weeks, away travel distance, time zones crossed | holdout 2024-26 error vs Vegas +0.830 -> +0.813, ATS 51.7% -> 51.8%; 2016-23 error +0.774 -> +0.769; full backtest ATS vs close 50.8% -> 51.0%, model-vs-close information t 1.9 -> 2.1 | shipped |
| 2026-11-04 | NFL: turnover luck (fumbles lost vs the league's 43.7% recovery share; INTs vs 19.3% of defended passes) | holdout 2023+ error vs Vegas +0.338 -> +0.338 (fumbles) / +0.345 (fumbles + INTs); 2018-22 slightly worse | not shipped (market and ratings already discount it) |
| 2026-11-04 | College: havoc rate (TFLs, sacks, forced fumbles, INTs, breakups per play; API data usable from 2016) | holdout 2024-26 error vs Vegas +0.813 -> +0.813; 2016-23 +0.769 -> +0.774 | not shipped |
| 2026-11-04 | NFL: special teams, net EPA on punts and kickoffs (field goals and extra points too noisy: all special teams made 2018-22 slightly worse) | passed the preset test: holdout 2023+ error vs Vegas +0.338 -> +0.320, model-only ATS 49.2% -> 49.6%; 2018-22 +0.169 -> +0.159. But the site's market-adjusted picks got worse on the same games: 2018-22 691-587 (54.1%) -> 675-603 (52.8%), 2023+ 453-459 (49.7%) -> 450-462 (49.3%) | not shipped (no gain in the picks we bet) |
| 2026-11-04 | Gameday betting: which posted picks still win at the gameday (closing) line? Subsets: line moved our way, unchanged, moved against; model-only number still 3+ (college) / 2+ (NFL) off the gameday line | college backtest, every pick 50.4% (2021-23) / 51.6% (2024-26); best subset (line moved our way) 51.4% / 52.8%; model still 3+ off 49.7% / 51.2%. NFL backtest 54.7% (2024) / 43.9% (2025-26), no subset holds up. Live 2026 first-posted picks graded at the close: college 123-102 (54.7%), NFL 29-27 (51.8%) | no gameday filter found; a bigger gameday disagreement does not mean a better pick |
| 2026-11-04 | College: detect QB changes from passing box scores (last game started by someone other than the season's main QB; plain and weighted by the main QB's share of attempts) | 2,418 of 11,260 games had a change on one side. Holdout 2024-26 error vs Vegas +0.808 -> +0.805 / +0.804, but model-only ATS 51.8% -> 51.7% / 51.6% and site picks 1214-1113 -> 1210-1117 / 1212-1115; 2016-23 error worse (+0.729 -> +0.732 / +0.731). Seen afterwards, not a preset test: our picks in QB-change games went 896-924 (49.2%) vs about 51.5% in the rest | not shipped; box scores cached in cache/cfb/passers_* |

## Queued

- College: skip picks in games where a team just changed quarterbacks (found after the fact at 49.2%;
  judge only on games after 2026-11-04, with the cached box scores)
