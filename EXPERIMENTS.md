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

## Not finished

- College: do model picks do better in low-attention games (weeknight non-power-conference, FCS)?
