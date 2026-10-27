# 1 · The research harness

## Setup

Four specialist agents and one lead agent worked on the same repository under fixed rules:

| Role | Remit |
|---|---|
| **Lead** | Owns `KNOWLEDGE` (the shared fact base), reads every round report, decides what gets explored next and what gets dropped |
| **Frequentist** | Random/fixed effects, NNLS transfer matrices, trimmed estimators |
| **Bayesian** | Hierarchical partial pooling, empirical-Bayes hyperparameters, calibrated intervals |
| **ML** | Ridge, XGBoost, MLP, stacking on residuals, adaptive online refits |
| **Time series** | ARIMA, Kalman trend, probit order-bias corrector, GRU/LSTM on the arrival sequence |
| **Auditor** (from round 2) | Tries to break the leader: leakage, edge cases, other arrival orders, operational failure modes |

Rules applied to every agent:

- The evaluator and data loader are read-only. The leaderboard is append-only JSONL ([`harness/leaderboard2.jsonl`](../harness/leaderboard2.jsonl), [`leaderboard3.jsonl`](../harness/leaderboard3.jsonl)).
- A model sees round-1 data for every zone, and round-2 data **only for zones already counted** (`data.pre_view`).
- **2018 is the development set and 2022 is the test set.** 2026 could look like either, so a model has to be robust on both.
- Features from earlier elections must come from an election *before* the one evaluated.

## The evaluator

[`evaluate2.py`](../harness/evaluate2.py) replays a past election night. At each checkpoint (1%, 2%, 5%, 10%, 20%, 40%, 70% of
zones in) the model receives the counted zones and projects the national result, every state, every zone, and a 90% interval.
It runs several arrival orders, each with several seeds:

- `real`: the actual 2022 order
- `como2022`: the 2022 order mapped onto 2018
- `aleatorio`: random
- `geo`: whole states arrive together
- `pior`: adversarial; the zones that swing most toward the right-wing candidate arrive first

[`evaluate3.py`](../harness/evaluate3.py) and [`partial.py`](../harness/partial.py) add the piece the auditor showed was missing:
**partially counted zones**. Each zone fills in over time following a curve calibrated to TSE's official 2022 counting log
(`f = clip(τ / (c·ts), 0, 1)^k`). The early sections within a zone carry a bias `β·(1−f)^m` (β < 0, as in 2022) plus section noise.

## Results by round

**Round 0 (lead).** Anchoring on round 1 does most of the work. A one-parameter uniform swing is 0.06–0.12 pp off
nationally when arrival is realistic. XGBoost predicting the *level* of the vote was worse early on (up to 1.2 pp).
The level is the wrong target.

**Round 1 (all four specialists).**

| Model | 2018 RMSE | 2022 RMSE | Interval coverage (2018 / 2022) | Verdict |
|---|---|---|---|---|
| Uniform swing | 0.77 | 0.21 | n/a | baseline |
| Frequentist random effects | **1.64** | 0.14 | **0.05** / 0.93 | overconfident; breaks on 2018 |
| Hierarchical Bayes v5 | 0.46 | 0.135 | 0.92 / 0.99 | most robust |
| ML stack (first version) | 0.24 | 0.10 | n/a | **leak**: a feature fitted on the evaluated year's final result |
| Time-series probit corrector | 0.77 | n/a | n/a | halves the adversarial-order error, no cost otherwise; kept as an alarm |
| ARIMA / Kalman / GRU / LSTM | n/a | n/a | n/a | no gain; dropped |

**Round 2 (Bayes vs. ML, with the auditor).**

| Model | 2018 | 2022 | Note |
|---|---|---|---|
| `r2_bayes_final` | 0.405 | 0.139 | with the probit order term |
| `r2_bayes_final_dint` | 0.406 | **0.120** | without it; chosen |
| `r2_ml_stack` (v5 + XGB + MLP on residuals) | 0.44 | 0.135 | 1–5% better per zone; inherited interval no longer valid |

The choice of `dint` was justified on 2018 data alone. With the real 2022 order applied to 2018, `dint` is 0.09 pp off and
the probit version 0.26. The probit term fires when there is nothing to correct, so it was demoted to an alarm.

The auditor's round-2 findings changed the project:

- **C1.** The live feed totals *sections*. At 5% counted in 2022, only 98 zones were complete, and 56 of those were abroad. A model built on complete zones only would sit idle for the first hour.
- **C2.** A zone with zero votes made the model crash.
- **C3.** Candidate numbers and the previous election were hard-coded.
- **C6.** Votes from abroad arrive in two bursts and on their own shift the projection by +1.5 pp.

**Round 3 (partial counts).** Summing partial zones as if they were final is 4.3 pp off on average and about 10 pp off at the start.
`r3_domingo` uses the round-2 hierarchy and treats each partial zone as a measurement with:

- a within-zone bias, estimated from how that zone's numbers change between readings;
- noise that shrinks as the zone fills in;
- the abroad zones handled separately;
- a **publish gate**.

| | 2018 | 2022 (run once, frozen) |
|---|---|---|
| National RMSE | 0.456 | **0.209** |
| Worst *published* error | 0.34 | 0.28 |
| Interval coverage | 0.94 | 0.94 |

The final audit then tried to break it:

- **Duplicate readings.** If the feed is polled faster than it updates, no zone advances between readings, the history resets, and the model *never publishes*. This would have been critical on the night. Fixed by hashing each snapshot in `live.py`.
- **A different bias shape.** With a homogeneous −5 pp·(1−f)³ within-zone bias, the model published −1.9 pp with a ±0.25 interval: confidently wrong. The cause is that the partial-zone correction was learned under 2022's bias shape.
  Mitigations: run the completed-zones-only model in parallel and publish only when the two intervals overlap; keep a minimum ±0.5 pp half-width while more than 30% of votes come from partial zones; never call a winner with 50% inside the interval.
- **Whole-state revisions.** A full-state correction at 10% counted produced one bad published point (+0.77) before it recovered. This was logged as a known risk.

Election night showed a milder version of the bias-shape risk (see [02](02-election-night.md)). The gate kept the published intervals honest: they covered the final result 6 out of 7 times.

## Round 4: the user's hypotheses, plus a literature check

| Hypothesis | Test | Result |
|---|---|---|
| Regions that swung more before will swing more again | Within-state swing deviation, 2018 vs 2022 | **No.** r = 0.02 by zone, 0.01 across states. The 0.78 seen at first came from votes abroad and zone size |
| Neighbourhoods vote alike | 64,983 geocoded polling places, swing after removing the city mean | **Yes, at short range.** r = 0.32 under 0.5 km, 0.25 at 1–2 km, 0.12 at 2–5 km, about 0 beyond 5 km |
| …and that helps live projection | Live simulation with neighbour pooling | 1–5% less error per polling place, 1–3% per zone, about 0 nationally; neighbourhood names add nothing beyond distance |
| Vote transfers by zone (third party → Lula) | Ecological regression across agents | Unstable (Tebet → Lula estimated at 0.20 by one agent and 0.49 by another); not publishable as fact |

Published live-projection systems (Washington Post, Decision Desk HQ, BBC exit-poll models) use the same core design:
model the change from a known baseline, use geographic and demographic covariance, and check calibration as a function of the share counted.

## What the harness taught me about working with agents

- **An adversarial auditor is worth more than another modeller.** Every important finding was a failure mode, not a better model:
  the leak, the 2022 contamination, partial zones, duplicate reads, bias shape.
- **An append-only leaderboard plus a shared knowledge base keeps rounds cumulative.** Each round started from the
  lead's synthesis rather than from scratch.
- **Track test-set contamination.** By round 2, model selection had looked at 2022, so 2022 differences below about 0.05 pp were treated as seed noise and not chased.
