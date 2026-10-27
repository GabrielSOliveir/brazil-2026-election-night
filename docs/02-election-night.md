# 2 · Election night, Oct 25 2026

## Plumbing

The TSE results site is a static JSON tree. The files needed:

| File | Content |
|---|---|
| `/oficial/ele2026/{eleição}/config/mun-e{eleição}-cm.json` | every state → municipality → zone |
| `/oficial/ele2026/{eleição}/dados/{uf}/{uf}-e{eleição}-ab.json` | per-state progress (cheap to poll) |
| `/oficial/ele2026/{eleição}/dados/{uf}/{uf}{mun}-z{zona}-c0001-e{eleição}-u.json` | one zone: sections total/counted, turnout, votes per candidate |
| `/oficial/ele2026/arquivo-urna/{pleito}/...` | per-section signed ballot-box bulletin (`.bu`, ASN.1 DER) |

The modelling container could not reach `resultados.tse.jus.br`, so the reader ran as JavaScript inside a browser tab on the
TSE site ([`tse/tse_feed.js`](../tse/tse_feed.js)):

1. Load the municipality/zone list once.
2. On each poll, re-fetch only zones that are not complete, using 4–6 workers with backoff on HTTP 429. The first full sweep at 24-way concurrency was rate-limited at 17:03.
3. Serialise to CSV (`uf,mun,zona,ts,st,...,c13,c22,...,hg`; `ts`/`st` = sections total/counted, `hg` = file hash).
4. Hand the snapshot to [`live/live.py`](../live/live.py).

`live.py` keeps model state across readings. It needs that state to estimate within-zone bias from how each zone moves between readings. On each reading it:

- refuses a snapshot whose hash matches the previous one;
- runs the main model (`r3_domingo`) and the completed-zones-only check (`r2_bayes_final_dint`);
- applies the publish gate;
- appends a line to the log ([`data/derived/live_projection_log.csv`](../data/derived/live_projection_log.csv)).

## Before the count: ballot-box bulletins

Voting hours were extended to 19:00 on the day, and TSE held the totals until then. Signed bulletins from closed sections were
already public, though. I wrote a DER walker that pulls the presidential race out of each bulletin
([`tse/bu_decoder.py`](../tse/bu_decoder.py)) and decoded **860 bulletins at 17:07**. Projected with the same swing model,
they pointed to **51.1–51.3%** for Lula, about 0.5 pp low. That is roughly what one would expect from a convenience sample of early-closing sections.

## The night

| Time | % sections | Zones complete / with data | Raw | Model | 90% | Check (complete only) | Published? |
|---|---|---|---|---|---|---|---|
| 19:05 | 0.6 | 77 / 388 | 53.85 | 52.23 | 50.88–53.58 | 50.76 | no (gate) |
| 19:18 | 1.9 | 82 / 1,051 | 51.90 | 52.11 | 50.55–53.67 | 51.00 | no |
| 19:30 | 4.8 | 121 / 1,620 | 51.23 | 52.33 | 51.79–52.87 | 51.42 | no |
| **19:42** | 11.7 | 304 / 2,475 | 50.29 | **51.32** | 50.82–51.82 | 51.55 | **yes**, intervals overlap |
| 19:55 | 17.8 | 513 / 2,872 | 49.58 | 51.22 | 50.72–51.72 ✗ | 51.82 | yes |
| 20:08 | 28.0 | 922 / 3,432 | **49.49** | 51.45 | 50.95–51.95 | 51.79 | yes |
| 20:22 | 39.4 | 1,495 / 3,990 | 50.33 | 51.52 | 51.02–52.02 | 51.80 | yes |
| 20:34 | 52.5 | 2,215 / 4,634 | 50.71 | 51.65 | 51.15–52.15 | 51.82 | yes |
| 20:46 | 69.8 | 3,220 / 5,393 | 50.92 | 51.69 | 51.19–52.19 | 51.81 | yes |
| 20:59 | 85.1 | 4,350 / 5,862 | 51.40 | 51.75 | 51.25–52.25 | 51.80 | yes, winner called |
| **Final** | 100 | | **51.79** | | | | |

Error vs. the final 51.785 over the 10 readings, as mean absolute error:

| Estimate | MAE |
|---|---|
| Raw count | 1.25 pp |
| Main model | 0.32 pp |
| Completed-zones check | 0.25 pp |

Regional projections were stable from the first reading: NE about 69%, SE about 47%, S about 37%, CO about 40%, N about 50%, abroad about 54%.

## Where it went wrong

Between 19:42 and 20:22 the main model sat 0.3–0.5 pp below the final. São Paulo explains most of it. Its partially counted
zones were showing a *negative* swing, because the early sections within each zone leaned toward Flávio. At the same time, its completed zones were
showing +3.7 pp. The partial-zone correction expected the 2022 bias shape, which fades quickly as a zone fills in. This year the bias persisted
longer, which is exactly the failure the round-3 auditor had simulated. The gate did what it was designed to do: intervals stayed
wide (±0.5) and covered the final in 6 of 7 published readings. The 19:55 miss was 0.07 pp.

## Lessons

1. **Model at the level the data arrives at.** The bias lived *inside* zones. Bulletins are per section and public during the
   count, so a section-level model with polling-place pooling would have removed most of it.
2. **A cross-check you gate on but never blend in wastes information.** Once a few hundred zones are complete, an
   inverse-variance blend of the two estimates would have been closer at every reading from 19:42 onward. Computed after the fact
   from the log, the blend is 0.18 pp off at 19:55 and at most 0.06 pp off from 20:08.
3. **Operational engineering is most of live modelling.** The work was in rate limits, duplicate reads, a container that cannot reach the
   source, and moving files between machines, not in fancier models.
