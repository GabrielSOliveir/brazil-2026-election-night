# Data

All source data is public, from TSE (Tribunal Superior Eleitoral): the open-data portal at https://dadosabertos.tse.jus.br and the
results site at https://resultados.tse.jus.br.

## Included

| Path | What | Built from |
|---|---|---|
| `tse/2018/`, `tse/2022/` | Presidential results per zone (`cand.csv` votes per candidate, `det.csv` turnout/blank/null), both rounds; 2022 official counting log | TSE open data, "Resultados" per municipality/zone; `Historico_Totalizacao_Presidente_BR_2T_2022.csv` |
| `derived/zones_2026_round{1,2}.csv` | 2026 presidential results per zone (6,292 zones), final | TSE results-site JSON, read with `tse/tse_feed.js` |
| `derived/live_snapshots/t2_HHMM.csv` | The 10 per-zone snapshots of the round-2 feed taken on election night | same, live |
| `derived/live_projection_log.csv` | What the model output at each snapshot (`live/replay.py` reproduces it) | `live/live.py` |
| `derived/bulletins_1707_860_sections.csv` | 860 ballot-box bulletins decoded at 17:07 on election night | `arquivo-urna/` bulletins + `TSE.extract` |

## Not included (download into `data/raw/`, outputs go to `data/work/`)

| File | Used by |
|---|---|
| `votacao_secao_2022_BR.zip` (votes per section) and `eleitorado_local_votacao_2022.zip` (polling places with coordinates) | `research/build_polling_places.py` → `research/neighborhood_tests.py`, `research/neighborhood_live_sim.py` |
