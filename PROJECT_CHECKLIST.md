# Flight Delays — Project Checklist (Path A)

**Repo:** [Machine_learning_Project](https://github.com/agarcia246/Machine_learning_Project)  
**Notebook:** `flight_delays.ipynb`  
**Row definition:** `(APT_ICAO, FLT_DATE)` daily airport panel.

---

## Team conventions (Phase 0) — agreed defaults

| Source | Grain | Join keys | Leakage rule (features for day *t*) |
|--------|--------|-----------|-------------------------------------|
| Traffic + ATC | airport × day | `APT_ICAO`, `FLT_DATE` | Lags use `shift(1)` per airport |
| ASMA | airport × month | `APT_ICAO`, `cal_year`, `cal_month` | Use **previous calendar month** after join |
| VFE | airport × month | same as ASMA | Use **previous calendar month** after join |
| ERT | country × day | `STATE_NAME`, `FLT_DATE` | Use **previous calendar day** after join |
| METAR | airport × day | `APT_ICAO`, `FLT_DATE` | Same-day weather (document) or prior day if strict |


## Done — classification core (Javier / team)

- [x] §1–11 — Load, base panel (traffic + ATC), targets, features, models
- [x] §12 — Champion + F1 threshold on validation
- [x] §13 — Locked test evaluation
- [x] §14 — Calibration plots
- [x] §15 — Regression (**teammate**)
- [ ] §16 — Permutation + SHAP (**teammate**)
- [ ] §17 — Temporal drift (**teammate**)

---

## Done — Path A setup (Phases 0–2)

- [x] Phase 0 — Conventions documented (this file + notebook §3b intro)
- [x] Phase 1 — §3b section scaffold in notebook (after base panel)
- [x] Phase 2 — Month/year keys, `asma_m` / `vfe_m` / `ert_country` prep, state-name mapping QA

---

## TODO — Path A data enrichment (Phases 3–7)

### Phase 3 — ASMA join
- [ ] Merge `asma_m` to `panel` on `APT_ICAO`, `cal_year`, `cal_month`
- [ ] Lag-1 month on all `asma_*` feature columns per airport
- [ ] EDA: missingness + short markdown

### Phase 4 — VFE join
- [ ] Merge `vfe_m` with `vfe_` prefix; lag-1 month
- [ ] Optional ratio features (CO₂ per flight, etc.)

### Phase 5 — ERT join
- [ ] Apply `STATE_NAME_MAP` to `ert_country`
- [ ] Aggregate to `STATE_NAME` × `FLT_DATE`; merge to panel
- [ ] Lag-1 day on `ert_*` columns

### Phase 6 — METAR
- [ ] Build `APT_ICAO → (lat, lon)` table
- [ ] Fetch/cache daily weather (`data/weather/`)
- [ ] Merge to panel; document missing airports

### Phase 7 — Panel QA
- [ ] Assert one row per `(APT_ICAO, FLT_DATE)`
- [ ] Missingness report for new columns
- [ ] Optional: `panel.to_parquet("data/panel_enriched.parquet")`

---

## TODO — Re-model & polish (Phases 8–11)

### Phase 8 — Retrain pipeline
- [ ] Re-run §4–§14 after enriched `panel`
- [ ] Compare val PR-AUC before vs after enrichment

### Phase 9 — Teammate sections
- [x] §15 regression
- [ ] §16, §17 (see notebook stubs)

### Phase 10 — Narrative
- [ ] Update abstract to match joins + lag rules
- [x] Expand README (setup, data sources, run instructions, agent context)
- [ ] Leakage paragraph in notebook

### Phase 11 — Submit
- [ ] Restart & run all
- [ ] Push to GitHub
- [ ] Ignore `data/weather/` cache in `.gitignore` if large

---

## Optional stretch (appendix in notebook)

- [ ] LightGBM + `libomp` on macOS
- [ ] Heavier `n_estimators` for final numbers
- [ ] Stacking / MLP
- [ ] Bootstrap CIs
- [ ] CO₂ narrative only with cited factors
