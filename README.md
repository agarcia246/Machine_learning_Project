# Machine Learning Project — European Flight Delays

Final group project for **ML: Foundations**. We predict **high ATC pre-departure delay days** at European airports using [EUROCONTROL](https://www.eurocontrol.int/) open operational data, with optional weather and performance enrichments.

**Canonical notebook:** [`flight_delays.ipynb`](flight_delays.ipynb)  
**Task tracker:** [`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) (Path A enrichment + teammate sections)  
**GitHub:** [agarcia246/Machine_learning_Project](https://github.com/agarcia246/Machine_learning_Project)

---

## Team

Alejandro Zapata, Alex Garcia, Andres Befeler, Javier Cruz, Jose Miguel Reyes, Nicolas Gonzalez

---

## Problem statement

**Unit of analysis:** one row per **`(APT_ICAO, FLT_DATE)`** — airport and calendar day.

| Task | Target | Definition |
|------|--------|------------|
| **Classification** (primary, §12–14) | `y_clf` | `1` if `delay_per_dep ≥` airport-specific trailing **365-day Q75** (computed with `shift(1)` so only past days count); else `0` |
| **Regression** (teammate, §15) | `y_reg` | `delay_per_dep` = `DLY_ATC_PRE_3 / FLT_DEP_3` when `FLT_DEP_3 ≥ MIN_DEPARTURES` (50) |

**Core delay metric:** `delay_per_dep` — average ATC pre-departure delay minutes per departure (from `atc_pre_departure_delays`).

We are on **Path A**: enrich the panel with ASMA, VFE, ERT, and METAR (see checklist). Phases **0–2** are scaffolded in the notebook; **joins (3–6)** and **retrain (8+)** are still open.

---

## Repository layout

```
ML/
├── flight_delays.ipynb      # Main deliverable — run this
├── PROJECT_CHECKLIST.md     # Path A phases + ownership checkboxes
├── requirements.txt         # Python dependencies
├── README.md                # This file (agent + teammate context)
├── data/                    # EUROCONTROL CSVs (committed, ~181 MB)
│   ├── airport_traffic/
│   ├── atc_pre_departure_delays/
│   ├── ert_dly_fir/
│   ├── asma_additional_time/
│   ├── vertical_flight_efficiency/
│   └── weather/             # (future) cached METAR — add to .gitignore if large
├── Actual_work.ipynb        # Older draft — do not treat as canonical
└── assignment_1__*.ipynb    # Unrelated individual assignment (bank marketing)
```

**Do not commit** unless asked: `flight_delays_executed.ipynb`, `.venv/`, large weather caches.

---

## Data sources (`data/`)

All folders use yearly CSVs (roughly **2016–2026**). Loaded via `read_concat(folder)` in the notebook.

| Folder | Grain | Role | In `panel` today? |
|--------|--------|------|-------------------|
| `airport_traffic/` | airport × day | Departures/arrivals (`FLT_DEP_1`, `FLT_TOT_1`, …) | Yes |
| `atc_pre_departure_delays/` | airport × day | `DLY_ATC_PRE_3`, `FLT_DEP_3` | Yes |
| `asma_additional_time/` | airport × **month** | Arrival sequencing / additional time | Prepared as `asma_m` (§3b) |
| `vertical_flight_efficiency/` | airport × **month** | CDO/CCO, CO₂ deltas | Prepared as `vfe_m` (§3b) |
| `ert_dly_fir/` | country/FIR × day | En-route delay by IATA cause codes | Prepared as `ert_country` (§3b) |
| METAR (external) | airport × day | Weather via `meteostat` | Not yet |

**Merge keys (daily panel):** `APT_ICAO` + `FLT_DATE` (dates normalized to midnight UTC).  
**Monthly joins (planned):** `APT_ICAO` + `cal_year` + `cal_month` (from `FLT_DATE`).  
**Country joins (planned):** `STATE_NAME` + `FLT_DATE` on ERT after mapping `ENTITY_NAME` → `STATE_NAME`.

---

## Notebook map (`flight_delays.ipynb`)

| Section | Status | What it does |
|---------|--------|----------------|
| §1 Setup | Done | Imports, `DATA_DIR`, `RANDOM_STATE=42` |
| §2 Load | Done | `read_concat` for all five EUROCONTROL folders |
| §3 Panel | Done | `panel` = `airport_traffic` ⋈ `atc_pre_departure_delays` |
| §3b Enrich | **Partial** | Phases 0–2: `cal_year`/`cal_month`, `asma_m`, `vfe_m`, `ert_country`, state-name QA |
| §4 Target | Done | `delay_per_dep`, `MIN_DEPARTURES=50` |
| §5 EDA | Done | Basic plots / missingness |
| §6 `y_clf` | Done | Trailing Q75 threshold per airport |
| §7 Features | Done | Calendar, lags (1/7/14/28d), rolls (7/30d), `state_yday` |
| §8 Split | Done | Train ≤ 2022, val = 2023, test > 2023 |
| §9–11 Models | Done | Dummy, logreg, tree, RF, HistGB; optional LightGBM |
| §12 Threshold | Done | Champion by val **PR-AUC**; **F1-optimal** `t_star` on val |
| §13 Test clf | Done | Locked test metrics + confusion matrices |
| §14 Calibration | Done | Reliability curves (val + test) |
| §15 Regression | **Teammate stub** | `pass` + TODO |
| §16 Interpretation | **Teammate stub** | Permutation + SHAP TODO |
| §17 Drift | **Teammate stub** | Val vs test / era slices TODO |
| Appendix | Reference | Optional stretch list |

After **Path A joins (phases 3–6)**, re-run **§4 → §14** so models use enriched features.

---

## Temporal split (do not change without team agreement)

| Split | `FLT_DATE` filter | Use |
|-------|-------------------|-----|
| Train | `≤ 2022-12-31` | Fit models |
| Validation | `2023-01-01` … `2023-12-31` | Pick model + threshold |
| Test | `> 2023-12-31` | **One** final report (no tuning) |

Variables after split: `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test` (classification); same `X` for regression on `y_reg`.

---

## Leakage rules (Path A — mandatory)

| Feature source | Safe use on day *t* |
|----------------|---------------------|
| Airport lags / rolls (§7) | Already `shift(1)` within `APT_ICAO` |
| `state_yday` | Prior-day median delay by `STATE_NAME` |
| ASMA / VFE (monthly) | Join then **lag 1 calendar month** per airport |
| ERT (country-day) | Join then **lag 1 calendar day** per `STATE_NAME` |
| METAR | Team choice: same-day (document) or prior day |

**Never** put in `X`: `y_clf`, `y_reg`, `thresh`, `delay_per_dep`, raw same-day delay columns used to define the target (`DLY_ATC_PRE_3`, etc.) — see `drop_cols` in §9.

**Known fixed bug:** removed duplicate lag column `lag_delay per dep_*` (space in name); only `lag_delay_per_dep_*` should exist.

---

## Key in-notebook objects (for agents)

| Name | Description |
|------|-------------|
| `panel` | Daily airport table (traffic + ATC); enrichment adds columns later |
| `df` | Filtered panel (`FLT_DEP_3 ≥ 50`, valid `y_clf`) + engineered features |
| `asma_m`, `vfe_m` | Monthly tables, prefixed columns, ready to merge in Phase 3–4 |
| `ert_country` | `ENTITY_TYPE == "COUNTRY (FIR)"`, `ert_*` columns |
| `MANUAL_STATE_NAME_MAP` | Panel `STATE_NAME` → ERT `ENTITY_NAME` fixes (extend as needed) |
| `fitted_models` | Dict of fitted sklearn `Pipeline`s from §11 |
| `champion`, `t_star` | Best model by val PR-AUC; threshold from val F1 sweep |
| `results_df` | Val metrics per classifier |

**Preprocessors:** `linear_preprocessor` (impute + scale + one-hot) for logreg; `tree_preprocessor` (ordinal cats) for trees.

---

## Environment setup

```bash
cd /path/to/Machine_learning_Project   # repo root — notebook uses Path.cwd() / "data"
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab flight_delays.ipynb
```

**macOS + LightGBM:** if `import lightgbm` fails with `libomp.dylib`, run `brew install libomp`. The notebook wraps LightGBM in `try/except (ImportError, OSError)` so runs still work without it.

**Run all cells** top-to-bottom after pulling changes. Expect several minutes if RF/ERT tables are large; SHAP/permutation (§16) should use subsamples.

---

## State name mapping (ERT ↔ panel)

ERT uses `ENTITY_NAME`; the panel uses `STATE_NAME`. Not all strings match exactly.

**Panel-only examples:** Israel, Luxembourg, Montenegro, Morocco, Republic of North Macedonia, Serbia  

**ERT-only examples:** North Macedonia, Spain Continental/Canarias, UK Continental/Oceanic, Turkiye, Portugal Continental, …

Extend `MANUAL_STATE_NAME_MAP` in §3b before Phase 5 merge. Inspect `state_name_map` and unmapped lists printed in that cell.

---

## Git workflow

- **Main branch:** `main` on `origin`
- **Canonical file to commit:** `flight_delays.ipynb`, `PROJECT_CHECKLIST.md`, `README.md`
- Commits from Cursor may add `Co-authored-by: Cursor`; human commits should be author-only (use normal terminal or `git commit-tree` if needed)

---

## What to work on next (quick pointer)

1. Open [`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) and claim a section.
2. **Data team:** Phases 3–6 (merge `asma_m`, `vfe_m`, `ert_country`, METAR) in §3b.
3. **Modeling team:** §15 regression, §16 SHAP, §17 drift.
4. After enrichment: Phase 8 — re-run §4–§14, update abstract.

---

## Other notebooks (ignore for submission)

| File | Note |
|------|------|
| `Actual_work.ipynb` | Early exploration; Spanish placeholders; not aligned with final pipeline |
| `assignment_1__alejandro_zapata[_zapata]_.ipynb` | UCI Bank Marketing — individual homework, different dataset |

---

## References

- EUROCONTROL Performance Review Unit — Open Performance Data (see dataset citation in notebook abstract)
- [Meteostat](https://meteostat.net/) — planned daily weather enrichment
- Course repo: [agarcia246/Machine_learning_Project](https://github.com/agarcia246/Machine_learning_Project)
