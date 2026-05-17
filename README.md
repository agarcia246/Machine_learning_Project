# European Flight Delays — ML Foundations Final Project

Predict **high ATC pre-departure delay days** and **delay intensity** at European airports using [EUROCONTROL](https://www.eurocontrol.int/) open operational data.


| **Main deliverable** | `[flight_delays.ipynb](flight_delays.ipynb)`                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| **Repository**       | [github.com/agarcia246/Machine_learning_Project](https://github.com/agarcia246/Machine_learning_Project) |


**Team:** Alejandro Zapata, Alex Garcia, Andres Befeler, Javier Cruz, Jose Miguel Reyes, Nicolas Gonzalez

---

## Overview

European air-traffic-management delay costs the EU economy an estimated **€10 billion per year** and contributes to avoidable aviation CO₂. We build a daily **airport × date** panel from EUROCONTROL Performance Review Unit data and tackle two coupled tasks:

1. **Classification** — Will tomorrow be a *high delay day* for this airport?
2. **Regression** — How many minutes of ATC pre-departure delay per departure should we expect?

Models use a **strict temporal split** (train ≤ 2022, validation = 2023, test > 2023). Hyperparameters are tuned on validation only; the test period is scored once for final reporting.

---

## Problem definition

**Unit of analysis:** one row per `(APT_ICAO, FLT_DATE)`.


| Task               | Target  | Definition                                                                                                                                |
| ------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Classification** | `y_clf` | `1` if `delay_per_dep` ≥ that airport's trailing **365-day 75th percentile** (computed with `shift(1)` so only past days count); else `0` |
| **Regression**     | `y_reg` | `delay_per_dep` when `FLT_DEP_3 ≥ 50`                                                                                                     |


**Core metric:** `delay_per_dep = DLY_ATC_PRE_3 / FLT_DEP_3` (average ATC pre-departure delay minutes per departure).

Airport-specific thresholds make “high delay” **relative to each airport's own history**, not a single global cutoff.

---

## Key results (current notebook run)

### Classification champion: tuned LightGBM (`lgbm_tuned`)

- Selected on validation **PR-AUC** (preferred under class imbalance).
- **F1-optimal threshold** on validation: `t* ≈ 0.48`.
- **Validation:** PR-AUC ≈ 0.61, ROC-AUC ≈ 0.72.
- **Locked test** (2024+): PR-AUC ≈ 0.49, ROC-AUC ≈ 0.69, F1 ≈ 0.47 (precision ≈ 0.38, recall ≈ 0.61).

Gradient boosting (LightGBM, HistGradientBoosting) clearly beats logistic regression and random forest on ranking metrics; tuning yields modest but consistent gains over default hyperparameters.

### Regression champion: random forest

- Selected on validation **RMSE**.
- **Test:** RMSE ≈ 1.15 min, MAE ≈ 0.37 min, R² ≈ 0.35.
- Tuned random forest achieves slightly better test RMSE (~1.14) but **untuned** random forest wins on validation RMSE, so it remains the reported champion under our selection rule.

Tree models substantially outperform a median dummy baseline; roughly **one-third of variance** in daily delay intensity is explained on the held-out test period.

---

## Data

All sources live under `[data/](data/)` as yearly CSVs (~2016–2026), loaded with `read_concat()` in the notebook.


| Folder                        | Grain             | Role                                              |
| ----------------------------- | ----------------- | ------------------------------------------------- |
| `airport_traffic/`            | airport × day     | Traffic volumes (`FLT_DEP_*`, `FLT_TOT_*`, …)     |
| `atc_pre_departure_delays/`   | airport × day     | `DLY_ATC_PRE_3`, `FLT_DEP_3` → `delay_per_dep`    |
| `asma_additional_time/`       | airport × month   | Arrival sequencing / additional time (enrichment) |
| `vertical_flight_efficiency/` | airport × month   | CDO/CCO, CO₂ deltas (enrichment)                  |
| `ert_dly_fir/`                | country/FIR × day | En-route delay by cause codes (enrichment)        |
| `weather/`                    | —                 | Reserved for METAR cache (gitignored when large)  |


**Daily panel merge keys:** `APT_ICAO` + `FLT_DATE`.  
**Planned monthly joins:** `APT_ICAO` + `cal_year` + `cal_month` (with **lag-1 month** after merge).  
**Planned country joins:** `STATE_NAME` + `FLT_DATE` on ERT (with **lag-1 day** after merge).

---

## Methodology

### Temporal split (do not change without team agreement)


| Split      | `FLT_DATE`   | Use                                                        |
| ---------- | ------------ | ---------------------------------------------------------- |
| Train      | ≤ 2022-12-31 | Fit models & preprocessors                                 |
| Validation | 2023         | Model selection, hyperparameters, classification threshold |
| Test       | > 2023-12-31 | **Single** locked evaluation                               |


### Features (baseline pipeline)

- Calendar encodings (month, weekday, holidays).
- Per-airport lags and rolling means of `delay_per_dep` (1 / 7 / 14 / 28 days; 7 / 30 days), all `**shift(1)`** to avoid leakage.
- `state_yday`: prior-day median delay by `STATE_NAME`.

### Models


| Task           | Baselines                                                                                         | Tuned (11.1 / 15.1)                     |
| -------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Classification | Dummy, logistic regression, decision tree, random forest, HistGradientBoosting, optional LightGBM | Logistic, HistGB, random forest, LightGBM |
| Regression     | Dummy (median), random forest, HistGradientBoosting, optional LightGBM                            | Random forest, HistGB, LightGBM           |


**Preprocessors:** `linear_preprocessor` (impute + scale + one-hot) for logistic regression; `tree_preprocessor` (ordinal categories) for tree models.

### Leakage rules

Never include in `X`: `y_clf`, `y_reg`, `thresh`, `delay_per_dep`, or same-day raw delay columns used to define the target. See `drop_cols` in notebook 9.

---

## Notebook map


| Section | Topic                                                  |
| ------- | ------------------------------------------------------ |
| 1–2    | Setup, load CSVs                                       |
| 3      | Build daily panel (traffic ⋈ ATC)                      |
| 3b     | Path A enrichment prep (ASMA, VFE, ERT, state mapping) |
| 4–5    | Regression target, EDA                                 |
| 6      | Classification target (`y_clf`)                        |
| 7–8    | Feature engineering, temporal split                    |
| 9–10   | Feature selection, preprocessing                       |
| 11     | Classification baselines                               |
| 11.1   | Classification hyperparameter tuning                   |
| 12–14  | Threshold, locked test eval, calibration               |
| 15     | Regression baselines & diagnostics                     |
| 15.1   | Regression hyperparameter tuning                       |
| 16     | Permutation importance + SHAP                          |




---

## Repository layout

```
Machine_learning_Project/
├── flight_delays.ipynb       # Main deliverable — run top to bottom
├── requirements.txt
├── README.md
├── data/                     # EUROCONTROL CSVs
│   ├── airport_traffic/
│   ├── atc_pre_departure_delays/
│   ├── asma_additional_time/
│   ├── vertical_flight_efficiency/
│   └── ert_dly_fir/
```


---

## Setup & run

```bash
cd Machine_learning_Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab flight_delays.ipynb
```

**macOS + LightGBM:** if `import lightgbm` fails with `libomp.dylib`, run `brew install libomp`. The notebook wraps LightGBM in `try/except` so the pipeline still runs without it.

**Run all cells** top-to-bottom after pulling changes. Full execution takes several minutes (tree models on ~1M+ panel rows). Keep 16 interpretation subsampled if runtime is an issue.

---


## References

- EUROCONTROL Performance Review Unit — [Open Performance Data](https://www.eurocontrol.int/)
- [Meteostat](https://meteostat.net/) — planned daily weather enrichment

