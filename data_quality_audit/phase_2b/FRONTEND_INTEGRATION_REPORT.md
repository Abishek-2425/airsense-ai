# AirSense AI — Phase 2B Frontend Integration & Synchronisation Report

**Document Date:** November 4, 2026  
**Auditor / Engineer:** Senior Machine Learning Engineer, Backend Architect & QA Lead  
**Application Component:** React 19 + TypeScript + Vite Single Page Application  
**Artifact Version:** v2.0.0  
**Build Verification:** Passed (`npm run build`, exit code 0)

---

## 1. Executive Summary

This report documents the comprehensive frontend updates executed under **Phase 2B** to synchronize the user interface with the promoted Release Candidate v2.0 (`aqi_model.pkl`, SHA256: `029f97a92da7c62a97b2f8a2ecf21ac4eae3235cdda3bd312ee43f854e9fa235`).

All user-facing views across the application's three core tabs (**Dashboard**, **Predict AQI**, and **Model & Methodology**) have been audited, updated, and verified against the backend API schemas and empirical model evaluation findings.

---

## 2. Updated TypeScript Interfaces (`frontend/src/types/index.ts`)

The TypeScript contract between frontend and backend was updated to support the Phase 2B operational features:

1. **`PredictRequest`**:
   - `pm25?: number` and `pm10?: number` made optional to reflect server-side ratio imputation.
   - `month?: number` added to allow seasonal ratio selection.
2. **`PredictResponse`**:
   - `input_values` updated with optional `pm25?: number | null`, `pm10?: number | null`, and `month?: number | null`.
   - `features_used?: Record<string, number>` added to display the actual features fed into the model.
   - `model_version?: string` added to display the deployed model version badge.
3. **`FormState` & `FormErrors`**:
   - Updated to support flexible input validation where either PM2.5 or PM10 can be omitted, with month auto-inferred from client system date.

---

## 3. Page-by-Page Integration Details

### 3.1 Dashboard (`frontend/src/pages/Dashboard.tsx`)
- **Metric Stat Cards**:
  - Out-of-sample $R^2$: Updated from legacy $0.8367$ to operational CPCB-scale **$0.8712$**.
  - Root Mean Squared Error (RMSE): Updated from $36.46$ to **$29.41$** ($-19.3\%$ error reduction).
  - Mean Absolute Error (MAE): Updated from $20.10$ to **$19.26$**.
  - Evaluation Sample Size ($N$): Synchronized to **$4,900$** held-out test calendar pairs.
- **Feature Importance Breakdown**:
  - Replaced legacy contaminated feature values with genuine empirical Gini importances:
    - $\text{PM}_{2.5}$: **$73.68\%$**
    - $\text{PM}_{10}$: **$11.84\%$**
    - $\text{CO}$: **$7.93\%$** (down from $40.37\%$ artifact)
    - $\text{SO}_2$: **$2.33\%$**
    - $\text{NO}_2$: **$2.19\%$**
    - $\text{O}_3$: **$2.03\%$**
  - Highlighted the **$85.52\%$ total particulate importance**, restoring alignment with CPCB atmospheric chemistry.

### 3.2 Predict AQI Page (`frontend/src/pages/PredictPage.tsx`)
- **Flexible Particulate Input**:
  - Added visual badge `Auto-estimated if empty` on PM2.5 and PM10 input labels.
  - Form validation updated: ensures at least one particulate measurement ($\text{PM}_{2.5}$ or $\text{PM}_{10}$) is provided. If one is omitted, the frontend automatically forwards the client calendar month (`new Date().getMonth() + 1`) to trigger server-side ratio imputation.
- **CO Anomaly Screening Guardrail**:
  - Updated CO training maximum threshold to **$17.0\ \text{mg/m}^3$** (down from distorted $175.81\ \text{mg/m}^3$).
  - Added warning banner when CO exceeds statutory ambient threshold ($17.0\ \text{mg/m}^3$), alerting the user that high readings may reflect local exhaust contamination.
- **Prediction Outcome Display**:
  - Displays operational predicted AQI ($0–500$ statutory scale).
  - If raw model output was clamped to statutory scale, displays explanatory note: `Raw model output: X.XX (capped to 500 statutory scale)`.
  - Displays model version tag (`v2.0.0`) and features utilized with imputation values (`features_used`).

### 3.3 Model & Methodology Page (`frontend/src/pages/ModelMethodologyPage.tsx`)
- **Seven-Stage Architecture Update**:
  - Step 2 updated to describe chronological embargo partitioning ($2015-01-01$ to $2019-12-04$ train, $2019-12-05$ embargo buffer, $2019-12-06$ to $2020-06-30$ test).
  - Step 3 updated to detail the $17.0\ \text{mg/m}^3$ CO anomaly screening and monthly $\text{PM}_{2.5}/\text{PM}_{10}$ ratio imputer.
  - Step 4 updated with operational scale metrics ($R^2 = 0.8712$, $\text{RMSE} = 29.41$).
- **Dedicated Phase 2B Upgrade Card**:
  - Added a dedicated card titled **Phase 2B Scientific Data Quality & Model Upgrades** detailing:
    1. Extreme CO anomaly screening (699 rows traced to Ahmedabad Maninagar).
    2. Atmospheric feature realism ($85.52\%$ particulate importance).
    3. Monthly ratio imputation (reducing missing-$\text{PM}_{10}$ test MAE from $26.84$ to $21.00$).
    4. Operational CPCB scale calibration ($\le 500$).

---

## 4. Build & Bundler Verification

A complete production build was executed using Vite and the TypeScript compiler:

```bash
npm --prefix frontend run build
```

### Execution Log:
```text
> frontend@0.0.0 build
> tsc -b && vite build

vite v8.3.5 building client environment for production...
transforming...
✓ 2519 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.68 kB │ gzip:   0.41 kB
dist/assets/index-Di-ou6DK.css   28.35 kB │ gzip:   5.81 kB
dist/assets/index-BM2KJbOe.js   662.11 kB │ gzip: 192.93 kB
✓ built in 1.96s
```

- **Exit Code:** 0
- **TypeScript Errors:** 0
- **Bundle Generation:** Complete (`dist/` generated cleanly)

---

## 5. Verification Checklist

| Item | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **API Contract** | Optional particulates supported in types | Verified | `PredictRequest` in `types/index.ts` |
| **Dashboard Metrics** | Reflects Release Candidate v2.0 metrics | Verified | $R^2 = 0.8712$, $\text{RMSE} = 29.41$, $\text{MAE} = 19.26$ in `Dashboard.tsx` |
| **Feature Importances** | Reflects empirical v2.0 values | Verified | $\text{PM}_{2.5}: 73.7\%, \text{PM}_{10}: 11.8\%, \text{CO}: 7.9\%$ in `Dashboard.tsx` |
| **CO Threshold** | Max observed CO updated to $17.0\ \text{mg/m}^3$ | Verified | `trainingMaxima['CO'] = 17.0` in `PredictPage.tsx` |
| **Particulate Inference** | Can submit with missing PM10 | Verified | Form validation allows missing PM10; submits month |
| **Methodology Page** | Documents chronological embargo and ratio imputation | Verified | Section C updated in `ModelMethodologyPage.tsx` |
| **Build Stability** | Zero TypeScript errors, clean production bundle | Verified | `npm run build` exited with code 0 |

