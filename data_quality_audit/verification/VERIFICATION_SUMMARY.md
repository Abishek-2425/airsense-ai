# AirSense AI — Phase 1B: Verification Summary & Decision Ledger

**Status**: Completed  
**Date**: November 2026  
**Auditor**: Senior Software Architect & Technical Project Auditor  
**Deliverable Directory**: `data_quality_audit/verification/`

---

## 1. Executive Summary

Phase 1B was conducted to independently evaluate and empirically verify seven critical findings regarding data quality, sensor validity, feature importance, API architecture, and documentation accuracy before making any modifications to the production system.

All verifications were executed under strict isolation constraints:
- `archive/` raw datasets were preserved without modification.
- `aqi_model.pkl` and `model_validation/aqi_model_candidate.pkl` were inspected in read-only mode.
- Retraining and imputation evaluations were performed in isolated experimental scripts.

---

## 2. Verification Findings Ledger

| # | Finding | Previous Audit Conclusion | Phase 1B Verification Result | Concrete Forensic Evidence | Recommended Action | Classification |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Ahmedabad CO Concentration** | Spikes concentrated in Ahmedabad; single station suspected. | **100% of extreme CO (>50 mg/m³) originates exclusively from station GJ001.** | `stations.csv` contains only `GJ001`. `city_day.csv` matches `station_day.csv` with $0.000$ diff across all columns. All top 20 national CO values ($98.33$–$175.81\ \text{mg/m}^3$) are from `GJ001`. | Flag `GJ001` observations in preprocessing; add $17\ \text{mg/m}^3$ advisory warning in backend. | **CONFIRMED** |
| **2** | **CO Sensor Contamination vs Calibration** | Values are unphysical; sensor defect suspected. | **Aggregation is authentic (diff < 0.005); ambient breathing air of 446 mg/m³ is toxicologically impossible; instrumental artifact strongly suspected.** | Hourly CO jumped from $2.0$ to $357\ \text{mg/m}^3$ in 1 hour; peak hourly CO was $446.41\ \text{mg/m}^3$ ($\approx 390\ \text{ppm}$). OSHA ceiling standard is $200\ \text{ppm}$. Unphysical $\text{CO}/\text{NO}_2$ ratio of $4.5$. | Exclude or cap unphysical $\text{CO} > 17\ \text{mg/m}^3$ observations during model training. | **STRONGLY SUSPECTED** |
| **3** | **Regional PM10 Missingness & Imputation** | Severe missingness in Lucknow/Chennai; median imputation invalid. | **Lucknow is 100% missing; current median imputer fails (R² = -0.062); monthly ratio imputation reduces error by 57.5% (R² = 0.744).** | Lucknow: $2,009 / 2,009$ days missing ($100\%$). Patna: $89.72\%$ missing. Chennai: $84.97\%$ missing. Benchmark on $17,476$ ground-truth pairs: Median MAE = $61.11$, Monthly Ratio MAE = $25.98$. | Replace global median imputer with physical $\text{PM}_{2.5}/\text{PM}_{10}$ seasonal ratio preprocessor. | **CONFIRMED** |
| **4** | **AQI Ground-Truth Values Exceeding 500** | 543 rows violate CPCB 500 ceiling; calculation questioned. | **Confirmed 543 rows; caused by un-capped linear extrapolation in scraper sub-index calculation (Ahmedabad accounts for 75.9%).** | CPCB NAQI terminates at $500$. Extrapolation slope $100/6$ on $\text{CO} = 132.47$ yields sub-index $2,041.17$ and daily AQI $2,049$. Delhi post-Diwali particulate extrapolation reached $716$. | Retain raw values in research data; clamp operational display AQI to $500$ with "Severe" advisory. | **CONFIRMED** |
| **5** | **Genuine CO Feature Importance & Distortion** | Production model is distorted by extreme CO, allocating excessive importance. | **Production model allocates 40.37% importance to CO; removing Ahmedabad collapses CO to 3.12% and restores PM2.5 to 83.39%.** | Inspection of `aqi_model.pkl` regressor. Controlled experiment without Ahmedabad: $\text{CO}$ importance falls by $-37.25\%$, while non-Ahmedabad test $R^2$ remains $0.8807$. | Retrain model using anomaly-screened data to eliminate unphysical reliance on CO. | **CONFIRMED** |
| **6** | **API Schema vs Pipeline Inconsistency** | API requires 6 inputs; pipeline imputer is unused. | **API rejects nulls with HTTP 422; pipeline contains SimpleImputer; naively allowing nulls causes dangerous under-prediction.** | `backend/schemas.py` lines 16–29 enforce finite floats. If $\text{PM}_{10}$ is omitted during a $350\ \mu\text{g/m}^3$ smog day, median imputer inserts $101.88$, deflating predicted AQI. | Maintain strict validation until domain-aware ratio preprocessor is integrated into backend. | **CONFIRMED** |
| **7** | **`HOW_IT_WORKS.md` Documentation Discrepancy** | Documented feature importances do not match model. | **Discrepancy confirmed: CO documented at ~8.9%, but actual serialized model has CO at 40.37%.** | `HOW_IT_WORKS.md` lines 94–101 report $\text{CO}: 8.9\%$, $\text{PM}_{10}: 17.8\%$, $\text{NO}_2: 12.5\%$. Real model has $\text{CO}: 40.37\%$, $\text{PM}_{10}: 4.26\%$, $\text{NO}_2: 2.94\%$. | Update documentation in Phase 2 to accurately describe model artifacts. | **CONFIRMED** |

---

## 3. Classification Summary

* **CONFIRMED (6 Findings)**:
  - Finding 1: Extreme CO Single-Station Provenance (`GJ001`).
  - Finding 3: Severe $\text{PM}_{10}$ Missingness and Ratio Estimation Superiority.
  - Finding 4: Origin and Mathematical Extrapolation of AQI $> 500$.
  - Finding 5: Genuine $\text{CO}$ Feature Importance ($40.37\%$) and Model Distortion.
  - Finding 6: API Schema vs Pipeline Inconsistency.
  - Finding 7: Feature Importance Documentation Discrepancy in `HOW_IT_WORKS.md`.
* **STRONGLY SUSPECTED (1 Finding)**:
  - Finding 2: Sensor Gain Calibration Drift / Localized Flue Impingement (supported by toxicological limits and stoichiometric ratios, though original hardware logs from 2018 cannot be physically retrieved).
* **UNSUPPORTED (0 Findings)**:
  - No audit findings were refuted or found to be false.

---

## 4. Final Project Readiness Decision

### Readiness Decision: **CONDITIONALLY READY FOR PHASE 2 (RETRAINING & REMEDIATION)**

The production system in its current state cannot be deployed to real-world end-users without remediation:
1. The production model `aqi_model.pkl` is heavily distorted, allocating $40.37\%$ of its decision variance to anomalous single-station $\text{CO}$ spikes.
2. Cities lacking $\text{PM}_{10}$ sensors cannot use the API without submitting fabricated values or suffering from broken median imputation.
3. The documentation in `HOW_IT_WORKS.md` misleads technical stakeholders regarding the true feature importance of the deployed model.

However, because all failure modes are thoroughly understood, mathematically proven, and have validated solutions, the project is **fully ready to proceed to Phase 2**.

### Prioritized Remediation Roadmap for Phase 2:
1. **Data Preprocessing & Anomaly Screening**:
   - Implement the physical monthly $\text{PM}_{2.5} / \text{PM}_{10}$ ratio imputer for missing particulate values.
   - Screen out or cap unphysical $\text{CO} > 17.0\ \text{mg/m}^3$ observations in the training split.
2. **Model Retraining**:
   - Retrain `RandomForestRegressor` with the clean, ratio-imputed dataset.
   - Verify that $\text{PM}_{2.5}$ importance exceeds $75\%$ and $\text{CO}$ importance is $< 5\%$.
3. **Backend & API Modernization**:
   - Add ratio-based preprocessing to the backend pipeline.
   - Update `backend/schemas.py` to permit optional particulate inputs while maintaining strict validation for unphysical values.
   - Clamp display predictions to $500$ with an explicit "Severe / Hazardous Emergency" advisory badge.
4. **Documentation Synchronization**:
   - Update `HOW_IT_WORKS.md` and `model_metadata.json` to reflect the newly trained model's true empirical metrics.

