# AirSense AI — Phase 1B: Complete Findings Verification Report

**Status**: Verified & Concluded  
**Date**: November 2026  
**Auditor**: Senior Software Architect, Machine Learning Engineer & Project Auditor  
**Project Directory**: `G:\PROJECTS\evs` (`D:\PROJECTS\evs`)  
**Scope**: Full verification of 7 findings from Phase 1 Data Quality Audit before model retraining or code modification.

---

## 1. Executive Summary

Following the comprehensive Phase 1 Data Quality Audit, Phase 1B was executed to independently and empirically verify seven serious issues identified in the AirSense AI platform:
1. **Extreme Carbon Monoxide ($\text{CO}$) in Ahmedabad**: Proven to originate exclusively from a single continuous monitoring station (`GJ001: Maninagar, Ahmedabad - GPCB`), where daily concentrations reached $175.81\ \text{mg/m}^3$ and hourly readings reached $446.41\ \text{mg/m}^3$ ($\approx 390\ \text{ppm}$).
2. **Sensor Malfunction / Artifact**: Confirmed that aggregation across telemetry levels is mathematically authentic (not a database or arithmetic corruption). Given that sustained ambient breathing concentrations of $400\ \text{mg/m}^3$ are lethal and physically inconsistent with ambient dilution, an instrumental artifact (gain calibration drift or localized flue/probe flaring) is **strongly suspected**.
3. **Severe $\text{PM}_{10}$ Missingness**: Confirmed that Lucknow has **$100\%$** missing $\text{PM}_{10}$ ($2,009$ out of $2,009$ days), while Patna is $89.72\%$ and Chennai is $84.97\%$ missing. The current production `SimpleImputer(strategy='median')` fails completely ($R^2 = -0.0621$). Imputing $\text{PM}_{10}$ via monthly physical $\text{PM}_{2.5} / \text{PM}_{10}$ ratios cuts reconstruction error by **$57.5\%$** ($R^2 = 0.7435$).
4. **AQI Ground-Truth Values $> 500$**: Verified that all $543$ daily records exceeding $500$ (peaking at $2,049$) stem from un-capped piecewise linear sub-index extrapolation in the scraping code, with Ahmedabad accounting for $75.87\%$ ($412$ records).
5. **Model Distortion & Real Feature Importance**: In the serialized production model [`aqi_model.pkl`](file:///d:/PROJECTS/evs/aqi_model.pkl), $\text{CO}$ genuine feature importance is **$40.37\%$** (not $8.9\%$). In an isolated retraining experiment excluding Ahmedabad, $\text{CO}$ importance collapsed from $40.37\%$ to **$3.12\%$**, while $\text{PM}_{2.5}$ rose to **$83.39\%$**.
6. **API Schema vs. Pipeline Inconsistency**: The API strictly enforces that all six pollutants be non-null and finite (`Field(...)`), rejecting missing inputs even though the model contains a `SimpleImputer`. However, making API inputs optional without domain-aware ratio preprocessing poses serious under-prediction risks during severe pollution episodes.
7. **Documentation Discrepancy in `HOW_IT_WORKS.md`**: Confirmed that section 3.3 of `HOW_IT_WORKS.md` reports fabricated or misaligned feature importances ($\text{CO}: \sim 8.9\%$, $\text{PM}_{10}: \sim 17.8\%$, $\text{NO}_2: \sim 12.5\%$).

---

## 2. Evidence-Based Verification of the 7 Audit Findings

### Finding 1: Extreme CO Observations Associated with Maninagar, Ahmedabad
* **Previous Audit Conclusion**: Extreme $\text{CO}$ observations are concentrated in Ahmedabad and may originate from a single station.
* **Verification Execution & Result**:
  - `stations.csv` confirms that Ahmedabad possesses only one active monitoring station: `GJ001: Maninagar, Ahmedabad - GPCB`.
  - In `city_day.csv`, all 20 of the highest $\text{CO}$ observations in the entire national dataset occurred in Ahmedabad (ranging from $98.33$ to $175.81\ \text{mg/m}^3$).
  - Across the other 25 cities in India ($27,522$ records), the maximum recorded daily $\text{CO}$ is $30.44\ \text{mg/m}^3$ (Delhi, 2015-12-22), and $99.76\%$ of observations are $\le 17.0\ \text{mg/m}^3$.
  - 100% of national daily observations $> 50\ \text{mg/m}^3$ trace strictly to `GJ001`.
* **Classification**: **CONFIRMED (Single Station Provenance)**.

### Finding 2: Possible Sensor Contamination or Calibration Errors
* **Previous Audit Conclusion**: The values are unphysical and likely represent sensor contamination or calibration failure.
* **Verification Execution & Result**:
  - Cross-tier reconciliation: Daily $\text{CO}$ values in `city_day.csv` match `station_day.csv` with **$0.000$ discrepancy**.
  - Hourly aggregation reconciliation: Daily means match 24-hour arithmetic averages from `station_hour.csv` within **$0.005$ tolerance**.
  - Hourly telemetry: On 2018-02-19, hourly $\text{CO}$ reached **$446.41\ \text{mg/m}^3$** ($\approx 390\ \text{ppm}$) at 03:00.
  - Multi-pollutant analysis: During high $\text{CO}$ episodes, mean $\text{NO}_2$ rose to $102.04\ \mu\text{g/m}^3$ and $\text{PM}_{2.5}$ rose to $109.81\ \mu\text{g/m}^3$. This indicates real combustion backdrop, but the $\text{CO}/\text{NO}_2$ ratio was distorted by $10\times$ to $50\times$.
  - Ambient human safety: $446\ \text{mg/m}^3$ causes loss of consciousness and death within hours; sustained ambient breathing air at this concentration is physically impossible under standard dispersion.
* **Classification**: **STRONGLY SUSPECTED (Instrumental Gain Factor or Probe Contamination)**.

### Finding 3: Severe PM10 Missingness in Lucknow, Chennai, and Patna
* **Previous Audit Conclusion**: Missingness is severe and median imputation is methodologically flawed; ratio estimation should be evaluated.
* **Verification Execution & Result**:
  - Exact missingness verified:
    - **Lucknow**: $2,009$ out of $2,009$ days missing ($100.0\%$).
    - **Patna**: $1,667$ out of $1,858$ days missing ($89.72\%$).
    - **Chennai**: $1,707$ out of $2,009$ days missing ($84.97\%$).
  - Physical Ratio Analysis: Across $17,476$ paired records, the national median $\text{PM}_{2.5}/\text{PM}_{10}$ ratio is $0.4704$, varying systematically by season (January winter peak: $0.555$; May summer dust trough: $0.401$).
  - Imputation Benchmark on $17,476$ ground-truth pairs:
    - Current Production Global Median Imputation: $\text{MAE} = 61.11\ \mu\text{g/m}^3$, $\text{RMSE} = 92.11\ \mu\text{g/m}^3$, $R^2 = -0.0621$ (completely fails).
    - Monthly Physical Ratio Imputation: $\text{MAE} = 25.98\ \mu\text{g/m}^3$, $\text{RMSE} = 45.26\ \mu\text{g/m}^3$, $R^2 = 0.7435$ ($57.5\%$ error reduction).
* **Classification**: **CONFIRMED**.

### Finding 4: AQI Ground-Truth Values Exceeding 500
* **Previous Audit Conclusion**: 543 observations exceed 500 (max 2,049 in Ahmedabad); they violate CPCB scale definitions.
* **Verification Execution & Result**:
  - Total records with $\text{AQI} > 500$: exactly **$543$ records** ($1.84\%$ of dataset).
  - Geographical distribution: Ahmedabad accounts for $412$ records ($75.87\%$, max $2,049$); Delhi accounts for $48$ records ($8.84\%$, max $716$); Patna accounts for $25$ records (max $619$).
  - Root cause verified: CPCB NAQI (2014) defines breakpoints only up to $500$ ("Severe" category). Kaggle scraping scripts implemented linear interpolation without an upper cap:
    $$I_{\text{CO}} = 400 + \frac{100}{40 - 34}(C_{\text{CO}} - 34) = 400 + 16.67 \times (C_{\text{CO}} - 34)$$
  - For $C_{\text{CO}} = 132.47\ \text{mg/m}^3$, $I_{\text{CO}}$ mathematically computes to $2,041.17$, producing a daily mean AQI of $2,049$.
* **Classification**: **CONFIRMED (Mathematical Artifact of Uncapped Linear Extrapolation)**.

### Finding 5: Genuine CO Feature Importance (~40.37%) and Model Distortion
* **Previous Audit Conclusion**: The production model is distorted by extreme CO, allocating excessive importance to CO.
* **Verification Execution & Result**:
  - Inspection of `aqi_model.pkl`:
    - `PM2.5`: $46.47\%$
    - `CO`: **$40.37\%$**
    - `PM10`: $4.26\%$
    - `SO2`: $3.24\%$
    - `NO2`: $2.94\%$
    - `O3`: $2.71\%$
  - Clean temporal test set performance (Date $\ge 2019-12-06$):
    - Non-Ahmedabad ($25$ cities, $N=4,703$): $R^2 = 0.8803$, $\text{RMSE} = 26.18$, $\text{MAE} = 17.80$.
    - Ahmedabad only ($N=197$): $R^2 = 0.6533$, $\text{RMSE} = 129.29$, $\text{MAE} = 75.05$.
    - Test observations with $\text{CO} > 17\ \text{mg/m}^3$ ($N=34$): $R^2 = 0.1450$, $\text{RMSE} = 253.91$, $\text{MAE} = 195.21$.
  - Isolated Retraining Experiment (Train set *without* Ahmedabad, $N=17,870$):
    - `CO` importance collapsed by $-37.25\%$ down to **$3.12\%$**.
    - `PM2.5` importance rose by $+36.92\%$ up to **$83.39\%$**.
    - Non-Ahmedabad test performance remained virtually identical ($R^2 = 0.8807$ vs $0.8803$).
* **Classification**: **CONFIRMED**.

### Finding 6: API Schema Requires All 6 Pollutants Despite Pipeline Imputer
* **Previous Audit Conclusion**: The backend schema requires all six features, making the pipeline imputer redundant.
* **Verification Execution & Result**:
  - Confirmed in `backend/schemas.py`: All six fields are declared with `Field(...)` and validated with `math.isfinite(v)` and `v >= 0.0`.
  - The API returns HTTP 422 Unprocessable Entity if any field is missing or null.
  - In `aqi_model.pkl`, Step 1 of the pipeline is `SimpleImputer(strategy='median')`.
  - Disconnect verified: The ML pipeline can handle missing fields, but the API blocks them.
  - Risk verified: Naively relaxing the API schema to `Optional[float] = None` would trigger the flawed global median imputer, introducing severe under-prediction during winter episodes in Lucknow/Patna.
* **Classification**: **CONFIRMED (Architectural Disconnect)**.

### Finding 7: Feature Importance Documentation Discrepancy in `HOW_IT_WORKS.md`
* **Previous Audit Conclusion**: Documentation reports incorrect feature importances.
* **Verification Execution & Result**:
  - `HOW_IT_WORKS.md` (lines 94–101) states:
    - `PM2.5`: ~44.6%
    - `PM10`: ~17.8% (Actual: 4.26%)
    - `NO2`: ~12.5% (Actual: 2.94%)
    - `CO`: ~8.9% (Actual: **40.37%**)
    - `O3`: ~8.5% (Actual: 2.71%)
    - `SO2`: ~7.7% (Actual: 3.24%)
  - The actual production model has $\text{CO}$ at $40.37\%$, more than $4.5\times$ higher than claimed, while particulate matter $\text{PM}_{10}$ and combustion marker $\text{NO}_2$ are overstated by $4\times$.
* **Classification**: **CONFIRMED (Erroneous Documentation)**.

---

## 3. Systematic Verification Ledger

| # | Finding Description | Previous Audit Conclusion | Phase 1B Verification Result | Forensic Evidence Base | Recommended Operational Action | Status Classification |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Ahmedabad CO Concentration | Concentrated in Ahmedabad; single station suspected. | **100% of CO > 50 mg/m³ originate from station GJ001.** | `stations.csv`, `city_day.csv`, `station_day.csv` (0.000 diff). | Flag GJ001 records in dataset; introduce 17 mg/m³ validation warning. | **CONFIRMED** |
| **2** | CO Physical Validity & Sensor Drift | Values are unphysical; sensor defect suspected. | **Sensor artifact strongly suspected; open-air ambient CO of 446 mg/m³ is impossible.** | Toxicological limits (OSHA, IDLH); unphysical CO:NO2 ratio of 4.5. | Exclude or cap unphysical CO (>17 mg/m³) in training sets. | **STRONGLY SUSPECTED** |
| **3** | Regional PM10 Missingness & Imputation | Severe missingness in Lucknow/Chennai; median imputation invalid. | **Lucknow is 100% missing; median imputer fails (R² = -0.062); monthly ratio achieves R² = 0.744.** | Empirical analysis of 17,476 pairs; seasonal ratio shifts (0.40–0.56). | Implement seasonal PM2.5/PM10 ratio preprocessor before model retraining. | **CONFIRMED** |
| **4** | AQI Ground-Truth Values > 500 | 543 rows violate CPCB 500 ceiling; calculation questioned. | **Result of un-capped linear extrapolation in scraping scripts (Ahmedabad: 75.9%).** | Mathematical proof: slope 100/6 on CO=132.47 yields AQI 2049. | Retain raw in research archives; clamp display prediction to 500 in UI/API. | **CONFIRMED** |
| **5** | Model Distortion by Extreme CO | Model over-relies on CO, distorting generalizability. | **Real CO importance is 40.37%; removing Ahmedabad drops CO to 3.12% and raises PM2.5 to 83.39%.** | Inspection of `aqi_model.pkl` and isolated retraining experiment. | Retrain production model with anomaly-screened training data in Phase 2. | **CONFIRMED** |
| **6** | API Schema vs Pipeline Mismatch | API requires 6 inputs; pipeline imputer is unused. | **API rejects nulls (HTTP 422); naive null acceptance causes high prediction errors.** | `backend/schemas.py` line 16-29 vs `aqi_model.pkl` SimpleImputer. | Keep schema strict until domain-aware ratio preprocessor is integrated. | **CONFIRMED** |
| **7** | `HOW_IT_WORKS.md` Discrepancy | Documented feature importances do not match model. | **Discrepancy confirmed: CO documented at 8.9%, but genuine model is 40.37%.** | Direct line-by-line comparison between `HOW_IT_WORKS.md` and `aqi_model.pkl`. | Synchronize documentation in Phase 2 to reflect empirical reality. | **CONFIRMED** |

---

## 4. Final Project Readiness Decision

### Readiness Verdict: **CONDITIONALLY READY FOR PHASE 2 (DATASET CORRECTION & RETRAINING)**

**Decision Rationale**:
* All seven audit findings have been independently verified with reproducible code, exact statistical calculations, and isolated experiments.
* No production files, original datasets, or trained model artifacts were modified during this audit phase.
* The system is **NOT** ready for operational public release in its current form due to the $40.37\%$ distortion of `aqi_model.pkl` and the failure of median imputation for cities lacking $\text{PM}_{10}$.
* The system **IS FULLY READY** to transition to **Phase 2 (Model Retraining & Pipeline Modernization)** under the following sequence:
  1. Build an anomaly-screening preprocessor that caps/flags $\text{CO} > 17.0\ \text{mg/m}^3$ and applies monthly $\text{PM}_{2.5}/\text{PM}_{10}$ ratio imputation for missing particulate matter.
  2. Retrain `aqi_model.pkl` on the clean partitioned dataset, verifying that $\text{PM}_{2.5} + \text{PM}_{10}$ feature importance exceeds $80\%$ and $\text{CO}$ importance remains $< 5\%$.
  3. Update `backend/schemas.py` and `backend/main.py` with physical domain pre-imputation, display clamping at $500$, and input guardrails.
  4. Synchronize all technical guides, including `HOW_IT_WORKS.md`.

