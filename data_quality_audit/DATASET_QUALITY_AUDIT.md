# AirSense AI — Dataset Quality & Consistency Audit Report

**Project**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)  
**Location**: `D:\PROJECTS\evs\archive\city_day.csv` (Audit Scope: `G:\PROJECTS\evs`)  
**Audit Phase**: Phase 1 — Comprehensive Dataset Quality, Integrity & Consistency Audit  
**Auditor**: Senior Software Architect, Machine Learning Engineer & Scientific Auditor  
**Date of Audit**: November 3, 2026  
**Status**: Complete — Non-Destructive Empirical Audit (Original Datasets and Production Artifacts Untouched)  

---

## Executive Summary

This audit constitutes an exhaustive, evidence-based investigation into the quality, integrity, and atmospheric plausibility of the primary dataset (`archive/city_day.csv`) utilized by the **AirSense AI** predictive modeling system. 

The audit confirms that the dataset contains **29,531 daily records across 26 major Indian cities**, from which **24,849 valid supervised consecutive-day observation pairs** ($t \to t+1$) can be formed. The calendar pairing and temporal embargo mechanisms implemented in the codebase are mathematically verified and free of date leakage.

However, the audit has identified **four major data-quality vulnerabilities** that directly impair model generalization and feature interpretability:

1. **Extreme Sensor Saturation & Calibration Contamination in Carbon Monoxide ($\text{CO}$)**:
   - In Ahmedabad (monitored solely by station `GJ001: Maninagar, GPCB`), $\text{CO}$ concentrations exhibit a median of **$16.24\ \text{mg/m}^3$** (compared to **$0.86\ \text{mg/m}^3$** across the other 25 cities) and spike to an extreme maximum of **$175.81\ \text{mg/m}^3$** ($65.4\%$ of Ahmedabad observations exceed $10\ \text{mg/m}^3$, and $48.1\%$ exceed the CPCB 8-hour Severe threshold of $17\ \text{mg/m}^3$).
   - Hourly forensic analysis reveals ambient outdoor spikes exceeding **$446\ \text{mg/m}^3$ ($\approx 390\ \text{ppm}$)**, which is physically impossible for ambient outdoor urban air without acute human lethality.
   - This single-station anomaly contaminated the multi-city dataset, leading the Random Forest model to assign an inflated **$40.37\%$ Gini feature importance** to $\text{CO}$.

2. **Severe, Non-Random Particulate Missingness ($\text{PM}_{10}$)**:
   - $\text{PM}_{10}$ is missing in **$37.72\%$** of the raw data ($11,140$ rows).
   - This missingness is highly geographically concentrated: in **Lucknow**, $\text{PM}_{10}$ is **$100.0\%$ missing** ($0$ non-null observations across $2,009$ days); in **Patna**, it is **$89.72\%$ missing**; in **Chennai**, **$84.97\%$ missing**; in **Ahmedabad**, **$79.74\%$ missing**; and in **Gurugram**, **$55.93\%$ missing**.
   - Imputing a static national training median ($101.88\ \mu\text{g/m}^3$) into entire cities completely flattened the predictive variance of $\text{PM}_{10}$, compressing its feature importance to just **$4.26\%$**.

3. **Sensor Zero Truncation**:
   - Carbon Monoxide ($\text{CO}$) contains **$2,328$ exact zero values** ($8.47\%$ of non-null rows). In ambient tropospheric air, true background $\text{CO}$ is typically $0.05$ to $0.15\ \text{mg/m}^3$ ($50$–$150\ \text{ppb}$). These values indicate sensor lower-detection-limit truncation or coarse decimal rounding by CAAQMS loggers.

4. **Un-Capped AQI Extrapolations in Ground Truth**:
   - The Indian NAQI scale is officially defined from $0$ to $500$. However, the raw CPCB ground truth contains **$543$ observations with $\text{AQI} > 500$**, reaching a maximum of **$2,049.0$** in Ahmedabad. These values are produced by linear extrapolation of sub-index equations beyond the Severe breakpoint without capping.

---

## 1. Dataset Provenance and Macro Statistics

### 1.1 Source Attribution & Archive Integrity
- **Primary Source**: Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring Station (CAAQMS) network.
- **Repository Location**: `archive/city_day.csv` (2,574,056 bytes) and compressed in `archive.zip` (76,469,579 bytes).
- **Total Record Count**: **29,531 rows**.
- **Total Columns**: **16 attributes**.
- **Temporal Coverage**: **January 1, 2015 to July 1, 2020** (2,008 calendar days).
- **Geographic Coverage**: **26 unique cities**, supported by **230 monitoring stations** across India.

### 1.2 Macro Column Inventory

| Column Name | Data Type | Physical Entity | Model Role | Non-Null Count | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `City` | String | Urban Agglomeration | Spatial Group Key | 29,531 | 0 | 0.00% |
| `Date` | Datetime | Observation Calendar Date | Temporal Index | 29,531 | 0 | 0.00% |
| `PM2.5` | Float | Fine Particulate ($\le 2.5\ \mu\text{m}$) | Model Feature #1 | 24,933 | 4,598 | 15.57% |
| `PM10` | Float | Coarse Particulate ($\le 10\ \mu\text{m}$) | Model Feature #2 | 18,391 | 11,140 | 37.72% |
| `NO` | Float | Nitric Oxide | Excluded | 25,949 | 3,582 | 12.13% |
| `NO2` | Float | Nitrogen Dioxide | Model Feature #3 | 25,946 | 3,585 | 12.14% |
| `NOx` | Float | Total Nitrogen Oxides | Excluded | 25,346 | 4,185 | 14.17% |
| `NH3` | Float | Ammonia | Excluded | 19,203 | 10,328 | 34.97% |
| `CO` | Float | Carbon Monoxide | Model Feature #4 | 27,472 | 2,059 | 6.97% |
| `SO2` | Float | Sulphur Dioxide | Model Feature #5 | 25,677 | 3,854 | 13.05% |
| `O3` | Float | Ground-Level Ozone | Model Feature #6 | 25,509 | 4,022 | 13.62% |
| `Benzene` | Float | Volatile Aromatic Hydrocarbon | Excluded | 23,908 | 5,623 | 19.04% |
| `Toluene` | Float | Volatile Aromatic Hydrocarbon | Excluded | 21,490 | 8,041 | 27.23% |
| `Xylene` | Float | Volatile Aromatic Hydrocarbon | Excluded | 11,422 | 18,109 | 61.32% |
| `AQI` | Float | Composite Air Quality Index | Shifted Target | 24,850 | 4,681 | 15.85% |
| `AQI_Bucket` | String | Statutory Descriptive Category | Target Label Check | 24,850 | 4,681 | 15.85% |

---

## 2. Exhaustive Missing-Value Analysis

### 2.1 Distribution of Missingness Across Pollutants
Missing values in atmospheric telemetry arise from sensor maintenance cycles, probe calibration flaring, electrical outages, and delayed network commissioning.

```
MISSINGNESS BY CRITERIA POLLUTANT IN city_day.csv (N = 29,531):
========================================================================
PM10     : [########################################] 37.72% (11,140 rows)
AQI      : [################] 15.85% (4,681 rows)
PM2.5    : [###############] 15.57% (4,598 rows)
O3       : [##############] 13.62% (4,022 rows)
SO2      : [#############] 13.05% (3,854 rows)
NO2      : [############] 12.14% (3,585 rows)
CO       : [#######] 6.97% (2,059 rows)
========================================================================
```

### 2.2 Geographic Heterogeneity in Missingness
Missingness is not uniformly distributed across the 26 cities; it reflects regional CAAQMS expansion schedules: `[VERIFIED: CITY_DATA_QUALITY.csv]`

```
+----------------------------------------------------------------------------------------------------+
|                                 CITY-WISE MISSINGNESS HIGHLIGHTS                                   |
+-------------------+-------------+---------------+---------------+---------------+------------------+
| City              | Total Rows  | PM10 Miss %   | PM2.5 Miss %  | O3 Miss %     | AQI Miss %       |
+-------------------+-------------+---------------+---------------+---------------+------------------+
| Lucknow           | 2,009       | 100.00% (ALL) | 5.08%         | 1.34%         | 5.77%            |
| Patna             | 1,858       |  89.72%       | 17.28%        | 14.96%        | 21.47%           |
| Chennai           | 2,009       |  84.97%       | 5.82%         | 2.44%         | 6.22%            |
| Ahmedabad         | 2,009       |  79.74%       | 31.26%        | 34.10%        | 33.60%           |
| Gurugram          | 1,679       |  55.93%       | 9.17%         | 6.91%         | 13.46%           |
| Ernakulam         |   162       |   2.47%       | 2.47%         | 100.00% (ALL) | 5.56%            |
| Jorapokhar        | 1,169       |  23.52%       | 67.58%        | 25.15%        | 34.05%           |
| Mumbai            | 2,009       |  62.02%       | 60.98%        | 60.33%        | 61.42%           |
| Delhi             | 2,009       |   3.83%       | 0.10%         | 4.18%         | 0.50% (Pristine) |
+-------------------+-------------+---------------+---------------+---------------+------------------+
```

#### Key Geographic Observations:
1. **The Lucknow $\text{PM}_{10}$ Void**: Across 5.5 years ($2,009$ calendar days), **not a single $\text{PM}_{10}$ measurement exists** for Lucknow. During model training, all $1,684$ training rows for Lucknow received the national median ($101.88\ \mu\text{g/m}^3$), meaning the model never learned the true coarse particulate dynamics of Uttar Pradesh's capital.
2. **The Ernakulam $\text{O}_3$ Void**: All $162$ records for Ernakulam lack Ozone data ($100\%$ missing).
3. **The Mumbai Temporal Commissioning Lag**: For Mumbai, missingness exceeds $60\%$ across all pollutants except $\text{CO}$ ($1.24\%$ missing) because continuous monitoring for particulates, $\text{NO}_2$, and $\text{SO}_2$ did not begin until mid-2018.

### 2.3 Temporal Evolution of Missingness
Evaluating missingness year-by-year illustrates the substantial maturation of the CPCB monitoring infrastructure: `[VERIFIED]`

```
+------+------------+-------------+-------------+-------------+-------------+-------------+-------------+
| Year | Total Rows | PM2.5 Miss% | PM10 Miss%  | NO2 Miss%   | CO Miss%    | SO2 Miss%   | AQI Miss%   |
+------+------------+-------------+-------------+-------------+-------------+-------------+-------------+
| 2015 | 2,801      | 33.7%       | 81.7%       | 15.9%       | 3.1%        | 17.3%       | 34.8%       |
| 2016 | 3,478      | 24.0%       | 66.7%       | 23.0%       | 11.8%       | 21.6%       | 26.0%       |
| 2017 | 4,689      | 29.8%       | 56.8%       | 27.0%       | 13.0%       | 29.1%       | 31.0%       |
| 2018 | 6,471      | 14.6%       | 34.1%       | 8.9%        | 6.6%        | 11.4%       | 11.5%       |
| 2019 | 7,446      | 4.2%        | 17.4%       | 3.6%        | 4.6%        | 4.4%        | 5.0%        |
| 2020 | 4,646      | 3.6%        | 7.8%        | 5.0%        | 4.0%        | 4.0%        | 4.8%        |
+------+------------+-------------+-------------+-------------+-------------+-------------+-------------+
```
*Crucial Implication*: In 2015, over **$81\%$ of $\text{PM}_{10}$ observations were missing**. The training partition (2015–2019) is heavily burdened by early-year missingness, whereas the test set (2019–2020) has under $8\%$ $\text{PM}_{10}$ missingness.

---

## 3. Outlier and Abnormal-Value Analysis

### 3.1 Summary Statistics and Percentile Distribution
Extracted directly from `DATA_QUALITY_SUMMARY.csv`: `[VERIFIED]`

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      POLLUTANT DISTRIBUTION & OUTLIER AUDIT                                       |
+----------+--------+---------+--------+--------+--------+--------+---------+---------+---------+----------+----------+
| Feature  | Valid N| Mean    | Std    | Min    | 1%     | 50%    | 95%     | 99%     | Max     | IQR      | IQR Outl |
+----------+--------+---------+--------+--------+--------+--------+---------+---------+---------+----------+----------+
| PM2.5    | 24,933 |   67.45 |  64.66 |   0.04 |   7.85 |  48.57 |  193.96 |  311.06 |  949.99 |    51.77 | 1,982    |
| PM10     | 18,391 |  118.13 |  90.61 |   0.01 |  13.22 |  95.68 |  303.34 |  444.61 | 1000.00 |    93.49 | 1,057    |
| NO2      | 25,946 |   28.56 |  24.47 |   0.01 |   1.39 |  21.69 |   74.13 |  117.12 |  362.21 |    25.87 | 1,188    |
| CO       | 27,472 |    2.25 |   6.96 |   0.00 |   0.00 |   0.89 |    8.02 |   34.53 |  175.81 |     0.94 | 2,475    |
| SO2      | 25,677 |   14.53 |  18.13 |   0.01 |   1.31 |   9.16 |   46.21 |  102.68 |  193.86 |     9.55 | 2,578    |
| O3       | 25,509 |   34.49 |  21.69 |   0.01 |   1.44 |  30.84 |   74.14 |  105.32 |  257.73 |    26.71 |   713    |
| AQI      | 24,850 |  166.46 | 140.70 |  13.00 |  33.00 | 118.00 |  407.00 |  661.51 | 2049.00 |   127.00 | 1,358    |
+----------+--------+---------+--------+--------+--------+--------+---------+---------+---------+----------+----------+
```

### 3.2 Four-Tier Taxonomy of Statistical Outliers

```
+--------------------------------------------------------------------------------------------------+
|                                    OUTLIER TAXONOMY MATRIX                                       |
+----------------------+-----------------------------+---------------------------------------------+
| Outlier Class        | Empirical Manifestation      | Atmospheric / Instrumental Plausibility     |
+----------------------+-----------------------------+---------------------------------------------+
| 1. Statistically     | PM2.5 in [300 - 600] ug/m3  | VALID: Winter meteorology & agricultural    |
| Unusual but Valid    | in Delhi / Patna / Lucknow  | stubble-burning inversions in Indo-Gangetic |
+----------------------+-----------------------------+---------------------------------------------+
| 2. Questionable /    | PM10 == 1000.00 ug/m3       | SENSOR CEILING: Instrument saturated at its |
| Sensor Saturation    | (Guwahati, Shillong)        | upper telemetry limit (BAM 1020 / TEOM)     |
+----------------------+-----------------------------+---------------------------------------------+
| 3. Severe Calibration| CO in [50 - 175.81] mg/m3   | INVALID AMBIENT: Physically lethal outdoor  |
| Flaring / Artifact   | in Ahmedabad (Maninagar)    | levels; sensor plume suction / voltage drift|
+----------------------+-----------------------------+---------------------------------------------+
| 4. Ground-Truth      | AQI in [501 - 2049]         | EQUATION ARTIFACT: Uncapped linear sub-index|
| Extrapolation Error  | (Ahmedabad, Amritsar)       | extrapolation beyond CPCB 500 boundary      |
+----------------------+-----------------------------+---------------------------------------------+
```

### 3.3 The Ahmedabad Carbon Monoxide ($\text{CO}$) Anomaly
- **The Finding**: The dataset reports a maximum $\text{CO}$ of **$175.81\ \text{mg/m}^3$** on `2017-10-25` in Ahmedabad.
- **Top 20 Observations**: All 20 highest $\text{CO}$ readings in the entire national dataset originate from **Ahmedabad** ($98.33$ to $175.81\ \text{mg/m}^3$).
- **Station Provenance**: Cross-referencing `archive/stations.csv` and `archive/station_day.csv` confirms that all Ahmedabad readings derive from a **single station: `GJ001: Maninagar, Ahmedabad - GPCB`**.
- **Atmospheric Impossibility**: $175.81\ \text{mg/m}^3 \approx 153\ \text{ppm}$. In `archive/city_hour.csv`, hourly readings on `2018-02-19` reach **$446.41\ \text{mg/m}^3$ ($\approx 390\ \text{ppm}$)**. Ambient outdoor exposure at $390\ \text{ppm}$ for 2 hours induces incapacitating carboxyhemoglobin saturation ($>35\%$), causing convulsions and death. Ambient open-air urban concentrations never exceed $10$–$20\ \text{mg/m}^3$ even in congested traffic jams.
- **Statistical Distortion**: In Ahmedabad, the Pearson correlation between $\text{CO}$ and ground-truth $\text{AQI}$ is **$r = 0.8415$** (compared to $r = 0.2798$ in Delhi). This single flawed station distorted the Random Forest model into assigning $\text{CO}$ a massive **$40.37\%$ feature importance**.

### 3.4 Particulate Saturation Ceilings
- In Guwahati on `2019-05-04`, $\text{PM}_{10}$ is recorded as exactly **$1,000.00\ \mu\text{g/m}^3$**.
- In Shillong on `2019-09-02` and `2019-09-03`, $\text{PM}_{10}$ is recorded as **$985.00\ \mu\text{g/m}^3$**.
- An exact measurement of $1,000.00\ \mu\text{g/m}^3$ represents the software truncation limit (upper dynamic range) of Beta Attenuation Monitors rather than an exact physical mass density.

---

## 4. Dataset Consistency & Calendar Continuity Audit

### 4.1 Duplicate Records & Conflicting Observations
- **Test Executed**: `df.duplicated(subset=['City', 'Date'])`.
- **Result**: **0 duplicate pairs**.
- **Integrity**: Every city has at most one observation per calendar day. No conflicting records exist on identical dates. `[VERIFIED]`

### 4.2 Calendar Day Discontinuities per City
While there are no duplicate dates, several cities have severe interruptions where monitoring was suspended for months: `[VERIFIED: CITY_DATA_QUALITY.csv]`
- **Ahmedabad, Bengaluru, Chennai, Delhi, Lucknow, Mumbai**: 2,009 rows covering 2,009 calendar days ($0$ missing calendar days; $100\%$ temporal continuity).
- **Hyderabad**: 2,006 rows ($3$ missing calendar days).
- **Patna**: 1,858 rows ($0$ missing calendar days between 2015-06-01 and 2020-07-01).
- **Late-Commissioned Cities**: Aizawl ($113$ days), Ernakulam ($162$ days), Kochi ($162$ days), Bhopal ($289$ days), Chandigarh ($304$ days), Shillong ($310$ days).

### 4.3 Ground-Truth AQI Bucket Alignment
Auditing statutory CPCB category intervals against `AQI_Bucket`: `[VERIFIED]`

```
+---------------+------------------+------------------+------------------+----------------------+
| AQI Bucket    | Expected Range   | Actual Min AQI   | Actual Max AQI   | Integrity Assessment |
+---------------+------------------+------------------+------------------+----------------------+
| Good          | 0 to 50          | 13.0             | 50.0             | Exact Conformity     |
| Satisfactory  | 51 to 100        | 51.0             | 100.0            | Exact Conformity     |
| Moderate      | 101 to 200       | 101.0            | 200.0            | Exact Conformity     |
| Poor          | 201 to 300       | 201.0            | 300.0            | Exact Conformity     |
| Very Poor     | 301 to 400       | 301.0            | 400.0            | Exact Conformity     |
| Severe        | 401 to 500       | 401.0            | 2,049.0          | UN-CAPPED EXTENSION  |
+---------------+------------------+------------------+------------------+----------------------+
```
*Conclusion*: All values $\le 500$ conform strictly to CPCB statutory thresholds. Values exceeding $500$ are categorized uniformly as `Severe`.

### 4.4 Verification of Consecutive Calendar-Day Pairing Logic
The consecutive calendar pairing logic in `train_aqi_model.py` and `model_validation/train_candidate_model.py` was subjected to programmatic assertions: `[VERIFIED: verify_pairing.py]`
1. **Intra-City Pairing**: Confirmed that `Next_City == City` for all $24,849$ pairs. No pairing crosses city boundaries.
2. **Exact 1-Day Difference**: Confirmed that `Next_Date - Date == 1 calendar day` for all $24,849$ pairs. Station gaps ($>1$ day) are dropped.
3. **Target Alignment**: Confirmed that `Target_AQI` strictly corresponds to tomorrow's observation ($t+1$).
4. **Temporal Embargo Verification**: In the candidate model, `Train_Max_Date` is `2019-12-04` (max target `2019-12-05`). The embargo date `2019-12-05` ($22$ rows) is dropped. `Test_Min_Date` is `2019-12-06`. **Zero calendar dates or targets leak into the test partition.**

---

## 5. Model-Input & Preprocessing Consistency Audit

```
+----------------------------------------------------------------------------------------------------------+
|                                    MODEL INPUT & SERVING CONTRACT MATRIX                                 |
+--------------------+-------------------------+-------------------------+---------------------------------+
| Dimension          | Training Pipeline       | Candidate Pipeline      | Production FastAPI Serving API  |
+--------------------+-------------------------+-------------------------+---------------------------------+
| Model Artifact     | aqi_model.pkl           | aqi_model_candidate.pkl | aqi_model.pkl                   |
| Feature Count      | Exactly 6 features      | 6 features + City/Date  | Exactly 6 features              |
| Feature Sequence   | PM2.5, PM10, NO2, CO,   | PM2.5, PM10, NO2, CO,   | PM2.5, PM10, NO2, CO,           |
|                    | SO2, O3                 | SO2, O3                 | SO2, O3                         |
| Preprocessing      | SimpleImputer(median)   | CityMonthAwareImputer   | SimpleImputer (in pipeline)     |
| Missing Handling   | Fills training median   | Fills City-Month median | REJECTS NaN / None with HTTP 422|
| Negative Inputs    | None in training set    | None in training set    | REJECTS negative with HTTP 422  |
| Maxima Guardrail   | Passive (in metadata)   | Passive (in metadata)   | Emits warning if input > maxima |
+--------------------+-------------------------+-------------------------+---------------------------------+
```

### Critical Serving Disconnect:
The production model artifact `aqi_model.pkl` contains `SimpleImputer` as Step 1. However, the serving API schema `backend/schemas.py` defines all 6 inputs as mandatory floats and rejects `NaN` with HTTP 422. Thus, the trained imputer is unreachable at runtime. If an IoT station drops $\text{PM}_{10}$ telemetry, the API aborts rather than falling back to the imputer.

---

## 6. Comprehensive Impact Assessment

```
+---------------------------------------------------------------------------------------------------------------+
|                                      DATA QUALITY RISK & IMPACT LEDGER                                        |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| ID  | Identified Issue              | Severity | Subsystem Affected                           | Safety/Model  |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| Q01 | Extreme CO Calibration Artifact| CRITICAL | Feature Importance & Regional Transfer       | Distorts ML   |
|     | (Ahmedabad Maninagar station) |          | (Distorts CO to 40.4% importance nationwide) | interpretation|
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| Q02 | 100% Missing PM10 in Lucknow  | HIGH     | Regional Accuracy & Feature Attribution      | Underpredicts |
|     | & >80% Missing in Patna/Chenn |          | (Blinds model to coarse dust dynamics)       | winter dust   |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| Q03 | Serving Schema Rejects NaNs   | MEDIUM   | Backend Inference Resilience                 | Breaks IoT    |
|     | (Dead code imputer in pipeline|          | (Fails on missing sensor streams)            | sensor inputs |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| Q04 | Ground-Truth AQI > 500        | MEDIUM   | Evaluation Metric Precision                  | Inflates RMSE |
|     | Extrapolation Spikes          |          | (Uncapped sub-indices skew squared loss)     | in Ahmedabad  |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
| Q05 | 2,328 Truncated Zero Values   | LOW      | Low-Concentration Sensitivity                | Negligible on |
|     | for Carbon Monoxide           |          | (Lacks resolution below 0.1 mg/m3)           | composite AQI |
+-----+-------------------------------+----------+----------------------------------------------+---------------+
```

---

## 7. Audit Conclusion & Suitability Verdict

```
========================================================================================
AUDIT SUITABILITY VERDICT: SUITABLE FOR CONTROLLED ENVIRONMENTAL FORECASTING EVALUATION
                   REQUIRES RIGOROUS SCREENING & IMPUTATION PRIOR TO DEPLOYMENT
========================================================================================
1. The dataset is MATHEMATICALLY SOUND for evaluating the existing research prototype.
   The pairings, chronological splits, and persistence baselines are authentic.
2. However, for a publication-grade or operational forecasting engine, the Ahmedabad 
   CO station anomaly (GJ001) MUST be winsorized or capped, Lucknow PM10 missingness 
   must be addressed via multivariate ratio imputation, and meteorological variables 
   must be integrated to resolve coastal failures.
========================================================================================
```

---
*Report certified by Senior Software Architect & Technical Auditor on November 3, 2026.*  
*AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)*

