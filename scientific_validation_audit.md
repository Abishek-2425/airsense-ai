# Scientific Validation Audit Report

**Project Title**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)  
**Location**: `G:\PROJECTS\evs`  
**Audit Date**: October 28, 2026  
**Auditor**: Automated Scientific Verification Agent  
**Audit Scope**: Training Pipeline, Serialized Model Artifact (`aqi_model.pkl`), Model Metadata (`model_metadata.json`), Dataset (`archive/city_day.csv`), FastAPI Backend (`backend/`), and React Frontend (`frontend/`).  
**Audit Status**: **Complete — Non-Destructive Inspection Only (No Code or Model Modified)**

---

## Executive Summary

A comprehensive, non-destructive scientific audit was conducted across seven key dimensions of the AirSense AI system. The evaluation confirmed that the numerical metrics reported in the project metadata ($R^2 = 0.8367$, $\text{RMSE} = 36.46$, $\text{MAE} = 20.10$ on 4,902 matched samples) are **mathematically authentic, exact, and reproducible from the serialized artifacts**. Furthermore, the user-facing application adheres to high standards of scientific transparency: pollutant units are correctly designated ($\text{CO}$ in $\text{mg/m}^3$, others in $\mu\text{g/m}^3$), the model outputs are clearly labeled as statistical regression estimates rather than official CPCB calculations or live sensor readings, and explicit statutory and medical disclaimers are prominently displayed.

However, the audit revealed **three critical scientific vulnerabilities** that must be documented for academic rigor:
1. **Temporal Leakage at Split Boundary**: The 80/20 chronological split occurred across a shared calendar date (`2019-12-05`), causing 20 training records to predict next-day AQI targets for `2019-12-06`—a date inside the test partition.
2. **COVID-19 Lockdown Distortion & Concept Drift**: The second half of the test partition (48.9% of samples, March 24 – June 30, 2020) occurred during the nationwide COVID-19 lockdown. Under these conditions, the persistence baseline **outperformed** the Random Forest model ($R^2 = 0.7307$ vs. $0.6336$), revealing significant concept drift.
3. **Severe Geographic Disparities & Zero-Shot Test Cities**: Three cities in the test set (**Aizawl**, **Ernakulam**, **Kochi**) had **zero** training samples. Furthermore, while the model performs well in high-variance northern cities (Delhi $R^2 = 0.8985$), it fails in southern coastal cities (Chennai $R^2 = -0.4509$, where predictions are worse than the historical mean).

---

## Audit Matrix by Dimension

| Dimension | Status | Key Findings | Severity |
| :--- | :---: | :--- | :---: |
| **1. Temporal Data Leakage** | **Deficiency Identified** | Shared boundary date (`2019-12-05`); train targets extend into test window (`2019-12-06`). | **Medium** |
| **2. Evaluation Correctness** | **Verified / Nuanced** | Reported metrics are exact; however, aggregate $R^2$ conceals degradation during lockdown. | **High** |
| **3. Feature & Target Alignment** | **Deficiency Identified** | Imputer uses single global median; 34% of training $\text{PM}_{10}$ is imputed; API rejects NaNs. | **High** |
| **4. Pollutant Units** | **Verified Correct** | $\text{CO}$ correctly labeled as $\text{mg/m}^3$; $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{O}_3$ as $\mu\text{g/m}^3$. Max CO ($175.81$) is an outlier. | **Low** |
| **5. AQI Categories** | **Verified Correct** | Category thresholds conform to CPCB bands; regression vs. sub-index distinction clearly stated. | **Informational** |
| **6. Model Generalisation** | **Deficiency Identified** | 3 zero-shot cities; negative $R^2$ in Chennai; test set omits peak post-monsoon burning season. | **High** |
| **7. Prediction Claims** | **Verified Correct** | Code and UI consistently frame output as an empirical estimate with clear academic disclaimers. | **Informational** |

---

## Detailed Findings

```
Severity Classification:
- [CRITICAL]: Invalidates published core claims or causes severe mathematical distortion.
- [HIGH]: Major methodological limitation, hidden domain failure, or systematic evaluation bias.
- [MEDIUM]: Procedural leakage or data-handling issue with measurable but bounded impact.
- [LOW]: Data cleanliness, boundary edge cases, or documentation inconsistencies.
- [VERIFIED]: Confirmed correct, reproducible, and compliant with scientific standards.
```

---

### Area 1: Temporal Data Leakage & Split Boundary Integrity

#### Finding 1.1: Shared Boundary Date (`2019-12-05`) and Target Leakage into Test Period
- **Severity**: `[MEDIUM]`
- **Status**: **Verified Issue**
- **Evidence**:
  The chronological split in [train_aqi_model.py](file:///G:/PROJECTS/evs/train_aqi_model.py#L91-L94) is defined by row indexing:
  $$\text{split\_index} = \text{int}(N \times 0.8) = \text{int}(24,849 \times 0.8) = 19,879$$
  Because `valid_paired_df` contains data from 26 cities sorted by date, exactly **22 records exist on calendar date `2019-12-05`**:
  - **20 records** (Amaravati, Bengaluru, Amritsar, Brajrajnagar, Hyderabad, Kolkata, Talcher, Chandigarh, Lucknow, Patna, Thiruvananthapuram, Chennai, Mumbai, Bhopal, Guwahati, Visakhapatnam, Ahmedabad, Coimbatore, Gurugram, Delhi) are assigned to the **training partition** (rows 19,859 to 19,878).
  - **2 records** (Jaipur, Jorapokhar) on the exact same calendar date are assigned to the **test partition** (rows 19,879 and 19,880).
  - For the 20 training records on `2019-12-05`, their ground-truth target (`Target_AQI`) corresponds to observation date **`2019-12-06`**, which falls inside the chronological window of the test set (`2019-12-05` to `2020-06-30`).
- **Impact**:
  1. The train and test sets do not possess disjoint date sets; `2019-12-05` is shared.
  2. Training targets extend into calendar dates evaluated in the test partition.
  3. Regional atmospheric systems (e.g. western disturbances and winter inversions across the Indo-Gangetic Plain) correlate air quality across northern cities on identical dates.
- **Empirical Quantification**:
  Evaluating `aqi_model.pkl` on a strictly date-isolated test set (introducing a 1-day embargo buffer: Train $\le \text{2019-12-04}$, Test $\ge \text{2019-12-06}$, $N=4,968$):
  - Model $R^2$ on clean test: **0.830353** (vs. reported **0.830380**, $\Delta = -0.000027$)
  - Model matched $R^2$: **0.836668** (vs. reported **0.836694**, $\Delta = -0.000026$)
  - Model matched RMSE: **36.4667** (vs. reported **36.4599**, $\Delta = +0.0068$)
  *Conclusion*: While the numerical distortion is negligible due to the small number of boundary rows (22 out of 24,849), the split technically violates temporal separation.
- **Recommended Fix**:
  Enforce date-based cutoffs rather than row-based indexing, with an explicit 1-day temporal buffer:
  ```python
  train_df = df[df['Date'] <= pd.to_datetime('2019-12-04')]
  test_df = df[df['Date'] >= pd.to_datetime('2019-12-06')]
  ```

#### Finding 1.2: City-Level Date Continuity and Pairing
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Positive**
- **Evidence**:
  The consecutive calendar day mask in [train_aqi_model.py](file:///G:/PROJECTS/evs/train_aqi_model.py#L60):
  ```python
  consecutive_mask = (raw_df['Next_Date'] - raw_df['Date']) == pd.Timedelta(days=1)
  ```
  strictly validates that observations are paired only when the consecutive day difference is exactly 1 calendar day. Multi-day station gaps (where monitoring was interrupted for $>1$ day) are eliminated.
- **Impact**: No false pairings across temporal gaps exist in the dataset.

---

### Area 2: Evaluation Correctness & Baseline Comparison

#### Finding 2.1: Mathematical Exactness of Reported Evaluation Metrics
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Positive**
- **Evidence**:
  Recomputation of scikit-learn evaluation metrics on `test_df` using the serialized `aqi_model.pkl`:
  - **Random Forest (Matched Test, $N=4,902$)**:
    - Computed: $R^2 = 0.836694$, $\text{RMSE} = 36.459891$, $\text{MAE} = 20.097779$
    - Metadata: $R^2 = 0.836694$, $\text{RMSE} = 36.459891$, $\text{MAE} = 20.097779$
    - *Discrepancy*: $0.000000$ (Exact match)
  - **Persistence Baseline ($\hat{y}_{t+1} = y_t$, Matched Test, $N=4,902$)**:
    - Computed: $R^2 = 0.772358$, $\text{RMSE} = 43.046835$, $\text{MAE} = 20.905141$
    - Metadata: $R^2 = 0.772358$, $\text{RMSE} = 43.046835$, $\text{MAE} = 20.905141$
    - *Discrepancy*: $0.000000$ (Exact match)
  - **Random Forest (All Test, $N=4,970$)**:
    - Computed: $R^2 = 0.830380$, $\text{RMSE} = 37.061493$, $\text{MAE} = 20.310667$
    - Metadata: $R^2 = 0.830380$, $\text{RMSE} = 37.061493$, $\text{MAE} = 20.310667$
    - *Discrepancy*: $0.000000$ (Exact match)
- **Impact**: All figures cited in the backend API, frontend dashboard, and project metadata are authentic and directly derived from the serialized model.

#### Finding 2.2: COVID-19 Lockdown Concept Drift Reverses Baseline Dominance
- **Severity**: `[HIGH]`
- **Status**: **Verified Issue**
- **Evidence**:
  India initiated Phase 1 of the national COVID-19 lockdown on **March 24, 2020**. Decomposing the matched test set into pre-lockdown and in-lockdown periods reveals:

  | Evaluation Period | Date Range | Samples | Model | $R^2$ Score | RMSE | MAE | Mean Actual AQI |
  | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
  | **Pre-Lockdown** | Dec 5, 2019 – Mar 23, 2020 | 2,542 (51.1%) | **Random Forest**<br>Persistence Baseline | **0.8190**<br>0.7251 | **45.68**<br>56.44 | **24.89**<br>29.69 | 158.04 |
  | **In-Lockdown** | Mar 24, 2020 – Jun 30, 2020 | 2,428 (48.9%) | Random Forest<br>**Persistence Baseline** | 0.6336<br>**0.7307** | 25.04<br>**21.37** | 15.51<br>**11.70** | 84.11 |

- **Impact**:
  1. During normal pre-lockdown conditions, the Random Forest model substantially outperformed the persistence baseline ($\Delta R^2 = +0.0939$, RMSE lowered by $10.76$ AQI points).
  2. During the lockdown, **the persistence baseline beat the Random Forest model across every single evaluation metric** ($\Delta R^2 = -0.0971$, RMSE higher by $3.67$ AQI points, MAE higher by $3.81$ AQI points).
  3. Under strict lockdown, vehicular and industrial emissions dropped abruptly, causing urban AQI to stabilize at unprecedented low levels (mean AQI halved from $158.04$ to $84.11$). Tomorrow's AQI became strongly auto-correlated with today's AQI. The Random Forest model, trained on 5 years of pre-lockdown emission synergies, experienced severe **concept drift**.
  4. Reporting an aggregate test $R^2 = 0.8367$ obscures the reality that the model underperformed a naive persistence baseline over nearly half the evaluation period.
- **Recommended Fix**:
  Disclose the sub-period evaluation breakdown in the project report and documentation. Highlight this as an academic case study in machine learning degradation during sudden anthropogenic emission interventions.

---

### Area 3: Feature and Target Alignment & Missing-Value Imputation

#### Finding 3.1: High $\text{PM}_{10}$ Missingness and Unconditional Global Median Imputation
- **Severity**: `[HIGH]`
- **Status**: **Verified Issue**
- **Evidence**:
  Audit of pollutant missingness across the dataset partitions:
  - Raw `city_day.csv`: $\text{PM}_{10}$ is null in **11,140 rows (37.72%)**.
  - Chronological Training Set: $\text{PM}_{10}$ is null in **6,767 rows (34.04%)**.
  - Chronological Test Set: $\text{PM}_{10}$ is null in **293 rows (5.90%)**.
  
  The training pipeline fits a single global `SimpleImputer(strategy='median')` across all training rows:
  $$\text{Imputed } \text{PM}_{10} = 101.88\ \mu\text{g/m}^3 \quad (\text{applied uniformly across all cities and seasons})$$
- **Impact**:
  1. Over one-third of the training data lacked measured $\text{PM}_{10}$. Imputing a static national median of $101.88\ \mu\text{g/m}^3$ injects severe distortion: for pristine or coastal cities (e.g. Aizawl, Thiruvananthapuram, where typical $\text{PM}_{10} < 40$), the imputed value creates an artificial pollution signal; for polluted winter cities (e.g. Delhi, where $\text{PM}_{10} > 350$), it drastically underestimates coarse particulate loading.
  2. This heavy imputation dilutes the explanatory signal of $\text{PM}_{10}$, explaining why its Gini feature importance is only **4.26%**, whereas $\text{PM}_{2.5}$ captures **46.47%**.
- **Recommended Fix**:
  In future model iterations, replace global median imputation with:
  - City- and month-stratified medians.
  - Ratio-based estimation ($\text{PM}_{10} \approx \text{PM}_{2.5} / \text{ratio}$ based on regional historical co-located observations).
  - Iterative multivariate imputation (`IterativeImputer` / MissForest).

#### Finding 3.2: Serving Pipeline Disconnect (Imputer Inaccessible at API)
- **Severity**: `[MEDIUM]`
- **Status**: **Verified Issue**
- **Evidence**:
  The serialized pipeline in `aqi_model.pkl` contains `SimpleImputer(strategy='median')` as step 1. However, the production API schema in [backend/schemas.py](file:///G:/PROJECTS/evs/backend/schemas.py#L16-L30) marks all six pollutant inputs as mandatory floats:
  ```python
  pm25: float = Field(..., description="PM2.5 concentration in ug/m3")
  pm10: float = Field(..., description="PM10 concentration in ug/m3")
  # validator rejects NaN and Infinity with HTTP 422
  ```
- **Impact**:
  If an end-user or IoT sensor station lacks $\text{PM}_{10}$ or $\text{O}_3$ telemetry, the API aborts with a validation error (422) instead of leveraging the pipeline's trained imputer. The imputer is active during training, but rendered dead code during inference.
- **Recommended Fix**:
  Allow optional inputs (`Optional[float] = None`) in `PredictRequest` if missing-sensor fallback is desired, or formally document that the serving contract requires complete 6-pollutant telemetry to prevent uncontrolled imputer bias.

---

### Area 4: Pollutant Units & Physical Plausibility

#### Finding 4.1: Source Column Units and Accounting
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Correct**
- **Evidence**:
  Cross-referencing CPCB statutory reporting standards (CPCB NAQI Guidelines 2014) and Kaggle dataset documentation against code implementations:
  - $\text{PM}_{2.5}$: $\mu\text{g/m}^3$ (24-hour average)
  - $\text{PM}_{10}$: $\mu\text{g/m}^3$ (24-hour average)
  - $\text{NO}_2$: $\mu\text{g/m}^3$ (24-hour average)
  - $\text{CO}$: $\mathbf{\text{mg/m}^3}$ (8-hour / daily average in CPCB standards)
  - $\text{SO}_2$: $\mu\text{g/m}^3$ (24-hour average)
  - $\text{O}_3$: $\mu\text{g/m}^3$ (8-hour / daily average)

  Inspecting [backend/schemas.py](file:///G:/PROJECTS/evs/backend/schemas.py#L16-L21), [frontend/src/pages/PredictPage.tsx](file:///G:/PROJECTS/evs/frontend/src/pages/PredictPage.tsx#L96), and [frontend/src/pages/ModelMethodologyPage.tsx](file:///G:/PROJECTS/evs/frontend/src/pages/ModelMethodologyPage.tsx#L490):
  $\text{CO}$ is consistently specified in $\text{mg/m}^3$, and the remaining five pollutants are specified in $\mu\text{g/m}^3$. No incorrect scaling factors or silent unit conversions were found.
- **Impact**: The system correctly handles pollutant units across the full stack.

#### Finding 4.2: Severe CO Outliers in Raw CPCB Data
- **Severity**: `[LOW]`
- **Status**: **Potential Concern**
- **Evidence**:
  Statistical distribution of $\text{CO}$ in raw `city_day.csv` ($N = 27,472$ non-null):
  - 50th percentile (median): $0.89\ \text{mg/m}^3$
  - 90th percentile: $2.63\ \text{mg/m}^3$
  - 99th percentile: $34.53\ \text{mg/m}^3$
  - 99.9th percentile: $90.10\ \text{mg/m}^3$
  - **Maximum observed**: **$175.81\ \text{mg/m}^3$** (recorded in Gurugram / Delhi)
- **Impact**:
  An ambient concentration of $175.81\ \text{mg/m}^3$ corresponds to $\approx 153\ \text{ppm}$. Sustained exposure at this level induces acute carboxyhemoglobin saturation ($>25\%$) causing unconsciousness or fatality. In ambient air quality datasets, such readings reflect probe calibration flaring or sensor malfunction rather than realistic ambient breathing air. Because `training_maxima` records $175.81\ \text{mg/m}^3$ as the upper threshold, inputs up to this extreme value are accepted without range warnings.
- **Recommended Fix**:
  Add an advisory warning for ambient $\text{CO} > 17.0\ \text{mg/m}^3$ (the CPCB 8-hour Severe threshold), alerting the user that the reading exceeds normal ambient atmospheric limits.

---

### Area 5: AQI Category Standards & Calculation Distinction

#### Finding 5.1: Category Boundary Alignment
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Correct**
- **Evidence**:
  The prototype categorization in [backend/decision_support.py](file:///G:/PROJECTS/evs/backend/decision_support.py#L15-L22) implements the exact numerical intervals of the Indian National Air Quality Index (NAQI):
  - $0 \le \text{AQI} \le 50$: **Good**
  - $51 \le \text{AQI} \le 100$: **Satisfactory**
  - $101 \le \text{AQI} \le 200$: **Moderate**
  - $201 \le \text{AQI} \le 300$: **Poor**
  - $301 \le \text{AQI} \le 400$: **Very Poor**
  - $401 \le \text{AQI} \le 500$: **Severe**

  Boundary transition arithmetic uses standard half-up integer rounding (`math.floor(val + 0.5)`), ensuring testable and deterministic transitions (e.g., $50.49 \to \text{Good}$, $50.50 \to \text{Satisfactory}$). Negative values are clamped to $0.0$ (`Good`) with an explanatory warning; predictions $>500$ are retained as numeric values with a `Severe` label and an out-of-range alert.
- **Impact**: Boundary classification is robust, reproducible, and standardized.

#### Finding 5.2: Framing Distinction Between Regression and CPCB Calculation
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Correct**
- **Evidence**:
  The codebase clearly emphasizes the mathematical distinction between this prototype and regulatory CPCB computation:
  - In [backend/decision_support.py](file:///G:/PROJECTS/evs/backend/decision_support.py#L4-L8):
    > *"The model outputs an estimated AQI by regression. The specified AQI bands are used only to label that estimate for this prototype; this is not a direct CPCB AQI calculation from pollutant sub-indices."*
  - In [frontend/src/pages/ModelMethodologyPage.tsx](file:///G:/PROJECTS/evs/frontend/src/pages/ModelMethodologyPage.tsx#L642-L656):
    > *"Crucial Distinction: Regression Estimate vs. CPCB AQI Calculation... CPCB calculates NAQI using piecewise linear breakpoints where overall AQI is the maximum sub-index... AirSense AI is a predictive forecasting system that trains an empirical ML regression model to forecast next-calendar-day AQI based on synergistic pollutant patterns."*
- **Impact**: The project maintains complete academic integrity and avoids misrepresenting its predictions as statutory CPCB determinations.

---

### Area 6: Model Generalisation & Geographic/Seasonal Limitations

#### Finding 6.1: Zero-Shot Cities in Test Partition
- **Severity**: `[HIGH]`
- **Status**: **Verified Issue**
- **Evidence**:
  Analysis of city distributions across the chronological split:
  - Total cities in Training: **23 cities**
  - Total cities in Test: **26 cities**
  - **Three cities present in the test set have exactly ZERO training samples**:
    1. **Aizawl**: $0$ train samples, $111$ test samples ($100\%$ out-of-sample)
    2. **Ernakulam**: $0$ train samples, $153$ test samples ($100\%$ out-of-sample)
    3. **Kochi**: $0$ train samples, $158$ test samples ($100\%$ out-of-sample)
  - Additionally, three cities have $<100$ training samples: Bhopal ($80$ train, $198$ test), Chandigarh ($92$ train, $207$ test), Shillong ($63$ train, $142$ test).
- **Impact**:
  Monitoring stations in Aizawl, Ernakulam, and Kochi were commissioned late in the CPCB program (late 2019 / 2020). Consequently, the model was evaluated on cities for which it possessed no historical baseline. Because the model architecture does not take city identity or spatial coordinates as features, it blindly transfers northern/inland pollutant relationships to northeastern hilly (Aizawl) and southwestern coastal (Kochi/Ernakulam) microclimates.
- **Recommended Fix**:
  Document that the evaluation partition includes zero-shot geographic transfer. Future models should include spatial features (latitude, longitude, elevation, coastal distance) to support generalized spatial transfer.

#### Finding 6.2: City-Level Model Failure (Negative $R^2$ in Chennai, Massive Error in Ahmedabad)
- **Severity**: `[HIGH]`
- **Status**: **Verified Issue**
- **Evidence**:
  Evaluating `aqi_model.pkl` on individual city subsets in the test partition:

  | City | Test Samples | $R^2$ Score | RMSE | MAE | Mean Actual AQI | Primary Issue |
  | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
  | **Delhi** | 208 | **+0.8985** | 34.16 | 25.18 | 203.05 | High variance, excellent fit |
  | **Kolkata** | 208 | **+0.8873** | 26.94 | 19.44 | 128.23 | Strong predictive accuracy |
  | **Hyderabad** | 208 | **+0.8808** | 10.91 | 8.58 | 84.56 | Low error, stable baseline |
  | **Visakhapatnam** | 202 | **+0.6664** | 22.94 | 17.47 | 88.51 | Moderate predictive utility |
  | **Ahmedabad** | 201 | **+0.6323** | **132.14** | **76.49** | 250.99 | **Extreme prediction errors ($\text{MAE} > 76$)** |
  | **Bengaluru** | 208 | **+0.5860** | 11.94 | 9.35 | 79.98 | Low variance compresses $R^2$ |
  | **Chennai** | 208 | **-0.4509** | **27.59** | **18.28** | 80.46 | **Model is worse than mean baseline ($R^2 < 0$)** |

- **Impact**:
  1. In **Chennai**, the Random Forest regression model produces a **negative $R^2$ ($-0.4509$)**. Predicting Chennai's next-day AQI using the historical mean of Chennai yields lower squared error than the trained machine learning model.
  2. In **Ahmedabad**, although $R^2$ is $0.6323$, the absolute error is catastrophic: an average error of **76.49 AQI points** and RMSE of **132.14**, which can swing a prediction across multiple health bands (e.g. predicting Moderate when actual is Very Poor).
  3. The headline metric of $R^2 = 0.8367$ is overwhelmingly driven by the large dynamic range of northern inland cities (Delhi, Kolkata, Lucknow), masking severe model failure in coastal cities where sea-breeze circulation decouples simple daily pollutant persistence.
- **Recommended Fix**:
  Publish the disaggregated city-by-city performance metrics table. Add localized warning notices in the UI when evaluating coastal stations like Chennai.

#### Finding 6.3: Seasonal Bias (Complete Absence of Monsoon and Post-Monsoon Burning Season)
- **Severity**: `[MEDIUM]`
- **Status**: **Verified Issue**
- **Evidence**:
  The chronological test set spans December 5, 2019 to June 30, 2020.
- **Impact**:
  The test set covers winter and pre-monsoon summer, but **completely omits July through November**. In India, October–November is the annual peak of hazardous air pollution (agricultural stubble burning in Punjab/Haryana, post-monsoon meteorological stagnation, and Diwali celebrations). The reported test metrics provide **zero empirical evidence** regarding how the model performs during India's most hazardous annual air pollution crisis.
- **Recommended Fix**:
  Acknowledge the seasonal truncation in the methodology documentation as an explicit limitation of the 2015–2020 dataset cut-off.

---

### Area 7: Prediction Claims & Regulatory Framing

#### Finding 7.1: Truth in Advertising Across Stack
- **Severity**: `[VERIFIED]`
- **Status**: **Verified Correct**
- **Evidence**:
  Comprehensive review of frontend labels, backend responses, and documentation:
  - Header: *"AI-Powered Air Quality Prediction & Decision Support System"*
  - Result view: *"Estimated Next-Day AQI"* (not "Official AQI" or "Measured AQI")
  - Model tag: *"RandomForestRegressor"*
  - Disclaimers on UI and API:
    - *"Academic research prototype. Continuous regression predictions and prototype category bands are for educational decision-support purposes only."*
    - *"Predictions are statistical approximations based on historical data patterns and should not be used as official statutory certifications, regulatory compliance audits, or clinical medical advice."*
    - Benchmark sample note: *"Historical reference test sample from model evaluation. Not live sensor readings."*
- **Impact**: The project exhibits complete ethical transparency and does not overstate its technological readiness.

---

## Prioritized Action Plan & Recommended Fixes

| Priority | Recommended Action | Implementation Scope | Complexity |
| :---: | :--- | :--- | :---: |
| **1** | **Disclose City-by-City Performance Table**: Add the disaggregated performance table (Delhi $R^2=0.90$ vs. Chennai $R^2=-0.45$) to `ModelMethodologyPage.tsx` and project reports to prevent misleading examiners. | Frontend / Documentation | Low |
| **2** | **Document COVID-19 Lockdown Impact**: Add an analytical note detailing how the lockdown artificially flattened AQI and allowed persistence to outperform Random Forest in March–June 2020. | Documentation / Metadata | Low |
| **3** | **Acknowledge Zero-Shot Cities & Seasonal Gaps**: Formally state that Aizawl, Ernakulam, and Kochi were unrepresented in training, and that July–November was not tested. | Methodology Section | Low |
| **4** | **Enforce Strict Date-Based Train/Test Split**: In the next pipeline refactor, eliminate row indexing at the boundary; split by date with a 1-day buffer (`Train <= 2019-12-04`, `Test >= 2019-12-06`). | Training Script | Low |
| **5** | **Refine Missing Value Imputation**: Transition from global median ($101.88$ for $\text{PM}_{10}$) to city-stratified or $\text{PM}_{2.5}$-correlated imputation. | Future Training | Medium |
| **6** | **Add Ambient CO Plausibility Alert**: Emit an advisory warning if $\text{CO} > 17.0\ \text{mg/m}^3$ indicating that the input exceeds plausible ambient breathing limits. | Backend Validation | Low |

---

## Conclusion

The AirSense AI system is technically sound, reproducible, and transparently presented. The reported aggregate metrics are mathematically authentic. However, this scientific audit has uncovered that aggregate metrics mask severe geographical divergence (Chennai $R^2 = -0.45$), concept drift during COVID-19 lockdowns, global imputation bias on $\text{PM}_{10}$, and a technical 1-day temporal split boundary overlap. Integrating the disclosures outlined in this audit report will elevate the project from an undergraduate prototype to an academically defensible, publication-grade scientific inquiry.
