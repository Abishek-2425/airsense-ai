# AirSense AI — Prioritized Audit Recommendations & Remediation Roadmap

**Project**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)  
**Location**: `D:\PROJECTS\evs` (Audit Scope: `G:\PROJECTS\evs`)  
**Audit Phase**: Phase 1 — Dataset Quality, Integrity & Consistency Audit  
**Auditor**: Senior Software Architect, Machine Learning Engineer & Scientific Auditor  
**Date of Audit**: November 3, 2026  
**Status**: Strategic & Technical Engineering Roadmap (Non-Destructive Guidance)  

---

## Executive Overview

Based on the empirical findings documented in `DATASET_QUALITY_AUDIT.md` and `POLLUTANT_UNIT_VERIFICATION.md`, this report establishes a **prioritized engineering and scientific roadmap**. 

Recommendations are divided into:
1. **Tier 1 (Mandatory Pre-Validation Fixes)**: Critical scientific and architectural discrepancies that must be addressed or formally resolved before final model validation.
2. **Tier 2 (Methodological & Pipeline Enhancements)**: Data engineering and model retraining improvements to eliminate feature distortion.
3. **Tier 3 (Acceptable Environmental Modeling Limitations)**: Known real-world environmental phenomena that can be legitimately documented as boundaries rather than refactored in code.

---

## 1. Prioritized Recommendations Matrix

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      RECOMMENDED ACTIONS PRIORITIZATION MATRIX                                     |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| ID  | Recommended Action            | Priority | Subsystem Affected    | Timeline / Phase   | Implementation Effort|
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T01 | Correct Feature Importance in | HIGH     | Documentation         | Immediate (Tier 1) | Very Low (1 hour)    |
|     | HOW_IT_WORKS.md               |          | (HOW_IT_WORKS.md)     |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T02 | Add Ambient CO Plausibility   | HIGH     | Backend Serving       | Immediate (Tier 1) | Low (2 hours)        |
|     | Guardrail (>17.0 mg/m3)       |          | (decision_support.py) |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T03 | Align Backend Schema for      | HIGH     | Backend API           | Immediate (Tier 1) | Medium (4 hours)     |
|     | Candidate & Missing Inputs    |          | (schemas.py, main.py) |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T04 | Author Missing Pre-Promotion  | HIGH     | Governance            | Immediate (Tier 1) | Low (2 hours)        |
|     | Review Document               |          | (model_validation/)   |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T05 | Cap / Winsorize CO at 25 mg/m3| HIGH     | Training Pipeline     | Next Retrain (T2)  | Medium (1 day)       |
|     | to Remove GJ001 Artifact      |          | (train_aqi_model.py)  |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T06 | Implement Particulate Ratio   | MEDIUM   | Feature Pipeline      | Next Retrain (T2)  | Medium (1 day)       |
|     | Imputation for PM10 (Lucknow) |          | (imputer.py)          |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T07 | Embed 26-City Disaggregated   | MEDIUM   | Frontend UI           | Next Sprint (T2)   | Low (3 hours)        |
|     | Table in Methodology Page     |          | (ModelMethodology.tsx)|                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
| T08 | Document Chennai Sea-Breeze & | LOW      | Technical Report      | Documentation (T3) | Very Low (1 hour)    |
|     | Post-Monsoon Burning Gaps     |          | (Methodology Section) |                    |                      |
+-----+-------------------------------+----------+-----------------------+--------------------+----------------------+
```

---

## 2. Tier 1: Mandatory Pre-Validation Fixes (Must Be Resolved)

### T01: Correct Feature Importance Discrepancy in `HOW_IT_WORKS.md`
- **Problem**: `HOW_IT_WORKS.md` lines 96–101 report fabricated feature importances ($\text{CO} \approx 8.9\%$, $\text{PM}_{10} \approx 17.8\%$). In reality, the trained production model `aqi_model.pkl` assigns **$40.37\%$ to $\text{CO}$** and only **$4.26\%$ to $\text{PM}_{10}$**.
- **Impact**: Any technical examiner inspecting both the documentation and the serialized pickle file will immediately identify the contradiction, undermining project credibility.
- **Action**: Update `HOW_IT_WORKS.md` to reflect the genuine Gini impurity values:
  - $\text{PM}_{2.5}$: **$46.47\%$**
  - $\text{CO}$: **$40.37\%$**
  - $\text{PM}_{10}$: **$4.26\%$**
  - $\text{SO}_2$: **$3.24\%$**
  - $\text{NO}_2$: **$2.94\%$**
  - $\text{O}_3$: **$2.71\%$**
- **Affected File**: `HOW_IT_WORKS.md` (Lines 96–101).

### T02: Add Ambient Carbon Monoxide Plausibility Alert in Backend
- **Problem**: The backend currently permits $\text{CO}$ values up to **$175.81\ \text{mg/m}^3$** without any range warning because $175.81$ was the maximum training value recorded in Ahmedabad. As proven in the unit verification audit, ambient outdoor $\text{CO} > 17\ \text{mg/m}^3$ is toxicologically lethal ($>15\ \text{ppm}$) and represents sensor flaring.
- **Action**: In `backend/decision_support.py`, add a domain-specific plausibility check:
  ```python
  CO_AMBIENT_SEVERE_THRESHOLD = 17.0  # CPCB 8-hour Severe Breakpoint

  if feature_inputs.get('CO', 0.0) > CO_AMBIENT_SEVERE_THRESHOLD:
      warnings.append(
          f"Advisory: Carbon Monoxide reading ({feature_inputs['CO']} mg/m3) exceeds normal ambient atmospheric limits "
          f"(>17.0 mg/m3). Readings above this level typically indicate localized exhaust plume contamination or sensor flaring."
      )
  ```
- **Affected File**: `backend/decision_support.py`.

### T03: Align Backend Serving Schema to Support Missing Sensors and Metadata
- **Problem**: The production model artifact contains `SimpleImputer`, and the candidate model contains `CityMonthAwareImputer`. However, `backend/schemas.py` defines `PredictRequest` with strictly mandatory floats, rejecting `NaN` or `None` with HTTP 422. Furthermore, it lacks `City` and `Date` fields.
- **Impact**: The imputer is dead code during inference. If an IoT station lacks $\text{PM}_{10}$ or $\text{O}_3$, the API aborts. For the candidate model, lack of city/date metadata disables Level 1 and Level 2 hierarchical imputation.
- **Action**:
  1. Allow optional float fields in `PredictRequest`: `pm10: Optional[float] = None`, `o3: Optional[float] = None`, etc.
  2. Add optional context fields: `city: Optional[str] = None`, `date: Optional[str] = None`.
  3. When `None` is submitted, pass `np.nan` to `pipeline.predict()`, allowing the trained imputer to activate cleanly.
- **Affected Files**: `backend/schemas.py`, `backend/predictor.py`.

### T04: Author the Missing Pre-Promotion Review Document
- **Problem**: Earlier references and project evaluation rubrics specify `model_validation/pre_promotion_review.md`, but this file does not exist in the repository.
- **Action**: Create `model_validation/pre_promotion_review.md` recording the formal architectural sign-off and documenting why candidate promotion is currently blocked (marginal accuracy gain, schema incompatibility, coastal limitations).
- **Affected File**: `model_validation/pre_promotion_review.md`.

---

## 3. Tier 2: Methodological & Pipeline Enhancements (Next Retraining)

### T05: Capping and Winsorization of Carbon Monoxide
- **Problem**: Single-station sensor flaring at Maninagar (`GJ001`) in Ahmedabad severely skewed the Random Forest regressor, artificially inflating $\text{CO}$ importance to $40.37\%$.
- **Action**: Before fitting future models, apply robust outlier winsorization in `train_aqi_model.py`:
  $$\text{CO}_{\text{train}} = \min\left(\text{CO}_{\text{train}},\ 25.0\ \text{mg/m}^3\right)$$
- **Expected Outcome**:
  - Restores atmospheric balance: $\text{PM}_{2.5}$ and $\text{PM}_{10}$ will regain their expected physical dominance.
  - Reduces generalization errors when predicting cities with normal $\text{CO}$ ranges.
- **Affected Files**: `train_aqi_model.py`, `model_validation/train_candidate_model.py`.

### T06: Replace Static Imputation with Particulate Ratio Imputation
- **Problem**: $\text{PM}_{10}$ is $100\%$ missing in Lucknow, $89.7\%$ in Patna, and $85\%$ in Chennai. Global median imputation ($101.88$) injects an artificial signal into pristine southern cities and drastically underestimates coarse dust in Gangetic winter.
- **Action**: In `imputer.py`, implement ratio-based estimation:
  $$\widehat{\text{PM}}_{10} = \begin{cases} 
  \text{PM}_{2.5} / 0.55 & \text{if } \text{PM}_{2.5} \text{ is non-null} \\
  \text{City-Month Median} & \text{otherwise}
  \end{cases}$$
  (Based on typical urban Indian $\text{PM}_{2.5} / \text{PM}_{10}$ ratios of $0.50$ to $0.65$ documented in CPCB particulate characterization studies).
- **Affected Files**: `model_validation/imputer.py`.

### T07: Embed 26-City Disaggregated Table in Frontend Methodology View
- **Problem**: The headline metric ($R^2 = 0.8367$) masks localized failure in Chennai ($R^2 = -0.4509$) and extreme RMSE in Ahmedabad ($132.14$).
- **Action**: Embed the complete 26-city disaggregated evaluation table generated in `candidate_metadata.json` into Section A of `frontend/src/pages/ModelMethodologyPage.tsx`.
- **Expected Outcome**: Full scientific transparency regarding geographical performance and model generalization bounds.
- **Affected Files**: `frontend/src/pages/ModelMethodologyPage.tsx`.

---

## 4. Tier 3: Acceptable Environmental & Project Limitations (Document Only)

The following issues reflect fundamental real-world data constraints that **should NOT be artificially altered or patched in code**, but must be clearly documented as environmental scope boundaries:

### L01: Coastal Atmospheric Decoupling in Chennai ($R^2 = -0.4509$)
- **Context**: In coastal megacities like Chennai, air quality is governed by maritime sea-breeze ventilation cycles and thermal marine boundary layers. Surface criteria pollutants alone without wind velocity and relative humidity are statistically insufficient to forecast next-day dispersion.
- **Recommendation**: Document Chennai as an explicit boundary condition illustrating the necessity of coupled meteorological-chemical modeling.

### L02: COVID-19 Lockdown Concept Drift (March–June 2020)
- **Context**: The second half of the test partition spans the national COVID-19 lockdown, where vehicular and industrial activity stopped abruptly. Under these conditions, air quality halved and stabilized, allowing the naive persistence baseline ($\hat{y}_{t+1} = y_t$) to outperform machine learning ($R^2 = 0.7307$ vs. $0.6336$).
- **Recommendation**: Highlight this as an exemplary case study in macroeconomic policy shocks causing concept drift in machine learning models.

### L03: Seasonal Truncation (Absence of October–November Stubble Burning)
- **Context**: The historical dataset terminates on July 1, 2020. The test partition spans December through June, completely omitting India's severe post-monsoon agricultural crop residue burning crisis (October–November).
- **Recommendation**: Explicitly note that model accuracy during severe stubble-burning episodes remains unverified out-of-sample due to dataset termination.

### L04: Zero-Shot Geographic Generalization (Aizawl, Ernakulam, Kochi)
- **Context**: These three cities were commissioned late in the CPCB CAAQMS program and have zero training samples ($100\%$ out-of-sample).
- **Recommendation**: Frame this as an empirical evaluation of zero-shot cross-city spatial transferability.

---

## 5. Implementation Roadmap Summary

```
========================================================================================
IMMEDIATE ACTION PLAN (Pre-Validation Checklist):
[ ] 1. Update HOW_IT_WORKS.md to correct feature importances (PM2.5: 46.5%, CO: 40.4%).
[ ] 2. Add CO > 17.0 mg/m3 advisory warning to backend/decision_support.py.
[ ] 3. Update backend/schemas.py to allow nullable floats for sensor drop resilience.
[ ] 4. Author model_validation/pre_promotion_review.md recording candidate governance.
[ ] 5. Embed 26-city performance comparison table into ModelMethodologyPage.tsx.
========================================================================================
```

---
*Report certified by Senior Software Architect & Technical Auditor on November 3, 2026.*  
*AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)*

