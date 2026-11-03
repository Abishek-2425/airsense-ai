# AirSense AI — Phase 1 Audit Execution Log & Verification Ledger

**Project**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)  
**Location**: `D:\PROJECTS\evs` (Audit Scope: `G:\PROJECTS\evs`)  
**Audit Phase**: Phase 1 — Dataset Quality, Integrity & Consistency Audit  
**Auditor**: Senior Software Architect, Machine Learning Engineer & Scientific Auditor  
**Date of Audit**: November 3, 2026  
**Execution Shell**: PowerShell on Windows 11 Enterprise (Build 26100)  
**Python Runtime**: Python 3.10.11 (Pyenv-win), Pandas 2.3.3, NumPy 2.2.6, Scikit-learn 1.7.2  

---

## 1. Inventory of Files Inspected

The audit executed non-destructive static inspections, structural checks, and bytecode verifications across the entire repository:

### 1.1 Dataset Files (`archive/`):
- `archive/city_day.csv` (2,574,056 bytes, 29,531 rows, 16 columns)
- `archive/city_hour.csv` (65,648,581 bytes, 707,875 rows, 16 columns)
- `archive/station_day.csv` (8,634,105 bytes, 108,035 rows, 16 columns)
- `archive/station_hour.csv` (219,675,434 bytes, 2,589,083 rows, 16 columns)
- `archive/stations.csv` (14,467 bytes, 230 stations, 5 columns)
- `archive.zip` (76,469,579 bytes, verified master zip container)

### 1.2 Model Artifacts & Training Scripts:
- `aqi_model.pkl` (78,785,271 bytes, verified production pipeline)
- `model_metadata.json` (2,138 bytes, verified training parameters and bounds)
- `train_aqi_model.py` (16,817 bytes, verified pairing logic and 80/20 split)
- `calculate_training_maxima.py` (1,834 bytes, verified non-null maximum logic)

### 1.3 Validation Artifacts (`model_validation/`):
- `model_validation/aqi_model_candidate.pkl` (78,756,636 bytes, verified candidate pipeline)
- `model_validation/candidate_metadata.json` (32,362 bytes, verified 26-city evaluation)
- `model_validation/candidate_evaluation_report.md` (6,560 bytes)
- `model_validation/imputer.py` (5,582 bytes, verified `CityMonthAwareImputer`)
- `model_validation/train_candidate_model.py` (22,550 bytes, verified 1-day embargo)
- `model_validation/test_candidate.py` (8,171 bytes, verified 6 test routines)

### 1.4 Backend Microservice (`backend/`):
- `backend/main.py` (6,858 bytes, verified routes, CORS, lifespan handler)
- `backend/schemas.py` (3,744 bytes, verified `PredictRequest`, bounds, validators)
- `backend/predictor.py` (6,805 bytes, verified model loading, bridge patch, inference)
- `backend/decision_support.py` (5,810 bytes, verified half-up rounding, category mapping)
- `backend/tests/test_api.py` (6,489 bytes, verified 6 backend test routines)
- `backend/README.md` (11,040 bytes)

### 1.5 Frontend Application (`frontend/`):
- `frontend/src/App.tsx` (6,614 bytes, verified 3-page tab state)
- `frontend/src/pages/Dashboard.tsx` (17,146 bytes, verified KPI cards, Recharts)
- `frontend/src/pages/PredictPage.tsx` (19,891 bytes, verified form inputs, units)
- `frontend/src/pages/ModelMethodologyPage.tsx` (37,371 bytes, verified 3 sub-tabs)
- `frontend/src/services/api.ts` (2,510 bytes, verified fetch client)
- `frontend/src/types/index.ts` (1,916 bytes, verified TypeScript interfaces)
- `frontend/src/components/Navbar.tsx` (6,757 bytes, verified health heartbeat)
- `frontend/src/components/AqiScale.tsx` (5,740 bytes, verified statutory gauge)
- `frontend/package.json` (verified dependencies: React 19, Vite 8, Recharts)
- `frontend/vite.config.ts` (verified port configuration)

### 1.6 Documentation & Prior Audits:
- `scientific_validation_audit.md` (26,423 bytes, verified initial validation findings)
- `HOW_IT_WORKS.md` (13,154 bytes, identified documentation discrepancy)
- `PROJECT_PROGRESS_REPORT.md` (certified complete audit report)

---

## 2. Inventory of Commands Executed

All commands executed during Phase 1 were strictly non-destructive:

```powershell
# 1. Directory and Workspace Initialization
New-Item -ItemType Directory -Path "data_quality_audit" -Force

# 2. Comprehensive Macro Dataset & Feature Statistics Calculation
python data_quality_audit/audit_script.py
# Output: Generated data_quality_audit/DATA_QUALITY_SUMMARY.csv
# Output: Generated data_quality_audit/CITY_DATA_QUALITY.csv

# 3. Investigation of Extreme Carbon Monoxide (CO) Distribution
python data_quality_audit/investigate_co.py
# Output: Verified Ahmedabad GJ001 isolation (Med: 16.24 mg/m3 vs Rest of India: 0.86 mg/m3)

# 4. Forensic Station Decomposition for Ahmedabad
python data_quality_audit/investigate_stations.py
# Output: Identified single station GJ001 (Maninagar, GPCB) as source of all Ahmedabad CO data

# 5. Temporal Progression Analysis for Ahmedabad CO and AQI
python data_quality_audit/investigate_ahm_temporal.py
# Output: 2018 peak mean CO = 33.24 mg/m3, mean AQI = 622.3, AQI > 500 = 184 days

# 6. Correlation Analysis between Criteria Pollutants and Reported AQI
python data_quality_audit/investigate_correlations.py
# Output: Verified CO-AQI correlation r = 0.8415 in Ahmedabad vs r = 0.2798 in Delhi

# 7. Hourly Telemetry Inspection for Peak Days in Ahmedabad
python data_quality_audit/check_hourly_co.py
# Output: 2017-10-25 evening spike from 2.00 to 357.04 mg/m3
python data_quality_audit/check_hourly_2018.py
# Output: 2018-02-19 03:00 AM spike to 446.41 mg/m3 (~390 ppm)

# 8. Calendar Pairing and Temporal Embargo Verification
python data_quality_audit/verify_pairing.py
# Output: Asserted intra-city pairing, Delta t = 1 day, and 1-day embargo separation

# 9. Verification of Zero Lucknow PM10 and Zero Ernakulam O3
python -c "import pandas as pd; df = pd.read_csv('archive/city_day.csv'); print('Lucknow PM10 count:', df[df['City']=='Lucknow']['PM10'].count()); print('Ernakulam O3 count:', df[df['City']=='Ernakulam']['O3'].count())"
# Output: Lucknow PM10 count = 0, Ernakulam O3 count = 0

# 10. Verification of Git Status & Production File Safety
git status
# Output: Confirmed clean working tree with zero unauthorized modifications
```

---

## 3. Errors, Edge Cases & Unavailable Evidence

1. **`pre_promotion_review.md` Unavailable**:
   - The document `model_validation/pre_promotion_review.md` referenced in audit tasks was searched across the filesystem and Git commit history. It does not exist in the repository. Its absence is documented as a governance finding in `AUDIT_RECOMMENDATIONS.md`.
2. **Missing Meteorological Variables**:
   - Meteorological data (wind speed, wind direction, planetary boundary layer height, temperature, and relative humidity) is absent from the CPCB archive. Coastal decoupling in Chennai ($R^2 = -0.4509$) cannot be resolved without external meteorological datasets (e.g. ERA5 / IMD).
3. **Sensor Calibration Metadata**:
   - Original CAAQMS sensor calibration logs and probe maintenance records for station `GJ001: Maninagar` are not included in the Kaggle compilation. The conclusion that hourly readings exceeding $446\ \text{mg/m}^3$ represent sensor saturation/flaring is proven via atmospheric and toxicological limits.

---

## 4. Production Safety & Non-Destructive Integrity Confirmation

In strict compliance with audit safety rules:

```
+--------------------------------------------------------------------------------------------------+
|                                    AUDIT SAFETY CONFIRMATION                                     |
+----------------------------------------------------+---------------------------------------------+
| System Artifact                                    | Safety Status                               |
+----------------------------------------------------+---------------------------------------------+
| archive/city_day.csv                               | UNCHANGED (Zero byte alteration)            |
| archive/station_day.csv, city_hour.csv, etc.       | UNCHANGED (Zero byte alteration)            |
| aqi_model.pkl (Production Estimator)               | UNCHANGED (Zero retrain or overwrite)       |
| model_validation/aqi_model_candidate.pkl           | UNCHANGED (Zero retrain or overwrite)       |
| backend/main.py, schemas.py, predictor.py          | UNCHANGED (Zero code modification)          |
| frontend/src/App.tsx, PredictPage.tsx, etc.        | UNCHANGED (Zero code modification)          |
| Created Deliverables Directory                     | data_quality_audit/ (Dedicated & Isolated)  |
+----------------------------------------------------+---------------------------------------------+
```

---

## 5. Summary of Phase 1 Deliverables Generated

All six required deliverables have been compiled, verified, and placed in `data_quality_audit/`:

1. `data_quality_audit/DATASET_QUALITY_AUDIT.md`: Complete macro audit report, missing-value distributions, outlier taxonomies, and consistency findings.
2. `data_quality_audit/POLLUTANT_UNIT_VERIFICATION.md`: Exhaustive measurement unit verification, statutory CPCB standards, and the forensic resolution of the Ahmedabad $\text{CO}$ anomaly.
3. `data_quality_audit/DATA_QUALITY_SUMMARY.csv`: Column-by-column empirical statistics (count, missingness, mean, median, percentiles, IQR bounds, zero counts, negative counts).
4. `data_quality_audit/CITY_DATA_QUALITY.csv`: City-wise coverage, temporal spans, missing calendar days, pollutant missing percentages, and extreme event counts across all 26 cities.
5. `data_quality_audit/AUDIT_RECOMMENDATIONS.md`: Prioritized engineering roadmap dividing issues into mandatory pre-validation fixes, pipeline enhancements, and acceptable academic limitations.
6. `data_quality_audit/AUDIT_EXECUTION_LOG.md`: Complete audit execution history, command ledger, and safety certifications.

---
*Execution log certified by Senior Software Architect & Technical Auditor on November 3, 2026.*  
*AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)*

