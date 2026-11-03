# AirSense AI — Phase 2B Documentation & Architecture Changelog

**Release Version:** v2.0.0 (Production Promoted)  
**Previous Version:** v1.0.0 (Legacy Production Pipeline)  
**Date:** November 4, 2026  
**Auditor / Architect:** Senior Machine Learning Engineer & Backend Architect

---

## 1. Version Overview

This changelog records all technical modifications, file updates, dataset screening operations, machine learning architectural upgrades, backend refactorings, frontend synchronisations, and documentation revisions executed during **Phase 2A** and **Phase 2B**.

```text
+--------------------------------------------------------------------------------------------------+
| Release v2.0.0 Key Transformations:                                                              |
| 1. Chronological Embargo Partitioning (Train <= 2019-12-04, Embargo 2019-12-05, Test >= 2019-12-06)|
| 2. Ahmedabad CO Sensor Contamination Screened (699 training rows with CO > 17.0 mg/m3 removed)   |
| 3. Target Capping to Statutory Scale (AQI <= 500)                                                |
| 4. Dynamic Monthly PM2.5/PM10 Ratio Imputation (mean ratio: 0.4644, winter peak: 0.5643)          |
| 5. Particulate Importance Restored (85.52% total weight; CO reduced from 40.37% to 7.93%)       |
| 6. Operational Scale CPCB Performance: R2 = 0.8712, RMSE = 29.41, MAE = 19.26 (N = 4,900)       |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. File-by-File Detailed Changelog

### 2.1 Model Artifacts & Preprocessing Code
| File Path | Nature of Change | Summary of Modifications |
| :--- | :---: | :--- |
| `aqi_model_legacy_v1.pkl` | **Created (Backup)** | Complete exact backup of legacy production model (SHA256: `3dbe4d75a94f2b1830eeddc98e0a60e50520d123190da9067dc4808629265968`). |
| `model_metadata_legacy_v1.json` | **Created (Backup)** | Complete exact backup of legacy metadata recording v1.0 parameters and metrics. |
| `model_validation/phase_2b/preprocessor.py` | **Created** | Implements standalone, serializable `AirSensePreprocessor` and `AirSenseCompositeModel` with monthly particulate ratio curve, fallback medians, and input validation. |
| `model_validation/phase_2b/train_release_candidate.py` | **Created & Executed** | Training script fitting Release Candidate v2.0 on clean chronological split with embargo, producing candidate artifacts and CSV evaluation tables. |
| `model_validation/phase_2b/release_candidate/candidate_pipeline.joblib` | **Created** | Release candidate serialized composite model (SHA256: `029f97a92da7c62a97b2f8a2ecf21ac4eae3235cdda3bd312ee43f854e9fa235`). |
| `aqi_model.pkl` | **Promoted** | Overwritten with Release Candidate v2.0 following verification of all 12 production promotion gates. |
| `model_metadata.json` | **Promoted** | Overwritten with Release Candidate v2.0 metadata recording genuine feature importances, training maxima, and operational evaluation metrics. |

### 2.2 Backend Microservice (`backend/`)
| File Path | Nature of Change | Summary of Modifications |
| :--- | :---: | :--- |
| `backend/preprocessor.py` | **Created** | Mirror copy of `model_validation/phase_2b/preprocessor.py` ensuring seamless deserialization and zero runtime dependency drift. Includes DataFrame column wrapping for clean prediction. |
| `backend/schemas.py` | **Updated** | Made `pm10` and `pm25` optional in `PredictRequest`; added `month` field (1–12); added `features_used` and `model_version` in `PredictResponse`; added `model_version` in `HealthResponse` and `ModelInfoResponse`. |
| `backend/predictor.py` | **Updated** | Updated lifespan loader to load `AirSenseCompositeModel`; added `predict_aqi_composite()` handling missing particulates; updated `get_feature_importances()` to retrieve genuine array from `model.regressor.feature_importances_`. |
| `backend/main.py` | **Updated** | Implemented composite prediction workflow; updated CO anomaly advisory threshold to $17.0\ \text{mg/m}^3$; capped operational `predicted_aqi` at statutory ceiling ($500.0$) while retaining unconstrained model output in `raw_predicted_aqi`. |
| `backend/README.md` | **Updated** | Rewritten to document v2.0 capabilities, ratio imputation, genuine feature importances, and operational metrics. |

### 2.3 Automated Test Suite (`tests/`)
| File Path | Nature of Change | Summary of Modifications |
| :--- | :---: | :--- |
| `tests/test_backend_api.py` | **Created** | Comprehensive 10-test battery covering `/health`, `/model-info`, `/example`, `/predict` (full inputs, missing PM10, missing PM2.5, missing both, CO anomaly advisory, negative validation, and upper bound clamping). All 10 tests passing. |
| `model_validation/phase_2b/test_fresh_process_loading.py` | **Created** | Standalone script validating fresh-process deserialization, checksum match, and inference reproducibility. |

### 2.4 Frontend Application (`frontend/`)
| File Path | Nature of Change | Summary of Modifications |
| :--- | :---: | :--- |
| `frontend/src/types/index.ts` | **Updated** | Added optional particulate fields and `month` to `PredictRequest`; added `features_used` and `model_version` to `PredictResponse`; updated `FormErrors` and `FormState`. |
| `frontend/src/pages/Dashboard.tsx` | **Updated** | Updated metric cards ($R^2=0.8712$, $\text{RMSE}=29.41$, $\text{MAE}=19.26$, $N=4,900$); updated feature importance chart with empirical v2.0 values ($\text{PM}_{2.5}: 73.7\%$, $\text{PM}_{10}: 11.8\%$, $\text{CO}: 7.9\%$). |
| `frontend/src/pages/PredictPage.tsx` | **Updated** | Supported optional PM10/PM2.5 inputs; added auto-estimation visual badge; updated form validation to require at least one particulate; updated CO maximum threshold to $17.0\ \text{mg/m}^3$; displayed features used and model version tag; updated raw output clamp description. |
| `frontend/src/pages/ModelMethodologyPage.tsx` | **Updated** | Updated 7-step pipeline methodology with chronological embargo split and ratio imputation; added dedicated Phase 2B scientific upgrades report card. |

### 2.5 Documentation & Audit Reports (`data_quality_audit/phase_2b/`)
| File Path | Nature of Change | Summary of Modifications |
| :--- | :---: | :--- |
| `README.md` (root) | **Created** | High-level project README with architecture flowcharts, v2.0 benchmarking tables, and local run instructions. |
| `HOW_IT_WORKS.md` | **Updated** | Comprehensive rewrite documenting Release Candidate v2.0 architecture, genuine feature importances, ratio estimation, and operational scale calibration. |
| `PHASE_2A_FINAL_VERIFICATION.md` | **Created** | 5-point verification report resolving Phase 2A findings, embargo split definition, ratio imputer superiority, and target cap validation. |
| `MODEL_TRAINING_REPORT.md` | **Created** | Technical report documenting chronological embargo training, training partition statistics, hyperparameters, and feature importance transformation. |
| `MODEL_EVALUATION_REPORT.md` | **Created** | Detailed evaluation report covering national, metropolitan, seasonal, and missingness simulation metrics across 4,900 held-out test days. |
| `BACKEND_INTEGRATION_REPORT.md` | **Created** | Architectural report documenting FastAPI backend refactoring, Pydantic schemas, and endpoint responses. |
| `PRODUCTION_PROMOTION_DECISION.md` | **Created** | Formal evaluation of 12 production promotion gates approving Release Candidate v2.0 promotion. |
| `FRONTEND_INTEGRATION_REPORT.md` | **Created** | Verification of React frontend synchronisation and successful Vite production compilation. |
| `TESTING_REPORT.md` | **Created** | Comprehensive test execution report documenting 100% pass rate across backend and frontend test suites. |
| `DOCUMENTATION_CHANGELOG.md` | **Created** | This comprehensive change log and version comparison document. |
| `PHASE_2B_FINAL_REPORT.md` | **Created** | Master technical closeout report summarizing the entire Phase 2A/2B lifecycle. |

---

## 3. Dataset Preservation Confirmation

Throughout all operations in Phase 2A and Phase 2B:
- **`archive/city_day.csv`** was untouched and preserved in its original byte-for-byte state.
- **`archive/city_hour.csv`**, **`station_day.csv`**, **`station_hour.csv`**, and **`stations.csv`** were untouched and read-only.
- All experimental training filtering was executed exclusively in memory and saved to isolated experimental artifacts under `data_quality_audit/` and `model_validation/`.
- No historical ground truth or sensor readings were silently deleted or modified.

