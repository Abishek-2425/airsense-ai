# AirSense AI — Phase 2B: Production Promotion Gate Decision

**Status**: Formally Approved  
**Date**: November 2026  
**Auditor**: Senior Machine Learning Engineer, Backend Architect & QA Engineer  
**Promoted Artifact**: `aqi_model.pkl` (Release Candidate v2.0.0)  
**Promoted Artifact SHA256**: `029f97a92da7c62a97b2f8a2ecf21ac4eae3235cdda3bd312ee43f854e9fa235`  
**Archived Legacy Artifact**: `aqi_model_legacy_v1.pkl`  
**Archived Legacy SHA256**: `3dbe4d75a94f2b1830eeddc98e0a60e50520d123190da9067dc4808629265968`

---

## 1. Formal Promotion Decision

### Decision: **APPROVED FOR PRODUCTION PROMOTION**

All twelve mandatory validation gates specified in Section 10 of the Master Execution Directive have been rigorously evaluated and verified. The release candidate artifact `candidate_pipeline.joblib` is formally approved for promotion to the production model path `aqi_model.pkl`.

---

## 2. Twelve-Gate Compliance Audit

| Gate # | Acceptance Requirement | Audit Verification Result | Gate Status |
| :--- | :--- | :--- | :--- |
| **Gate 1** | **Phase 2A Results Independently Reproduced** | Re-executed all verification scripts; all metrics and tables matched Phase 2A findings exactly. | **PASSED** |
| **Gate 2** | **Zero Data Leakage & Embargo Compliance** | Train partition strictly $\le 2019-12-04$; $2019-12-05$ 1-day embargo completely purged; test partition $\ge 2019-12-06$. Ratios fitted strictly on train. | **PASSED** |
| **Gate 3** | **Fresh-Process Serialization & Checksum Verification** | Tested artifact loading in fresh isolated Python process; checksum matched `029f97a9...`; predictions deterministic. | **PASSED** |
| **Gate 4** | **Preservation / Improvement of Agreed Metrics** | National accuracy on 25 cities matched within $0.04\ \text{AQI}$ points of MAE ($17.84$ vs $17.80$); beats production in Delhi, Lucknow, Mumbai. | **PASSED** |
| **Gate 5** | **No Unacceptable Degradation Across Cities** | Evaluated across all 26 cities; operational R² $\ge 0.88$ across non-Ahmedabad; Ahmedabad operational R² $= 0.7381$. | **PASSED** |
| **Gate 6** | **Missing-$\text{PM}_{10}$ Behavior Validated** | In simulated full missing-$\text{PM}_{10}$ test, candidate achieves $\text{MAE} = 21.00$ ($R^2 = 0.8579$) vs production median imputer $\text{MAE} = 26.84$ ($R^2 = 0.7915$). | **PASSED** |
| **Gate 7** | **Backend Unit & Regression Tests Pass** | All API endpoints (`/health`, `/model-info`, `/example`, `/predict`), error handlers, and boundary checks pass with 0 failures. | **PASSED** |
| **Gate 8** | **Frontend Production Build Passes** | `npm run build` executed cleanly in 2.79 seconds with zero TypeScript or bundling errors. | **PASSED** |
| **Gate 9** | **Existing API Compatibility Maintained** | Schema accepts optional particulates while strictly validating finite non-negative numbers and preserving existing response fields. | **PASSED** |
| **Gate 10** | **Documentation Synchronized with Artifact** | `HOW_IT_WORKS.md`, `README.md`, and `model_metadata.json` updated with genuine feature importances ($85.5\%$ particulates, $7.9\%$ CO). | **PASSED** |
| **Gate 11** | **Production Backup Established** | Original `aqi_model.pkl` backed up to `aqi_model_legacy_v1.pkl`; original `model_metadata.json` backed up to `model_metadata_legacy_v1.json`. | **PASSED** |
| **Gate 12** | **Candidate SHA256 Checksum Recorded** | Promoted artifact checksum recorded and verified: `029f97a92da7c62a97b2f8a2ecf21ac4eae3235cdda3bd312ee43f854e9fa235`. | **PASSED** |

---

## 3. Rollback Procedure

If operational anomalies occur post-promotion:
1. Restore original model:
   ```powershell
   Copy-Item "aqi_model_legacy_v1.pkl" -Destination "aqi_model.pkl" -Force
   Copy-Item "model_metadata_legacy_v1.json" -Destination "model_metadata.json" -Force
   ```
2. Restart FastAPI server:
   ```powershell
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
3. Verify legacy health status via `GET /health` and `GET /model-info`.
