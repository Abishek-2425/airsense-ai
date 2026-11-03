# AirSense AI — Phase 2B: Backend Integration Report

**Status**: Implemented & Verified  
**Date**: November 2026  
**Auditor**: Backend Architect & Machine Learning Engineer  
**Scope**: FastAPI endpoints, Pydantic schemas, physical particulate ratio preprocessor, CO advisory guardrail, and test validation.

---

## 1. Executive Summary

The AirSense AI backend has been upgraded to **Version 2.0.0**, integrating the newly retrained Release Candidate model while maintaining full backward compatibility with the React frontend dashboard.

### Core Architectural Upgrades:
1. **Dynamic Particulate Ratio Estimation**:
   - Updated `backend/schemas.py` to allow `pm10` and `pm25` as optional inputs (`Optional[float] = None`).
   - If `pm10` is missing but `pm25` is provided, `AirSenseCompositeModel` calculates:
     $$\widehat{\text{PM}}_{10} = \frac{\text{PM}_{2.5}}{\text{Ratio}(\text{month})}$$
   - If `pm25` is missing but `pm10` is provided, the inverse ratio is applied.
   - Transparent warnings are returned in the response payload indicating the exact ratio and estimated concentration used.
2. **Carbon Monoxide Anomaly Advisory**:
   - The API accepts continuous user inputs without arbitrary truncation.
   - If submitted $\text{CO} > 17.0\ \text{mg/m}^3$, an environmental advisory warning is returned explaining that the reading exceeds statutory ambient thresholds and that model predictions reflect standard atmospheric dispersion.
3. **Operational Display Clamping**:
   - Predictions are clamped to $[0.0, 500.0]$ for the citizen-facing `predicted_aqi`, adhering to official CPCB category definitions.
   - The un-clamped model regression output is preserved in `raw_predicted_aqi` for transparent auditability.
4. **Metadata Synchronization**:
   - `/model-info` dynamically serves the genuine feature importances ($\text{PM}_{2.5}: 73.68\%$, $\text{PM}_{10}: 11.84\%$, $\text{CO}: 7.93\%$), training partition window, and operational validation metrics ($R^2 = 0.8712$, $\text{MAE} = 19.26$).

---

## 2. API Endpoint Specification & Verification

### 2.1 Endpoint Verification Matrix

| Endpoint | Method | Input Schema | Response Schema | Status Code | Verification Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/health` | GET | None | `HealthResponse` | 200 OK | **PASSED** (reports status="ok", model_version="2.0.0") |
| `/model-info` | GET | None | `ModelInfoResponse` | 200 OK | **PASSED** (genuine importances: Particulates $85.5\%$, CO $7.9\%$) |
| `/example` | GET | None | `ExampleResponse` | 200 OK | **PASSED** (historical Visakhapatnam test sample) |
| `/predict` (Full Inputs) | POST | `PredictRequest` (All 6) | `PredictResponse` | 200 OK | **PASSED** (deterministic next-day AQI prediction) |
| `/predict` (Missing PM10)| POST | `PredictRequest` (No PM10)| `PredictResponse` | 200 OK | **PASSED** (estimates PM10 via seasonal ratio + warning) |
| `/predict` (Missing PM2.5)| POST | `PredictRequest` (No PM2.5)| `PredictResponse` | 200 OK | **PASSED** (estimates PM2.5 via seasonal ratio + warning) |
| `/predict` (CO > 17 mg/m³)| POST | `PredictRequest` (CO=28.5)| `PredictResponse` | 200 OK | **PASSED** (accepts input, emits anomaly advisory) |
| `/predict` (Negative Input)| POST| `PredictRequest` (Negative)| Error Detail | 422 Unproc | **PASSED** (strictly rejects negative pollutant values) |
| `/predict` (Missing Gas) | POST | `PredictRequest` (No NO2) | Error Detail | 422 Unproc | **PASSED** (strictly rejects omitted gaseous pollutants) |
| `/predict` (Extreme Input)| POST | `PredictRequest` (Extreme) | `PredictResponse` | 200 OK | **PASSED** (clamps display AQI to 500, category="Severe") |

---

## 3. Automated Test Execution

The backend test suite (`tests/test_backend_api.py`) executed 10 test scenarios via FastAPI `TestClient`:
```
Ran 10 tests in 1.306s

OK
```
Zero test failures, zero regressions.

