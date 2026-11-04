# AI-Powered Air Quality Prediction & Decision Support System - Backend API (Release v2.0)

Production backend implementation of the AirSense AI system. This service provides a robust FastAPI backend that serves next-calendar-day Air Quality Index (AQI) predictions and environmental decision-support guidance from current-day pollutant measurements.

---

## 1. Project Purpose & Current Capabilities (Release Candidate v2.0)

- **Next-Day AQI Regression**: Predicts the next calendar day's AQI from current-day concentrations of six key criteria pollutants (`PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`).
- **Flexible Particulate Input with Seasonal Ratio Imputation**: Supports missing `PM10` or `PM2.5` via training-derived monthly ratio estimation ($\widehat{\text{PM}}_{10} = \text{PM}_{2.5} / r_{\text{month}}$), maintaining high accuracy even when stations lack coarse particulate analyzers.
- **Pre-trained Composite Model**: Loads `AirSenseCompositeModel` (`AirSensePreprocessor` + `RandomForestRegressor`) once at application startup.
- **Rule-Based Decision Support**: Maps numeric AQI predictions into standard CPCB category bands with deterministic half-up rounding and returns targeted environmental and public health recommendations.
- **Input Validation & Bound Verification**: Validates numeric bounds, prevents non-finite and negative inputs, alerts users if concentrations exceed training maxima, and issues runtime warnings for CO observations exceeding statutory ambient screening levels ($17.0\ \text{mg/m}^3$).
- **Transparency & Diagnostics**: Exposes genuine model information, feature importances (restored particulate weight: 85.52%), training bounds, and documented historical test examples via RESTful endpoints.

---

## 2. Project Structure

```text
backend/
├── __init__.py
├── main.py              # FastAPI application, lifespan handler, CORS, REST routers
├── preprocessor.py      # Serializable AirSensePreprocessor & AirSenseCompositeModel
├── schemas.py           # Pydantic v2 schemas supporting optional particulates
├── predictor.py         # Thread-safe pipeline loading, model caching & inference
├── decision_support.py  # Classification, half-up rounding, and health advisories
├── README.md            # Backend technical documentation
└── tests/
    └── test_backend_api.py # Automated test suite (10/10 passing tests)
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 and 3.11)
- PowerShell or command shell on Windows / Linux / macOS

### Install Dependencies
From the project root:

```bash
pip install -r requirements.txt
```

---

## 4. Running the Tests & Server

### Run Backend Test Suite
Run the 10 automated test cases via `unittest`:

```bash
python -m unittest tests/test_backend_api.py
```

### Start the FastAPI Server
Start Uvicorn on port `8000`:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 5. Model Features & Order

The pre-trained Random Forest model strictly requires six pollutant features in this exact order:

1. `PM2.5` (Fine Particulate Matter $\le 2.5\ \mu\text{m}$, in $\mu\text{g/m}^3$)
2. `PM10` (Coarse Particulate Matter $\le 10\ \mu\text{m}$, in $\mu\text{g/m}^3$)
3. `NO2` (Nitrogen Dioxide, in $\mu\text{g/m}^3$)
4. `CO` (Carbon Monoxide, in $\text{mg/m}^3$)
5. `SO2` (Sulphur Dioxide, in $\mu\text{g/m}^3$)
6. `O3` (Ground-Level Ozone, in $\mu\text{g/m}^3$)

In Release Candidate v2.0, `pm10` and `pm25` are optional in `/predict` requests. If either is omitted, it is dynamically estimated using training-fitted monthly ratios.

---

## 6. Reported Model Evaluation Metrics (v2.0)

Evaluated across the 4,900 held-out chronological test observations (`2019-12-06` to `2020-06-30`):

| Evaluation Metric | Random Forest v2.0 (Operational $\le 500$) | Persistence Baseline | Comparative Advantage |
| :--- | :---: | :---: | :--- |
| **Coefficient of Determination ($R^2$)** | **0.8712** | 0.8284 | **+4.28% variance explained** |
| **Root Mean Squared Error (RMSE)** | **29.41** | 34.04 | **-13.6% error reduction** |
| **Mean Absolute Error (MAE)** | **19.26** | 19.60 | **-0.34 AQI points lower error** |
| **Non-Ahmedabad Cities ($N=4,703$) MAE** | **17.84** | 18.02 | **Consistent across all 25 cities** |
| **Delhi ($N=208$) $R^2$** | **0.9027** | 0.8991 | **Superior in high-particulate episodes** |
| **Missing-$\text{PM}_{10}$ Simulation MAE** | **21.00** | 26.84 *(v1 imputer)* | **-5.84 points error reduction** |

---

## 7. Operational AQI Category Bands & Rules

| AQI Band | Category | Summary Description |
| :--- | :--- | :--- |
| 0–50 | **Good** | Minimal health impact; ideal for outdoor activities |
| 51–100 | **Satisfactory** | Minor breathing discomfort to sensitive people |
| 101–200 | **Moderate** | Breathing discomfort to people with lungs, asthma, and heart diseases |
| 201–300 | **Poor** | Breathing discomfort to most people on prolonged exposure |
| 301–400 | **Very Poor** | Respiratory illness on prolonged exposure |
| 401–500 | **Severe** | Affects healthy people and seriously impacts those with existing diseases |

### Boundary & Range Handling Rules
1. **Negative Clamping**: Negative outputs are clamped to `0.0`.
2. **Statutory Scale Ceiling**: Outputs exceeding 500 are clamped to `500.0` for the operational forecast, with the unconstrained raw model output preserved in `raw_predicted_aqi`.
3. **Fractional Value Rounding**: Standard arithmetic half-up rounding (`math.floor(val + 0.5)`) prevents floating-point boundary jitter.

---

## 8. Training Maxima & Anomaly Guardrails

- `PM2.5`: **949.99** $\mu\text{g/m}^3$
- `PM10`: **917.08** $\mu\text{g/m}^3$
- `NO2`: **362.21** $\mu\text{g/m}^3$
- `CO`: **17.00** $\text{mg/m}^3$ *(screened threshold)*
- `SO2`: **186.08** $\mu\text{g/m}^3$
- `O3`: **257.73** $\mu\text{g/m}^3$

---

## 9. Feature Importance (Release Candidate v2.0)

Genuine Gini importance extracted from the retrained ensemble:
- `PM2.5`: **73.68%**
- `PM10`: **11.84%**
- `CO`: **7.93%**
- `SO2`: **2.33%**
- `NO2`: **2.19%**
- `O3`: **2.03%**

Total particulate importance accounts for **85.52%** of predictive weight, accurately capturing Indian urban atmospheric physics.
