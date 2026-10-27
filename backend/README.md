# AI-Powered Air Quality Prediction & Decision Support System - Backend API

Stage 2 implementation of the final-year Environmental Studies (EVS) project. This service provides a small, robust FastAPI backend that serves next-calendar-day Air Quality Index (AQI) predictions and environmental decision-support guidance from current-day pollutant measurements.

---

## 1. Project Purpose & Current Capabilities

- **Next-Day AQI Regression**: Predicts the next calendar day's AQI from current-day concentrations of six key criteria pollutants (`PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`).
- **Pre-trained Pipeline Integration**: Loads the pre-trained scikit-learn `Pipeline` (`SimpleImputer` + `RandomForestRegressor`) once at application startup.
- **Rule-Based Decision Support**: Maps the numeric AQI regression prediction to prototype bands and returns concise, deterministic environmental and public health recommendations.
- **Input Validation & Bound Verification**: Validates numeric bounds, prevents non-finite and negative inputs, and alerts users if any pollutant exceeds observed training data maxima.
- **Transparency & Diagnostics**: Exposes genuine model information, feature importances, training bounds, and documented historical test examples via RESTful endpoints.

---

## 2. Project Structure

```text
evs/
├── archive/
│   ├── city_day.csv
│   ├── city_hour.csv
│   ├── station_day.csv
│   ├── station_hour.csv
│   └── stations.csv
├── archive.zip
├── train_aqi_model.py
├── calculate_training_maxima.py
├── aqi_model.pkl
├── model_metadata.json
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── predictor.py
│   ├── decision_support.py
│   ├── README.md
│   └── tests/
│       ├── __init__.py
│       └── test_api.py
└── static/
    └── index.html
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PowerShell or command shell on Windows / Linux / macOS

### Install Dependencies
From the project root (`evs/`):

```bash
pip install -r requirements.txt
```

---

## 4. Running the Tests & Server

### Run Focused Tests
Run the 6 focused tests via `pytest`:

```bash
pytest backend/tests/test_api.py -v
```

### Start the FastAPI Server
Start Uvicorn on port `8000`:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Root Web Placeholder**: [http://localhost:8000/](http://localhost:8000/)

---

## 5. Model Features & Order

The pre-trained Random Forest model strictly requires six pollutant features in this exact order:

1. `PM2.5` (Particulate Matter $\le 2.5\ \mu\text{m}$, in $\mu\text{g/m}^3$)
2. `PM10` (Particulate Matter $\le 10\ \mu\text{m}$, in $\mu\text{g/m}^3$)
3. `NO2` (Nitrogen Dioxide, in $\mu\text{g/m}^3$)
4. `CO` (Carbon Monoxide, in $\text{mg/m}^3$)
5. `SO2` (Sulphur Dioxide, in $\mu\text{g/m}^3$)
6. `O3` (Ozone, in $\mu\text{g/m}^3$)

JSON requests submitted to `/predict` map `pm25`, `pm10`, `no2`, `co`, `so2`, and `o3` to these respective feature names.

---

## 6. Reported Model Evaluation Metrics

From the Stage 1 training report on the matched chronological test set (4,902 samples, `2019-12-05` to `2020-06-30`):

| Model | Evaluation Split | $R^2$ Score | RMSE | MAE | Test Samples |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **Matched Chronological Test Set** | **0.8367** | **36.46** | **20.10** | **4,902** |
| Random Forest Regressor | All Chronological Test Rows | 0.8304 | 37.06 | 20.31 | 4,970 |
| Persistence Baseline ($\hat{y}_{t+1} = y_t$) | Matched Chronological Test Set | 0.7724 | 43.05 | 20.91 | 4,902 |

> **Evaluation Context**: The persistence baseline predicts tomorrow's AQI using today's AQI. The Random Forest model demonstrates a meaningful improvement over the persistence baseline, lowering RMSE by ~6.6 AQI points. Note that $R^2$ represents the proportion of target variance explained by the model, not a percentage prediction accuracy.

---

## 7. Prototype AQI Category Bands & Rules

> **Important Disclaimer**:
> The model outputs an estimated AQI by regression. The specified AQI bands are used only to label that estimate for this prototype; this is not a direct CPCB AQI calculation from pollutant sub-indices.

### Required Bands

| AQI Band | Category | Summary Description |
| :--- | :--- | :--- |
| 0–50 | **Good** | Minimal health impact; ideal for outdoor activities |
| 51–100 | **Satisfactory** | Minor breathing discomfort to sensitive people |
| 101–200 | **Moderate** | Breathing discomfort to people with lungs, asthma, and heart diseases |
| 201–300 | **Poor** | Breathing discomfort to most people on prolonged exposure |
| 301–400 | **Very Poor** | Respiratory illness on prolonged exposure |
| 401–500 | **Severe** | Affects healthy people and seriously impacts those with existing diseases |

### Boundary & Range Handling Rules
1. **Negative Prediction Clamping**: If raw regression output $< 0.0$, the displayed `predicted_aqi` is clamped to `0.0`, category is assigned as `Good`, and a warning is included noting that the model produced a negative estimate.
2. **Above-500 Prediction Handling**: If raw regression output $> 500.0$, the exact numeric prediction remains visible, category is assigned as `Severe`, and a warning notes that the prediction exceeds the supported 0–500 band range.
3. **Fractional Value Rounding**: To determine category transitions deterministically, the prediction is rounded to the nearest integer using standard half-up arithmetic (`math.floor(val + 0.5)`). For example, `50.49` falls in `Good` ($\le 50$), while `50.50` transitions to `Satisfactory` ($51$).

---

## 8. Training Maxima & Out-of-Range Warnings

When pollutant inputs exceed the maximum non-null values observed during model training, the prediction is still computed, but a clear warning is appended to `warnings`.

### Observed Training Partition Maxima (2015-01-01 to 2019-12-05)
- `PM2.5`: **949.99** $\mu\text{g/m}^3$
- `PM10`: **917.08** $\mu\text{g/m}^3$
- `NO2`: **362.21** $\mu\text{g/m}^3$
- `CO`: **175.81** $\text{mg/m}^3$
- `SO2`: **186.08** $\mu\text{g/m}^3$
- `O3`: **257.73** $\mu\text{g/m}^3$

---

## 9. Feature Importance & Caveat

Feature importance values returned by `GET /model-info` are genuine Gini/impurity scores extracted directly from `RandomForestRegressor.feature_importances_`:

- `PM2.5`: **0.464730** (46.5%)
- `CO`: **0.403710** (40.4%)
- `PM10`: **0.042632** (4.3%)
- `SO2`: **0.032438** (3.2%)
- `NO2`: **0.029412** (2.9%)
- `O3`: **0.027078** (2.7%)

> **Feature Importance Caveat**:
> Feature importance shows how the trained model uses its inputs; it does not establish that a pollutant causes an AQI outcome.

---

## 10. API Endpoints & Example Responses

### `GET /health`
Returns service readiness and confirms the saved model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `GET /model-info`
Returns architecture metadata, training maxima, metrics, and feature importances.

**Actual Response**:
```json
{
  "model_name": "RandomForestRegressor",
  "target": "Next-calendar-day AQI",
  "features": [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3"
  ],
  "feature_importances": [
    {
      "feature": "PM2.5",
      "importance": 0.46473
    },
    {
      "feature": "PM10",
      "importance": 0.042632
    },
    {
      "feature": "NO2",
      "importance": 0.029412
    },
    {
      "feature": "CO",
      "importance": 0.40371
    },
    {
      "feature": "SO2",
      "importance": 0.032438
    },
    {
      "feature": "O3",
      "importance": 0.027078
    }
  ],
  "training_maxima": {
    "PM2.5": 949.99,
    "PM10": 917.08,
    "NO2": 362.21,
    "CO": 175.81,
    "SO2": 186.08,
    "O3": 257.73
  },
  "metrics": {
    "r2": 0.8367,
    "rmse": 36.46,
    "mae": 20.1,
    "test_samples": 4902
  },
  "feature_importance_note": "Feature importance shows how the trained model uses its inputs; it does not establish that a pollutant causes an AQI outcome.",
  "dataset_summary": {
    "total_raw_rows": 29531,
    "valid_calendar_pairs": 29505,
    "valid_target_samples": 24849,
    "cities_count": 26,
    "train_samples": 19879,
    "test_samples": 4970,
    "train_date_range": {
      "start": "2015-01-01",
      "end": "2019-12-05"
    },
    "test_date_range": {
      "start": "2019-12-05",
      "end": "2020-06-30"
    }
  }
}
```

---

### `GET /example`
Returns the historical reference test example from Visakhapatnam (`2020-06-22`).

**Actual Response**:
```json
{
  "city": "Visakhapatnam",
  "date": "2020-06-22",
  "input_values": {
    "pm25": 33.17,
    "pm10": 108.22,
    "no2": 42.45,
    "co": 0.73,
    "so2": 13.65,
    "o3": 34.85
  },
  "today_aqi": 95.0,
  "actual_next_day_aqi": 100.0,
  "reported_model_prediction": 112.63,
  "reported_absolute_error": 12.63,
  "note": "Historical reference test sample from model evaluation. Not live sensor readings."
}
```

---

### `POST /predict`
Submits current-day pollutant readings and returns the next-day prediction, category, and recommendations.

**Request Payload**:
```json
{
  "pm25": 33.17,
  "pm10": 108.22,
  "no2": 42.45,
  "co": 0.73,
  "so2": 13.65,
  "o3": 34.85
}
```

**Actual Response**:
```json
{
  "predicted_aqi": 112.63,
  "raw_predicted_aqi": 112.63,
  "category": "Moderate",
  "recommendations": [
    "Air quality is acceptable; however, moderate health concern exists for sensitive groups.",
    "Children, older adults, and individuals with respiratory or cardiovascular conditions should reduce prolonged outdoor exertion.",
    "Keep windows closed during peak traffic hours if residing near busy roads."
  ],
  "warnings": [],
  "input_values": {
    "pm25": 33.17,
    "pm10": 108.22,
    "no2": 42.45,
    "co": 0.73,
    "so2": 13.65,
    "o3": 34.85
  },
  "model_name": "RandomForestRegressor"
}
```

---

## 11. Current Limitations & Scope

- **Offline Machine Learning Model**: Uses historical tabular data from `archive/city_day.csv`; there is no live streaming or IoT sensor ingestion.
- **Estimated AQI Regression**: The model predicts AQI as a continuous regression estimate; it does not calculate sub-indices per the official CPCB break-point formula.
- **Decision Support Prototyping**: Recommendations are deterministic, rule-based suggestions intended for student decision support and should not be used as medical or statutory emergency advisories.
- **Stateless API**: No database, user authentication, or session management is implemented in Stage 2.
- **Frontend Dashboard**: A simple placeholder is served at `/`; the full interactive UI will be integrated in Stage 3.
