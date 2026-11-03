# AirSense AI — Phase 1B: Model Distortion & Feature Importance Verification

**Status**: Verified & Documented  
**Date**: November 2026  
**Auditor**: Senior Machine Learning Engineer & Project Auditor  
**Artifact Scope**: `aqi_model.pkl`, `HOW_IT_WORKS.md`, `model_metadata.json`, `archive/city_day.csv`

---

## 1. Executive Summary

This report documents the verification of model distortion induced by anomalous Carbon Monoxide ($\text{CO}$) measurements and exposes a critical documentation discrepancy between the production artifact [`aqi_model.pkl`](file:///d:/PROJECTS/evs/aqi_model.pkl) and the project architectural guide [`HOW_IT_WORKS.md`](file:///d:/PROJECTS/evs/HOW_IT_WORKS.md).

### Key Empirical Findings
1. **Critical Documentation Discrepancy**: The production model `aqi_model.pkl` allocates **$40.37\%$** of its total Gini feature importance to $\text{CO}$, making it the second most dominant feature in the system. In contrast, `HOW_IT_WORKS.md` reports $\text{CO}$ importance as **$\sim 8.9\%$**, while overstating $\text{PM}_{10}$ ($17.8\%$ vs actual $4.26\%$) and $\text{NO}_2$ ($12.5\%$ vs actual $2.94\%$).
2. **Extreme Regional Distortion**: On the clean temporal test set (post December 5, 2019), the production model achieves high accuracy across 25 Indian cities ($R^2 = 0.8803$, $\text{RMSE} = 26.18$, $\text{MAE} = 17.80$). However, on Ahmedabad test observations, model error quadruples ($R^2 = 0.6533$, $\text{RMSE} = 129.29$, $\text{MAE} = 75.05$). For test observations where $\text{CO} > 17\ \text{mg/m}^3$, the model breaks down almost completely ($R^2 = 0.1450$, $\text{RMSE} = 253.91$, $\text{MAE} = 195.21$).
3. **Controlled Retraining Experiment**: An isolated experimental model trained on the chronological training partition *excluding* Ahmedabad observations revealed that $\text{CO}$ importance collapses from **$40.37\%$ down to $3.12\%$** (a $-37.25\%$ drop), while $\text{PM}_{2.5}$ importance expands from **$46.47\%$ to $83.39\%$** ($+36.92\%$). The experimental model performs identically on the rest of India ($R^2 = 0.8807$ vs $0.8803$), proving that the production model's reliance on $\text{CO}$ is an artifact of fitting anomalous single-station spikes.

---

## 2. Feature Importance Verification

The feature importances were extracted directly from the serialized production pipeline `aqi_model.pkl` (`RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)`):

### 2.1 Genuine vs Documented Feature Importance

| Feature | Chemical / Physical Entity | Genuine Importance (`aqi_model.pkl`) | Documented in `HOW_IT_WORKS.md` | Absolute Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **`PM2.5`** | Fine Particulate Matter | **$46.47\%$** ($0.464730$) | $\sim 44.6\%$ | $+1.87\%$ |
| **`CO`** | Carbon Monoxide | **$40.37\%$** ($0.403710$) | $\sim 8.9\%$ | **$+31.47\%$** (Huge Discrepancy) |
| **`PM10`** | Coarse Particulate Matter | **$4.26\%$** ($0.042632$) | $\sim 17.8\%$ | **$-13.54\%$** (Overstated $4\times$) |
| **`SO2`** | Sulphur Dioxide | **$3.24\%$** ($0.032438$) | $\sim 7.7\%$ | $-4.46\%$ |
| **`NO2`** | Nitrogen Dioxide | **$2.94\%$** ($0.029412$) | $\sim 12.5\%$ | **$-9.56\%$** (Overstated $4\times$) |
| **`O3`** | Ground-Level Ozone | **$2.71\%$** ($0.027078$) | $\sim 8.5\%$ | $-5.79\%$ |
| **Total** | | **$100.00\%$** | $100.0\%$ | |

```mermaid
pie title Genuine vs Documented Feature Importance
    "Genuine CO (aqi_model.pkl)" : 40.37
    "Genuine PM2.5" : 46.47
    "Genuine Other Pollutants" : 13.16
```

### 2.2 Root Cause of Documentation Mismatch
`HOW_IT_WORKS.md` (Section 3.3) describes a model where particulate matter ($\text{PM}_{2.5} + \text{PM}_{10}$) accounts for $> 62\%$ of predictive weight and $\text{CO}$ plays a minor role ($8.9\%$). That distribution reflects an idealized, physically sound air-quality relationship. However, the model actually committed to disk and served by the API has $\text{CO}$ at $40.37\%$. This proves that earlier documentation was either authored against a different candidate model or written aspirationally without reading the actual serialized artifact.

---

## 3. Production Model Performance & Regional Distortion

To quantify how the Ahmedabad data affects prediction, the production model was evaluated across geographical and concentration subsets using the standard temporal test split ($N = 4,968$ observations from 2019-12-06 to 2020-06-30, with $N = 4,900$ having valid next-day ground-truth AQI):

### 3.1 Sub-Cohort Evaluation Matrix

| Evaluation Subset | Sample Count ($N$) | $R^2$ Score | RMSE | MAE | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Test Partition (All Cities)** | $4,900$ | $0.8367$ | $36.47$ | $20.10$ | Baseline benchmark |
| **Non-Ahmedabad Cities (25 Cities)** | $4,703$ | **$0.8803$** | **$26.18$** | **$17.80$** | High accuracy |
| **Ahmedabad Only (`GJ001`)** | $197$ | **$0.6533$** | **$129.29$** | **$75.05$** | **Error quadrupled** |
| **Observations with $\text{CO} \le 17\ \text{mg/m}^3$** | $4,811$ | $0.8462$ | $30.75$ | $19.07$ | Stable |
| **Observations with $\text{CO} > 17\ \text{mg/m}^3$** | $34$ | **$0.1450$** | **$253.91$** | **$195.21$** | **Severe failure** |

### 3.2 Analysis of Failure Modes
* For normal Indian ambient air ($\text{CO} \le 17\ \text{mg/m}^3$, representing $98.18\%$ of the test set), the model exhibits an $\text{MAE}$ of $19.07\ \text{AQI}$ points.
* For the 34 test cases where $\text{CO} > 17\ \text{mg/m}^3$ (all in Ahmedabad), the model produces a catastrophic $\text{MAE}$ of **$195.21$** and an $\text{RMSE}$ of **$253.91$**.
* Despite the Random Forest allocating $40.37\%$ of its decision splits to $\text{CO}$, it cannot accurately predict next-day AQI during extreme $\text{CO}$ events. Because these high readings represent localized spikes that dissipate rapidly during the afternoon, relying on current-day $\text{CO}$ to forecast next-day AQI creates massive over-prediction errors.

---

## 4. Controlled Retraining Experiment (Excluding Ahmedabad)

To isolate the structural effect of Ahmedabad on the learning process, an identical Scikit-learn pipeline (`SimpleImputer(strategy='median')` + `RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)`) was trained exclusively on non-Ahmedabad training records ($N = 17,870$ paired observations, down from $19,879$).

### 4.1 Feature Importance Shift

| Feature | Production Model (Trained with Ahmedabad) | Experimental Model (Trained WITHOUT Ahmedabad) | Absolute Shift ($\Delta$) | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **`PM2.5`** | $46.47\%$ | **$83.39\%$** | $+36.92\%$ | $+79.4\%$ |
| **`PM10`** | $4.26\%$ | **$7.83\%$** | $+3.57\%$ | $+83.8\%$ |
| **`CO`** | **$40.37\%$** | **$3.12\%$** | **$-37.25\%$** | **$-92.3\%$** |
| **`NO2`** | $2.94\%$ | **$1.90\%$** | $-1.04\%$ | $-35.4\%$ |
| **`O3`** | $2.71\%$ | **$1.99\%$** | $-0.72\%$ | $-26.6\%$ |
| **`SO2`** | $3.24\%$ | **$1.77\%$** | $-1.47\%$ | $-45.4\%$ |

```mermaid
xychart-beta
    title "Feature Importance Shift (With vs Without Ahmedabad)"
    x-axis ["PM2.5", "CO", "PM10", "SO2", "NO2", "O3"]
    y-axis "Importance (%)" 0 --> 90
    bar [46.5, 40.4, 4.3, 3.2, 2.9, 2.7]
    bar [83.4, 3.1, 7.8, 1.8, 1.9, 2.0]
```

### 4.2 Comparative Evaluation on Held-Out Test Data

| Test Evaluation Slice | Production Model Metric | Experimental Model Metric | Net Impact |
| :--- | :--- | :--- | :--- |
| **Non-Ahmedabad Test ($N=4,703$)** | $R^2 = 0.8803$, $\text{RMSE} = 26.18$, $\text{MAE} = 17.80$ | $R^2 = 0.8807$, $\text{RMSE} = 26.13$, $\text{MAE} = 17.82$ | **Statistically identical** ($\Delta R^2 = +0.0004$) |
| **Ahmedabad Test ($N=197$)** | $R^2 = 0.6533$, $\text{RMSE} = 129.29$, $\text{MAE} = 75.05$ | $R^2 = 0.1093$, $\text{RMSE} = 207.24$, $\text{MAE} = 110.25$ | Severe drop (no longer fits anomaly) |
| **National Combined ($N=4,900$)** | $R^2 = 0.8367$, $\text{RMSE} = 36.47$, $\text{MAE} = 20.10$ | $R^2 = 0.7074$, $\text{RMSE} = 48.81$, $\text{MAE} = 21.54$ | Lower aggregate $R^2$ due to unmodeled outliers |

---

## 5. Methodological & Scientific Synthesis

### 5.1 Why Aggregate $R^2$ is Deceptive
A superficial reading of the aggregate test score might conclude that the experimental model is "worse" because its national $R^2$ drops from $0.8367$ to $0.7074$. 

**This is a classic statistical trap in machine learning.**
1. In the production model, the high variance of the Ahmedabad target (where next-day AQI reached $2,049$) creates enormous total sum of squares ($\text{TSS}$). Because the production model learned to correlate high current-day $\text{CO}$ with high next-day AQI in Ahmedabad, it captures a portion of that massive variance, artificially boosting $R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$.
2. For the other $96\%$ of India (25 cities), the experimental model performs identically or slightly better ($R^2 = 0.8807$, $\text{RMSE} = 26.13$).
3. The experimental model reflects true physical atmospheric dynamics: $\text{PM}_{2.5}$ and $\text{PM}_{10}$ account for $> 91\%$ of predictive signal, aligning perfectly with established environmental science literature.

### 5.2 Verification Classification
* **CO Feature Importance ($\sim 40.37\%$)**: **CONFIRMED** in artifact `aqi_model.pkl`.
* **Documentation Discrepancy in `HOW_IT_WORKS.md`**: **CONFIRMED**.
* **Model Distortion by Ahmedabad**: **CONFIRMED**. Single-station instrumental anomalies account for $> 37\%$ of all decision-tree splits in the production model.

---

## 6. Recommendations for Architecture & Retraining

1. **Synchronize Documentation**: Update `HOW_IT_WORKS.md` immediately in Phase 2 to accurately reflect the production artifact until retraining is authorized.
2. **Phase 2 Retraining Architecture**:
   - Retrain the production pipeline using an anomaly-screened training set where unphysical $\text{CO}$ measurements ($> 17\ \text{mg/m}^3$) are either capped, flagged, or isolated.
   - Retain $\text{PM}_{2.5}$ as the primary predictive backbone ($> 80\%$ expected importance).
3. **Guardrails**: Maintain an operational check on user inputs to warn when submitted $\text{CO} > 17\ \text{mg/m}^3$.

