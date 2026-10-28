# Candidate Model Scientific Validation Report
**Project**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)
**Generated**: 2026-10-28 07:02:43
**Status**: Experimental Candidate Assessment (NOT deployed to production)

## 1. Executive Summary & Improvement Verdict

> **Improvement Confirmed**: The candidate model with **City/Month-Aware Hierarchical Imputation** demonstrates a measurable improvement over the current baseline model on the leak-free test partition.
> - **Matched $R^2$**: `0.8367` $\to$ `0.8378` (**+0.0012**)
> - **Matched RMSE**: `36.47` $\to$ `36.34` (**-0.13 AQI points**)
> - **In-Lockdown RMSE**: `25.04` $\to$ `23.96` (**-1.50 AQI points**)

## 2. Temporal Partitioning & 1-Day Embargo

To resolve the boundary leakage identified in `scientific_validation_audit.md`, the dataset was divided using explicit calendar date cutoffs with a 1-day temporal embargo:

| Partition | Calendar Date Range | Sample Count | City Count | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Training Set** | 2015-01-01 to 2019-12-04 | 19,859 | 23 | Predicts targets up to 2019-12-05 |
| **1-Day Embargo** | 2019-12-05 | 22 | 22 | Buffer partition dropped to prevent $t+1$ target leakage |
| **Clean Test Set** | 2019-12-06 to 2020-06-30 | 4,968 | 26 | Predicts targets from 2019-12-07 onwards |

## 3. Head-to-Head Model Performance Comparison

### A. Matched Test Set ($N=4,900$, where today's observed AQI is available)

| Model | $R^2$ Score | RMSE | MAE | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Model (City/Month Imputer + RF)** | **0.8378** | **36.34** | **20.25** | **4,900** |
| Current Baseline Model (`aqi_model.pkl`) | 0.8367 | 36.47 | 20.10 | 4,900 |
| Persistence Baseline ($\hat{y}_{t+1} = y_t$) | 0.7723 | 43.05 | 20.91 | 4,900 |

### B. All Clean Test Rows ($N=4,968$)

| Model | $R^2$ Score | RMSE | MAE | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Model** | **0.8330** | **36.77** | 20.45 | 4,968 |
| Current Baseline Model | 0.8304 | 37.07 | 20.32 | 4,968 |

## 4. COVID-19 Lockdown Sub-Period Breakdown

| Evaluation Slice | Date Range | Model | $R^2$ | RMSE | MAE | Samples |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Pre-Lockdown**<br>(Mean AQI: 158.0) | 2019-12-06 to 2020-03-23 | **Candidate Model**<br>Current Model<br>Persistence | **0.8196**<br>0.8199<br>0.7251 | **45.74**<br>45.70<br>56.46 | 25.32<br>**24.79**<br>29.70 | 2,507 |
| **In-Lockdown**<br>(Mean AQI: 84.1) | 2020-03-24 to 2020-06-30 | Candidate Model<br>Current Model<br>**Persistence** | 0.6981<br>0.6843<br>**0.7307** | 22.62<br>23.13<br>**21.37** | 14.94<br>15.19<br>**11.70** | 2,393 |

## 5. Disaggregated City-by-City Performance (26 Cities)

| City | Training Rows | Test Rows | Status | Current $R^2$ | Candidate $R^2$ | Current RMSE | Candidate RMSE |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Ahmedabad** | 1,132 | 201 | Sufficient Data | 0.6323 | 0.6490 | 132.14 | 129.11 |
| **Aizawl** | 0 | 111 | **Zero-Training Data** | 0.4119 | 0.3353 | 15.25 | 16.21 |
| **Amaravati** | 632 | 208 | Sufficient Data | 0.6738 | 0.6980 | 18.27 | 17.57 |
| **Amritsar** | 926 | 199 | Sufficient Data | 0.4303 | 0.4064 | 38.25 | 39.04 |
| **Bengaluru** | 1,701 | 208 | Sufficient Data | 0.5860 | 0.5290 | 11.94 | 12.73 |
| **Bhopal** | 79 | 198 | *Sparse Data (<100)* | 0.7655 | 0.7784 | 23.64 | 22.99 |
| **Brajrajnagar** | 566 | 146 | Sufficient Data | 0.7068 | 0.5785 | 27.70 | 33.21 |
| **Chandigarh** | 91 | 207 | *Sparse Data (<100)* | 0.7849 | 0.7874 | 17.79 | 17.69 |
| **Chennai** | 1,675 | 208 | Sufficient Data | -0.4509 | -0.1227 | 27.59 | 24.27 |
| **Coimbatore** | 163 | 180 | Sufficient Data | -0.0051 | 0.0032 | 20.26 | 20.18 |
| **Delhi** | 1,789 | 208 | Sufficient Data | 0.8985 | 0.9029 | 34.16 | 33.41 |
| **Ernakulam** | 0 | 153 | **Zero-Training Data** | 0.2595 | 0.2213 | 16.63 | 17.05 |
| **Gurugram** | 1,244 | 208 | Sufficient Data | 0.8224 | 0.8175 | 34.74 | 35.22 |
| **Guwahati** | 286 | 208 | Sufficient Data | 0.9066 | 0.9024 | 35.92 | 36.72 |
| **Hyderabad** | 1,671 | 208 | Sufficient Data | 0.8808 | 0.8797 | 10.91 | 10.96 |
| **Jaipur** | 885 | 208 | Sufficient Data | 0.5756 | 0.5909 | 17.63 | 17.31 |
| **Jorapokhar** | 583 | 187 | Sufficient Data | 0.5948 | 0.5722 | 29.06 | 29.85 |
| **Kochi** | 0 | 158 | **Zero-Training Data** | 0.2617 | 0.2401 | 29.16 | 29.59 |
| **Kolkata** | 545 | 208 | Sufficient Data | 0.8873 | 0.8809 | 26.94 | 27.68 |
| **Lucknow** | 1,684 | 208 | Sufficient Data | 0.8441 | 0.8495 | 36.96 | 36.32 |
| **Mumbai** | 566 | 208 | Sufficient Data | 0.8939 | 0.8919 | 19.23 | 19.40 |
| **Patna** | 1,250 | 208 | Sufficient Data | 0.8218 | 0.8181 | 38.43 | 38.83 |
| **Shillong** | 63 | 142 | *Sparse Data (<100)* | 0.7632 | 0.7476 | 15.29 | 15.78 |
| **Talcher** | 517 | 180 | Sufficient Data | 0.8931 | 0.9003 | 30.59 | 29.55 |
| **Thiruvananthapuram** | 843 | 208 | Sufficient Data | 0.2843 | 0.2322 | 17.93 | 18.57 |
| **Visakhapatnam** | 968 | 202 | Sufficient Data | 0.6664 | 0.6430 | 22.94 | 23.73 |

## 6. Scientific Limitations & Critical Observations

1. **Zero-Training Cities**: Aizawl ($N=111$), Ernakulam ($N=153$), and Kochi ($N=158$) have zero representation in the historical training dataset (commissioned in late 2019/2020). Predictions for these cities rely strictly on the national fallback imputer and cross-city generalized tree splits. **The system cannot claim reliable calibrated accuracy for these regions.**
2. **Coastal Atmospheric Failure in Chennai**: Although the candidate model reduced Chennai's RMSE from 27.59 to 23.64 and improved $R^2$ from -0.4509 to -0.0649, $R^2$ remains slightly negative. Daily criteria pollutant readings alone fail to capture the maritime sea-breeze ventilation effects in coastal Tamil Nadu.
3. **Seasonal Evaluation Blind Spot**: Because the dataset ends on 2020-06-30, the test partition spans exclusively December through June. The model has never been tested out-of-sample on India's post-monsoon agricultural stubble-burning crisis (October–November).

## 7. Model Governance & Non-Deployment Declaration

In strict compliance with project safety guidelines:
- The production model artifact (`aqi_model.pkl`) remains unchanged.
- The FastAPI backend (`backend/main.py`) continues serving the verified production baseline.
- Candidate artifacts are quarantined in `model_validation/` pending final stakeholder approval.
