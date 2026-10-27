# AirSense AI — Frontend Dashboard

React + Vite + TypeScript frontend for the **AI-Powered Air Quality Prediction & Decision Support System** (Environmental Studies Final-Year Capstone Project).

---

## 1. Overview & Features

- **Dashboard**: Real-time FastAPI connectivity status, core model performance metrics (R²: 0.8367, RMSE: 36.46, MAE: 20.10, N=4,902), feature-importance preview, benchmark case study card, and system architecture workflow.
- **AQI Predictor**: Interactive input form for six criteria pollutants (`PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`) with inline validation, range warnings, demonstration preset loader, visual AQI scale meter (0–500+), category badges, and rule-based decision support advisories.
- **Model Insights**: In-depth analytics including Gini impurity feature importances visualized via Recharts, observed training partition maxima, and chronological dataset partitioning details.
- **Historical Example**: Ground-truth benchmark evaluation study from Visakhapatnam (`2020-06-22` -> `2020-06-23`), displaying observed AQI (95.0), actual next-day AQI (100.0), model prediction (112.63), and absolute error (12.63).
- **About & Methodology**: Academic context, pipeline workflow, comparison between empirical regression prediction and official CPCB sub-index breakpoint calculation, and statutory disclaimers.

---

## 2. Technology Stack

- **Framework**: React 19 + Vite 6
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Data Visualization**: Recharts
- **HTTP Client**: Native Fetch with AbortController timeout handling

---

## 3. Getting Started

### Prerequisites
- Node.js v18+ (tested on Node v22)
- Backend running at `http://localhost:8000`

### Installation
From the `frontend/` directory:

```bash
npm install
```

### Configuration
The backend API URL defaults to `http://localhost:8000`. You can configure it via `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

### Start Development Server
```bash
npm run dev
```

The application will be available at [http://localhost:5173](http://localhost:5173).

### Production Build & Type Check
```bash
npm run build
```

---

## 4. Academic Disclaimers

> **CPCB Formula vs. Regression Estimate**:
> The model outputs an estimated AQI by regression. The specified AQI bands are used only to label that estimate for this prototype; this is not a direct CPCB AQI calculation from pollutant sub-indices.
>
> **Statutory & Medical Notice**:
> This prototype is for educational decision-support purposes only and does not provide statutory regulatory certification or personalized medical advice.
