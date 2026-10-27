import React from 'react';
import {
  Info,
  Layers,
  ShieldAlert,
  BookOpen,
  School
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-8 pb-16 max-w-5xl">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-teal-700 text-xs font-semibold uppercase tracking-wider mb-1">
          <Info className="w-4 h-4" />
          <span>Documentation &amp; Context</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">
          About the Project &amp; Scientific Methodology
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Academic context, machine-learning pipeline architecture, and environmental decision-support methodology.
        </p>
      </div>

      {/* Project Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-5">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-100">
          <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
            <School className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Academic Project Overview</span>
            <h2 className="text-lg font-bold text-slate-900">
              AI-Powered Air Quality Prediction &amp; Decision Support System
            </h2>
          </div>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">
          This system was developed as a final-year Environmental Studies (EVS) capstone project. Rapid industrialization,
          vehicular emissions, and seasonal agricultural burning contribute to severe ambient air pollution across Indian urban
          corridors. Conventional air monitoring systems report historical or current AQI reactively. This project investigates
          the feasibility of training a machine learning regression pipeline to predict next-calendar-day AQI using multi-pollutant
          sensor readings, enabling municipal authorities and sensitive populations to take proactive mitigation measures.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Target Phenomenon</span>
            <span className="text-sm font-bold text-slate-800 block mt-1">Next-Calendar-Day AQI</span>
            <p className="text-[11px] text-slate-500 mt-1">Predicts 24-hour forward ambient air quality</p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Input Regressors</span>
            <span className="text-sm font-bold text-slate-800 block mt-1">6 Criteria Pollutants</span>
            <p className="text-[11px] text-slate-500 mt-1">PM2.5, PM10, NO2, CO, SO2, O3</p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Core Estimator</span>
            <span className="text-sm font-bold text-slate-800 block mt-1">Random Forest Pipeline</span>
            <p className="text-[11px] text-slate-500 mt-1">100 trees with training-fit median imputer</p>
          </div>
        </div>
      </div>

      {/* Methodology Section */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
        <div className="flex items-center gap-2.5 pb-2">
          <Layers className="w-5 h-5 text-teal-600" />
          <h2 className="text-lg font-bold text-slate-900">Seven-Stage Pipeline Architecture</h2>
        </div>

        <div className="space-y-4 text-sm text-slate-600">
          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              1
            </span>
            <div>
              <strong className="text-slate-800 block">Dataset Ingestion &amp; Consecutive Calendar Day Pairing</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Ingested 29,531 daily records across 26 Indian cities from the Central Pollution Control Board (CPCB) archive
                (2015–2020). Observations were sorted strictly by city and date. Next-day targets were paired only when the
                consecutive date difference was exactly 1 calendar day, ensuring temporal validity and yielding 24,849 valid training pairs.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              2
            </span>
            <div>
              <strong className="text-slate-800 block">Chronological Train/Test Partitioning</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Random shuffling was strictly avoided to prevent temporal leakage. The dataset was partitioned chronologically (80% train:
                19,879 records from 2015-01-01 to 2019-12-05; 20% test: 4,970 records from 2019-12-05 to 2020-06-30).
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              3
            </span>
            <div>
              <strong className="text-slate-800 block">Leakage-Free Preprocessing &amp; Median Imputation</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Criteria pollutants (especially PM10, which was introduced later in certain stations) contain missing sensor values.
                A <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">SimpleImputer(strategy='median')</code> was fitted strictly
                on the training partition, ensuring zero statistical test leakage.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              4
            </span>
            <div>
              <strong className="text-slate-800 block">Random Forest Regression Training</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Trained an ensemble of 100 decision trees (<code className="text-xs bg-slate-100 px-1 py-0.5 rounded">min_samples_leaf=2</code>)
                yielding an out-of-sample R² of 0.8367, RMSE of 36.46, and MAE of 20.10. This significantly outperformed the
                meteorological persistence baseline (R² = 0.7724, RMSE = 43.05).
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              5
            </span>
            <div>
              <strong className="text-slate-800 block">Decision Support &amp; Bound Verification Engine</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Maps numeric AQI predictions into six prototype category bands (Good, Satisfactory, Moderate, Poor, Very Poor, Severe),
                clamps negative regression estimates, alerts if inputs exceed historical training maxima, and generates actionable public
                health advisories.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              6
            </span>
            <div>
              <strong className="text-slate-800 block">FastAPI Asynchronous Microservice</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Lightweight Python service loading the serialized pipeline once at startup with CORS, Pydantic v2 input validation,
                and RESTful endpoints for health check, prediction, model insights, and benchmark validation.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3.5">
            <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              7
            </span>
            <div>
              <strong className="text-slate-800 block">Modern React + Vite Frontend Dashboard</strong>
              <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                Interactive web client with Tailwind CSS styling, Recharts visualizations, accessible category badges, inline input validation,
                and live backend connectivity indicators.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Crucial Technical Distinction */}
      <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 space-y-4 shadow-sm">
        <div className="flex items-center gap-2.5 text-teal-400">
          <BookOpen className="w-5 h-5" />
          <h2 className="text-base font-bold">Crucial Distinction: Regression Estimate vs. CPCB AQI Calculation</h2>
        </div>

        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          The Central Pollution Control Board (CPCB) calculates the official National Air Quality Index (NAQI) using a
          deterministic sub-index formula: each pollutant's 24-hour average concentration is mapped through piecewise linear
          breakpoints, and the overall AQI is taken as the maximum sub-index (provided minimum monitoring criteria are met).
        </p>

        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          In contrast, <strong>AirSense AI is a predictive forecasting system</strong>. It does not calculate same-day AQI from
          piecewise breakpoint tables; rather, it trains an empirical machine learning regression model to forecast the
          <em>next-calendar-day</em> overall AQI value based on the synergistic pattern of today's six criteria pollutants. The prototype
          bands are used strictly to categorize this numeric estimate for intuitive decision support.
        </p>
      </div>

      {/* Disclaimers */}
      <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 space-y-3">
        <div className="flex items-center gap-2 text-amber-900 font-bold text-sm">
          <ShieldAlert className="w-5 h-5 text-amber-600" />
          <span>Statutory &amp; Medical Disclaimer</span>
        </div>
        <p className="text-xs text-amber-900 leading-relaxed">
          This system is an academic research demonstration developed for an undergraduate Environmental Studies curriculum.
          Predictions are statistical approximations based on historical data patterns and should not be used as official statutory
          certifications, regulatory compliance audits, or clinical medical advice. Individuals experiencing acute respiratory or
          cardiovascular symptoms must consult certified healthcare professionals or refer to official government bulletins.
        </p>
      </div>
    </div>
  );
};
