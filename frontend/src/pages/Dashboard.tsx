import React from 'react';
import {
  ArrowRight,
  TrendingUp,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Database,
  Cpu,
  ShieldCheck,
  BarChart2,
  Wind
} from 'lucide-react';
import type { NavTab, MethodologySection, ModelInfoResponse, ExampleResponse } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

interface DashboardProps {
  onNavigate: (tab: NavTab, section?: MethodologySection) => void;
  modelInfo: ModelInfoResponse | null;
  example: ExampleResponse | null;
}

export const Dashboard: React.FC<DashboardProps> = ({
  onNavigate,
  modelInfo,
  example,
}) => {
  const metrics = modelInfo?.metrics || {
    r2: 0.8367,
    rmse: 36.46,
    mae: 20.10,
    test_samples: 4902,
  };

  const featureImportanceData = modelInfo?.feature_importances
    ? [...modelInfo.feature_importances].sort((a, b) => b.importance - a.importance)
    : [
        { feature: 'PM2.5', importance: 0.46473 },
        { feature: 'CO', importance: 0.40371 },
        { feature: 'PM10', importance: 0.042632 },
        { feature: 'SO2', importance: 0.032438 },
        { feature: 'NO2', importance: 0.029412 },
        { feature: 'O3', importance: 0.027078 },
      ];

  const colors = ['#0d9488', '#0f766e', '#14b8a6', '#2dd4bf', '#5eead4', '#99f6e4'];

  return (
    <div className="space-y-8 pb-12">
      {/* Hero Banner */}
      <section className="bg-gradient-to-r from-slate-900 via-teal-950 to-slate-900 text-white rounded-2xl p-6 sm:p-10 shadow-lg relative overflow-hidden">
        <div className="absolute right-0 top-0 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/20 border border-teal-400/30 text-teal-300 text-xs font-semibold tracking-wide uppercase">
            <Cpu className="w-3.5 h-3.5" />
            <span>Final-Year Environmental Studies Project</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
            AI-Powered Air Quality Prediction &amp; Decision Support System
          </h1>

          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            A machine-learning decision support platform predicting next-calendar-day Air Quality Index (AQI)
            from multi-criteria pollutant readings. Trained on 5 years of historical Indian air quality data to empower
            proactive environmental health mitigation.
          </p>

          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('predict')}
              className="inline-flex items-center gap-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold px-5 py-2.5 rounded-xl shadow-md transition-all hover:gap-3 cursor-pointer"
            >
              <span>Launch AQI Predictor</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('model-methodology', 'performance')}
              className="inline-flex items-center gap-2 bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 font-medium px-4 py-2.5 rounded-xl transition-colors cursor-pointer"
            >
              <BarChart2 className="w-4 h-4 text-teal-400" />
              <span>Model &amp; Methodology</span>
            </button>
          </div>
        </div>
      </section>

      {/* Model Performance KPI Cards */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Model Evaluation Metrics</h2>
            <p className="text-xs text-slate-500">
              Evaluated on the matched chronological test partition (4,902 samples, Dec 2019 – Jun 2020)
            </p>
          </div>
          <span className="text-xs text-teal-700 bg-teal-50 border border-teal-200 font-semibold px-2.5 py-1 rounded-md">
            Baseline Improvement: +8.3% R²
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: R2 */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs hover:border-teal-300 transition-colors">
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">R² Coefficient</span>
              <TrendingUp className="w-4 h-4 text-teal-600" />
            </div>
            <div className="text-3xl font-extrabold text-slate-900">
              {metrics.r2.toFixed(4)}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Persistence baseline: <span className="font-semibold text-slate-700">0.7724</span>
            </p>
          </div>

          {/* Card 2: RMSE */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs hover:border-teal-300 transition-colors">
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">RMSE Error</span>
              <Sliders className="w-4 h-4 text-emerald-600" />
            </div>
            <div className="text-3xl font-extrabold text-slate-900">
              {metrics.rmse.toFixed(2)}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Persistence baseline: <span className="font-semibold text-slate-700">43.05</span> AQI
            </p>
          </div>

          {/* Card 3: MAE */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs hover:border-teal-300 transition-colors">
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">MAE Error</span>
              <ShieldCheck className="w-4 h-4 text-cyan-600" />
            </div>
            <div className="text-3xl font-extrabold text-slate-900">
              {metrics.mae.toFixed(2)}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Average absolute error: ~20.1 AQI points
            </p>
          </div>

          {/* Card 4: Evaluation Samples */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs hover:border-teal-300 transition-colors">
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Test Sample Size</span>
              <Database className="w-4 h-4 text-blue-600" />
            </div>
            <div className="text-3xl font-extrabold text-slate-900">
              {metrics.test_samples.toLocaleString()}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Total historical dataset: <span className="font-semibold text-slate-700">29,531</span> rows
            </p>
          </div>
        </div>
      </section>

      {/* Two Column Layout: Feature Importance & Historical Test Sample */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Feature Importance Preview */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-slate-900">Pollutant Feature Importance</h3>
              <p className="text-xs text-slate-500">
                Random Forest Gini impurity contribution across the 6 criteria pollutants
              </p>
            </div>
            <button
              onClick={() => onNavigate('model-methodology', 'performance')}
              className="text-xs font-semibold text-teal-700 hover:text-teal-900 flex items-center gap-1 cursor-pointer"
            >
              <span>Deep Dive</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImportanceData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <XAxis type="number" domain={[0, 0.5]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} fontSize={11} />
                <YAxis dataKey="feature" type="category" fontSize={12} tickLine={false} />
                <Tooltip
                  formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, 'Importance']}
                  contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                  {featureImportanceData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="text-[11px] text-slate-500 bg-slate-50 p-2.5 rounded-lg border border-slate-100">
            <strong>Key Observation:</strong> Fine particulate matter (<span className="font-medium text-slate-700">PM2.5</span>)
            and carbon monoxide (<span className="font-medium text-slate-700">CO</span>) together account for over 86% of the
            tree-split importance for predicting tomorrow's AQI.
          </p>
        </div>

        {/* Right: Real Test Example Preview */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-teal-800 bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200">
                Verified Historical Test Sample
              </span>
              <span className="text-xs text-slate-400">Visakhapatnam · 2020-06-22</span>
            </div>
            <h3 className="font-bold text-slate-900 text-lg">Benchmark Prediction Sample</h3>
            <p className="text-xs text-slate-500">
              Extracted from the held-out chronological test partition for ground-truth validation.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-2">
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block uppercase font-medium">Today's AQI</span>
              <span className="text-lg font-bold text-slate-800">
                {example ? example.today_aqi.toFixed(0) : '95'}
              </span>
              <span className="text-[10px] text-slate-400 block">Observed</span>
            </div>

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block uppercase font-medium">Actual Next-Day</span>
              <span className="text-lg font-bold text-slate-800">
                {example ? example.actual_next_day_aqi.toFixed(0) : '100'}
              </span>
              <span className="text-[10px] text-slate-400 block">Ground Truth</span>
            </div>

            <div className="bg-teal-50 p-3 rounded-lg border border-teal-200 text-center">
              <span className="text-[10px] text-teal-700 block uppercase font-medium">Model Predicted</span>
              <span className="text-lg font-extrabold text-teal-800">
                {example ? example.reported_model_prediction.toFixed(2) : '112.63'}
              </span>
              <span className="text-[10px] text-teal-600 block">Moderate Band</span>
            </div>

            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block uppercase font-medium">Abs. Error</span>
              <span className="text-lg font-bold text-slate-800">
                {example ? example.reported_absolute_error.toFixed(2) : '12.63'}
              </span>
              <span className="text-[10px] text-slate-400 block">AQI Points</span>
            </div>
          </div>

          <div className="text-xs text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1">
            <div className="font-semibold text-slate-700 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Input Readings for Test Date:</span>
            </div>
            <p className="text-slate-500 text-[11px]">
              PM2.5: 33.17 µg/m³ · PM10: 108.22 µg/m³ · NO2: 42.45 µg/m³ · CO: 0.73 mg/m³ · SO2: 13.65 µg/m³ · O3: 34.85 µg/m³
            </p>
          </div>

          <div className="pt-2 flex justify-end">
            <button
              onClick={() => onNavigate('model-methodology', 'example')}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 hover:text-teal-900 cursor-pointer"
            >
              <span>View Full Historical Evaluation Details</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </section>

      {/* 5-Step System Workflow Overview */}
      <section className="bg-white p-6 sm:p-8 rounded-xl border border-slate-200 shadow-xs space-y-6">
        <div>
          <h2 className="text-lg font-bold text-slate-900">System Architecture &amp; Prediction Pipeline</h2>
          <p className="text-xs text-slate-500">
            End-to-end dataflow from criteria pollutant observations to deterministic health advisories
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative">
          {[
            {
              step: '1',
              title: 'Multi-Pollutant Input',
              desc: 'Collect current-day concentrations for 6 pollutants (PM2.5, PM10, NO2, CO, SO2, O3).',
              icon: Wind,
            },
            {
              step: '2',
              title: 'Bound Validation',
              desc: 'Pydantic verifies non-negativity and alerts if readings exceed training data maxima.',
              icon: Sliders,
            },
            {
              step: '3',
              title: 'Median Imputation',
              desc: 'Missing inputs are imputed using learned training medians to prevent temporal leakage.',
              icon: Database,
            },
            {
              step: '4',
              title: 'Random Forest',
              desc: 'Ensemble of 100 decision trees estimates next-calendar-day continuous AQI.',
              icon: Cpu,
            },
            {
              step: '5',
              title: 'Decision Support',
              desc: 'Maps estimate to prototype category bands and returns rule-based mitigation guidance.',
              icon: ShieldCheck,
            },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.step}
                className="bg-slate-50/70 border border-slate-200/80 rounded-xl p-4 flex flex-col justify-between space-y-3 relative group hover:border-teal-400 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center">
                    {item.step}
                  </span>
                  <Icon className="w-5 h-5 text-slate-400 group-hover:text-teal-600 transition-colors" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900 mb-1">{item.title}</h4>
                  <p className="text-[11px] text-slate-500 leading-normal">{item.desc}</p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Disclaimer Callout */}
        <div className="bg-amber-50/80 border border-amber-200 p-4 rounded-xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-900 leading-relaxed">
            <strong>Important Academic Prototype Disclaimer:</strong> This application predicts future AQI as a continuous regression
            estimate trained on historical CPCB records. Category bands are applied to this estimate for decision support. This does not
            represent an official regulatory CPCB calculation from real-time sub-indices, nor does it provide personalized medical advice.
          </div>
        </div>
      </section>
    </div>
  );
};
