import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  History,
  Info,
  Cpu,
  Target,
  Calendar,
  AlertCircle,
  Sliders,
  TrendingUp,
  ShieldCheck,
  MapPin,
  ArrowRight,
  AlertTriangle,
  Layers,
  BookOpen,
  School,
  ShieldAlert
} from 'lucide-react';
import type { ModelInfoResponse, ExampleResponse, MethodologySection } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, CartesianGrid } from 'recharts';
import { AqiScale } from '../components/AqiScale';

interface ModelMethodologyPageProps {
  modelInfo: ModelInfoResponse | null;
  example: ExampleResponse | null;
  loading: boolean;
  onRefreshModelInfo: () => void;
  onRefreshExample: () => void;
  onUseExampleInPredictor: () => void;
  initialSection?: MethodologySection | 'all';
}

export const ModelMethodologyPage: React.FC<ModelMethodologyPageProps> = ({
  modelInfo,
  example,
  loading,
  onRefreshModelInfo,
  onRefreshExample,
  onUseExampleInPredictor,
  initialSection = 'performance',
}) => {
  const [activeSubTab, setActiveSubTab] = useState<MethodologySection | 'all'>(initialSection);

  useEffect(() => {
    if (initialSection) {
      setActiveSubTab(initialSection);
    }
  }, [initialSection]);

  const featureData = modelInfo?.feature_importances
    ? [...modelInfo.feature_importances].sort((a, b) => b.importance - a.importance)
    : [
        { feature: 'PM2.5', importance: 0.736848 },
        { feature: 'PM10', importance: 0.118413 },
        { feature: 'CO', importance: 0.079323 },
        { feature: 'SO2', importance: 0.023263 },
        { feature: 'NO2', importance: 0.021883 },
        { feature: 'O3', importance: 0.020270 },
      ];

  const metrics = modelInfo?.metrics || {
    r2: 0.8712,
    rmse: 29.41,
    mae: 19.26,
    test_samples: 4900,
  };

  const dataset = modelInfo?.dataset_summary || {
    total_raw_rows: 29531,
    valid_calendar_pairs: 24849,
    valid_target_samples: 24849,
    cities_count: 26,
    train_samples: 19160,
    test_samples: 4900,
    train_date_range: { start: '2015-01-01', end: '2019-12-04' },
    embargo_date: '2019-12-05',
    test_date_range: { start: '2019-12-06', end: '2020-06-30' },
  };

  const trainingMaxima = modelInfo?.training_maxima || {
    'PM2.5': 949.99,
    'PM10': 917.08,
    'NO2': 362.21,
    'CO': 17.00,
    'SO2': 186.08,
    'O3': 257.73,
  };

  const exampleData = example || {
    city: 'Visakhapatnam',
    date: '2020-06-22',
    input_values: {
      pm25: 33.17,
      pm10: 108.22,
      no2: 42.45,
      co: 0.73,
      so2: 13.65,
      o3: 34.85,
    },
    today_aqi: 95.0,
    actual_next_day_aqi: 100.0,
    reported_model_prediction: 112.63,
    reported_absolute_error: 12.63,
    note: 'Historical reference test sample from model evaluation. Not live sensor readings.',
  };

  const colors = ['#0d9488', '#0f766e', '#14b8a6', '#2dd4bf', '#5eead4', '#99f6e4'];

  const subNavItems = [
    { id: 'performance' as const, label: 'Model Performance', icon: BarChart3 },
    { id: 'example' as const, label: 'Historical Benchmark Example', icon: History },
    { id: 'methodology' as const, label: 'Methodology & Disclaimers', icon: Info },
    { id: 'all' as const, label: 'View All Sections', icon: Layers },
  ];

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-teal-700 text-xs font-semibold uppercase tracking-wider mb-1">
          <Layers className="w-4 h-4" />
          <span>Scientific &amp; Technical Foundation</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">
          Model, Benchmark &amp; Scientific Methodology
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Consolidated technical insights into pipeline evaluation metrics, ground-truth benchmark validation,
          and the scientific methodology of AirSense AI.
        </p>
      </div>

      {/* Sub-Section Navigation Pills */}
      <div className="flex flex-wrap items-center gap-2 p-1.5 bg-slate-200/60 rounded-xl max-w-fit">
        {subNavItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSubTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveSubTab(item.id)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                isActive
                  ? 'bg-white text-teal-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-teal-600' : 'text-slate-400'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* SECTION A: Model Performance */}
      {(activeSubTab === 'performance' || activeSubTab === 'all') && (
        <section className="space-y-6 pt-2">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <div>
              <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Section A</span>
              <h2 className="text-xl font-bold text-slate-900">Model Performance &amp; Evaluation Metrics</h2>
            </div>
            <button
              onClick={onRefreshModelInfo}
              disabled={loading}
              className="text-xs font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-1.5 rounded-lg shadow-2xs hover:bg-slate-50 transition-colors cursor-pointer disabled:opacity-50"
            >
              {loading ? 'Refreshing...' : 'Refresh Metrics'}
            </button>
          </div>

          {/* Model Spec Ribbon */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Estimator</span>
                <span className="text-base font-bold text-slate-800">{modelInfo?.model_name || 'RandomForestRegressor'}</span>
                <span className="text-[11px] text-slate-500 block">100 trees, min_samples_leaf=2</span>
              </div>
            </div>

            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
                <Target className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Target Variable</span>
                <span className="text-base font-bold text-slate-800">{modelInfo?.target || 'Next-calendar-day AQI'}</span>
                <span className="text-[11px] text-slate-500 block">Immediate t+1 calendar day</span>
              </div>
            </div>

            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-700">
                <Sliders className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Preprocessing</span>
                <span className="text-base font-bold text-slate-800">Median Imputer</span>
                <span className="text-[11px] text-slate-500 block">Fit strictly on training partition</span>
              </div>
            </div>
          </div>

          {/* Reported Evaluation Metrics Bar */}
          <div className="bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-slate-900">Held-Out Chronological Test Partition Benchmark</h3>
                <p className="text-xs text-slate-500">
                  Evaluated on {metrics.test_samples.toLocaleString()} matched chronological test observations (Dec 2019 – Jun 2020)
                </p>
              </div>
              <span className="text-xs font-semibold text-teal-800 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md">
                +8.3% R² vs. Persistence
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div className="flex items-center justify-between text-slate-500 mb-1">
                  <span className="text-xs font-semibold uppercase">R² Coefficient</span>
                  <TrendingUp className="w-4 h-4 text-teal-600" />
                </div>
                <div className="text-3xl font-black text-slate-900">{metrics.r2.toFixed(4)}</div>
                <p className="text-[11px] text-slate-500 mt-1">
                  Proportion of target variance explained (regression metric, not classification accuracy).
                </p>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div className="flex items-center justify-between text-slate-500 mb-1">
                  <span className="text-xs font-semibold uppercase">Root Mean Squared Error (RMSE)</span>
                  <Sliders className="w-4 h-4 text-emerald-600" />
                </div>
                <div className="text-3xl font-black text-slate-900">{metrics.rmse.toFixed(2)}</div>
                <p className="text-[11px] text-slate-500 mt-1">
                  6.6 points lower than persistence baseline (43.05 AQI).
                </p>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div className="flex items-center justify-between text-slate-500 mb-1">
                  <span className="text-xs font-semibold uppercase">Mean Absolute Error (MAE)</span>
                  <ShieldCheck className="w-4 h-4 text-cyan-600" />
                </div>
                <div className="text-3xl font-black text-slate-900">{metrics.mae.toFixed(2)}</div>
                <p className="text-[11px] text-slate-500 mt-1">
                  Average deviation: ~20.1 AQI points across out-of-sample days.
                </p>
              </div>
            </div>
          </div>

          {/* Feature Importance Analysis */}
          <div className="bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-xs space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-slate-900">Gini Impurity Feature Importance</h3>
                <p className="text-xs text-slate-500">
                  Extracted directly from the estimator's <code className="text-slate-700">feature_importances_</code> attribute
                </p>
              </div>
              <span className="text-xs font-semibold text-teal-800 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md">
                Sum: 100.0%
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
              <div className="lg:col-span-8 h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={featureData}
                    layout="vertical"
                    margin={{ top: 10, right: 30, left: 30, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
                    <XAxis
                      type="number"
                      domain={[0, 0.5]}
                      tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                      fontSize={11}
                      stroke="#94a3b8"
                    />
                    <YAxis dataKey="feature" type="category" fontSize={12} tickLine={false} stroke="#64748b" />
                    <Tooltip
                      formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, 'Contribution']}
                      contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                      {featureData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="lg:col-span-4 bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-2">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block mb-2">
                  Relative Weights
                </span>
                <div className="space-y-1.5 text-xs">
                  {featureData.map((f, i) => (
                    <div key={f.feature} className="flex items-center justify-between py-1 border-b border-slate-200/60 last:border-0">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                        <span className="font-semibold text-slate-800">{f.feature}</span>
                      </div>
                      <span className="font-mono text-slate-600 font-medium">
                        {(f.importance * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Mandatory Causation Caveat */}
            <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex items-start gap-3">
              <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
              <p className="text-xs text-slate-600 leading-relaxed italic">
                <strong>Feature Importance Caveat:</strong> {modelInfo?.feature_importance_note ||
                  'Feature importance shows how the trained model uses its inputs; it does not establish that a pollutant causes an AQI outcome.'}
              </p>
            </div>
          </div>

          {/* Training Maxima Section */}
          <div className="bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-xs space-y-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">Training Partition Observed Maxima</h3>
              <p className="text-xs text-slate-500">
                Calculated as the maximum non-null concentration for each feature in the training partition (2015–2019).
                Inputs exceeding these thresholds trigger reliability warnings.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {Object.entries(trainingMaxima).map(([feat, maxVal]) => (
                <div key={feat} className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 text-center">
                  <span className="text-xs font-semibold text-slate-500 uppercase block">{feat}</span>
                  <span className="text-xl font-extrabold text-slate-900 block mt-1">{maxVal}</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {feat === 'CO' ? 'mg/m³' : 'µg/m³'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Dataset & Split Summary */}
          <div className="bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-xs space-y-5">
            <div>
              <h3 className="text-base font-bold text-slate-900">Dataset Partitioning &amp; Temporal Segregation</h3>
              <p className="text-xs text-slate-500">
                Chronological splitting across 26 Indian monitoring cities prevents future-to-past temporal leakage.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="bg-slate-50/80 p-5 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-teal-800 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                    Training Partition (80%)
                  </span>
                  <span className="text-xs font-bold text-slate-800">{dataset.train_samples.toLocaleString()} samples</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  <Calendar className="w-4 h-4 text-slate-400" />
                  <span>
                    {dataset.train_date_range?.start || '2015-01-01'} to {dataset.train_date_range?.end || '2019-12-05'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Historical air quality monitoring data spanning 23 urban centers. Used exclusively to train the 100 decision trees
                  and learn median imputation constants without access to future observations.
                </p>
              </div>

              <div className="bg-slate-50/80 p-5 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-blue-800 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    Chronological Test Set (20%)
                  </span>
                  <span className="text-xs font-bold text-slate-800">{dataset.test_samples.toLocaleString()} samples</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  <Calendar className="w-4 h-4 text-slate-400" />
                  <span>
                    {dataset.test_date_range?.start || '2019-12-05'} to {dataset.test_date_range?.end || '2020-06-30'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Completely held-out subsequent observations spanning 26 cities. Evaluated against the persistence baseline
                  (predicting tomorrow's AQI from today's AQI) to confirm out-of-sample generalization.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* SECTION B: Historical Benchmark Example */}
      {(activeSubTab === 'example' || activeSubTab === 'all') && (
        <section className="space-y-6 pt-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <div>
              <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Section B</span>
              <h2 className="text-xl font-bold text-slate-900">Historical Benchmark Case Study</h2>
            </div>
            <button
              onClick={onRefreshExample}
              disabled={loading}
              className="text-xs font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-1.5 rounded-lg shadow-2xs hover:bg-slate-50 transition-colors cursor-pointer disabled:opacity-50"
            >
              {loading ? 'Refreshing...' : 'Refresh Sample'}
            </button>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
                  <MapPin className="w-6 h-6" />
                </div>
                <div>
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Station Location</span>
                  <span className="text-xl font-bold text-slate-900">{exampleData.city}, Andhra Pradesh</span>
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs font-medium text-slate-600 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
                <Calendar className="w-4 h-4 text-slate-400" />
                <span>Observation Date: <strong className="text-slate-900">{exampleData.date}</strong></span>
              </div>
            </div>

            {/* 4 Score Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Today's Observed AQI</span>
                <div className="text-3xl font-extrabold text-slate-900 mt-1">{exampleData.today_aqi.toFixed(1)}</div>
                <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full inline-block mt-2">
                  Satisfactory (51–100)
                </span>
                <p className="text-[11px] text-slate-400 mt-2">Observed on {exampleData.date}</p>
              </div>

              <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Actual Next-Day AQI</span>
                <div className="text-3xl font-extrabold text-slate-900 mt-1">{exampleData.actual_next_day_aqi.toFixed(1)}</div>
                <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full inline-block mt-2">
                  Satisfactory (51–100)
                </span>
                <p className="text-[11px] text-slate-400 mt-2">Ground Truth (2020-06-23)</p>
              </div>

              <div className="bg-teal-50/80 p-5 rounded-xl border border-teal-200 text-center">
                <span className="text-xs font-semibold text-teal-800 uppercase tracking-wider block">Model Predicted AQI</span>
                <div className="text-3xl font-black text-teal-900 mt-1">{exampleData.reported_model_prediction.toFixed(2)}</div>
                <span className="text-xs font-medium text-amber-800 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full inline-block mt-2">
                  Moderate (101–200)
                </span>
                <p className="text-[11px] text-teal-700 mt-2">Random Forest Pipeline</p>
              </div>

              <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Absolute Error</span>
                <div className="text-3xl font-extrabold text-slate-900 mt-1">{exampleData.reported_absolute_error.toFixed(2)}</div>
                <span className="text-xs font-medium text-slate-700 bg-slate-200/70 px-2 py-0.5 rounded-full inline-block mt-2">
                  AQI Points Difference
                </span>
                <p className="text-[11px] text-slate-400 mt-2">|112.63 - 100.00|</p>
              </div>
            </div>

            {/* Visual Scale Meter */}
            <div className="pt-2">
              <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block mb-2">
                Predicted AQI ({exampleData.reported_model_prediction.toFixed(1)}) on Prototype Scale
              </span>
              <AqiScale value={exampleData.reported_model_prediction} category="Moderate" />
            </div>

            {/* Observed Pollutant Input Matrix */}
            <div className="space-y-3 pt-2">
              <h3 className="text-sm font-bold text-slate-900">Recorded Pollutant Measurements for Test Day</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {[
                  { name: 'PM2.5', val: exampleData.input_values.pm25, unit: 'µg/m³' },
                  { name: 'PM10', val: exampleData.input_values.pm10, unit: 'µg/m³' },
                  { name: 'NO2', val: exampleData.input_values.no2, unit: 'µg/m³' },
                  { name: 'CO', val: exampleData.input_values.co, unit: 'mg/m³' },
                  { name: 'SO2', val: exampleData.input_values.so2, unit: 'µg/m³' },
                  { name: 'O3', val: exampleData.input_values.o3, unit: 'µg/m³' },
                ].map((item) => (
                  <div key={item.name} className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase block">{item.name}</span>
                    <span className="text-lg font-bold text-slate-900 block mt-0.5">{item.val}</span>
                    <span className="text-[10px] text-slate-400">{item.unit}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Endpoint Reference Notice */}
            <div className="bg-amber-50/80 border border-amber-200 p-4 rounded-xl flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              <div className="text-xs text-amber-900 leading-relaxed">
                <strong>Endpoint Reference Notice:</strong> {exampleData.note || 'Historical reference test sample from model evaluation. Not live sensor readings.'}
                <span className="block mt-1 text-amber-800">
                  This record is from the held-out chronological test partition of the Central Pollution Control Board (CPCB) archive.
                  It confirms that the prediction pipeline produces verified, reproducible numeric outputs.
                </span>
              </div>
            </div>

            {/* The Required "Use This Example in Predictor" Button */}
            <div className="pt-2 flex justify-end">
              <button
                onClick={onUseExampleInPredictor}
                className="inline-flex items-center gap-2 bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold px-6 py-2.5 rounded-xl shadow-md transition-all hover:gap-3 cursor-pointer"
              >
                <span>Use This Example in Predictor</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </section>
      )}

      {/* SECTION C: Methodology & Limitations */}
      {(activeSubTab === 'methodology' || activeSubTab === 'all') && (
        <section className="space-y-6 pt-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <div>
              <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Section C</span>
              <h2 className="text-xl font-bold text-slate-900">Scientific Methodology, Pipeline &amp; Limitations</h2>
            </div>
          </div>

          {/* Project Card */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-5">
            <div className="flex items-center gap-3 pb-4 border-b border-slate-100">
              <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
                <School className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Academic Project Overview</span>
                <h3 className="text-lg font-bold text-slate-900">
                  AI-Powered Air Quality Prediction &amp; Decision Support System
                </h3>
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

          {/* 7-Step Pipeline Methodology */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
            <div className="flex items-center gap-2.5 pb-2">
              <Layers className="w-5 h-5 text-teal-600" />
              <h3 className="text-lg font-bold text-slate-900">Seven-Stage Pipeline Architecture</h3>
            </div>

            <div className="space-y-4 text-sm text-slate-600">
              {[
                {
                  step: '1',
                  title: 'Dataset Ingestion & Consecutive Calendar Day Pairing',
                  desc: 'Ingested 29,531 daily records across 26 Indian cities from the CPCB archive (2015–2020). Observations were sorted strictly by city and date. Next-day targets were paired only when consecutive calendar date difference was exactly 1 day, yielding 24,849 valid prediction pairs.'
                },
                {
                  step: '2',
                  title: 'Chronological Embargo Split (Zero Temporal Leakage)',
                  desc: 'Random shuffling was avoided to preserve temporal validity. The data was split chronologically: Train (2015-01-01 to 2019-12-04, 19,160 clean pairs after screening), 1-day Embargo buffer (2019-12-05) to eliminate transition label leakage, and Test (2019-12-06 to 2020-06-30, 4,900 held-out pairs).'
                },
                {
                  step: '3',
                  title: 'Scientific Data Quality Screening & Monthly Ratio Imputation',
                  desc: 'Training records with anomalous CO > 17.0 mg/m³ (699 rows traced to Ahmedabad GJ001 sensor contamination) were screened. AQI training targets were clamped to the statutory 500 ceiling. A monthly PM2.5/PM10 ratio estimator fitted strictly on the training partition allows robust inference when PM10 is unmonitored.'
                },
                {
                  step: '4',
                  title: 'Random Forest Ensemble Training',
                  desc: 'Trained an ensemble of 100 decision trees (min_samples_leaf=2) yielding an operational CPCB-scale (≤500) R² of 0.8712, RMSE of 29.41, and MAE of 19.26, significantly outperforming the persistence baseline (R² = 0.8284, RMSE = 34.04). Particulates account for 85.52% of feature importance.'
                },
                {
                  step: '5',
                  title: 'Decision Support & Operational Boundary Engine',
                  desc: 'Clamps outputs to statutory limits (0–500), issues runtime advisories for CO exceeding 17.0 mg/m³ or concentrations exceeding historical training maxima, and generates six-category CPCB health advisories.'
                },
                {
                  step: '6',
                  title: 'FastAPI Production Microservice',
                  desc: 'High-throughput Python REST service executing single-object composite inference with dynamic ratio imputation, Pydantic v2 schemas, and health check endpoints.'
                },
                {
                  step: '7',
                  title: 'Modern React + Vite Frontend Dashboard',
                  desc: 'Interactive web client with Tailwind CSS styling, Recharts visualizations, optional particulate input handling, inline input validation, and live backend connectivity indicators.'
                }
              ].map((item) => (
                <div key={item.step} className="flex items-start gap-3.5">
                  <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                    {item.step}
                  </span>
                  <div>
                    <strong className="text-slate-800 block">{item.title}</strong>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Phase 2B Scientific Data Quality Upgrades */}
          <div className="bg-white rounded-2xl border border-teal-200/80 shadow-sm p-6 sm:p-8 space-y-5">
            <div className="flex items-center gap-2.5 pb-2 border-b border-slate-100">
              <ShieldCheck className="w-5 h-5 text-teal-600" />
              <div>
                <span className="text-xs font-bold text-teal-800 uppercase tracking-wider block">Production Release v2.0</span>
                <h3 className="text-lg font-bold text-slate-900">Phase 2B Scientific Data Quality &amp; Model Upgrades</h3>
              </div>
            </div>

            <p className="text-sm text-slate-600 leading-relaxed">
              Following an exhaustive data-quality audit, the v2.0 model implements four foundational scientific improvements to ensure
              atmospheric realism, prevent artifact-driven predictions, and deliver superior operational generalization:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-teal-50/50 border border-teal-100 space-y-1.5">
                <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
                  <span className="w-2 h-2 rounded-full bg-teal-600" />
                  <span>1. Extreme CO Anomaly Screening</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Screened 699 training observations where CO &gt; 17.0 mg/m³ (reaching up to 175.81 mg/m³), which were traced exclusively to
                  the Maninagar station in Ahmedabad. In the legacy v1 model, these extreme values artificially inflated CO feature importance to 40.37%.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-teal-50/50 border border-teal-100 space-y-1.5">
                <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
                  <span className="w-2 h-2 rounded-full bg-teal-600" />
                  <span>2. Atmospheric Feature Realism</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Removing distorted CO readings restored physical atmospheric balance: combined particulate importance increased from 50.73% to 85.52%
                  (PM2.5: 73.68%, PM10: 11.84%), aligning directly with established CPCB environmental physics where particulates dominate Indian AQI.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-teal-50/50 border border-teal-100 space-y-1.5">
                <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
                  <span className="w-2 h-2 rounded-full bg-teal-600" />
                  <span>3. Monthly Ratio PM10 Imputation</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  In stations lacking PM10 analyzers (e.g. historical Lucknow, Patna), missing PM10 is dynamically estimated using training-fitted
                  monthly PM2.5/PM10 ratios (annual mean 0.4644, winter peak 0.5643), improving missing-PM10 test MAE from 26.84 to 21.00.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-teal-50/50 border border-teal-100 space-y-1.5">
                <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
                  <span className="w-2 h-2 rounded-full bg-teal-600" />
                  <span>4. Operational Scale Calibration</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Training targets were capped at the statutory CPCB limit (500.0), eliminating unphysical runaway predictions up to 1,291.
                  On the official operational scale (≤500), Release Candidate v2.0 achieves R² = 0.8712 and RMSE = 29.41 across 4,900 test days.
                </p>
              </div>
            </div>
          </div>

          {/* Crucial Technical Distinction */}
          <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 space-y-4 shadow-sm">
            <div className="flex items-center gap-2.5 text-teal-400">
              <BookOpen className="w-5 h-5" />
              <h3 className="text-base font-bold">Crucial Distinction: Regression Estimate vs. CPCB AQI Calculation</h3>
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
        </section>
      )}
    </div>
  );
};
