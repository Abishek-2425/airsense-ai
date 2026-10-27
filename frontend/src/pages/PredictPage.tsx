import React, { useState } from 'react';
import {
  Wind,
  Sparkles,
  RotateCcw,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Info
} from 'lucide-react';
import type { PredictRequest, PredictResponse, ModelInfoResponse, ExampleResponse } from '../types';
import { api } from '../services/api';
import { AqiScale, getCategoryStyles } from '../components/AqiScale';

interface PredictPageProps {
  modelInfo: ModelInfoResponse | null;
  example: ExampleResponse | null;
}

interface FormState {
  pm25: string;
  pm10: string;
  no2: string;
  co: string;
  so2: string;
  o3: string;
}

interface FormErrors {
  pm25?: string;
  pm10?: string;
  no2?: string;
  co?: string;
  so2?: string;
  o3?: string;
}

const DEFAULT_EXAMPLE: FormState = {
  pm25: '33.17',
  pm10: '108.22',
  no2: '42.45',
  co: '0.73',
  so2: '13.65',
  o3: '34.85',
};

export const PredictPage: React.FC<PredictPageProps> = ({ modelInfo, example }) => {
  const [formData, setFormData] = useState<FormState>(DEFAULT_EXAMPLE);
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const trainingMaxima = modelInfo?.training_maxima || {
    'PM2.5': 949.99,
    'PM10': 917.08,
    'NO2': 362.21,
    'CO': 175.81,
    'SO2': 186.08,
    'O3': 257.73,
  };

  const pollutantConfigs = [
    {
      id: 'pm25' as keyof FormState,
      name: 'PM2.5',
      label: 'Fine Particulate Matter (PM2.5)',
      unit: 'µg/m³',
      desc: 'Inhalable particles ≤ 2.5 micrometers from combustion, vehicle exhaust, and smoke.',
      maxObserved: trainingMaxima['PM2.5'],
      step: '0.01',
    },
    {
      id: 'pm10' as keyof FormState,
      name: 'PM10',
      label: 'Coarse Particulate Matter (PM10)',
      unit: 'µg/m³',
      desc: 'Inhalable particles ≤ 10 micrometers from dust, construction, and mechanical road abrasion.',
      maxObserved: trainingMaxima['PM10'],
      step: '0.01',
    },
    {
      id: 'no2' as keyof FormState,
      name: 'NO2',
      label: 'Nitrogen Dioxide (NO₂)',
      unit: 'µg/m³',
      desc: 'Toxic gas from motor vehicles, industrial boilers, and thermal power facilities.',
      maxObserved: trainingMaxima['NO2'],
      step: '0.01',
    },
    {
      id: 'co' as keyof FormState,
      name: 'CO',
      label: 'Carbon Monoxide (CO)',
      unit: 'mg/m³',
      desc: 'Colorless, odorless gas emitted by incomplete combustion in engines and biomass burning.',
      maxObserved: trainingMaxima['CO'],
      step: '0.01',
    },
    {
      id: 'so2' as keyof FormState,
      name: 'SO2',
      label: 'Sulphur Dioxide (SO₂)',
      unit: 'µg/m³',
      desc: 'Gas from fuel combustion containing sulphur (coal, petroleum refining, smelters).',
      maxObserved: trainingMaxima['SO2'],
      step: '0.01',
    },
    {
      id: 'o3' as keyof FormState,
      name: 'O3',
      label: 'Ground-Level Ozone (O₃)',
      unit: 'µg/m³',
      desc: 'Secondary photochemical pollutant formed by NOx and VOC reaction under sunlight.',
      maxObserved: trainingMaxima['O3'],
      step: '0.01',
    },
  ];

  const handleInputChange = (field: keyof FormState, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};
    let isValid = true;

    pollutantConfigs.forEach((p) => {
      const rawVal = formData[p.id].trim();
      if (!rawVal) {
        newErrors[p.id] = `${p.name} is required`;
        isValid = false;
        return;
      }
      const num = parseFloat(rawVal);
      if (isNaN(num) || !isFinite(num)) {
        newErrors[p.id] = `Must be a valid finite number`;
        isValid = false;
      } else if (num < 0) {
        newErrors[p.id] = `Concentration cannot be negative`;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);

    if (!validateForm()) return;

    setSubmitting(true);
    try {
      const payload: PredictRequest = {
        pm25: parseFloat(formData.pm25),
        pm10: parseFloat(formData.pm10),
        no2: parseFloat(formData.no2),
        co: parseFloat(formData.co),
        so2: parseFloat(formData.so2),
        o3: parseFloat(formData.o3),
      };

      const res = await api.predict(payload);
      setResult(res);
    } catch (err: any) {
      setApiError(err.message || 'An error occurred while connecting to the prediction API.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleLoadExample = () => {
    if (example?.input_values) {
      setFormData({
        pm25: example.input_values.pm25.toString(),
        pm10: example.input_values.pm10.toString(),
        no2: example.input_values.no2.toString(),
        co: example.input_values.co.toString(),
        so2: example.input_values.so2.toString(),
        o3: example.input_values.o3.toString(),
      });
    } else {
      setFormData(DEFAULT_EXAMPLE);
    }
    setErrors({});
    setApiError(null);
  };

  const handleReset = () => {
    setFormData({
      pm25: '',
      pm10: '',
      no2: '',
      co: '',
      so2: '',
      o3: '',
    });
    setErrors({});
    setApiError(null);
    setResult(null);
  };

  const styles = result ? getCategoryStyles(result.category) : null;

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-teal-700 text-xs font-semibold uppercase tracking-wider mb-1">
          <Wind className="w-4 h-4" />
          <span>Inference Engine</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">
          Predict Next-Calendar-Day AQI
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Submit current-day concentrations for all six criteria pollutants. The pre-trained Random Forest
          pipeline will estimate tomorrow's Air Quality Index and provide rule-based environmental decision support.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Form (7 cols on lg) */}
        <div className="lg:col-span-7 bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-xs space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-slate-100">
            <div>
              <h2 className="text-base font-bold text-slate-900">Pollutant Concentration Inputs</h2>
              <p className="text-xs text-slate-500">All 6 fields accept non-negative decimal measurements</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleLoadExample}
                className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium px-2.5 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
                title="Populate with Visakhapatnam test example"
              >
                <Sparkles className="w-3.5 h-3.5 text-teal-600" />
                <span>Load Benchmark Sample</span>
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="text-xs text-slate-500 hover:text-slate-800 p-1.5 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer"
                title="Reset fields"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {pollutantConfigs.map((p) => {
                const val = formData[p.id];
                const numVal = parseFloat(val);
                const isOutOfRange = !isNaN(numVal) && numVal > p.maxObserved;
                const fieldError = errors[p.id];

                return (
                  <div key={p.id} className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label htmlFor={p.id} className="text-xs font-semibold text-slate-800 flex items-center gap-1">
                        <span>{p.name}</span>
                        <span className="text-slate-400 font-normal">({p.unit})</span>
                      </label>
                      <span className="text-[10px] text-slate-400">
                        Max: {p.maxObserved}
                      </span>
                    </div>

                    <div className="relative">
                      <input
                        id={p.id}
                        type="number"
                        step={p.step}
                        min="0"
                        value={val}
                        onChange={(e) => handleInputChange(p.id, e.target.value)}
                        placeholder={`e.g. ${p.maxObserved ? (p.maxObserved * 0.1).toFixed(1) : '10.0'}`}
                        className={`w-full px-3 py-2 text-sm rounded-lg border transition-all focus:outline-none focus:ring-2 ${
                          fieldError
                            ? 'border-red-400 bg-red-50/50 focus:ring-red-200 text-red-900'
                            : isOutOfRange
                            ? 'border-amber-400 bg-amber-50/30 focus:ring-amber-200'
                            : 'border-slate-300 bg-white focus:border-teal-500 focus:ring-teal-100'
                        }`}
                      />
                      {isOutOfRange && (
                        <div
                          className="absolute right-2.5 top-2.5 text-amber-500"
                          title={`Reading exceeds historical training maximum (${p.maxObserved} ${p.unit})`}
                        >
                          <AlertTriangle className="w-4 h-4" />
                        </div>
                      )}
                    </div>

                    {fieldError ? (
                      <p className="text-[11px] text-red-600 font-medium">{fieldError}</p>
                    ) : isOutOfRange ? (
                      <p className="text-[10px] text-amber-700">
                        Exceeds training max ({p.maxObserved} {p.unit}) — prediction may have higher uncertainty.
                      </p>
                    ) : (
                      <p className="text-[11px] text-slate-400 line-clamp-1" title={p.desc}>
                        {p.desc}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>

            {/* API Error Message Alert */}
            {apiError && (
              <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 flex items-start gap-2.5 text-xs text-red-800">
                <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <span className="font-bold">Prediction Request Failed:</span>
                  <p>{apiError}</p>
                </div>
              </div>
            )}

            {/* Action Button */}
            <div className="pt-2 flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                FastAPI inference endpoint: <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-600">POST /predict</code>
              </span>
              <button
                type="submit"
                disabled={submitting}
                className={`inline-flex items-center gap-2 bg-teal-600 hover:bg-teal-500 text-white font-semibold text-sm px-6 py-2.5 rounded-xl shadow-md transition-all cursor-pointer ${
                  submitting ? 'opacity-70 cursor-not-allowed' : 'hover:shadow-lg'
                }`}
              >
                {submitting ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Processing Inference...</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    <span>Predict Next-Day AQI</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Right Column: Prediction Results Card (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-6">
          {result ? (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6 overflow-hidden relative animate-in fade-in duration-300">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Prediction Outcome</span>
                <span className="text-xs text-slate-400 font-mono">{result.model_name}</span>
              </div>

              {/* Main Score Header */}
              <div className="flex items-baseline justify-between">
                <div>
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                    Estimated Next-Day AQI
                  </span>
                  <div className="text-5xl font-black text-slate-900 tracking-tight mt-1">
                    {result.predicted_aqi.toFixed(1)}
                  </div>
                  {result.raw_predicted_aqi !== result.predicted_aqi && (
                    <span className="text-xs text-slate-400">
                      Raw model output: {result.raw_predicted_aqi.toFixed(2)} (clamped to 0)
                    </span>
                  )}
                </div>

                {/* Category Badge */}
                {styles && (
                  <div className={`px-4 py-2 rounded-xl border text-center font-bold text-sm shadow-xs ${styles.badge}`}>
                    <span>{result.category}</span>
                  </div>
                )}
              </div>

              {/* Visual Scale Meter */}
              <AqiScale value={result.predicted_aqi} category={result.category} />

              {/* Recommendations Box */}
              <div className="space-y-2 pt-2">
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-teal-600" />
                  <span>Decision Support &amp; Health Advisories</span>
                </h4>
                <div className="space-y-2">
                  {result.recommendations.map((rec, i) => (
                    <div
                      key={i}
                      className="text-xs text-slate-700 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80 leading-relaxed"
                    >
                      {rec}
                    </div>
                  ))}
                </div>
              </div>

              {/* Warnings Box if any */}
              {result.warnings && result.warnings.length > 0 && (
                <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1.5">
                  <div className="font-bold flex items-center gap-1.5 text-amber-800">
                    <AlertTriangle className="w-4 h-4 text-amber-600" />
                    <span>Prediction Warnings &amp; Range Notices</span>
                  </div>
                  <ul className="list-disc pl-4 space-y-1 text-[11px] text-amber-900">
                    {result.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Submitted Inputs Summary */}
              <div className="pt-2 border-t border-slate-100">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">
                  Submitted Pollutant Inputs
                </span>
                <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">PM2.5</span>
                    <span className="font-semibold text-slate-700">{result.input_values.pm25}</span>
                  </div>
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">PM10</span>
                    <span className="font-semibold text-slate-700">{result.input_values.pm10}</span>
                  </div>
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">NO₂</span>
                    <span className="font-semibold text-slate-700">{result.input_values.no2}</span>
                  </div>
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">CO</span>
                    <span className="font-semibold text-slate-700">{result.input_values.co}</span>
                  </div>
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">SO₂</span>
                    <span className="font-semibold text-slate-700">{result.input_values.so2}</span>
                  </div>
                  <div className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <span className="text-[10px] text-slate-400 block">O₃</span>
                    <span className="font-semibold text-slate-700">{result.input_values.o3}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-dashed border-slate-300 p-8 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-slate-100 mx-auto flex items-center justify-center text-slate-400">
                <Wind className="w-6 h-6" />
              </div>
              <div className="max-w-xs mx-auto space-y-1">
                <h3 className="font-bold text-slate-800 text-sm">Awaiting Pollutant Inputs</h3>
                <p className="text-xs text-slate-500">
                  Fill in the concentrations on the left or click "Load Benchmark Sample" to evaluate tomorrow's AQI.
                </p>
              </div>

              <div className="pt-2 text-left bg-slate-50/70 p-3.5 rounded-xl border border-slate-200 space-y-2 text-xs text-slate-600">
                <div className="font-semibold text-slate-700 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5 text-teal-600" />
                  <span>Demonstration Preset:</span>
                </div>
                <p className="text-[11px] text-slate-500">
                  Clicking "Load Benchmark Sample" loads historical Visakhapatnam readings which produce approximately{' '}
                  <span className="font-bold text-slate-800">112.63 (Moderate)</span>.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
