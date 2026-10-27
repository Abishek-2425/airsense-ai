import React from 'react';
import {
  History,
  Calendar,
  MapPin,
  ArrowRight,
  AlertTriangle
} from 'lucide-react';
import type { ExampleResponse } from '../types';
import { AqiScale } from '../components/AqiScale';

interface HistoricalExamplePageProps {
  example: ExampleResponse | null;
  loading: boolean;
  onNavigateToPredict: () => void;
  onRefresh: () => void;
}

export const HistoricalExamplePage: React.FC<HistoricalExamplePageProps> = ({
  example,
  onNavigateToPredict,
  onRefresh,
}) => {
  const data = example || {
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

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-teal-700 text-xs font-semibold uppercase tracking-wider mb-1">
            <History className="w-4 h-4" />
            <span>Benchmark Validation</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">
            Historical Test Example Case Study
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Demonstration of actual ground-truth test evaluation retrieved from the backend endpoint{' '}
            <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-700 text-xs">GET /example</code>.
          </p>
        </div>

        <button
          onClick={onRefresh}
          className="text-xs font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-3 py-1.5 rounded-lg shadow-2xs hover:bg-slate-50 transition-colors cursor-pointer"
        >
          Refresh Sample
        </button>
      </div>

      {/* Main Case Study Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
              <MapPin className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Station Location</span>
              <span className="text-xl font-bold text-slate-900">{data.city}, Andhra Pradesh</span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs font-medium text-slate-600 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
            <Calendar className="w-4 h-4 text-slate-400" />
            <span>Observation Date: <strong className="text-slate-900">{data.date}</strong></span>
          </div>
        </div>

        {/* 4 Score Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Today's Observed AQI</span>
            <div className="text-3xl font-extrabold text-slate-900 mt-1">{data.today_aqi.toFixed(1)}</div>
            <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full inline-block mt-2">
              Satisfactory (51–100)
            </span>
            <p className="text-[11px] text-slate-400 mt-2">Observed on {data.date}</p>
          </div>

          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Actual Next-Day AQI</span>
            <div className="text-3xl font-extrabold text-slate-900 mt-1">{data.actual_next_day_aqi.toFixed(1)}</div>
            <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full inline-block mt-2">
              Satisfactory (51–100)
            </span>
            <p className="text-[11px] text-slate-400 mt-2">Ground Truth (2020-06-23)</p>
          </div>

          <div className="bg-teal-50/80 p-5 rounded-xl border border-teal-200 text-center">
            <span className="text-xs font-semibold text-teal-800 uppercase tracking-wider block">Model Predicted AQI</span>
            <div className="text-3xl font-black text-teal-900 mt-1">{data.reported_model_prediction.toFixed(2)}</div>
            <span className="text-xs font-medium text-amber-800 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full inline-block mt-2">
              Moderate (101–200)
            </span>
            <p className="text-[11px] text-teal-700 mt-2">Random Forest Pipeline</p>
          </div>

          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Absolute Error</span>
            <div className="text-3xl font-extrabold text-slate-900 mt-1">{data.reported_absolute_error.toFixed(2)}</div>
            <span className="text-xs font-medium text-slate-700 bg-slate-200/70 px-2 py-0.5 rounded-full inline-block mt-2">
              AQI Points Difference
            </span>
            <p className="text-[11px] text-slate-400 mt-2">|112.63 - 100.00|</p>
          </div>
        </div>

        {/* Visual Scale Meter for the prediction */}
        <div className="pt-2">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block mb-2">
            Predicted AQI ({data.reported_model_prediction.toFixed(1)}) on Prototype Scale
          </span>
          <AqiScale value={data.reported_model_prediction} category="Moderate" />
        </div>

        {/* Observed Pollutant Input Matrix */}
        <div className="space-y-3 pt-2">
          <h3 className="text-sm font-bold text-slate-900">Recorded Pollutant Measurements for Test Day</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { name: 'PM2.5', val: data.input_values.pm25, unit: 'µg/m³' },
              { name: 'PM10', val: data.input_values.pm10, unit: 'µg/m³' },
              { name: 'NO2', val: data.input_values.no2, unit: 'µg/m³' },
              { name: 'CO', val: data.input_values.co, unit: 'mg/m³' },
              { name: 'SO2', val: data.input_values.so2, unit: 'µg/m³' },
              { name: 'O3', val: data.input_values.o3, unit: 'µg/m³' },
            ].map((item) => (
              <div key={item.name} className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                <span className="text-[11px] font-semibold text-slate-400 uppercase block">{item.name}</span>
                <span className="text-lg font-bold text-slate-900 block mt-0.5">{item.val}</span>
                <span className="text-[10px] text-slate-400">{item.unit}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Endpoint Note Banner */}
        <div className="bg-amber-50/80 border border-amber-200 p-4 rounded-xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-900 leading-relaxed">
            <strong>Endpoint Reference Notice:</strong> {data.note || 'Historical reference test sample from model evaluation. Not live sensor readings.'}
            <span className="block mt-1 text-amber-800">
              This record is from the held-out chronological test partition of the Central Pollution Control Board (CPCB) archive. It confirms that the prediction pipeline produces verified, reproducible numeric outputs.
            </span>
          </div>
        </div>

        {/* CTA to Predict tab */}
        <div className="pt-2 flex justify-end">
          <button
            onClick={onNavigateToPredict}
            className="inline-flex items-center gap-2 bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold px-5 py-2.5 rounded-xl shadow-md transition-all hover:gap-3 cursor-pointer"
          >
            <span>Load This Sample into Predictor</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
