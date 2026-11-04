import React, { useState, useEffect, useCallback } from 'react';
import type { NavTab, HealthResponse, ModelInfoResponse, ExampleResponse, MethodologySection, FormState } from './types';
import { api } from './services/api';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { PredictPage } from './pages/PredictPage';
import { ModelMethodologyPage } from './pages/ModelMethodologyPage';
import { Wind, AlertCircle, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
  const [methodologySection, setMethodologySection] = useState<MethodologySection | 'all'>('performance');
  const [prefillValues, setPrefillValues] = useState<FormState | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(false);
  const [example, setExample] = useState<ExampleResponse | null>(null);
  const [exampleLoading, setExampleLoading] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  // Fetch health status
  const fetchHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const res = await api.getHealth();
      setHealth(res);
      setInitError(null);
    } catch {
      setHealth({ status: 'unreachable', model_loaded: false });
      setInitError(
        'Unable to connect to the FastAPI backend at http://localhost:8000. ' +
        'Please verify that the server is running (`uvicorn backend.main:app --port 8000`).'
      );
    } finally {
      setHealthLoading(false);
    }
  }, []);

  // Fetch model info
  const fetchModelInfo = useCallback(async () => {
    setModelInfoLoading(true);
    try {
      const res = await api.getModelInfo();
      setModelInfo(res);
    } catch (err) {
      console.warn('Could not load model info from API, using fallback values:', err);
    } finally {
      setModelInfoLoading(false);
    }
  }, []);

  // Fetch historical example
  const fetchExample = useCallback(async () => {
    setExampleLoading(true);
    try {
      const res = await api.getExample();
      setExample(res);
    } catch (err) {
      console.warn('Could not load example from API, using fallback values:', err);
    } finally {
      setExampleLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
    fetchExample();
  }, [fetchHealth, fetchModelInfo, fetchExample]);

  const handleNavigate = (tab: NavTab, section?: MethodologySection | 'all') => {
    if (section) {
      setMethodologySection(section);
    }
    setCurrentTab(tab);
  };

  const handleUseExampleInPredictor = () => {
    let exampleMonth = String(new Date().getMonth() + 1);
    if (example?.date) {
      const parts = example.date.split('-');
      if (parts.length >= 2) {
        const parsedM = parseInt(parts[1], 10);
        if (!isNaN(parsedM) && parsedM >= 1 && parsedM <= 12) {
          exampleMonth = String(parsedM);
        }
      }
    }

    if (example?.input_values) {
      setPrefillValues({
        pm25: example.input_values.pm25.toString(),
        pm10: example.input_values.pm10.toString(),
        no2: example.input_values.no2.toString(),
        co: example.input_values.co.toString(),
        so2: example.input_values.so2.toString(),
        o3: example.input_values.o3.toString(),
        month: exampleMonth,
      });
    } else {
      setPrefillValues({
        pm25: '33.17',
        pm10: '108.22',
        no2: '42.45',
        co: '0.73',
        so2: '13.65',
        o3: '34.85',
        month: exampleMonth,
      });
    }
    setCurrentTab('predict');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      {/* Top Navigation */}
      <Navbar
        currentTab={currentTab}
        onTabChange={(tab) => handleNavigate(tab)}
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={fetchHealth}
      />

      {/* Backend Disconnected Alert Banner */}
      {initError && (
        <div className="bg-rose-50 border-b border-rose-200 text-rose-900 px-4 py-2.5 text-xs">
          <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{initError}</span>
            </div>
            <button
              onClick={fetchHealth}
              className="inline-flex items-center gap-1 font-semibold text-rose-800 hover:text-rose-950 underline shrink-0 cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      )}

      {/* Main Content View */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentTab === 'dashboard' && (
          <Dashboard
            onNavigate={handleNavigate}
            modelInfo={modelInfo}
            example={example}
          />
        )}

        {currentTab === 'predict' && (
          <PredictPage
            modelInfo={modelInfo}
            example={example}
            prefilledValues={prefillValues}
          />
        )}

        {currentTab === 'model-methodology' && (
          <ModelMethodologyPage
            modelInfo={modelInfo}
            example={example}
            loading={modelInfoLoading || exampleLoading}
            onRefreshModelInfo={fetchModelInfo}
            onRefreshExample={fetchExample}
            onUseExampleInPredictor={handleUseExampleInPredictor}
            initialSection={methodologySection}
          />
        )}
      </main>

      {/* Professional Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-xs text-slate-500 space-y-3 text-center sm:text-left sm:flex sm:items-center sm:justify-between sm:space-y-0">
          <div>
            <div className="font-semibold text-slate-800 flex items-center justify-center sm:justify-start gap-1.5">
              <Wind className="w-4 h-4 text-teal-600" />
              <span>AirSense AI · Environmental Studies Final-Year Project</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              AI-Powered Air Quality Prediction &amp; Decision Support System
            </p>
          </div>

          <div className="text-[11px] text-slate-400 max-w-md">
            Academic research prototype. Continuous regression predictions and prototype category bands are for
            educational decision-support purposes only.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
