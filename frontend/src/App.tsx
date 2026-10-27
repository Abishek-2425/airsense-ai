import React, { useState, useEffect, useCallback } from 'react';
import type { NavTab, HealthResponse, ModelInfoResponse, ExampleResponse } from './types';
import { api } from './services/api';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { PredictPage } from './pages/PredictPage';
import { ModelInsightsPage } from './pages/ModelInsightsPage';
import { HistoricalExamplePage } from './pages/HistoricalExamplePage';
import { AboutPage } from './pages/AboutPage';
import { Wind, AlertCircle, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
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

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      {/* Top Navigation */}
      <Navbar
        currentTab={currentTab}
        onTabChange={setCurrentTab}
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
            onNavigate={setCurrentTab}
            modelInfo={modelInfo}
            example={example}
          />
        )}

        {currentTab === 'predict' && (
          <PredictPage
            modelInfo={modelInfo}
            example={example}
          />
        )}

        {currentTab === 'model-info' && (
          <ModelInsightsPage
            modelInfo={modelInfo}
            loading={modelInfoLoading}
            onRefresh={fetchModelInfo}
          />
        )}

        {currentTab === 'example' && (
          <HistoricalExamplePage
            example={example}
            loading={exampleLoading}
            onNavigateToPredict={() => setCurrentTab('predict')}
            onRefresh={fetchExample}
          />
        )}

        {currentTab === 'about' && <AboutPage />}
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
