import type {
  HealthResponse,
  ModelInfoResponse,
  ExampleResponse,
  PredictRequest,
  PredictResponse,
} from '../types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 10000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds. Please check if the FastAPI backend is responsive.`);
    }
    throw new Error(`Network error connecting to ${url}: ${error.message || 'Server unreachable'}`);
  } finally {
    clearTimeout(timeoutId);
  }
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetchWithTimeout(`${BASE_URL}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}`);
    }
    return res.json();
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetchWithTimeout(`${BASE_URL}/model-info`);
    if (!res.ok) {
      throw new Error(`Failed to fetch model info (status ${res.status})`);
    }
    return res.json();
  },

  async getExample(): Promise<ExampleResponse> {
    const res = await fetchWithTimeout(`${BASE_URL}/example`);
    if (!res.ok) {
      throw new Error(`Failed to fetch historical example (status ${res.status})`);
    }
    return res.json();
  },

  async predict(payload: PredictRequest): Promise<PredictResponse> {
    const res = await fetchWithTimeout(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let errorDetail = `Prediction failed with status ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson.errors && Array.isArray(errorJson.errors)) {
          errorDetail = errorJson.errors.join('; ');
        } else if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // fallback
      }
      throw new Error(errorDetail);
    }

    return res.json();
  },
};
