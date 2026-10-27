export interface PredictRequest {
  pm25: number;
  pm10: number;
  no2: number;
  co: number;
  so2: number;
  o3: number;
}

export interface PredictResponse {
  predicted_aqi: number;
  raw_predicted_aqi: number;
  category: string;
  recommendations: string[];
  warnings: string[];
  input_values: {
    pm25: number;
    pm10: number;
    no2: number;
    co: number;
    so2: number;
    o3: number;
  };
  model_name: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelMetrics {
  r2: number;
  rmse: number;
  mae: number;
  test_samples: number;
}

export interface DatasetSummary {
  total_raw_rows: number;
  valid_calendar_pairs: number;
  valid_target_samples: number;
  cities_count: number;
  train_samples: number;
  test_samples: number;
  train_date_range?: {
    start: string;
    end: string;
  };
  test_date_range?: {
    start: string;
    end: string;
  };
}

export interface ModelInfoResponse {
  model_name: string;
  target: string;
  features: string[];
  feature_importances: FeatureImportance[];
  training_maxima: Record<string, number>;
  metrics: ModelMetrics;
  feature_importance_note: string;
  dataset_summary?: DatasetSummary;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ExampleResponse {
  city: string;
  date: string;
  input_values: {
    pm25: number;
    pm10: number;
    no2: number;
    co: number;
    so2: number;
    o3: number;
  };
  today_aqi: number;
  actual_next_day_aqi: number;
  reported_model_prediction: number;
  reported_absolute_error: number;
  note?: string;
}

export type NavTab = 'dashboard' | 'predict' | 'model-info' | 'example' | 'about';
