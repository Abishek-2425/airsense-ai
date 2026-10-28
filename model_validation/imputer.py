"""
City and Month Aware Hierarchical Imputer for Air Quality Pollutant Concentrations.

Fits hierarchical median statistics strictly on the training partition:
Level 1: (City, Month) median if sufficient observations exist (count >= min_samples)
Level 2: (City) median if city exists in training set
Level 3: Global (National) training median as fallback

During transform:
- If City and Date/Month are present in input DataFrame, uses Level 1 -> Level 2 -> Level 3.
- If input lacks City and Month (e.g. standalone 6-feature array or dict), falls back cleanly to Level 3.
- Returns strictly the 6 numerical features in required order:
  ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']


class CityMonthAwareImputer(BaseEstimator, TransformerMixin):
    """
    Hierarchical imputer designed to capture seasonal and geographic baseline shifts
    without data leakage.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, min_samples: int = 5):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.min_samples = min_samples
        self.city_month_medians_: Dict[str, Dict[str, float]] = {}
        self.city_medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self.statistics_: np.ndarray = np.array([])

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fits hierarchical medians strictly on training data.
        X must be a pandas DataFrame to extract City and Month. If numpy array,
        only global medians are learned.
        """
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_cols[:X.shape[1]])
        else:
            df = X.copy()

        # 1. Global medians across all training rows
        self.global_medians_ = {
            col: float(df[col].dropna().median()) if col in df.columns else 0.0
            for col in self.feature_cols
        }
        self.statistics_ = np.array([self.global_medians_[col] for col in self.feature_cols])

        # 2. City & (City, Month) hierarchical medians if metadata is present
        if 'City' in df.columns:
            # City medians
            for col in self.feature_cols:
                if col in df.columns:
                    city_grp = df.groupby('City')[col].median().to_dict()
                    self.city_medians_[col] = city_grp
                else:
                    self.city_medians_[col] = {}

            # City-Month medians
            month_col = None
            if 'Month' in df.columns:
                month_col = df['Month']
            elif 'Date' in df.columns:
                month_col = pd.to_datetime(df['Date']).dt.month

            if month_col is not None:
                df_temp = df.copy()
                df_temp['_Month'] = month_col
                df_temp['_City_Month'] = df_temp['City'].astype(str) + '_' + df_temp['_Month'].astype(str)

                for col in self.feature_cols:
                    if col in df_temp.columns:
                        # Only keep medians where sample count >= min_samples
                        counts = df_temp.groupby('_City_Month')[col].count()
                        medians = df_temp.groupby('_City_Month')[col].median()
                        valid_keys = counts[counts >= self.min_samples].index
                        self.city_month_medians_[col] = medians.loc[valid_keys].to_dict()
                    else:
                        self.city_month_medians_[col] = {}

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms input by imputing missing values.
        Returns a 2D numpy array of shape (N, 6) in exact feature_cols order.
        """
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_cols[:X.shape[1]])
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            df = X.copy()

        has_city = 'City' in df.columns
        has_month = 'Month' in df.columns or 'Date' in df.columns

        month_series = None
        if has_month:
            if 'Month' in df.columns:
                month_series = df['Month'].astype(str)
            else:
                month_series = pd.to_datetime(df['Date']).dt.month.astype(str)

        output_df = pd.DataFrame(index=df.index)

        for col in self.feature_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
            else:
                series = pd.Series(np.nan, index=df.index)

            # Hierarchical imputation
            # Level 1: City_Month
            if has_city and month_series is not None and col in self.city_month_medians_:
                cm_key = df['City'].astype(str) + '_' + month_series
                l1_val = cm_key.map(self.city_month_medians_[col])
                series = series.fillna(l1_val)

            # Level 2: City
            if has_city and col in self.city_medians_:
                l2_val = df['City'].astype(str).map(self.city_medians_[col])
                series = series.fillna(l2_val)

            # Level 3: Global fallback
            g_val = self.global_medians_.get(col, 0.0)
            series = series.fillna(g_val)

            output_df[col] = series

        return output_df[self.feature_cols].values
