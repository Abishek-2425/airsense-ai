"""
AirSense AI — Phase 2B Robust Preprocessing Module

Provides serializable, leakage-free particulate ratio imputation and input validation
for operational next-day AQI prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_ORDER = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
CO_ANOMALY_THRESHOLD = 17.0
AQI_MAX_STATUTORY = 500.0


class AirSensePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for AirSense AI.
    Implements physical PM2.5 / PM10 monthly ratio imputation for missing PM10,
    with a documented fallback hierarchy and zero data leakage.
    """
    def __init__(self):
        self.month_ratios_: Dict[int, float] = {}
        self.national_ratio_: float = 0.4704
        self.global_medians_: Dict[str, float] = {}
        self.feature_names_: List[str] = FEATURE_ORDER

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits ratio and median statistics strictly on historical training data.
        """
        # Ensure Month is available
        if 'Month' not in X.columns:
            if 'Date' in X.columns:
                months = pd.to_datetime(X['Date']).dt.month
            else:
                months = pd.Series([1] * len(X))
        else:
            months = X['Month']

        # Calculate pollutant global medians
        for col in self.feature_names_:
            if col in X.columns:
                self.global_medians_[col] = float(X[col].dropna().median())
            else:
                self.global_medians_[col] = 0.0

        # Fit PM2.5/PM10 ratio on paired non-null training observations
        paired_mask = (X['PM10'] > 0) & (X['PM2.5'] > 0) & (X['PM2.5'] <= 1.2 * X['PM10'])
        paired = X[paired_mask].copy()
        paired['Ratio'] = paired['PM2.5'] / paired['PM10']
        paired['Month'] = months[paired_mask]

        if len(paired) > 0:
            self.national_ratio_ = float(paired['Ratio'].median())
            self.month_ratios_ = {int(m): float(r) for m, r in paired.groupby('Month')['Ratio'].median().items()}
        else:
            self.national_ratio_ = 0.4704
            self.month_ratios_ = {m: 0.4704 for m in range(1, 13)}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms DataFrame by filling missing PM10 using monthly ratios,
        and filling any remaining missing features with training medians.
        """
        X_out = X.copy()

        # Check for Month
        if 'Month' in X_out.columns:
            months = X_out['Month']
        elif 'Date' in X_out.columns:
            months = pd.to_datetime(X_out['Date']).dt.month
        else:
            # Default to current month or 1
            months = pd.Series([1] * len(X_out), index=X_out.index)

        # 1. PM10 Ratio Imputation
        missing_pm10 = X_out['PM10'].isnull()
        for idx in X_out[missing_pm10].index:
            pm25 = X_out.loc[idx, 'PM2.5'] if 'PM2.5' in X_out.columns else np.nan
            m = int(months.loc[idx]) if pd.notnull(months.loc[idx]) else 1

            if pd.notnull(pm25) and pm25 > 0:
                r = self.month_ratios_.get(m, self.national_ratio_)
                X_out.loc[idx, 'PM10'] = float(pm25 / r)
            else:
                X_out.loc[idx, 'PM10'] = self.global_medians_.get('PM10', 101.88)

        # 2. PM2.5 Fallback Imputation (if PM2.5 is missing but PM10 is given)
        if 'PM2.5' in X_out.columns:
            missing_pm25 = X_out['PM2.5'].isnull()
            for idx in X_out[missing_pm25].index:
                pm10 = X_out.loc[idx, 'PM10']
                m = int(months.loc[idx]) if pd.notnull(months.loc[idx]) else 1
                if pd.notnull(pm10) and pm10 > 0:
                    r = self.month_ratios_.get(m, self.national_ratio_)
                    X_out.loc[idx, 'PM2.5'] = float(pm10 * r)
                else:
                    X_out.loc[idx, 'PM2.5'] = self.global_medians_.get('PM2.5', 52.20)

        # 3. Global median fallback for remaining pollutants
        for col in self.feature_names_:
            if col in X_out.columns:
                med = self.global_medians_.get(col, 0.0)
                X_out[col] = X_out[col].fillna(med)

        return X_out[self.feature_names_]

    def process_single_input(
        self,
        pm25: Optional[float],
        pm10: Optional[float],
        no2: float,
        co: float,
        so2: float,
        o3: float,
        month: int = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Validates, imputes, and orders a single operational prediction request.
        Returns (feature_vector_1x6, warnings_list).
        """
        warnings = []

        # Validate non-particulate required pollutants
        for name, val in [('NO2', no2), ('CO', co), ('SO2', so2), ('O3', o3)]:
            if val is None or not np.isfinite(val):
                raise ValueError(f"Pollutant '{name}' must be a finite number.")
            if val < 0.0:
                raise ValueError(f"Pollutant '{name}' cannot be negative ({val}).")

        # Validate provided particulates
        if pm25 is not None:
            if not np.isfinite(pm25): raise ValueError("PM2.5 must be a finite number.")
            if pm25 < 0.0: raise ValueError(f"PM2.5 cannot be negative ({pm25}).")

        if pm10 is not None:
            if not np.isfinite(pm10): raise ValueError("PM10 must be a finite number.")
            if pm10 < 0.0: raise ValueError(f"PM10 cannot be negative ({pm10}).")

        # Check CO anomaly threshold
        if co > CO_ANOMALY_THRESHOLD:
            warnings.append(
                f"Carbon Monoxide (CO={co:.2f} mg/m³) exceeds statutory ambient threshold "
                f"({CO_ANOMALY_THRESHOLD} mg/m³). Model predictions reflect standard ambient dispersion."
            )

        # Particulate handling & estimation
        final_pm25 = pm25
        final_pm10 = pm10

        if final_pm10 is None and final_pm25 is not None:
            r = self.month_ratios_.get(month, self.national_ratio_)
            final_pm10 = float(final_pm25 / r)
            warnings.append(
                f"PM10 was not provided. Estimated as {final_pm10:.2f} µg/m³ using "
                f"training-derived monthly ratio (PM2.5/PM10 = {r:.3f} for month {month})."
            )
        elif final_pm25 is None and final_pm10 is not None:
            r = self.month_ratios_.get(month, self.national_ratio_)
            final_pm25 = float(final_pm10 * r)
            warnings.append(
                f"PM2.5 was not provided. Estimated as {final_pm25:.2f} µg/m³ using "
                f"training-derived monthly ratio (PM2.5/PM10 = {r:.3f} for month {month})."
            )
        elif final_pm10 is None and final_pm25 is None:
            # Both missing
            final_pm25 = self.global_medians_.get('PM2.5', 52.20)
            final_pm10 = self.global_medians_.get('PM10', 101.88)
            warnings.append(
                "Neither PM2.5 nor PM10 was provided. Fallback global historical medians applied."
            )

        features = np.array([[final_pm25, final_pm10, no2, co, so2, o3]], dtype=np.float64)
        return features, warnings



class AirSenseCompositeModel:
    """
    Complete self-contained composite inference artifact for AirSense AI Release Candidate.
    Combines the fitted AirSensePreprocessor and RandomForestRegressor.
    """
    def __init__(self, preprocessor: AirSensePreprocessor, regressor, version: str = "2.0.0"):
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.version = version
        self.feature_names = FEATURE_ORDER
        self.co_anomaly_threshold = CO_ANOMALY_THRESHOLD
        self.aqi_max_statutory = AQI_MAX_STATUTORY

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Accepts DataFrame with features (and optional Month/Date),
        transforms features and predicts next-day AQI.
        """
        X_trans = self.preprocessor.transform(X)
        return self.regressor.predict(X_trans[self.feature_names])

    def predict_single(
        self,
        pm25: Optional[float],
        pm10: Optional[float],
        no2: float,
        co: float,
        so2: float,
        o3: float,
        month: int = 1
    ) -> Dict[str, Any]:
        """
        Operational single-prediction interface for FastAPI backend.
        """
        features, warnings = self.preprocessor.process_single_input(
            pm25=pm25, pm10=pm10, no2=no2, co=co, so2=so2, o3=o3, month=month
        )
        features_df = pd.DataFrame(features, columns=self.feature_names)
        raw_pred = float(self.regressor.predict(features_df)[0])
        clamped_pred = float(np.clip(raw_pred, 0.0, self.aqi_max_statutory))

        # Categorize
        if clamped_pred <= 50: category = "Good"
        elif clamped_pred <= 100: category = "Satisfactory"
        elif clamped_pred <= 200: category = "Moderate"
        elif clamped_pred <= 300: category = "Poor"
        elif clamped_pred <= 400: category = "Very Poor"
        else: category = "Severe"

        return {
            'raw_predicted_aqi': round(raw_pred, 2),
            'predicted_aqi': round(clamped_pred, 2),
            'aqi_category': category,
            'warnings': warnings,
            'features_used': {
                name: round(float(features[0, i]), 2) for i, name in enumerate(self.feature_names)
            },
            'model_version': self.version
        }
