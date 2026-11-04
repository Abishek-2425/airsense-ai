"""
AirSense AI — Phase 2B Robust Preprocessing Module (v2.0.0 Production Candidate)

Provides serializable, leakage-free particulate ratio imputation, strict input validation,
for operational next-day AQI prediction.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_ORDER = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
CO_ANOMALY_THRESHOLD = 17.0  # Project-defined CO screening threshold derived from Maninagar, Ahmedabad audit
AQI_MAX_STATUTORY = 500.0

_OMITTED = object()


def validate_month_integer(month: Any, allow_none: bool = False) -> int:
    """
    Strictly validates that 'month' is a genuine integer between 1 and 12.
    Explicitly rejects booleans (True/False), numpy booleans, floats (including 3.0),
    strings ('3'), None (unless allow_none=True), and out-of-range values.
    """
    if month is None:
        if allow_none:
            return datetime.now().month
        raise ValueError("Month cannot be None when a month value is supplied.")

    # In Python, bool is a subclass of int: isinstance(True, int) is True!
    if isinstance(month, (bool, np.bool_)):
        raise ValueError(
            f"Invalid month value {month} (type: {type(month).__name__}): month must be a genuine integer between 1 and 12."
        )

    # Reject floats (even 3.0, np.float64(3.0)), strings, etc. Must be genuine int or numpy integer.
    if not isinstance(month, (int, np.integer)):
        raise ValueError(
            f"Invalid month value {repr(month)} (type: {type(month).__name__}): month must be a genuine integer between 1 and 12."
        )

    if not (1 <= month <= 12):
        raise ValueError(
            f"Month must be an integer between 1 and 12 (received: {month})."
        )

    return int(month)


class AirSensePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for AirSense AI.
    Implements physical PM2.5 / PM10 monthly ratio imputation for missing PM10,
    with a documented fallback hierarchy, strict input validation, and zero data leakage.

    Note on CO Threshold:
    CO_ANOMALY_THRESHOLD (17.0 mg/m³) is a project-defined screening threshold derived
    from the historical dataset audit (Maninagar, Ahmedabad uncalibrated sensor contamination).
    It is not a statutory regulatory standard, nor are the historical dataset's raw CO units
    independently verified.
    """
    def __init__(self):
        self.month_ratios_: Dict[int, float] = {}
        self.national_ratio_: Optional[float] = None
        self.global_medians_: Dict[str, float] = {}
        self.feature_names_: List[str] = FEATURE_ORDER
        self._is_fitted: bool = False
        self.provenance_: Optional[str] = None
        self.ratio_provenance_: Dict[int, str] = {}

    @property
    def is_fitted_(self) -> bool:
        """
        Returns True if the preprocessor is fitted.
        Maintains backward compatibility for deserialized artifacts whose __dict__
        already holds fitted statistics (month_ratios_ and global_medians_).
        """
        if getattr(self, '_is_fitted', False):
            return True
        return bool(getattr(self, 'month_ratios_', None) and getattr(self, 'global_medians_', None))

    @is_fitted_.setter
    def is_fitted_(self, value: bool):
        self._is_fitted = bool(value)

    @classmethod
    def from_frozen_statistics(cls, metadata_or_path: Any = None) -> "AirSensePreprocessor":
        """
        Constructs and returns an AirSensePreprocessor fitted from frozen training metadata.
        Strictly validates that all 12 monthly ratios and all 6 pollutant medians are present,
        numeric, and positive. Zero silent fallback constants or fabricated values are used.
        """
        import json

        if metadata_or_path is None:
            candidate_paths = [
                Path(__file__).resolve().parent.parent / "model_validation" / "phase_2b" / "release_candidate" / "preprocessor_metadata.json",
                Path(__file__).resolve().parent.parent / "model_metadata.json",
                Path("model_validation/phase_2b/release_candidate/preprocessor_metadata.json"),
                Path("model_metadata.json"),
            ]
            meta = None
            resolved_path = None
            for p in candidate_paths:
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    resolved_path = str(p)
                    break
            if meta is None:
                raise RuntimeError("Cannot construct preprocessor from frozen statistics: metadata file not found.")
        elif isinstance(metadata_or_path, (str, Path)):
            resolved_path = str(metadata_or_path)
            with open(metadata_or_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        elif isinstance(metadata_or_path, dict):
            resolved_path = "supplied_dict"
            meta = metadata_or_path
        else:
            raise TypeError(f"Expected dict, str, or Path for metadata_or_path, received {type(metadata_or_path).__name__}")

        prep_block = meta.get("preprocessing", meta)

        # 1. National ratio
        nat_ratio = prep_block.get("national_ratio", prep_block.get("national_pm_ratio"))
        if nat_ratio is None or not isinstance(nat_ratio, (int, float, np.number)) or isinstance(nat_ratio, bool) or not np.isfinite(nat_ratio) or nat_ratio <= 0:
            raise ValueError(f"Invalid or missing national ratio in metadata: {nat_ratio}. Must be a finite positive number.")

        # 2. Monthly ratios (all 12 calendar months required)
        monthly = prep_block.get("month_ratios", {})
        if not monthly or not isinstance(monthly, dict):
            raise ValueError("Missing or invalid 'month_ratios' dictionary in metadata.")

        month_ratios_dict = {}
        for m in range(1, 13):
            val = monthly.get(str(m), monthly.get(m))
            if val is None or not isinstance(val, (int, float, np.number)) or isinstance(val, bool) or not np.isfinite(val) or val <= 0:
                raise ValueError(f"Incomplete monthly ratios in metadata: month {m} is missing or invalid ({val}).")
            month_ratios_dict[m] = float(val)

        # 3. Global pollutant medians (all 6 criteria pollutants required)
        medians = prep_block.get("global_medians", prep_block.get("global_pollutant_medians", prep_block.get("imputer_medians", {})))
        if not medians or not isinstance(medians, dict):
            raise ValueError("Missing or invalid pollutant medians dictionary in metadata.")

        global_medians_dict = {}
        for col in FEATURE_ORDER:
            if col not in medians:
                raise ValueError(f"Incomplete preprocessor metadata: missing median for feature '{col}'.")
            val = medians[col]
            if not isinstance(val, (int, float, np.number)) or isinstance(val, bool) or not np.isfinite(val) or val <= 0:
                raise ValueError(f"Invalid global median for '{col}': {val}. Must be a finite positive number.")
            global_medians_dict[col] = float(val)

        instance = cls()
        instance.national_ratio_ = float(nat_ratio)
        instance.month_ratios_ = month_ratios_dict
        instance.global_medians_ = global_medians_dict
        instance.ratio_provenance_ = {m: f"frozen_metadata:{Path(resolved_path).name}" for m in range(1, 13)}
        instance.provenance_ = f"frozen_metadata:{Path(resolved_path).name}"
        instance.is_fitted_ = True
        return instance

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits ratio and median statistics strictly on historical training data.
        """
        if 'PM10' not in X.columns or 'PM2.5' not in X.columns:
            raise ValueError("Training DataFrame X must contain both 'PM10' and 'PM2.5' columns.")

        # Ensure Month is available
        if 'Month' not in X.columns:
            if 'Date' in X.columns:
                months = pd.to_datetime(X['Date'], errors='coerce').dt.month
            else:
                months = pd.Series([1] * len(X), index=X.index)
        else:
            months = X['Month']

        # Calculate pollutant global medians
        for col in self.feature_names_:
            if col in X.columns:
                med = X[col].dropna().median()
                self.global_medians_[col] = float(med) if pd.notnull(med) else 0.0
            else:
                self.global_medians_[col] = 0.0

        # Fit PM2.5/PM10 ratio on paired non-null training observations
        paired_mask = (X['PM10'] > 0) & (X['PM2.5'] > 0) & (X['PM2.5'] <= 1.2 * X['PM10'])
        paired = X[paired_mask].copy()
        paired['Ratio'] = (paired['PM2.5'] / paired['PM10']).to_numpy()
        paired['Month'] = months[paired_mask].to_numpy()

        if len(paired) == 0:
            raise ValueError(
                "Cannot fit AirSensePreprocessor: training data contains zero valid non-null "
                "particulate pairs satisfying (PM10 > 0, PM2.5 > 0, and PM2.5 <= 1.2 * PM10)."
            )

        self.national_ratio_ = float(paired['Ratio'].median())
        learned_month_ratios = {}
        for m, r in paired.groupby('Month')['Ratio'].median().items():
            try:
                learned_month_ratios[int(m)] = float(r)
            except (ValueError, TypeError):
                pass

        self.month_ratios_ = {}
        self.ratio_provenance_ = {}
        for m in range(1, 13):
            if m in learned_month_ratios:
                self.month_ratios_[m] = learned_month_ratios[m]
                self.ratio_provenance_[m] = "learned_training_median"
            else:
                # Documented fallback to learned national ratio (never a hardcoded constant)
                self.month_ratios_[m] = self.national_ratio_
                self.ratio_provenance_[m] = "fallback_learned_national_ratio"

        self.provenance_ = "runtime_fitted"
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms DataFrame by filling missing PM10 using monthly ratios,
        and filling remaining missing features with training medians.
        """
        if not self.is_fitted_:
            raise RuntimeError("AirSensePreprocessor instance is not fitted. Call 'fit()' or use 'from_frozen_statistics()'.")

        X_out = X.copy()
        n_rows = len(X_out)

        # Handle Month/Date
        if 'Month' in X_out.columns:
            months_raw = X_out['Month'].tolist()
            validated_list = []
            for pos, val in enumerate(months_raw):
                try:
                    v = validate_month_integer(val, allow_none=False)
                    validated_list.append(v)
                except ValueError as ve:
                    idx_lbl = X_out.index[pos]
                    raise ValueError(f"Invalid month at index {idx_lbl} in DataFrame column 'Month': {ve}") from ve
        elif 'Date' in X_out.columns:
            date_series = pd.to_datetime(X_out['Date'], errors='coerce')
            if date_series.isnull().any():
                bad_pos = int(date_series.isnull().to_numpy().nonzero()[0][0])
                bad_idx = X_out.index[bad_pos]
                raise ValueError(f"Invalid or unparseable Date at index {bad_idx}: '{X_out.iloc[bad_pos]['Date']}'")
            validated_list = [int(m) for m in date_series.dt.month.to_numpy()]
        else:
            # Genuinely omitted Month/Date columns: use current month
            cur_month = datetime.now().month
            validated_list = [cur_month] * n_rows

        # Particulate arrays for index-independent, positional handling
        pm25_vals = X_out['PM2.5'].to_numpy(copy=True, dtype=float) if 'PM2.5' in X_out.columns else np.full(n_rows, np.nan)
        pm10_vals = X_out['PM10'].to_numpy(copy=True, dtype=float) if 'PM10' in X_out.columns else np.full(n_rows, np.nan)

        orig_missing_pm10 = np.isnan(pm10_vals)
        orig_missing_pm25 = np.isnan(pm25_vals)

        pm10_median = self.global_medians_.get('PM10', 101.82)
        pm25_median = self.global_medians_.get('PM2.5', 52.15)

        for i in range(n_rows):
            m = validated_list[i]
            r = self.month_ratios_.get(m, self.national_ratio_)

            # 1. PM10 Imputation
            if orig_missing_pm10[i]:
                if not orig_missing_pm25[i] and not np.isnan(pm25_vals[i]) and pm25_vals[i] > 0:
                    pm10_vals[i] = float(pm25_vals[i] / r)
                else:
                    pm10_vals[i] = float(pm10_median)

            # 2. PM2.5 Imputation
            if orig_missing_pm25[i]:
                # Only apply ratio if PM10 was genuinely provided by user, not freshly imputed from median
                if not orig_missing_pm10[i] and not np.isnan(pm10_vals[i]) and pm10_vals[i] > 0:
                    pm25_vals[i] = float(pm10_vals[i] * r)
                else:
                    pm25_vals[i] = float(pm25_median)

        X_out['PM2.5'] = pm25_vals
        X_out['PM10'] = pm10_vals

        for col in self.feature_names_:
            if col not in ('PM2.5', 'PM10') and col in X_out.columns:
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
        month: Any = _OMITTED
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Validates, imputes, and orders a single operational prediction request.
        Returns (feature_vector_1x6, warnings_list).
        """
        if not self.is_fitted_:
            raise RuntimeError("AirSensePreprocessor instance is not fitted. Call 'fit()' or use 'from_frozen_statistics()'.")

        if month is _OMITTED:
            month_int = datetime.now().month
        else:
            month_int = validate_month_integer(month, allow_none=False)

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

        # Check project-defined CO screening threshold
        if co > CO_ANOMALY_THRESHOLD:
            warnings.append(
                f"Carbon Monoxide (CO={co:.2f} mg/m³) exceeds the project-defined CO screening threshold "
                f"({CO_ANOMALY_THRESHOLD:.1f} mg/m³). Model predictions may have increased uncertainty "
                f"above this level."
            )

        # Particulate handling & estimation
        final_pm25 = pm25
        final_pm10 = pm10

        if final_pm10 is None and final_pm25 is not None:
            r = self.month_ratios_.get(month_int, self.national_ratio_)
            final_pm10 = float(final_pm25 / r)
            warnings.append(
                f"PM10 was not provided. Estimated as {final_pm10:.2f} µg/m³ using "
                f"training-derived monthly ratio (PM2.5/PM10 = {r:.3f} for month {month_int})."
            )
        elif final_pm25 is None and final_pm10 is not None:
            r = self.month_ratios_.get(month_int, self.national_ratio_)
            final_pm25 = float(final_pm10 * r)
            warnings.append(
                f"PM2.5 was not provided. Estimated as {final_pm25:.2f} µg/m³ using "
                f"training-derived monthly ratio (PM2.5/PM10 = {r:.3f} for month {month_int})."
            )
        elif final_pm10 is None and final_pm25 is None:
            # Both missing
            final_pm25 = self.global_medians_.get('PM2.5', 52.15)
            final_pm10 = self.global_medians_.get('PM10', 101.82)
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
        month: Any = _OMITTED
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
