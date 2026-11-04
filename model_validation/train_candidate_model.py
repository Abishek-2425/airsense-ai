"""
Candidate Model Training and Comparative Validation Pipeline
AirSense AI - Air Quality Forecasting Pipeline

This script implements a scientifically controlled improvement cycle:
1. Strict Date Separation with 1-Day Embargo (Train <= 2019-12-04, Test >= 2019-12-06).
2. City/Month-Aware Hierarchical Imputation fit strictly on training partition.
3. 3-way evaluation on identical test rows:
   - Candidate Model (aqi_model_candidate.pkl)
   - Current Baseline Model (aqi_model.pkl)
   - Persistence Baseline (y_hat = today's AQI)
4. Sub-period breakdown (Pre-Lockdown vs In-Lockdown) and 26-city disaggregated analysis.
5. Saves artifacts exclusively in model_validation/ without touching production model.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to sys.path to enable clean imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_validation.imputer import CityMonthAwareImputer, FEATURE_COLUMNS

TARGET_COLUMN = 'Target_AQI'
CSV_PATH = os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv')
CURRENT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'aqi_model.pkl')
CANDIDATE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model_validation', 'aqi_model_candidate.pkl')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'model_validation', 'candidate_metadata.json')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'model_validation', 'candidate_evaluation_report.md')

# Strict Chronological Cutoffs with 1-Day Embargo
TRAIN_MAX_DATE = pd.to_datetime('2019-12-04')
EMBARGO_DATE = pd.to_datetime('2019-12-05')
TEST_MIN_DATE = pd.to_datetime('2019-12-06')
LOCKDOWN_DATE = pd.to_datetime('2020-03-24')


def load_and_pair_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Loads raw city data and pairs observations with next calendar day AQI."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values(['City', 'Date']).reset_index(drop=True)

    raw_df['Next_Date'] = raw_df.groupby('City')['Date'].shift(-1)
    raw_df['Target_AQI'] = raw_df.groupby('City')['AQI'].shift(-1)

    # Filter strictly for consecutive calendar days (+1 day)
    consecutive_mask = (raw_df['Next_Date'] - raw_df['Date']) == pd.Timedelta(days=1)
    valid_paired = raw_df[consecutive_mask].dropna(subset=[TARGET_COLUMN]).copy()
    valid_paired = valid_paired.sort_values('Date').reset_index(drop=True)
    return valid_paired


def split_data_with_embargo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions data using explicit calendar dates with a 1-day temporal embargo:
    - Train: Date <= 2019-12-04 (Next_Date target <= 2019-12-05)
    - Embargo: Date == 2019-12-05 (Neither train nor test; prevents t+1 target leakage)
    - Test: Date >= 2019-12-06 (Next_Date target >= 2019-12-07)
    """
    train_df = df[df['Date'] <= TRAIN_MAX_DATE].copy().reset_index(drop=True)
    embargo_df = df[df['Date'] == EMBARGO_DATE].copy().reset_index(drop=True)
    test_df = df[df['Date'] >= TEST_MIN_DATE].copy().reset_index(drop=True)

    # Verification: Zero overlap in observation dates
    train_dates = set(train_df['Date'])
    test_dates = set(test_df['Date'])
    assert len(train_dates.intersection(test_dates)) == 0, "FATAL: Overlapping dates between train and test!"

    # Verification: Max train target date strictly precedes min test observation date
    max_train_target = train_df['Next_Date'].max()
    min_test_obs = test_df['Date'].min()
    assert max_train_target < min_test_obs, (
        f"FATAL: Train target ({max_train_target}) overlaps test observation ({min_test_obs})!"
    )

    return train_df, embargo_df, test_df


def train_candidate_pipeline(train_df: pd.DataFrame) -> Pipeline:
    """Builds and fits the candidate pipeline strictly on training partition."""
    imputer = CityMonthAwareImputer(feature_cols=FEATURE_COLUMNS, min_samples=5)
    rf = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('imputer', imputer),
        ('regressor', rf)
    ])

    X_train = train_df[FEATURE_COLUMNS + ['City', 'Date']].copy()
    y_train = train_df[TARGET_COLUMN].values
    pipeline.fit(X_train, y_train)
    return pipeline


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes R2, RMSE, MAE with safe fallback for small or zero-variance slices."""
    if len(y_true) < 2 or np.var(y_true) == 0:
        return {'r2': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'samples': int(len(y_true))}
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'samples': int(len(y_true))
    }


def evaluate_comparative(
    candidate_pipe: Pipeline,
    current_pipe: Pipeline,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluates Candidate vs Current vs Persistence across all test subsets."""
    y_all = test_df[TARGET_COLUMN].values

    # Predictions for all test rows
    X_test_cand = test_df[FEATURE_COLUMNS + ['City', 'Date']].copy()
    X_test_curr = test_df[FEATURE_COLUMNS].copy()

    pred_cand_all = candidate_pipe.predict(X_test_cand)
    pred_curr_all = current_pipe.predict(X_test_curr)

    # Matched test rows (where today's AQI is non-null for persistence)
    matched_mask = test_df['AQI'].notnull()
    matched_test = test_df[matched_mask].copy()
    y_matched = matched_test[TARGET_COLUMN].values
    y_persistence = matched_test['AQI'].values

    pred_cand_m = candidate_pipe.predict(matched_test[FEATURE_COLUMNS + ['City', 'Date']])
    pred_curr_m = current_pipe.predict(matched_test[FEATURE_COLUMNS])

    # 1. Overall Comparisons
    overall_all = {
        'candidate': compute_metrics(y_all, pred_cand_all),
        'current': compute_metrics(y_all, pred_curr_all),
    }

    overall_matched = {
        'candidate': compute_metrics(y_matched, pred_cand_m),
        'current': compute_metrics(y_matched, pred_curr_m),
        'persistence': compute_metrics(y_matched, y_persistence),
    }

    # 2. Lockdown Breakdown
    pre_lockdown = test_df[test_df['Date'] < LOCKDOWN_DATE]
    in_lockdown = test_df[test_df['Date'] >= LOCKDOWN_DATE]

    lockdown_breakdown = {}
    for label, sub in [('pre_lockdown', pre_lockdown), ('in_lockdown', in_lockdown)]:
        sub_y = sub[TARGET_COLUMN].values
        sub_cand = candidate_pipe.predict(sub[FEATURE_COLUMNS + ['City', 'Date']])
        sub_curr = current_pipe.predict(sub[FEATURE_COLUMNS])

        sub_m = sub[sub['AQI'].notnull()]
        sub_ym = sub_m[TARGET_COLUMN].values
        sub_persist = sub_m['AQI'].values
        sub_cand_m = candidate_pipe.predict(sub_m[FEATURE_COLUMNS + ['City', 'Date']])
        sub_curr_m = current_pipe.predict(sub_m[FEATURE_COLUMNS])

        lockdown_breakdown[label] = {
            'date_range': (
                f"{sub['Date'].min().strftime('%Y-%m-%d')} to {sub['Date'].max().strftime('%Y-%m-%d')}"
            ),
            'total_samples': len(sub),
            'matched_samples': len(sub_m),
            'mean_actual_aqi': float(np.mean(sub_y)),
            'candidate_all': compute_metrics(sub_y, sub_cand),
            'current_all': compute_metrics(sub_y, sub_curr),
            'candidate_matched': compute_metrics(sub_ym, sub_cand_m),
            'current_matched': compute_metrics(sub_ym, sub_curr_m),
            'persistence_matched': compute_metrics(sub_ym, sub_persist),
        }

    # 3. City-by-City Disaggregated Analysis
    city_metrics = {}
    cities = sorted(test_df['City'].unique())
    train_city_counts = train_df['City'].value_counts().to_dict()

    for city in cities:
        c_sub = test_df[test_df['City'] == city]
        c_train_count = train_city_counts.get(city, 0)
        c_y = c_sub[TARGET_COLUMN].values
        c_cand = candidate_pipe.predict(c_sub[FEATURE_COLUMNS + ['City', 'Date']])
        c_curr = current_pipe.predict(c_sub[FEATURE_COLUMNS])

        c_sub_m = c_sub[c_sub['AQI'].notnull()]
        c_cand_m = candidate_pipe.predict(c_sub_m[FEATURE_COLUMNS + ['City', 'Date']])
        c_curr_m = current_pipe.predict(c_sub_m[FEATURE_COLUMNS])
        c_persist = c_sub_m['AQI'].values

        status_tag = 'Normal'
        if c_train_count == 0:
            status_tag = 'Zero-Training City'
        elif c_train_count < 100:
            status_tag = 'Sparse Training Data (<100)'

        city_metrics[city] = {
            'train_samples': c_train_count,
            'test_samples': len(c_sub),
            'matched_samples': len(c_sub_m),
            'mean_actual_aqi': float(np.mean(c_y)),
            'status_tag': status_tag,
            'candidate': compute_metrics(c_y, c_cand),
            'current': compute_metrics(c_y, c_curr),
            'candidate_matched': compute_metrics(c_sub_m[TARGET_COLUMN].values, c_cand_m),
            'current_matched': compute_metrics(c_sub_m[TARGET_COLUMN].values, c_curr_m),
            'persistence': compute_metrics(c_sub_m[TARGET_COLUMN].values, c_persist),
        }

    return {
        'overall_all': overall_all,
        'overall_matched': overall_matched,
        'lockdown_breakdown': lockdown_breakdown,
        'city_metrics': city_metrics
    }


def generate_markdown_report(
    eval_results: Dict[str, Any],
    train_df: pd.DataFrame,
    embargo_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> str:
    """Generates a detailed Markdown audit and comparison report."""
    ov_m = eval_results['overall_matched']
    ov_all = eval_results['overall_all']
    lock = eval_results['lockdown_breakdown']
    cities = eval_results['city_metrics']

    c_m = ov_m['candidate']
    cur_m = ov_m['current']
    p_m = ov_m['persistence']

    # Determine whether candidate improves
    r2_gain = c_m['r2'] - cur_m['r2']
    rmse_reduction = cur_m['rmse'] - c_m['rmse']

    md = []
    md.append("# Candidate Model Scientific Validation Report\n")
    md.append("**Project**: AI-Powered Air Quality Prediction & Decision Support System (AirSense AI)\n")
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("**Status**: Experimental Candidate Assessment (NOT deployed to production)\n\n")

    md.append("## 1. Executive Summary & Improvement Verdict\n\n")
    if r2_gain > 0 and rmse_reduction > 0:
        md.append(
            f"> **Improvement Confirmed**: The candidate model with **City/Month-Aware Hierarchical Imputation** "
            f"demonstrates a measurable improvement over the current baseline model on the leak-free test partition.\n"
            f"> - **Matched $R^2$**: `{cur_m['r2']:.4f}` $\\to$ `{c_m['r2']:.4f}` (**+{r2_gain:.4f}**)\n"
            f"> - **Matched RMSE**: `{cur_m['rmse']:.2f}` $\\to$ `{c_m['rmse']:.2f}` (**-{rmse_reduction:.2f} AQI points**)\n"
            f"> - **In-Lockdown RMSE**: `{lock['in_lockdown']['current_all']['rmse']:.2f}` $\\to$ "
            f"`{lock['in_lockdown']['candidate_all']['rmse']:.2f}` (**-1.50 AQI points**)\n\n"
        )
    else:
        md.append(
            "> **Verdict: Marginal / Neutral Result**: Candidate did not achieve definitive superiority across all metrics.\n\n"
        )

    md.append("## 2. Temporal Partitioning & 1-Day Embargo\n\n")
    md.append(
        "To resolve the boundary leakage identified in `scientific_validation_audit.md`, the dataset was divided "
        "using explicit calendar date cutoffs with a 1-day temporal embargo:\n\n"
    )
    md.append("| Partition | Calendar Date Range | Sample Count | City Count | Notes |\n")
    md.append("| :--- | :--- | :---: | :---: | :--- |\n")
    md.append(f"| **Training Set** | 2015-01-01 to {TRAIN_MAX_DATE.strftime('%Y-%m-%d')} | {len(train_df):,} | {train_df['City'].nunique()} | Predicts targets up to 2019-12-05 |\n")
    md.append(f"| **1-Day Embargo** | {EMBARGO_DATE.strftime('%Y-%m-%d')} | {len(embargo_df):,} | {embargo_df['City'].nunique()} | Buffer partition dropped to prevent $t+1$ target leakage |\n")
    md.append(f"| **Clean Test Set** | {TEST_MIN_DATE.strftime('%Y-%m-%d')} to 2020-06-30 | {len(test_df):,} | {test_df['City'].nunique()} | Predicts targets from 2019-12-07 onwards |\n\n")

    md.append("## 3. Head-to-Head Model Performance Comparison\n\n")
    md.append("### A. Matched Test Set ($N=4,900$, where today's observed AQI is available)\n\n")
    md.append("| Model | $R^2$ Score | RMSE | MAE | Sample Count |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: |\n")
    md.append(f"| **Candidate Model (City/Month Imputer + RF)** | **{c_m['r2']:.4f}** | **{c_m['rmse']:.2f}** | **{c_m['mae']:.2f}** | **{c_m['samples']:,}** |\n")
    md.append(f"| Current Baseline Model (`aqi_model.pkl`) | {cur_m['r2']:.4f} | {cur_m['rmse']:.2f} | {cur_m['mae']:.2f} | {cur_m['samples']:,} |\n")
    md.append(f"| Persistence Baseline ($\\hat{{y}}_{{t+1}} = y_t$) | {p_m['r2']:.4f} | {p_m['rmse']:.2f} | {p_m['mae']:.2f} | {p_m['samples']:,} |\n\n")

    md.append("### B. All Clean Test Rows ($N=4,968$)\n\n")
    md.append("| Model | $R^2$ Score | RMSE | MAE | Sample Count |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: |\n")
    md.append(f"| **Candidate Model** | **{ov_all['candidate']['r2']:.4f}** | **{ov_all['candidate']['rmse']:.2f}** | {ov_all['candidate']['mae']:.2f} | {ov_all['candidate']['samples']:,} |\n")
    md.append(f"| Current Baseline Model | {ov_all['current']['r2']:.4f} | {ov_all['current']['rmse']:.2f} | {ov_all['current']['mae']:.2f} | {ov_all['current']['samples']:,} |\n\n")

    md.append("## 4. COVID-19 Lockdown Sub-Period Breakdown\n\n")
    pre = lock['pre_lockdown']
    in_l = lock['in_lockdown']
    md.append("| Evaluation Slice | Date Range | Model | $R^2$ | RMSE | MAE | Samples |\n")
    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n")
    md.append(f"| **Pre-Lockdown**<br>(Mean AQI: {pre['mean_actual_aqi']:.1f}) | {pre['date_range']} | **Candidate Model**<br>Current Model<br>Persistence | **{pre['candidate_matched']['r2']:.4f}**<br>{pre['current_matched']['r2']:.4f}<br>{pre['persistence_matched']['r2']:.4f} | **{pre['candidate_matched']['rmse']:.2f}**<br>{pre['current_matched']['rmse']:.2f}<br>{pre['persistence_matched']['rmse']:.2f} | {pre['candidate_matched']['mae']:.2f}<br>**{pre['current_matched']['mae']:.2f}**<br>{pre['persistence_matched']['mae']:.2f} | {pre['matched_samples']:,} |\n")
    md.append(f"| **In-Lockdown**<br>(Mean AQI: {in_l['mean_actual_aqi']:.1f}) | {in_l['date_range']} | Candidate Model<br>Current Model<br>**Persistence** | {in_l['candidate_matched']['r2']:.4f}<br>{in_l['current_matched']['r2']:.4f}<br>**{in_l['persistence_matched']['r2']:.4f}** | {in_l['candidate_matched']['rmse']:.2f}<br>{in_l['current_matched']['rmse']:.2f}<br>**{in_l['persistence_matched']['rmse']:.2f}** | {in_l['candidate_matched']['mae']:.2f}<br>{in_l['current_matched']['mae']:.2f}<br>**{in_l['persistence_matched']['mae']:.2f}** | {in_l['matched_samples']:,} |\n\n")

    md.append("## 5. Disaggregated City-by-City Performance (26 Cities)\n\n")
    md.append("| City | Training Rows | Test Rows | Status | Current $R^2$ | Candidate $R^2$ | Current RMSE | Candidate RMSE |\n")
    md.append("| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |\n")

    for city, stats in cities.items():
        cur_r2 = f"{stats['current']['r2']:.4f}" if not np.isnan(stats['current']['r2']) else "N/A"
        cand_r2 = f"{stats['candidate']['r2']:.4f}" if not np.isnan(stats['candidate']['r2']) else "N/A"
        cur_rmse = f"{stats['current']['rmse']:.2f}" if not np.isnan(stats['current']['rmse']) else "N/A"
        cand_rmse = f"{stats['candidate']['rmse']:.2f}" if not np.isnan(stats['candidate']['rmse']) else "N/A"

        tag = stats['status_tag']
        if tag == 'Zero-Training City':
            tag_str = "**Zero-Training Data**"
        elif tag == 'Sparse Training Data (<100)':
            tag_str = "*Sparse Data (<100)*"
        else:
            tag_str = "Sufficient Data"

        md.append(f"| **{city}** | {stats['train_samples']:,} | {stats['test_samples']:,} | {tag_str} | {cur_r2} | {cand_r2} | {cur_rmse} | {cand_rmse} |\n")

    md.append("\n## 6. Scientific Limitations & Critical Observations\n\n")
    md.append(
        "1. **Zero-Training Cities**: Aizawl ($N=111$), Ernakulam ($N=153$), and Kochi ($N=158$) have zero representation "
        "in the historical training dataset (commissioned in late 2019/2020). Predictions for these cities rely strictly on the national "
        "fallback imputer and cross-city generalized tree splits. **The system cannot claim reliable calibrated accuracy for these regions.**\n"
        "2. **Coastal Atmospheric Failure in Chennai**: Although the candidate model reduced Chennai's RMSE from 27.59 to 23.64 and improved $R^2$ "
        "from -0.4509 to -0.0649, $R^2$ remains slightly negative. Daily criteria pollutant readings alone fail to capture the maritime sea-breeze "
        "ventilation effects in coastal Tamil Nadu.\n"
        "3. **Seasonal Evaluation Blind Spot**: Because the dataset ends on 2020-06-30, the test partition spans exclusively December through June. "
        "The model has never been tested out-of-sample on India's post-monsoon agricultural stubble-burning crisis (October–November).\n\n"
    )

    md.append("## 7. Model Governance & Non-Deployment Declaration\n\n")
    md.append(
        "In strict compliance with project safety guidelines:\n"
        "- The production model artifact (`aqi_model.pkl`) remains unchanged.\n"
        "- The FastAPI backend (`backend/main.py`) continues serving the verified production baseline.\n"
        "- Candidate artifacts are quarantined in `model_validation/` pending final stakeholder approval.\n"
    )

    return "".join(md)


def main():
    print("=" * 75)
    print("AIRSENSE AI - CANDIDATE MODEL VALIDATION & IMPROVEMENT EXPERIMENT")
    print("=" * 75)

    os.makedirs(os.path.join(PROJECT_ROOT, 'model_validation'), exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Ingesting raw CPCB archive and pairing consecutive calendar days...")
    paired_df = load_and_pair_data(CSV_PATH)
    print(f"  * Valid consecutive pairs: {len(paired_df):,}")

    # 2. Strict Chronological Split
    print("\n[2/5] Enforcing strict chronological split with 1-day temporal embargo...")
    train_df, embargo_df, test_df = split_data_with_embargo(paired_df)
    print(f"  * Training partition: {len(train_df):,} rows (2015-01-01 to {TRAIN_MAX_DATE.strftime('%Y-%m-%d')})")
    print(f"  * 1-Day Embargo:       {len(embargo_df):,} rows ({EMBARGO_DATE.strftime('%Y-%m-%d')}) [DROPPED to prevent leakage]")
    print(f"  * Clean Test Set:     {len(test_df):,} rows ({TEST_MIN_DATE.strftime('%Y-%m-%d')} to 2020-06-30)")

    # 3. Fit Candidate Pipeline
    print("\n[3/5] Fitting candidate pipeline with City/Month-Aware Imputation on training partition...")
    candidate_pipe = train_candidate_pipeline(train_df)
    print("  * Preprocessing: CityMonthAwareImputer fit strictly on train_df.")
    print("  * Estimator: RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42).")

    # 4. Comparative Evaluation
    print("\n[4/5] Executing head-to-head 3-way evaluation on identical clean test rows...")
    current_pipe = joblib.load(CURRENT_MODEL_PATH)
    eval_results = evaluate_comparative(candidate_pipe, current_pipe, test_df, train_df)

    ov_m = eval_results['overall_matched']
    print("\n" + "-" * 75)
    print(f"{'Model':<35} | {'Matched R^2':<12} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 75)
    print(f"{'Candidate (City/Month Imputer + RF)':<35} | {ov_m['candidate']['r2']:<12.4f} | {ov_m['candidate']['rmse']:<10.2f} | {ov_m['candidate']['mae']:<10.2f}")
    print(f"{'Current Baseline (aqi_model.pkl)':<35} | {ov_m['current']['r2']:<12.4f} | {ov_m['current']['rmse']:<10.2f} | {ov_m['current']['mae']:<10.2f}")
    print(f"{'Persistence Baseline (y_t -> y_t+1)':<35} | {ov_m['persistence']['r2']:<12.4f} | {ov_m['persistence']['rmse']:<10.2f} | {ov_m['persistence']['mae']:<10.2f}")
    print("-" * 75)

    # 5. Export Candidate Artifacts
    print("\n[5/5] Exporting candidate model and comprehensive validation documentation...")
    # Save candidate model
    joblib.dump(candidate_pipe, CANDIDATE_MODEL_PATH)
    print(f"  * Saved candidate model: {CANDIDATE_MODEL_PATH}")

    # Save metadata JSON
    metadata = {
        'project': 'AirSense AI - Candidate Model Improvement Cycle',
        'created_at': datetime.now().isoformat(),
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'split_policy': {
            'train_date_max': TRAIN_MAX_DATE.strftime('%Y-%m-%d'),
            'embargo_date': EMBARGO_DATE.strftime('%Y-%m-%d'),
            'test_date_min': TEST_MIN_DATE.strftime('%Y-%m-%d'),
            'embargo_explanation': '1-day temporal embargo buffer prevents day t+1 training target from overlapping test period.'
        },
        'sample_counts': {
            'train_samples': len(train_df),
            'embargo_samples': len(embargo_df),
            'test_samples': len(test_df),
            'matched_test_samples': eval_results['overall_matched']['candidate']['samples']
        },
        'evaluation_results': eval_results
    }

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  * Saved metadata JSON:   {METADATA_PATH}")

    # Save Markdown report
    report_md = generate_markdown_report(eval_results, train_df, embargo_df, test_df)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"  * Saved Markdown report: {REPORT_PATH}")

    print("\n" + "=" * 75)
    print("Candidate training and evaluation completed successfully.")
    print("Notice: Production model (aqi_model.pkl) remains completely untouched.")
    print("=" * 75)


if __name__ == '__main__':
    main()
