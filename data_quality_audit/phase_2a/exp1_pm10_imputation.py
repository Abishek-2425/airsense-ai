import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day['Month'] = city_day['Date'].dt.month
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

# Define Seasons based on Indian Meteorological Department (IMD):
# Winter: Dec (12), Jan (1), Feb (2)
# Summer/Pre-Monsoon: Mar (3), Apr (4), May (5)
# Monsoon: Jun (6), Jul (7), Aug (8)
# Post-Monsoon/Autumn: Sep (9), Oct (10), Nov (11)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

city_day['Season'] = city_day['Month'].apply(get_season)

# Set up chronological partitions
# Full Training (for Test eval): <= 2019-12-04
# Embargo: 2019-12-05
# Held-Out Test: >= 2019-12-06
train_full = city_day[city_day['Date'] <= pd.to_datetime('2019-12-04')].copy().reset_index(drop=True)
test_full = city_day[city_day['Date'] >= pd.to_datetime('2019-12-06')].copy().reset_index(drop=True)

# Internal split for validation:
# Internal Train: <= 2018-12-31
# Internal Val: 2019-01-02 to 2019-12-04
train_val_train = city_day[city_day['Date'] <= pd.to_datetime('2018-12-31')].copy().reset_index(drop=True)
train_val_eval = city_day[(city_day['Date'] >= pd.to_datetime('2019-01-02')) & (city_day['Date'] <= pd.to_datetime('2019-12-04'))].copy().reset_index(drop=True)


class PM10ImputerExperiment:
    """
    Evaluates 5 imputation strategies strictly fit on training data.
    """
    def __init__(self, train_df):
        self.train_df = train_df.copy()
        self.stats = {}
        self._fit()

    def _fit(self):
        df = self.train_df
        # 1. Strategy A: Global median
        self.global_median_pm10 = df['PM10'].median()
        self.global_median_pm25 = df['PM2.5'].median()

        # 2. Strategy B: City median
        self.city_median_pm10 = df.groupby('City')['PM10'].median().to_dict()

        # 3. Strategy C: City-Month median
        self.city_month_median_pm10 = df.groupby(['City', 'Month'])['PM10'].median().to_dict()

        # Training paired observations for ratio calculations (PM2.5 > 0, PM10 > 0, PM2.5 <= 1.2 * PM10)
        paired = df[(df['PM10'] > 0) & (df['PM2.5'] > 0)].copy()
        paired = paired[paired['PM2.5'] <= 1.2 * paired['PM10']]
        paired['ratio'] = paired['PM2.5'] / paired['PM10']

        # National ratio median
        self.national_ratio = paired['ratio'].median()

        # 4. Strategy D: Month ratio median
        self.month_ratio = paired.groupby('Month')['ratio'].median().to_dict()

        # 5. Strategy E: City-Month ratio median
        self.city_month_ratio = paired.groupby(['City', 'Month'])['ratio'].median().to_dict()
        self.city_ratio = paired.groupby('City')['ratio'].median().to_dict()

    def impute(self, eval_df, strategy='A'):
        """
        Predicts PM10 for eval_df using the specified strategy.
        Tracks fallback frequencies.
        """
        preds = []
        fallbacks = {
            'exact_match': 0,
            'fallback_city': 0,
            'fallback_month': 0,
            'fallback_national': 0,
            'fallback_global_median': 0
        }

        for _, row in eval_df.iterrows():
            city = row['City']
            month = row['Month']
            pm25 = row['PM2.5']

            if strategy == 'A':
                # Global median
                preds.append(self.global_median_pm10)
                fallbacks['exact_match'] += 1

            elif strategy == 'B':
                # City median with global fallback
                if city in self.city_median_pm10 and not np.isnan(self.city_median_pm10[city]):
                    preds.append(self.city_median_pm10[city])
                    fallbacks['exact_match'] += 1
                else:
                    preds.append(self.global_median_pm10)
                    fallbacks['fallback_global_median'] += 1

            elif strategy == 'C':
                # City-Month median with City, then Global fallback
                if (city, month) in self.city_month_median_pm10 and not np.isnan(self.city_month_median_pm10[(city, month)]):
                    preds.append(self.city_month_median_pm10[(city, month)])
                    fallbacks['exact_match'] += 1
                elif city in self.city_median_pm10 and not np.isnan(self.city_median_pm10[city]):
                    preds.append(self.city_median_pm10[city])
                    fallbacks['fallback_city'] += 1
                else:
                    preds.append(self.global_median_pm10)
                    fallbacks['fallback_global_median'] += 1

            elif strategy == 'D':
                # Monthly PM2.5/PM10 ratio imputation
                if pd.notnull(pm25) and pm25 > 0:
                    r = self.month_ratio.get(month, self.national_ratio)
                    preds.append(pm25 / r)
                    fallbacks['exact_match'] += 1
                else:
                    # PM2.5 missing -> fallback to month median PM10, then global
                    val = self.month_ratio.get(month, None)
                    preds.append(self.global_median_pm10)
                    fallbacks['fallback_global_median'] += 1

            elif strategy == 'E':
                # City-Month PM2.5/PM10 ratio with full fallback hierarchy
                if pd.notnull(pm25) and pm25 > 0:
                    if (city, month) in self.city_month_ratio and not np.isnan(self.city_month_ratio[(city, month)]):
                        r = self.city_month_ratio[(city, month)]
                        preds.append(pm25 / r)
                        fallbacks['exact_match'] += 1
                    elif city in self.city_ratio and not np.isnan(self.city_ratio[city]):
                        r = self.city_ratio[city]
                        preds.append(pm25 / r)
                        fallbacks['fallback_city'] += 1
                    elif month in self.month_ratio and not np.isnan(self.month_ratio[month]):
                        r = self.month_ratio[month]
                        preds.append(pm25 / r)
                        fallbacks['fallback_month'] += 1
                    else:
                        r = self.national_ratio
                        preds.append(pm25 / r)
                        fallbacks['fallback_national'] += 1
                else:
                    # PM2.5 missing -> fallback to City-Month median PM10 -> City -> Global
                    if (city, month) in self.city_month_median_pm10 and not np.isnan(self.city_month_median_pm10[(city, month)]):
                        preds.append(self.city_month_median_pm10[(city, month)])
                        fallbacks['fallback_city'] += 1
                    elif city in self.city_median_pm10 and not np.isnan(self.city_median_pm10[city]):
                        preds.append(self.city_median_pm10[city])
                        fallbacks['fallback_city'] += 1
                    else:
                        preds.append(self.global_median_pm10)
                        fallbacks['fallback_global_median'] += 1

        return np.array(preds), fallbacks


def evaluate_imputation(eval_df, imputer, fold_name="Validation"):
    # Filter to ground truth PM10 non-null
    target_eval = eval_df[eval_df['PM10'].notnull()].copy().reset_index(drop=True)
    y_true = target_eval['PM10'].values
    n_samples = len(target_eval)

    strategies = {
        'Strategy A (Global Median)': 'A',
        'Strategy B (City Median)': 'B',
        'Strategy C (City-Month Median)': 'C',
        'Strategy D (Monthly Ratio)': 'D',
        'Strategy E (City-Month Ratio)': 'E'
    }

    overall_results = []
    preds_dict = {}

    for name, code in strategies.items():
        preds, fallbacks = imputer.impute(target_eval, strategy=code)
        preds_dict[code] = preds
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)

        overall_results.append({
            'Fold': fold_name,
            'Strategy': name,
            'Code': code,
            'N': n_samples,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Fallbacks': fallbacks
        })

    # Breakdowns by Season
    season_results = []
    for s in ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']:
        idx = target_eval['Season'] == s
        if idx.sum() > 0:
            y_s = y_true[idx]
            for name, code in strategies.items():
                p_s = preds_dict[code][idx]
                season_results.append({
                    'Fold': fold_name,
                    'Season': s,
                    'Strategy': name,
                    'N': int(idx.sum()),
                    'MAE': mean_absolute_error(y_s, p_s),
                    'RMSE': np.sqrt(mean_squared_error(y_s, p_s)),
                    'R2': r2_score(y_s, p_s)
                })

    # Breakdowns by High-PM2.5 Pollution Episode (PM2.5 >= 120 ug/m3)
    episode_results = []
    # Only evaluate episode breakdown where PM2.5 is non-null
    has_pm25 = target_eval['PM2.5'].notnull()
    high_pm25 = (target_eval['PM2.5'] >= 120) & has_pm25
    normal_pm25 = (target_eval['PM2.5'] < 120) & has_pm25

    for ep_label, mask in [('High PM2.5 (>=120)', high_pm25), ('Normal PM2.5 (<120)', normal_pm25)]:
        if mask.sum() > 0:
            y_ep = y_true[mask]
            for name, code in strategies.items():
                p_ep = preds_dict[code][mask]
                episode_results.append({
                    'Fold': fold_name,
                    'Episode': ep_label,
                    'Strategy': name,
                    'N': int(mask.sum()),
                    'MAE': mean_absolute_error(y_ep, p_ep),
                    'RMSE': np.sqrt(mean_squared_error(y_ep, p_ep)),
                    'R2': r2_score(y_ep, p_ep)
                })

    # Top Cities Breakdown
    city_counts = target_eval['City'].value_counts()
    top_cities = city_counts.head(8).index.tolist()
    city_results = []
    for c in top_cities:
        c_mask = target_eval['City'] == c
        y_c = y_true[c_mask]
        for name, code in strategies.items():
            p_c = preds_dict[code][c_mask]
            city_results.append({
                'Fold': fold_name,
                'City': c,
                'Strategy': name,
                'N': int(c_mask.sum()),
                'MAE': mean_absolute_error(y_c, p_c),
                'RMSE': np.sqrt(mean_squared_error(y_c, p_c)),
                'R2': r2_score(y_c, p_c)
            })

    return overall_results, season_results, episode_results, city_results


# Run Experiments on Validation Fold and Test Fold
print("Fitting Imputer on Internal Training Set (<= 2018-12-31)...")
imputer_val = PM10ImputerExperiment(train_val_train)
val_ov, val_se, val_ep, val_ci = evaluate_imputation(train_val_eval, imputer_val, fold_name="Internal Validation (2019)")

print("Fitting Imputer on Full Training Set (<= 2019-12-04)...")
imputer_test = PM10ImputerExperiment(train_full)
test_ov, test_se, test_ep, test_ci = evaluate_imputation(test_full, imputer_test, fold_name="Held-Out Test (2020)")

# Save Results
all_overall = pd.DataFrame(val_ov + test_ov)
all_season = pd.DataFrame(val_se + test_se)
all_episode = pd.DataFrame(val_ep + test_ep)
all_city = pd.DataFrame(val_ci + test_ci)

all_overall.to_csv(os.path.join(RESULTS_DIR, 'exp1_pm10_imputation_overall.csv'), index=False)
all_season.to_csv(os.path.join(RESULTS_DIR, 'exp1_pm10_imputation_by_season.csv'), index=False)
all_episode.to_csv(os.path.join(RESULTS_DIR, 'exp1_pm10_imputation_by_episode.csv'), index=False)
all_city.to_csv(os.path.join(RESULTS_DIR, 'exp1_pm10_imputation_by_city.csv'), index=False)

print("\n" + "="*80)
print("EXPERIMENT 1: OVERALL PM10 IMPUTATION EVALUATION (HELD-OUT TEST SET)")
print("="*80)
print(all_overall[all_overall['Fold'] == 'Held-Out Test (2020)'][['Strategy', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 1: HIGH PM2.5 EPISODE EVALUATION (HELD-OUT TEST SET)")
print("="*80)
print(all_episode[(all_episode['Fold'] == 'Held-Out Test (2020)') & (all_episode['Episode'] == 'High PM2.5 (>=120)')][['Strategy', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("FALLBACK FREQUENCY REPORT (STRATEGY E ON HELD-OUT TEST)")
print("="*80)
for row in test_ov:
    if row['Code'] == 'E':
        print(json.dumps(row['Fallbacks'], indent=2))

