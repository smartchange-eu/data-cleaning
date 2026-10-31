# public version of the preprocessing script
# original dataset file names and source variable names have been replaced
# with generic names to avoid disclosing dataset-specific information.

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# file paths
data_path = "DATA/abcd/input.csv"
log_path = os.path.join("RESULTS-3", "ABCD_cleaning_log.csv")
output_path = os.path.join("RESULTS-3", "ABCD_edited.csv")
os.makedirs("RESULTS-3", exist_ok=True)
non_imputed_path = os.path.join("RESULTS-3", "ABCD_non_imputed.csv")

# load input
df = pd.read_csv(data_path, delimiter=',', decimal='.')
df['orig_index'] = np.arange(len(df))

def detect_outliers_by_age(df, features, contamination=0.01, min_group_size=30):
    df = df.copy()
    df['Outlier'] = False
    df['AGE_INT'] = df['AGE'].fillna(-1).astype(int).clip(lower=0)
    for age_int, group in df.groupby('AGE_INT'):
        if len(group) < min_group_size:
            continue
        X = group[features].dropna()
        if len(X) < min_group_size:
            continue
        Xs = StandardScaler().fit_transform(X)
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = model.fit_predict(Xs)
        out_idx = X.index[preds == -1]
        df.loc[out_idx, 'Outlier'] = True
    df.drop(columns=['AGE_INT'], inplace=True)
    return df

# input columns
baseline = ['source_sex', 'source_birth_date']
f3_cols = [
    'wave1_age_years', 'wave1_age_alternate', 'wave1_bmi', 'wave1_height', 'wave1_weight', 'wave1_hip_measure', 'wave1_waist',
    'wave1_glucose', 'wave1_cholesterol', 'wave1_triglycerides', 'wave1_hdl', 'wave1_ldl',
    'wave1_diastolic_bp', 'wave1_systolic_bp', 'wave1_resting_hr', 'wave1_sleep_weekday', 'wave1_sleep_weekend',
    'wave1_vegetables', 'wave1_fruits', 'wave1_sweetened_beverages', 'wave1_no_meat', 'wave1_mother_education', 'wave1_family_diabetes', 'wave1_income'
]
f4_cols = [
    'wave2_age_primary', 'wave2_age_alternate', 'wave2_bmi', 'wave2_height', 'wave2_weight', 'wave2_hip', 'wave2_waist',
    'wave2_glucose', 'wave2_cholesterol', 'wave2_triglycerides', 'wave2_hdl', 'wave2_ldl', 'wave2_diastolic_bp', 'wave2_systolic_bp', 'wave2_resting_hr',
    'wave2_sleep_weekday', 'wave2_sleep_weekend', 'wave2_vegetables', 'wave2_fruits', 'wave2_sweetened_beverages',
    'wave2_no_meat', 'wave2_mother_education', 'wave2_father_education', 'wave2_vigorous_minutes', 'wave2_moderate_minutes', 'wave2_valid_days', 'wave2_income'
]

# separate study waves
f3_df = df[baseline + ['orig_index'] + f3_cols].copy()
f3_df['period'] = 'F3'
f4_df = df[baseline + ['orig_index'] + f4_cols].copy()
f4_df['period'] = 'F4'
long = pd.concat([f3_df, f4_df], ignore_index=True)

# participant metadata
long['ORIGIN'] = 'ABCD'
long['ID_num'] = long.groupby('orig_index').ngroup() + 1
long['ID'] = 'ABCD-' + long['ID_num'].astype(str)
long['SEX'] = long['source_sex'].map({1: 'Male', 2: 'Female'})

def compute_age(row):
    if row['period'] == 'F3':
        val = row['wave1_age_years'] if pd.notna(row.get('wave1_age_years')) else row.get('wave1_age_alternate')
    else:
        val = row['wave2_age_primary'] if pd.notna(row.get('wave2_age_primary')) else row.get('wave2_age_alternate')
    try:
        return round(float(val), 2)
    except:
        return np.nan

# fill missing age from the other wave
long['AGE'] = long.apply(compute_age, axis=1)

def impute_age(group):
    non_missing = group['AGE'].dropna()
    if len(non_missing) == 1:
        known_idx = non_missing.index[0]
        known_val = non_missing.iloc[0]
        known_period = group.loc[known_idx, 'period']
        if known_period == 'F3':
            group['AGE'] = group['AGE'].fillna(round(known_val + 6.5, 2))
        elif known_period == 'F4':
            group['AGE'] = group['AGE'].fillna(round(known_val - 6.5, 2))
    return group
long = long.groupby('ID', group_keys=False).apply(impute_age)

# harmonized measurements
long['BMI'] = np.where(long['period'] == 'F3', long['wave1_bmi'], long['wave2_bmi'])
long['HEIGHT'] = np.where(long['period'] == 'F3', long['wave1_height'], long['wave2_height'])
long['WEIGHT'] = np.where(long['period'] == 'F3', long['wave1_weight'], long['wave2_weight'])
long['WAIST CIRC'] = np.where(long['period'] == 'F3', long['wave1_waist'], long['wave2_waist'])
long['GLUCOSE'] = np.where(long['period'] == 'F3', long['wave1_glucose'], long['wave2_glucose'])
long['CHOLESTEROL'] = np.where(long['period'] == 'F3', long['wave1_cholesterol'], long['wave2_cholesterol'])
long['TRIGLYCERIDES'] = np.where(long['period'] == 'F3', long['wave1_triglycerides'], long['wave2_triglycerides'])
long['HDL'] = np.where(long['period'] == 'F3', long['wave1_hdl'], long['wave2_hdl'])
long['LDL'] = np.where(long['period'] == 'F3', long['wave1_ldl'], long['wave2_ldl'])
long['DIASTOLIC BP'] = np.where(long['period'] == 'F3', long['wave1_diastolic_bp'], long['wave2_diastolic_bp'])
long['SYSTOLIC BP'] = np.where(long['period'] == 'F3', long['wave1_systolic_bp'], long['wave2_systolic_bp'])

# physical activity
long['wave2_vigorous_minutes'] = pd.to_numeric(long['wave2_vigorous_minutes'], errors='coerce')
long['wave2_moderate_minutes'] = pd.to_numeric(long['wave2_moderate_minutes'], errors='coerce')
long['wave2_valid_days'] = pd.to_numeric(long['wave2_valid_days'], errors='coerce')
long['VIGOROUS ACT'] = np.where(long['period'] == 'F4', (long['wave2_vigorous_minutes'] / long['wave2_valid_days'] / 60).round(2), np.nan)
long['MODERATE ACT'] = np.where(long['period'] == 'F4', (long['wave2_moderate_minutes'] / long['wave2_valid_days'] / 60).round(2), np.nan)
long['RESTING HEART RATE'] = np.where(long['period'] == 'F3', long['wave1_resting_hr'], long['wave2_resting_hr'])

# diet
long['VEGETABLES'] = np.where(long['period'] == 'F3', long['wave1_vegetables'], long['wave2_vegetables'])
long['FRUITS'] = np.where(long['period'] == 'F3', long['wave1_fruits'], long['wave2_fruits'])
long['SWEETENED BEVERAGES'] = np.where(long['period'] == 'F3', long['wave1_sweetened_beverages'], long['wave2_sweetened_beverages'])

def transform_food_value(val):
    if pd.isna(val):
        return np.nan
    try:
        float_val = float(val)
    except:
        return np.nan
    if float_val == 1:
        return 0
    elif float_val == 2:
        return round((0.14 + 0.29 + 0.40) / 3, 2)
    elif float_val == 3:
        return round((0.86 + 0.71 + 0.57) / 3, 2)
    elif float_val == 4:
        return 1
    else:
        return float_val
for col in ['VEGETABLES', 'FRUITS', 'SWEETENED BEVERAGES']:
    long[col] = long[col].apply(transform_food_value)

# parental education
long['MOTHER EDUC'] = np.where(long['period'] == 'F3', long['wave1_mother_education'], long['wave2_mother_education'])
long['FATHER EDUC'] = long['wave2_father_education']
mother_educ_filled = long.groupby('orig_index')['MOTHER EDUC'].transform(lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan)
father_educ_filled = long.groupby('orig_index')['FATHER EDUC'].transform(lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan)
educ_map = {"1": 6, "2": 10, "3": 12, "4": 16, "5": 7}
long['MOTHER EDUC'] = mother_educ_filled.map(educ_map)
long['FATHER EDUC'] = father_educ_filled.map(educ_map)

# parental occupation
occupation_map = df[['orig_index', 'source_mother_occupation', 'source_father_occupation']].copy()
occupation_map.rename(columns={'source_mother_occupation': 'MOTHER OCCUP', 'source_father_occupation': 'FATHER OCCUP'}, inplace=True)
long = long.merge(occupation_map, on='orig_index', how='left')

# income
long['INCOME'] = np.where(long['period'] == 'F3', long['wave1_income'], long['wave2_income'])

# missing income response
long['INCOME'] = pd.to_numeric(long['INCOME'], errors='coerce')
long.loc[long['INCOME'] == 5, 'INCOME'] = np.nan

# fill income across waves
long['INCOME'] = long.groupby('orig_index')['INCOME'].transform(lambda x: x.ffill().bfill())

# income scale

def map_income_to_5(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 4
    if x == 4:
        return 5
    return np.nan
long['INCOME'] = long['INCOME'].apply(map_income_to_5).astype('float')

# average daily sleep
cols_to_numeric = ['wave1_sleep_weekday', 'wave1_sleep_weekend', 'wave2_sleep_weekday', 'wave2_sleep_weekend']
for col in cols_to_numeric:
    long[col] = pd.to_numeric(long[col], errors='coerce')
long['SLEEP'] = np.where(
    long['period'] == 'F3',
    ((long['wave1_sleep_weekday'] * 5 + long['wave1_sleep_weekend'] * 2) / 7).round(2),
    ((long['wave2_sleep_weekday'] * 5 + long['wave2_sleep_weekend'] * 2) / 7).round(2)
)

# family diabetes
df['wave1_family_diabetes'] = pd.to_numeric(df['wave1_family_diabetes'], errors='coerce')
fd_map = df[['orig_index', 'wave1_family_diabetes']].copy()
fd_map['FAMILY DIABETES'] = np.where(fd_map['wave1_family_diabetes'] == 2, 1, 0)
long = long.merge(fd_map[['orig_index', 'FAMILY DIABETES']], on='orig_index', how='left')
long['mvpa'] = long['MODERATE ACT'] + long['VIGOROUS ACT']
long['intensity_minutes'] = long['MODERATE ACT'] + 2 * long['VIGOROUS ACT']
long['harmonized_pa'] = long['intensity_minutes']

# select output columns
final_cols = [
    'ORIGIN','ID','SEX','AGE','BMI','HEIGHT','WEIGHT',
    'WAIST CIRC','GLUCOSE','CHOLESTEROL','TRIGLYCERIDES',
    'HDL','LDL','DIASTOLIC BP','SYSTOLIC BP',
    'VIGOROUS ACT','MODERATE ACT','mvpa','intensity_minutes','harmonized_pa','RESTING HEART RATE',
    'VEGETABLES','FRUITS', 'SWEETENED BEVERAGES',
    'MOTHER EDUC','FATHER EDUC','MOTHER OCCUP','FATHER OCCUP','INCOME',
    'SLEEP','FAMILY DIABETES'
]
data_processed = long[final_cols].copy()
data_processed['COUNTRY'] = 'Netherlands'

# sort records
data_processed['ID_num'] = data_processed['ID'].str.extract(r'ABCD-(\d+)').astype(int)
data_processed['period'] = np.where(data_processed['AGE'] < 10, 'F3', 'F4')
data_processed = data_processed.sort_values(by=['ID_num', 'period'])
data_processed = data_processed.drop(columns=['ID_num', 'period'], errors='ignore')

# outlier detection
# convert measurements to numeric
data_processed['HEIGHT'] = pd.to_numeric(data_processed['HEIGHT'], errors='coerce')
data_processed['WEIGHT'] = pd.to_numeric(data_processed['WEIGHT'], errors='coerce')
data_processed['BMI'] = pd.to_numeric(data_processed['BMI'], errors='coerce')
data_processed['WAIST CIRC'] = pd.to_numeric(data_processed['WAIST CIRC'], errors='coerce')
data_processed['GLUCOSE'] = pd.to_numeric(data_processed['GLUCOSE'], errors='coerce')
data_processed['CHOLESTEROL'] = pd.to_numeric(data_processed['CHOLESTEROL'], errors='coerce')
data_processed['TRIGLYCERIDES'] = pd.to_numeric(data_processed['TRIGLYCERIDES'], errors='coerce')
data_processed['HDL'] = pd.to_numeric(data_processed['HDL'], errors='coerce')
data_processed['LDL'] = pd.to_numeric(data_processed['LDL'], errors='coerce')
data_processed['DIASTOLIC BP'] = pd.to_numeric(data_processed['DIASTOLIC BP'], errors='coerce')
data_processed['SYSTOLIC BP'] = pd.to_numeric(data_processed['SYSTOLIC BP'], errors='coerce')
data_processed['RESTING HEART RATE'] = pd.to_numeric(data_processed['RESTING HEART RATE'], errors='coerce')
data_processed['VIGOROUS ACT'] = pd.to_numeric(data_processed['VIGOROUS ACT'], errors='coerce')
data_processed['MODERATE ACT'] = pd.to_numeric(data_processed['MODERATE ACT'], errors='coerce')
exclude = ['ORIGIN', 'ID', 'SEX', 'AGE', 'COUNTRY', 'Z_SCORE_PA']
features = [c for c in data_processed.select_dtypes(include=np.number).columns if c not in exclude]
df_flagged = detect_outliers_by_age(data_processed, features)

# remove flagged records
data_processed = df_flagged.loc[~df_flagged['Outlier']].drop(columns='Outlier')
numeric_cols = [col for col in data_processed.columns if col not in ['ORIGIN', 'ID', 'SEX']]
protected_cols = ['ORIGIN', 'ID', 'SEX', 'AGE', 'COUNTRY']
numeric_cols = [col for col in data_processed.columns if col not in protected_cols]

# convert numeric fields
for col in numeric_cols:
    data_processed[col] = data_processed[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)', expand=False)
    data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')

# save before z-scores
data_non_imputed = data_processed.copy()
data_non_imputed.to_csv(non_imputed_path, index=False, float_format='%.2f')

# physical activity z-scores
# age groups

def age_group(age):
    if 4.5 <= age <= 6.5:
        return 'young'
    elif 10.5 <= age <= 12.5:
        return 'older'
    else:
        return 'other'
data_processed['AGE_GROUP'] = data_processed['AGE'].apply(age_group)

# z-score within age and sex groups
data_processed['Z_SCORE_PA'] = data_processed.groupby(['AGE_GROUP', 'SEX'])['harmonized_pa'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# remove helper column
data_processed = data_processed.drop(columns=['AGE_GROUP'])

# socioeconomic z-scores
# variables
affluence_vars = ['MOTHER EDUC', 'FATHER EDUC', 'MOTHER OCCUP', 'FATHER OCCUP', 'INCOME']
z_col_names = {
    'MOTHER EDUC': 'Z_SCORE_MOTHER EDUC',
    'FATHER EDUC': 'Z_SCORE_FATHER EDUC',
    'MOTHER OCCUP': 'Z_SCORE_MOTHER_OCCUP',
    'FATHER OCCUP': 'Z_SCORE_FATHER_OCCUP',
    'INCOME': 'Z_SCORE_INCOME'
}

# calculate z-scores
for col in affluence_vars:
    if col in data_processed.columns:
        mean = data_processed[col].mean(skipna=True)
        std = data_processed[col].std(skipna=True)
        z_name = z_col_names[col]
        data_processed[z_name] = data_processed[col].apply(lambda x: (x - mean) / std if pd.notna(x) else np.nan)

# average socioeconomic z-scores
z_score_cols = list(z_col_names.values())
data_processed['Z_SCORE_AFFLUENCE'] = data_processed[z_score_cols].mean(axis=1, skipna=True)

# summary statistics
log_data = []
for col in data_processed.columns:
    origins = "ABCD"
    n_present = data_processed[col].notnull().sum()
    n_missing = data_processed[col].isnull().sum()
    if col in numeric_cols or col in ['AGE', 'Z_SCORE_PA']:
        stats = data_processed[col].describe()
        log_data.append({
            "Variable": col,
            "Datasets": origins,
            "Present": n_present,
            "Missing": n_missing,
            "Mean": round(stats.get("mean", np.nan), 2),
            "Min": round(stats.get("min", np.nan), 2),
            "25%": round(stats.get("25%", np.nan), 2),
            "50%": round(stats.get("50%", np.nan), 2),
            "75%": round(stats.get("75%", np.nan), 2),
            "Max": round(stats.get("max", np.nan), 2)
        })
    else:
        log_data.append({
            "Variable": col,
            "Datasets": origins,
            "Present": n_present,
            "Missing": n_missing,
            "Mean": "N/A", "Min": "N/A", "25%": "N/A",
            "50%": "N/A", "75%": "N/A", "Max": "N/A"
        })
log_df = pd.DataFrame(log_data)
log_df.to_csv(log_path, index=False)
print(f"Non-imputed saved to '{non_imputed_path}'")
print(f"Cleaning log saved to '{log_path}'")

# standardize output names
data_processed.columns = (
    data_processed.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
missing_per_row = data_processed.isna().sum(axis=1)
data_processed = data_processed[missing_per_row < 27].copy()
data_processed = data_processed.replace("nan", "")

# save output
data_processed.to_csv(output_path, index=False, float_format='%.2f')

# format numeric values
df_out = pd.read_csv(output_path)

def clean_number(x):
    try:
        val = float(x)
        return str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip('0').rstrip('.')
    except:
        return x
for col in df_out.columns:
    df_out[col] = df_out[col].apply(clean_number)
df_out = df_out.fillna('')
df_out.to_csv(output_path, index=False)
print(f"Processed dataset saved to '{output_path}'")
