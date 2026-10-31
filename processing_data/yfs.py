# yfs data preprocessing and harmonization
# this script processes cohort data into the standardized smartchange format
# original input filenames and source column names have been replaced with generic names
# the input file must use the generic column names defined in this script
# no original participant data are included

import pandas as pd
import numpy as np
import os

# define file paths
input_folder = "DATA/yfs"
output_folder = "RESULTS"
input_file = "input.xlsx"
output_file = "YFS_edited.csv"

os.makedirs(output_folder, exist_ok=True)

# load the dataset
yfs_data = pd.read_excel(f"{input_folder}/{input_file}")

# rename columns to final lowercase + underscore names
yfs_data = yfs_data.rename(columns={
    'participant_id': 'id',
    'sex_input': 'sex',
    'age_input': 'age',

    'bmi_input': 'bmi',
    'height_input': 'height',
    'weight_input': 'weight',
    'waist_circ_input': 'waist_circ',

    'glucose_input': 'glucose',
    'cholesterol_source': 'cholesterol',
    'triglycerides_input':'triglycerides',
    'hdl_source':'hdl',
    'ldl_source':'ldl',

    'systolic_bp_source':'systolic_bp',
    'diastolic_phase_iv':'diast_kiv',
    'diastolic_phase_v':'diast_kv',

    'smoking_input': 'smoking',
    'alcohol_input': 'alcohol',
    'resting_heart_rate_source': 'resting_heart_rate',

    'peak_vo2_source': 'vo2_max',

    'diabetes_indicator_1': 'diabetes1',
    'diabetes_indicator_2': 'diabetes2',

    'family_income_quintile':'family_income',
    'income_quintile':'income',
    'education_years': 'education',

    'activity_duration_category': 'pa_inthour',
    'activity_frequency_category':'pa_freque',
    'activity_intensity_category':'pa_intens',
    'daily_steps':'pa_steps',
    'activity_intensity_frequency': 'pa_intfreq',
    'organized_activity_frequency': 'pa_organi',

    'fruits_input':'fruits',
    'vegetables_primary': 'vegetab',
    'vegetables_fresh':'vegetabf',
    'vegetables_cooked':'vegetabc',
    'sausage_frequency': 'sausage',
    'deli_meat_frequency':'delimeat',
    'meat_frequency':  'meat',
    'chicken_frequency': 'chicken',
    'blood_food_frequency': 'blood',
    'rice_frequency':  'rice',
    'porridge_frequency':'porridge',
    'bread_frequency': 'bread',
    'sweetened_beverage_frequency':'beverage',

    'mother_education_years': 'mother_educ',
    'father_education_years': 'father_educ',
})

# store original (id, age) combinations before any transformations
original_ids_ages = set(zip(yfs_data['id'], yfs_data['age']))

# diabolic bp based on age -> diastolic_bp
def select_diastolic_bp(row):
    if pd.notnull(row['diast_kiv']) and row['age'] < 13:
        return row['diast_kiv']
    if pd.notnull(row['diast_kv']) and row['age'] >= 13:
        return row['diast_kv']
    if pd.notnull(row['diast_kiv']):
        return row['diast_kiv']
    if pd.notnull(row['diast_kv']):
        return row['diast_kv']
    return np.nan

yfs_data['diastolic_bp'] = yfs_data.apply(select_diastolic_bp, axis=1)

# capitalize sex
yfs_data['sex'] = yfs_data['sex'].str.capitalize()

# frequency -> portions/day mapping
portion_mapping = {
    1: 1.00,
    2: 0.86,
    3: 0.29,
    4: 0.14,
    5: 0.07,
    6: 0.00
}

# fruits
yfs_data['fruits'] = yfs_data['fruits'].map(portion_mapping)

# vegetables (possibly two columns)
def process_vegetables(row):
    if pd.notnull(row['vegetab']):
        return portion_mapping.get(row['vegetab'], np.nan)
    vals = [row['vegetabf'], row['vegetabc']]
    vals = [v for v in vals if pd.notnull(v)]
    if vals:
        return portion_mapping.get(round(np.mean(vals)), np.nan)
    return np.nan

yfs_data['vegetables'] = yfs_data.apply(process_vegetables, axis=1)

# smoking: 1/2/3 -> current smoker (1), 4 -> ex-smoker (2), 5 -> never smoker (0)
def process_smoking(v):
    if pd.isnull(v):
        return np.nan
    if any(str(n) in str(v) for n in [1, 2, 3]):
        return 1
    if "4" in str(v):
        return 2
    if "5" in str(v):
        return 0
    return np.nan

yfs_data['smoking'] = yfs_data['smoking'].apply(process_smoking)

# diabetes -> 1 if either diabetes1 or diabetes2 is 1, else 0
yfs_data['diabetes'] = ((yfs_data['diabetes1'] == 1) | (yfs_data['diabetes2'] == 1)).astype(int)
yfs_data.drop(columns=['diabetes1', 'diabetes2'], inplace=True)

# steps
yfs_data['steps'] = yfs_data['pa_steps']

# sweetened_beverages
yfs_data['sweetened_beverages'] = yfs_data['beverage'].map(portion_mapping)

# processed_meats = sausage + delimeat (keep nan if both missing)
sa = yfs_data['sausage'].map(portion_mapping)
dm = yfs_data['delimeat'].map(portion_mapping)
proc = sa.fillna(0) + dm.fillna(0)
proc[sa.isna() & dm.isna()] = np.nan
yfs_data['processed_meats'] = proc

# red_meats = meat + chicken + blood (keep nan if all missing)
m  = yfs_data['meat'].map(portion_mapping)
ch = yfs_data['chicken'].map(portion_mapping)
bl = yfs_data['blood'].map(portion_mapping)
red = m.fillna(0) + ch.fillna(0) + bl.fillna(0)
red[m.isna() & ch.isna() & bl.isna()] = np.nan
yfs_data['red_meats'] = red

# cereals = rice + porridge + bread (keep nan if all missing)
r  = yfs_data['rice'].map(portion_mapping)
p  = yfs_data['porridge'].map(portion_mapping)
b  = yfs_data['bread'].map(portion_mapping)
cer = r.fillna(0) + p.fillna(0) + b.fillna(0)
cer[r.isna() & p.isna() & b.isna()] = np.nan
yfs_data['cereals'] = cer

# origin
yfs_data['origin'] = 'YFS'
yfs_data['country'] = 'Finland'

# vpa_indicator from pa_intens -> keep only 1, 2, 3
def map_vpa_indicator(v):
    if pd.isnull(v):
        return np.nan
    s = str(v).strip()
    first = s.split(':', 1)[0].strip()
    if first in ['1', '2', '3']:
        return int(first)
    return np.nan

yfs_data['vpa_indicator'] = yfs_data['pa_intens'].apply(map_vpa_indicator)

# z_score_pa_fallback

def extract_category(v):
    if pd.isnull(v):
        return np.nan

    try:
        return float(str(v).split(':', 1)[0].strip())
    except:
        return np.nan

yfs_data['pa_intens_num'] = yfs_data['pa_intens'].apply(extract_category)
yfs_data['pa_intfreq_num'] = yfs_data['pa_intfreq'].apply(extract_category)
yfs_data['pa_organi_num'] = yfs_data['pa_organi'].apply(extract_category)

# reverse pa_organi:
yfs_data['pa_organi_num'] = 7 - yfs_data['pa_organi_num']

fallback_vars = [
    'pa_intens_num',
    'pa_intfreq_num',
    'pa_organi_num'
]

z_fallback_cols = []

for var in fallback_vars:
    zcol = f'z_{var}'
    z_fallback_cols.append(zcol)

    mu = yfs_data[var].mean(skipna=True)
    sigma = yfs_data[var].std(skipna=True)

    yfs_data[zcol] = (
        (yfs_data[var] - mu) / sigma
        if sigma > 0 else np.nan
    )

yfs_data['z_score_pa_fallback'] = yfs_data[z_fallback_cols].mean(
    axis=1,
    skipna=True
)

# temporary alcohol cap pending feedback from yfs partners
yfs_data['alcohol'] = pd.to_numeric(yfs_data['alcohol'], errors='coerce')
yfs_data.loc[yfs_data['alcohol'] > 125, 'alcohol'] = 125

# map pa_inthour -> mvpa with random within ranges
def map_mvpa_from_inthour(v):
    if pd.isnull(v):
        return np.nan
    cat = str(v).split(':', 1)[0].strip()
    if cat == '6':
        return np.random.uniform(1, 2)
    if cat == '5':
        return np.random.uniform(0.6, 0.9)
    if cat == '4':
        return np.random.uniform(0.3, 0.4)
    if cat == '3':
        return 0.14
    if cat == '2':
        return 0.07
    if cat == '1':
        return 0.00
    return np.nan

# map pa_freque -> mvpa fallback, used only when pa_inthour is missing
def map_mvpa_from_freque(v):
    if pd.isnull(v):
        return np.nan

    s = str(v).strip().lower()
    cat = s.split(':', 1)[0].strip()

    if cat == '1':
        return 0.00

    if cat == '2':
        if 'less than monthly' in s:
            return np.random.uniform(0, 0.02)
        return np.random.uniform(0, 0.03)

    if cat == '3':
        return 0.03

    if cat == '4':
        return np.random.uniform(0.07, 0.1)

    if cat == '5':
        return 0.15

    if cat == '6':
        if '5-6 days' in s:
            return np.random.uniform(0.7, 0.9)
        if '2-4 days' in s:
            return np.random.uniform(0.3, 0.6)
        return np.random.uniform(0.3, 0.9)

    if cat == '7':
        return np.random.uniform(0.5, 1.3)

    return np.nan

# first try pa_inthour, then use pa_freque only if pa_inthour is missing
yfs_data['mvpa'] = yfs_data['pa_inthour'].apply(map_mvpa_from_inthour)
mvpa_from_freque = yfs_data['pa_freque'].apply(map_mvpa_from_freque)
yfs_data['mvpa'] = yfs_data['mvpa'].fillna(mvpa_from_freque)

yfs_data.drop(columns=[
    'pa_inthour',
    'pa_freque',
    'pa_intens',
    'pa_intfreq',
    'pa_organi',
    'pa_intens_num',
    'pa_intfreq_num',
    'pa_organi_num',
    'z_pa_intens_num',
    'z_pa_intfreq_num',
    'z_pa_organi_num'
], inplace=True)

# keep only original (id, age) combos
yfs_data = yfs_data[
    yfs_data[['id', 'age']].apply(tuple, axis=1)
    .isin(original_ids_ages)
]

# desired column order (base variables)
desired_columns = [
    "origin", "country", "id", "sex", "age", "bmi", "height", "weight", "waist_circ",
    "glucose", "cholesterol", "triglycerides", "hdl", "ldl",
    "diastolic_bp", "systolic_bp", "smoking", "alcohol", "resting_heart_rate",
    "vo2_max", "diabetes", "steps", "mvpa", "vpa_indicator", "z_score_pa_fallback",
    "fruits", "vegetables", "sweetened_beverages", "processed_meats",
    "red_meats", "cereals", "mother_educ", "father_educ",
    "family_income", "income", "education"
]
yfs_data = yfs_data.reindex(columns=[c for c in desired_columns if c in yfs_data.columns])

# drop rows with fewer than 9 non-null columns
yfs_data = yfs_data[yfs_data.notna().sum(axis=1) >= 9]

# map quintile strings to numeric for family_income and income
quintile_map = {
    '5th quintile': 5,
    '4th quintile': 4,
    '3rd quintile': 3,
    '2nd quintile': 2,
    '1st quintile': 1
}
yfs_data['family_income'] = yfs_data['family_income'].map(quintile_map)
yfs_data['income']        = yfs_data['income'].map(quintile_map)

# merge family_income into income (prefer family_income when present)
yfs_data['income'] = yfs_data['family_income'].fillna(yfs_data['income'])
yfs_data.drop(columns=['family_income'], inplace=True)

# ensure the four affluence vars are numeric
affluence_vars = ['mother_educ', 'father_educ', 'income', 'education']
yfs_data[affluence_vars] = yfs_data[affluence_vars].apply(pd.to_numeric, errors='coerce')

# compute individual z-scores
z_cols = []
for var in affluence_vars:
    zcol = f"z_score_{var}"
    z_cols.append(zcol)
    mu = yfs_data[var].mean()
    sigma = yfs_data[var].std()
    yfs_data[zcol] = (yfs_data[var] - mu) / sigma

# compute overall affluence index
yfs_data['z_score_affluence'] = yfs_data[z_cols].mean(axis=1, skipna=True)

# coerce numerics and format
numeric_cols = [c for c in yfs_data.columns if c not in ['origin', 'country', 'id', 'sex']]
for col in numeric_cols:
    yfs_data[col] = pd.to_numeric(yfs_data[col], errors='coerce')
    yfs_data[col] = (
        yfs_data[col]
        .round(2)
        .map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if pd.notnull(x) else '')
    )

# prepend "yfs-" to every id
yfs_data['id'] = 'YFS-' + yfs_data['id'].astype(str)

# save
yfs_data.to_csv(f"{output_folder}/{output_file}", index=False)
print("YFS dataset processed and saved successfully!")