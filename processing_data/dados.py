# public dados preprocessing script
# source column names and input filename are generic
# dados-fix.py runs first and creates the intermediate selected.csv file

import os
import subprocess
import sys
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(script_dir, 'dados-fix.py')], check=True)
input_file = 'DATA/dados/input.csv'
output_file = 'RESULTS/DADOS_edited.csv'
dados = pd.read_csv(input_file, sep=';', decimal=',')

# convert the source measurements
numeric_columns = [
    'glucose_raw', 'cholesterol_raw', 'triglycerides_raw', 'hdl_raw', 'ldl_raw',
    'moderate_minutes', 'vigorous_minutes', 'valid_days', 'sleep_total',
    'food_1', 'food_2', 'food_3', 'food_9', 'food_13', 'food_14', 'food_19',
    'mother_education_code', 'father_education_code', 'vo2_measure',
    'diastolic_raw', 'systolic_raw', 'mother_diabetes_1', 'mother_diabetes_2',
    'father_diabetes_1', 'father_diabetes_2', 'cars_code', 'bedroom_code',
    'computers_code', 'internet_code'
]
for col in numeric_columns:
    if col in dados.columns:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

# set cohort fields and standardize measurements
dados['origin'] = 'DADOS'
dados['country'] = 'Spain'
dados = dados.rename(columns={'participant_code': 'id'})
dados['id'] = 'DADOS-' + dados['id'].astype(str)
dados['sex'] = dados['sex_code'].map({0: 'Female', 1: 'Male'})
rename_base = {'age_raw': 'age', 'height_raw': 'height', 'weight_raw': 'weight', 'waist_raw': 'waist_circ'}
if 'bmi_raw' in dados.columns:
    rename_base['bmi_raw'] = 'bmi'
dados = dados.rename(columns=rename_base)
for col in ['bmi', 'height', 'weight', 'waist_circ']:
    if col in dados.columns:
        dados[col] = pd.to_numeric(dados[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# convert laboratory units
dados['glucose'] = dados['glucose_raw'] * 0.0555
dados['cholesterol'] = dados['cholesterol_raw'] / 38.67
dados['triglycerides'] = dados['triglycerides_raw'] / 88.57
dados['hdl'] = dados['hdl_raw'] / 38.67
dados['ldl'] = dados['ldl_raw'] / 38.67
dados = dados.rename(columns={'diastolic_raw': 'diastolic_bp', 'systolic_raw': 'systolic_bp'})

# recode alcohol and activity
alcohol_mapping = {0: 0, 1: round((0.5 * 10) / 7, 2), 2: round((3 * 10) / 7, 2), 3: 10}
dados['alcohol'] = dados['food_19'].map(alcohol_mapping)
dados['moderate_minutes'] = pd.to_numeric(dados['moderate_minutes'], errors='coerce')
dados['vigorous_minutes'] = pd.to_numeric(dados['vigorous_minutes'], errors='coerce')
dados['moderate_act'] = (dados['moderate_minutes'] / dados['valid_days'] / 60).round(2)
dados['vigorous_act'] = (dados['vigorous_minutes'] / dados['valid_days'] / 60).round(2)
dados = dados.rename(columns={'heart_rate_raw': 'resting_heart_rate', 'shuttle_speed_raw': 'shuttle_run_speed'})

# recode food frequencies
freq_map = {0: 0, 1: round(0.5 / 7, 2), 2: round(1 / 7, 2), 3: round(3 / 7, 2), 4: round(5.5 / 7, 2), 5: 1, 6: 1.5}
dados['vegetables'] = dados['food_2'].map(freq_map)
dados['fruits'] = dados['food_1'].map(freq_map)
def avg_portion(row):
    vals = []
    for col in ['food_9', 'food_13', 'food_14']:
        if col in row.index and pd.notnull(row[col]):
            try:
                value = int(row[col])
            except Exception:
                value = np.nan
            vals.append(freq_map.get(value, np.nan))
    return np.mean(vals) if vals else np.nan
dados['cereals'] = dados.apply(avg_portion, axis=1)
dados['sweetened_beverages'] = dados['food_3'].map(freq_map)
dados['sleep'] = (dados['sleep_total'] / dados['valid_days']).round(2)

# recode psychosocial and family measures
dados = dados.rename(columns={'loneliness_raw': 'lonely'})
def fam_diab(row):
    mother = row['mother_diabetes_1'] == 1 or row['mother_diabetes_2'] == 1
    father = row['father_diabetes_1'] == 1 or row['father_diabetes_2'] == 1
    return 1 if (mother or father) else 0
dados['family_diabetes'] = dados.apply(fam_diab, axis=1)
dados = dados.rename(columns={'vo2_measure': 'vo2_max'})
ed_map = {0: 0, 1: 6, 2: 12, 3: 16, 4: 19}
dados['mother_educ'] = dados['mother_education_code'].map(ed_map)
dados['father_educ'] = dados['father_education_code'].map(ed_map)
dados['family_cars'] = dados['cars_code'].map({0: 1, 1: 2, 2: 3, 3: 3})
dados['bedroom'] = dados['bedroom_code'].map({0: 1, 1: 2})
dados['family_computers'] = dados['computers_code'].map({0: 1, 1: 2, 2: 3, 3: 4})
dados['internet'] = dados['internet_code'].map({0: 1, 1: 2})

# integrate the additional values by participant id
selected = pd.read_csv('DATA/dados/selected.csv', dtype={'id': str})
sel_lookup = selected.set_index('id')[['moderate_act', 'vigorous_act', 'vo2_max', 'sleep']]
for col in ['moderate_act', 'vigorous_act', 'vo2_max', 'sleep']:
    dados[col] = dados['id'].map(sel_lookup[col]).fillna(dados[col])
for col in ['moderate_act', 'vigorous_act', 'vo2_max', 'sleep']:
    dados[col] = pd.to_numeric(dados[col], errors='coerce')
a = dados['moderate_act']
b = dados['vigorous_act']
dados['mvpa'] = np.where(a.isna() & b.isna(), np.nan, a.fillna(0) + b.fillna(0))
dados['mvpa'] = dados['mvpa'].round(2)

# calculate affluence z-scores
affluence_vars = ['mother_educ', 'father_educ', 'family_cars', 'bedroom', 'family_computers', 'internet']
z_cols = []
for var in affluence_vars:
    col = pd.to_numeric(dados[var], errors='coerce')
    mu = col.mean()
    sigma = col.std(ddof=0)
    zcol = f'z_score_{var}'
    z_cols.append(zcol)
    if sigma == 0 or pd.isna(sigma):
        dados[zcol] = np.nan
    else:
        dados[zcol] = col.apply(lambda x: (x - mu) / sigma if pd.notnull(x) else np.nan)
dados['z_score_affluence'] = dados[z_cols].mean(axis=1, skipna=True)

# select and format the harmonized columns
desired = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
    'diastolic_bp', 'systolic_bp', 'alcohol', 'moderate_act', 'vigorous_act',
    'mvpa', 'shuttle_run_speed', 'vegetables', 'fruits', 'vo2_max',
    'family_diabetes', 'cereals', 'sweetened_beverages', 'sleep', 'lonely',
    'mother_educ', 'father_educ', 'family_cars', 'bedroom', 'family_computers',
    'internet', 'z_score_mother_educ', 'z_score_father_educ',
    'z_score_family_cars', 'z_score_bedroom', 'z_score_family_computers',
    'z_score_internet', 'z_score_affluence'
]
final = dados[[col for col in desired if col in dados.columns]].copy()
final = final.astype('object')
for col in final.columns:
    if col not in ['origin', 'country', 'id', 'sex']:
        final[col] = pd.to_numeric(final[col], errors='coerce').apply(lambda x: f'{x:.2f}' if pd.notnull(x) else '')
os.makedirs('RESULTS', exist_ok=True)
final.to_csv(output_file, index=False)
print('DADOS_edited.csv has been created.')
