# slofit students data preprocessing and harmonization
# original input filenames and source column names have been replaced with generic names
# the input file must use the generic column names defined in this script
# no original participant data are included

import pandas as pd
import numpy as np
import os

# define file paths
input_folder = 'DATA/slofit_students'
output_folder = 'RESULTS'
input_file = 'input.csv'
output_file = 'SLOfit_Students_edited.csv'

# load the semicolon-delimited dataset and clean column names
slofit_data = pd.read_csv(os.path.join(input_folder, input_file), delimiter=';', on_bad_lines='skip')
slofit_data.columns = slofit_data.columns.str.strip()

# fix comma-decimal columns
for col in ['participant_age', 'body_mass_index']:
    slofit_data[col] = slofit_data[col].astype(str).str.replace(',', '.', regex=False)
    slofit_data[col] = pd.to_numeric(slofit_data[col], errors='coerce')

# remember original participant and age combinations
original_ids_ages = set(zip(slofit_data['participant_id'], slofit_data['participant_age']))

# rename generic source columns to harmonized variables
slofit_data = slofit_data.rename(columns={
    'participant_id': 'id',
    'participant_sex': 'sex',
    'participant_age': 'age',
    'body_mass_index': 'bmi',
    'body_height': 'height',
    'body_weight': 'weight',
    'run_600m_time': 't600',
    'standing_long_jump': 'standing_lj'
})

# convert height from mm to cm and weight from tenths of kg to kg
slofit_data['height'] = pd.to_numeric(slofit_data['height'], errors='coerce') / 10
slofit_data['weight'] = pd.to_numeric(slofit_data['weight'], errors='coerce') / 10
slofit_data['standing_lj'] = pd.to_numeric(slofit_data['standing_lj'], errors='coerce')

# normalize sex and add cohort metadata
slofit_data['sex'] = slofit_data['sex'].astype(str).str.strip().str.capitalize()
slofit_data['origin'] = 'SLOfit Students'
slofit_data['country'] = 'Slovenia'

# keep the original participant-age combinations
slofit_data = slofit_data[slofit_data[['id', 'age']].apply(tuple, axis=1).isin(original_ids_ages)].copy()

# convert the 600-metre run time to numeric
slofit_data['t600'] = pd.to_numeric(slofit_data['t600'], errors='coerce')

# remove records containing only the minimum identifying information
required = ['origin', 'id', 'sex', 'age']
mask = slofit_data.notna().sum(axis=1) > slofit_data[required].notna().sum(axis=1)
slofit_data = slofit_data[mask].copy()

# estimate shuttle-run speed and vo2 max from the 600-metre run
# sex codes: 1 = male, 2 = female
def estimate_vo2max(age, sex, height, weight, t600):
    if pd.isna(age) or pd.isna(height) or pd.isna(weight) or pd.isna(t600):
        return np.nan, np.nan
    if height <= 0 or weight <= 0:
        return np.nan, np.nan

    bmi = weight / (height / 100) ** 2
    a2, a3 = age ** 2, age ** 3

    if sex == 1:
        srt = (
            12.47157
            - 0.01532 * t600
            - 0.65344 * age
            + 0.07034 * a2
            - 0.00192 * a3
            + 0.02428 * height
            - 0.02831 * weight
            + 0.00363 * bmi
        )
    else:
        srt = (
            12.13319
            - 0.01203 * t600
            - 0.23114 * age
            + 0.03633 * a2
            - 0.00118 * a3
            + 0.00813 * height
            - 0.01158 * weight
            - 0.01690 * bmi
        )

    if pd.isna(srt):
        return np.nan, np.nan

    level = max(1, int((srt - 8.0) / 0.5) + 1)
    laps = level * 9
    vo2 = 41.77 + 0.49 * laps - 0.0029 * laps ** 2 - 0.62 * bmi

    if pd.isna(vo2):
        return np.nan, np.nan
    return round(srt, 2), round(vo2, 2)

# populate estimated variables and count successful transformations
slofit_data['shuttle_run_speed'] = np.nan
slofit_data['vo2_max'] = np.nan
n_t600_present = 0
n_shuttle_filled = 0
n_vo2_filled = 0
n_failed_transform = 0

for idx, row in slofit_data.iterrows():
    t = row['t600']
    if pd.notna(t):
        n_t600_present += 1
        sex_code = 1 if row['sex'] == 'Male' else 2
        s, v = estimate_vo2max(row['age'], sex_code, row['height'], row['weight'], t)

        if pd.notna(s):
            slofit_data.at[idx, 'shuttle_run_speed'] = s
            n_shuttle_filled += 1
        else:
            n_failed_transform += 1

        if pd.notna(v):
            slofit_data.at[idx, 'vo2_max'] = v
            n_vo2_filled += 1

# select the final harmonized columns in the original order
desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight',
    'standing_lj', 't600', 'shuttle_run_speed', 'vo2_max'
]
final_cols = [c for c in desired_columns if c in slofit_data.columns]
slofit_data = slofit_data[final_cols].copy()

# format numeric values with up to two decimal places
def fmt(x):
    if pd.isna(x) or x == '':
        return ''
    try:
        f = float(x)
    except Exception:
        return x
    s = f'{f:.2f}'
    return s.rstrip('0').rstrip('.')

for c in slofit_data.columns:
    if c not in ['origin', 'country', 'id', 'sex']:
        slofit_data[c] = slofit_data[c].apply(fmt)

# prefix participant ids and save the output
slofit_data['id'] = 'SLOFIT-STUDENTS-' + slofit_data['id'].astype(str)
os.makedirs(output_folder, exist_ok=True)
slofit_data.to_csv(os.path.join(output_folder, output_file), index=False)
print('\nSLOfit Students dataset processed and saved successfully!')
