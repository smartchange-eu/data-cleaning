# slofit lifelong data preprocessing and harmonization
# this script converts cohort measurements to the standardized smartchange format
# original filenames and source column names have been replaced with generic names
# the input excel files must use the generic source columns specified below
# no original participant data or identifier crosswalks are included

import os
import pandas as pd
import numpy as np

# file paths
input_folder = 'DATA/slofit_lifelong'
output_folder = 'RESULTS'
input_file = 'measurements.xlsx'
health_file = 'health_problems.xlsx'
output_file = 'SLOfit_lifelong_edited.csv'

# load the main measurement dataset
slofit_data = pd.read_excel(os.path.join(input_folder, input_file))
slofit_data.columns = slofit_data.columns.str.strip()

# remember the original participant and age combinations
original_ids_ages = set(zip(slofit_data['participant_id'], slofit_data['participant_age']))

# rename generic source columns to harmonized output names
slofit_data = slofit_data.rename(columns={
    'participant_id': 'id',
    'participant_sex': 'sex',
    'participant_age': 'age',
    'body_mass_index': 'bmi',
    'body_height': 'height',
    'body_weight': 'weight',
    'waist_measurement': 'waist_circ',
    'flexibility_measurement': 'sit_and_reach',
    'standing_jump_measurement': 'standing_lj',
    'six_minute_walk_distance': '6_min_walking_test',
    'six_hundred_meter_time': 't600'
})

# standardize sex and add cohort information
slofit_data['sex'] = slofit_data['sex'].astype(str).str.strip().str.capitalize()
slofit_data['origin'] = 'SLOfit lifelong'
slofit_data['country'] = 'Slovenia'

# keep participant and age combinations from the source data
slofit_data = slofit_data[
    slofit_data[['id', 'age']].apply(tuple, axis=1).isin(original_ids_ages)
].copy()

# select and order the measurement columns
desired = [
    'origin', 'country', 'id', 'sex', 'age',
    'bmi', 'height', 'weight', 'waist_circ', 'sit_and_reach',
    'standing_lj', '6_min_walking_test', 't600'
]
cols = [c for c in desired if c in slofit_data.columns]
slofit_data = slofit_data.reindex(columns=cols)

# convert measurement columns to numeric values
numeric_cols = [c for c in slofit_data.columns if c not in ['origin', 'country', 'sex', 'id']]
for col in numeric_cols:
    slofit_data[col] = pd.to_numeric(slofit_data[col], errors='coerce')

# treat zero values as missing for these performance tests
if 'standing_lj' in slofit_data.columns:
    slofit_data.loc[slofit_data['standing_lj'] == 0, 'standing_lj'] = np.nan
if '6_min_walking_test' in slofit_data.columns:
    slofit_data.loc[slofit_data['6_min_walking_test'] == 0, '6_min_walking_test'] = np.nan

# remove rows containing only basic participant information
req = ['origin', 'id', 'sex', 'age']
mask = slofit_data.notna().sum(axis=1) > slofit_data[req].notna().sum(axis=1)
slofit_data = slofit_data[mask].copy()

# combine duplicate records for the same participant and age
agg_dict = {
    'origin': 'first',
    'sex': 'first',
    'bmi': 'mean',
    'height': 'mean',
    'weight': 'mean',
    'waist_circ': 'mean',
    'sit_and_reach': 'mean',
    'standing_lj': 'mean',
    '6_min_walking_test': 'mean',
    't600': 'mean'
}
slofit_data = slofit_data.groupby(['id', 'age'], as_index=False).agg(agg_dict)
slofit_data['country'] = 'Slovenia'

# load health records and identify the original diabetes categories
health_data = pd.read_excel(os.path.join(input_folder, health_file))
health_data.columns = health_data.columns.str.strip()
diabetes_records = health_data[health_data['health_condition_code'].isin([17, 18])]

def get_diabetes_value(participant_id):
    records = diabetes_records[diabetes_records['participant_id'] == participant_id]
    if records.empty:
        return ''
    if records['response'].astype(str).str.strip().str.lower().eq('yes').any():
        return 1
    return 0

slofit_data['diabetes'] = slofit_data['id'].apply(get_diabetes_value)

# estimate shuttle-run speed and vo2max from the 600-meter test
# sex codes used in the equations: 1 = male, 2 = female
def estimate_vo2max(age, sex, height, weight, t600):
    if pd.isna(age) or pd.isna(height) or pd.isna(weight) or pd.isna(t600):
        return np.nan, np.nan
    if height <= 0 or weight <= 0:
        return np.nan, np.nan

    bmi = weight / (height / 100) ** 2
    a2, a3 = age ** 2, age ** 3

    if sex == 1:
        srt = (
            12.47157 - 0.01532 * t600 - 0.65344 * age
            + 0.07034 * a2 - 0.00192 * a3 + 0.02428 * height
            - 0.02831 * weight + 0.00363 * bmi
        )
    elif sex == 2:
        srt = (
            12.13319 - 0.01203 * t600 - 0.23114 * age
            + 0.03633 * a2 - 0.00118 * a3 + 0.00813 * height
            - 0.01158 * weight - 0.01690 * bmi
        )
    else:
        return np.nan, np.nan

    if pd.isna(srt):
        return np.nan, np.nan

    level = max(1, int((srt - 8.0) / 0.5) + 1)
    laps = level * 9
    vo2 = 41.77 + 0.49 * laps - 0.0029 * laps ** 2 - 0.62 * bmi
    if pd.isna(vo2):
        return np.nan, np.nan
    return round(srt, 2), round(vo2, 2)

# calculate performance estimates and track missing results
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
        speed, vo2 = estimate_vo2max(row['age'], sex_code, row['height'], row['weight'], t)
        if pd.notna(speed):
            slofit_data.at[idx, 'shuttle_run_speed'] = speed
            n_shuttle_filled += 1
        else:
            n_failed_transform += 1
        if pd.notna(vo2):
            slofit_data.at[idx, 'vo2_max'] = vo2
            n_vo2_filled += 1

# format numeric values to a maximum of two decimal places
def format_cell(value):
    if value == '' or pd.isna(value):
        return ''
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return f'{number:.2f}'.rstrip('0').rstrip('.')

num_cols = [c for c in slofit_data.columns if c not in ['origin', 'country', 'sex', 'id']]
for col in num_cols:
    slofit_data[col] = slofit_data[col].apply(format_cell)

# add the original cohort-specific identifier prefix
slofit_data['id'] = 'SLOFIT-LIFELONG-' + slofit_data['id'].astype(str)

# save the harmonized dataset
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_file)
slofit_data.to_csv(output_path, index=False)
print(f'SLOfit lifelong dataset saved to {output_path}')
