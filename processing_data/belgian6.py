# public version of the belgian wave 6 preprocessing script
# original filenames and source column names have been replaced with generic names
# input files must use the generic column names shown in this script
# original datasets are not included

import os
import pandas as pd
import numpy as np

input_folder = 'DATA/belgian_wave6'
output_folder = 'RESULTS'
output_file = 'Phase6_edited.csv'

# load the seven input files
antromotdiet = pd.read_csv(os.path.join(input_folder, 'anthropometry.csv'))
diet = pd.read_csv(os.path.join(input_folder, 'diet.csv'))
physact = pd.read_csv(os.path.join(input_folder, 'physical_activity.csv'))
whosmok = pd.read_csv(os.path.join(input_folder, 'smoking.csv'))
med2 = pd.read_csv(os.path.join(input_folder, 'medical.csv'))
scl90 = pd.read_csv(os.path.join(input_folder, 'wellbeing.csv'))
lbppsy = pd.read_csv(os.path.join(input_folder, 'stress.csv'))

# use a shared participant id column
datasets = [antromotdiet, diet, physact, whosmok, med2, scl90, lbppsy]
for df in datasets:
    if 'participant_id_alt' in df.columns:
        df.rename(columns={'participant_id_alt': 'participant_id'}, inplace=True)

def map_sex(value):
    val = str(value)
    return 'Male' if (val.startswith('2') and val.endswith('1')) or (val.startswith('3') and val.endswith('1')) else 'Female'

def process_smoking(row):
    current = row['current_smoking']
    previous = row['previous_smoking_info']
    if isinstance(previous, str):
        has_info = previous.strip() != ''
    else:
        has_info = not pd.isna(previous)
    if current in [1, 2]:
        return 2
    if current == 0:
        return 1 if has_info else 0
    return np.nan

# harmonize anthropometry and clinical measures
antromotdiet = antromotdiet.rename(columns={
    'sex_source': 'sex', 'body_mass_index': 'bmi', 'waist_measurement': 'waist_circ',
    'body_height': 'height', 'body_weight': 'weight', 'blood_glucose': 'glucose',
    'total_cholesterol': 'cholesterol', 'blood_triglycerides': 'triglycerides',
    'hdl_cholesterol': 'hdl', 'ldl_cholesterol': 'ldl',
    'diastolic_pressure': 'diastolic_bp', 'systolic_pressure': 'systolic_bp',
    'walking_test_distance': '6_min_walking_test'
})
antromotdiet['sex'] = antromotdiet['participant_id'].apply(map_sex)

# harmonize dietary measures
diet = diet.rename(columns={
    'participant_age': 'age', 'alcohol_intake': 'alcohol', 'sugar_intake': 'sugar',
    'protein_intake': 'protein', 'fat_intake': 'fat',
    'vegetable_intake': 'vegetables', 'fruit_intake': 'fruits', 'dairy_intake': 'dairy'
})

# recode smoking
whosmok['smoking'] = whosmok.apply(process_smoking, axis=1)

# convert weekly activity and sleep to daily values
physact = physact.rename(columns={
    'vigorous_activity_weekly': 'vigorous_act',
    'moderate_activity_weekly': 'moderate_act'
})
physact['vigorous_act'] = physact['vigorous_act'] / 7
physact['moderate_act'] = physact['moderate_act'] / 7
physact['sleep'] = physact['sleep_weekly'] / 7

# harmonize medical and questionnaire measures
med2 = med2.rename(columns={'diabetes_status': 'diabetes', 'family_diabetes_status': 'family_diabetes'})
scl90['loneliness_item'] = pd.to_numeric(scl90['loneliness_item'], errors='coerce')
scl90['lonely'] = scl90['loneliness_item'] - 1
lbppsy = lbppsy.rename(columns={'stress_category': 'stress'})

# merge on participant id
merged = pd.merge(antromotdiet, diet[['participant_id', 'age', 'alcohol', 'sugar', 'protein', 'fat', 'vegetables', 'fruits', 'dairy']], on='participant_id', how='outer')
merged = pd.merge(merged, whosmok[['participant_id', 'smoking']], on='participant_id', how='outer')
merged = pd.merge(merged, physact[['participant_id', 'vigorous_act', 'moderate_act', 'sleep']], on='participant_id', how='outer')
merged = pd.merge(merged, med2[['participant_id', 'diabetes', 'family_diabetes']], on='participant_id', how='outer')
merged = pd.merge(merged, scl90[['participant_id', 'lonely']], on='participant_id', how='outer')
merged = pd.merge(merged, lbppsy[['participant_id', 'stress']], on='participant_id', how='outer')
merged['mvpa'] = merged['vigorous_act'] + merged['moderate_act']
merged['origin'] = 'LGS_Wave_6'
merged['country'] = 'Belgium'
merged = merged.rename(columns={'participant_id': 'id'})

desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
    'diastolic_bp', 'systolic_bp', 'vigorous_act', 'moderate_act', 'mvpa', 'sleep',
    'vegetables', 'fruits', 'alcohol', 'smoking', 'resting_heart_rate', 'sit_and_reach',
    'diabetes', 'family_diabetes', 'lonely', 'stress', '6_min_walking_test'
]
final_columns = [col for col in desired_columns if col in merged.columns]
merged = merged.reindex(columns=final_columns)
merged['id'] = 'BE-KUL-' + merged['id'].astype(str)

# convert numerical columns
numeric_cols = [col for col in merged.columns if col not in ['origin', 'country', 'sex', 'id']]
for col in numeric_cols:
    try:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    except Exception as e:
        print(f'Error converting column {col}: {e}')

# convert blood measurements to mmol/l
if 'cholesterol' in merged.columns:
    merged['cholesterol'] = merged['cholesterol'] / 38.67
if 'hdl' in merged.columns:
    merged['hdl'] = merged['hdl'] / 38.67
if 'ldl' in merged.columns:
    merged['ldl'] = merged['ldl'] / 38.67
if 'triglycerides' in merged.columns:
    merged['triglycerides'] = merged['triglycerides'] / 88.57
if 'glucose' in merged.columns:
    merged['glucose'] = merged['glucose'] * 0.0555

# round values and remove rows with too few values
for col in numeric_cols:
    merged[col] = merged[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    merged[col] = merged[col].apply(lambda x: ('{:.2f}'.format(x).rstrip('0').rstrip('.')) if pd.notnull(x) else '')
merged.replace('', np.nan, inplace=True)
merged = merged.dropna(thresh=4)

# save the harmonized dataset
os.makedirs(output_folder, exist_ok=True)
merged.to_csv(os.path.join(output_folder, output_file), index=False)
print('Merged dataset created successfully!')
