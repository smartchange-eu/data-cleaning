# public version of the belgian wave 2-4 preprocessing script
# original input filenames and source column names have been replaced
# input files must use the generic column names in this script
# original datasets and participant data are not included

import pandas as pd
import numpy as np
import os

folder_path = 'DATA/belgian_wave234'
results_folder = 'RESULTS'

blood = pd.read_csv(os.path.join(folder_path, 'blood.csv'))
diet = pd.read_csv(os.path.join(folder_path, 'diet.csv'))
antro = pd.read_csv(os.path.join(folder_path, 'anthropometry.csv'))
fysio = pd.read_csv(os.path.join(folder_path, 'physiology.csv'))
med = pd.read_csv(os.path.join(folder_path, 'medical.csv'))
motor = pd.read_csv(os.path.join(folder_path, 'motor.csv'))
fc = pd.read_csv(os.path.join(folder_path, 'activity.csv'))
sleep = pd.read_csv(os.path.join(folder_path, 'sleep.csv'))
life = pd.read_csv(os.path.join(folder_path, 'life.csv'))

antro = antro['participant_id source_weight_7 source_weight_8 source_weight_9 source_height_7 source_height_8 source_height_9 source_bmi_7 source_bmi_8 source_bmi_9 source_waist_8 source_waist_9'.split()]
antro_long = antro.melt(id_vars=['participant_id'], var_name='Variable', value_name='Value')
antro_long['Year'] = antro_long['Variable'].str.extract('_(\\d)').astype(int)
antro_long['Variable'] = antro_long['Variable'].str.replace('_\\d', '', regex=True)
antro_pivot = antro_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()
antro_pivot['source_age'] = antro_pivot['Year'].map({7: 30, 8: 35, 9: 40})

blood = blood[['participant_id', 'source_glucose_9', 'source_cholesterol_9', 'source_triglycerides_9', 'source_hdl_9', 'source_ldl_9']]
blood['Year'] = 9

diet = diet[['participant_id', 'source_energy_9', 'source_alcohol_pct_9', 'source_sugar_pct_9', 'source_protein_pct_9', 'source_fat_pct_9']]
diet['Year'] = 9

fysio = fysio['participant_id source_oxygen_7 source_oxygen_8 source_oxygen_9 source_pulse_7 source_pulse_8'.split()]
fysio_long = fysio.melt(id_vars='participant_id', var_name='Variable', value_name='Value')
fysio_long['Year'] = fysio_long['Variable'].str.extract('_(\\d)').astype(int)
fysio_long['Variable'] = fysio_long['Variable'].str.replace('_\\d', '', regex=True)
fysio_pivot = fysio_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()

med_diabetes = med[['participant_id', 'source_mother_diabetes', 'source_father_diabetes']].copy()

med_for_melt = med[[
    'participant_id', 'source_medication_7', 'source_medication_8', 'source_medication_9',
    'source_smoking_7', 'source_smoking_8', 'source_smoking_9',
    'source_systolic_7', 'source_systolic_8', 'source_systolic_9',
    'source_diastolic_7', 'source_diastolic_8', 'source_diastolic_9'
]].copy()

med_long = med_for_melt.melt(id_vars=['participant_id'], var_name='Variable', value_name='Value')
med_long['Year'] = med_long['Variable'].str.extract('_(\\d)').astype(int)
med_long['Variable'] = med_long['Variable'].str.replace('_\\d', '', regex=True)
med_pivot = med_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()

def recode_smoking(x):
    if pd.isna(x):
        return 0
    try:
        x = float(x)
    except ValueError:
        return 0

    if x == 0:
        return 0
    elif x == 1:
        return 2
    elif x > 1:
        return 1
    else:
        return 0

med_pivot['source_smoking'] = (
    pd.to_numeric(med_pivot['source_smoking'], errors='coerce')
    .apply(recode_smoking)
)

motor = motor[['participant_id', 'source_flexibility_7', 'source_flexibility_8', 'source_flexibility_9']]
motor_long = motor.melt(id_vars='participant_id', var_name='Variable', value_name='Value')
motor_long['Year'] = motor_long['Variable'].str.extract('_(\\d)').astype(int)
motor_long['Variable'] = motor_long['Variable'].str.replace('_\\d', '', regex=True)
motor_pivot = motor_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()

fc = fc[['participant_id', 'source_vigorous_8', 'source_vigorous_9', 'source_moderate_8', 'source_moderate_9']]
fc_long = fc.melt(id_vars='participant_id', var_name='Variable', value_name='Value')
fc_long['Year'] = fc_long['Variable'].str.extract('_(\\d)').astype(int)
fc_long['Variable'] = fc_long['Variable'].str.replace('_\\d', '', regex=True)
fc_pivot = fc_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()

sleep = sleep[['participant_id', 'source_sleep_7', 'source_sleep_8', 'source_sleep_9']]
sleep_long = sleep.melt(id_vars='participant_id', var_name='Variable', value_name='Value')
sleep_long['Year'] = sleep_long['Variable'].str.extract('_(\\d)').astype(int)
sleep_long['Variable'] = sleep_long['Variable'].str.replace('_\\d', '', regex=True)
sleep_pivot = sleep_long.pivot(index=['participant_id', 'Year'], columns='Variable', values='Value').reset_index()
sleep_pivot = sleep_pivot.rename(columns={'source_sleep': 'source_sleep_daily'})
sleep_pivot['source_sleep_daily'] = pd.to_numeric(sleep_pivot['source_sleep_daily'].replace('', np.nan), errors='coerce') / 7

merged_df = pd.concat(
    [antro_pivot, blood, diet, fysio_pivot, med_pivot, motor_pivot, fc_pivot, sleep_pivot],
    axis=0,
    ignore_index=True
)
merged_df = merged_df.groupby(['participant_id', 'Year']).first().reset_index()
merged_df = merged_df.drop(columns=['Year'])

merged_df.insert(0, 'origin', 'LGS_Wave234')
merged_df.insert(1, 'country', 'Belgium')
merged_df['sex'] = 'Male'

merged_df = merged_df.merge(med_diabetes, on='participant_id', how='left')

def compute_family_diabetes(row):
    father = int(row['source_father_diabetes']) if pd.notnull(row['source_father_diabetes']) else 0
    mother = int(row['source_mother_diabetes']) if pd.notnull(row['source_mother_diabetes']) else 0

    if father == 0 and mother == 0:
        return 0
    else:
        return 1

merged_df['family_diabetes_temp'] = merged_df.apply(compute_family_diabetes, axis=1)
merged_df = merged_df.drop(columns=['source_mother_diabetes', 'source_father_diabetes'])

column_mapping = {
    'participant_id': 'id',
    'source_weight': 'weight',
    'source_height': 'height',
    'source_bmi': 'bmi',
    'source_waist': 'waist_circ',
    'source_glucose_9': 'glucose',
    'source_cholesterol_9': 'cholesterol',
    'source_triglycerides_9': 'triglycerides',
    'source_hdl_9': 'hdl',
    'source_ldl_9': 'ldl',
    'source_alcohol_pct_9': 'alcohol',
    'source_sugar_pct_9': 'sugar',
    'source_protein_pct_9': 'protein',
    'source_fat_pct_9': 'fat',
    'source_pulse': 'resting_heart_rate',
    'source_oxygen': 'vo2',
    'source_diastolic': 'diastolic_bp',
    'source_systolic': 'systolic_bp',
    'source_smoking': 'smoking',
    'source_flexibility': 'sit_and_reach',
    'source_vigorous': 'vigorous_act',
    'source_moderate': 'moderate_act',
    'source_medication': 'medication',
    'source_age': 'age',
    'source_sleep_daily': 'sleep',
    'family_diabetes_temp': 'family_diabetes',
    'source_energy_9': 'energy_kcal',
}

merged_df = merged_df.rename(columns=column_mapping)

if {'alcohol', 'energy_kcal'}.issubset(merged_df.columns):
    merged_df['alcohol'] = (
        (merged_df['alcohol'] / 100)
        * merged_df['energy_kcal']
        / 7
    ).round(2)

for col in merged_df.columns:
    if col not in ['origin', 'country', 'sex', 'family_diabetes', 'id']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

if 'vigorous_act' in merged_df.columns:
    merged_df['vigorous_act'] = merged_df['vigorous_act'].apply(lambda x: x / 7 if pd.notnull(x) else x)
if 'moderate_act' in merged_df.columns:
    merged_df['moderate_act'] = merged_df['moderate_act'].apply(lambda x: x / 7 if pd.notnull(x) else x)

merged_df['mvpa'] = merged_df[['vigorous_act', 'moderate_act']].apply(
    lambda row: np.nan if row.isna().all() else row.fillna(0).sum(),
    axis=1
)

if 'weight' in merged_df.columns and 'vo2' in merged_df.columns:
    merged_df['vo2'] = merged_df.apply(
        lambda row: (row['vo2'] * 1000 / row['weight']) if pd.notnull(row['vo2']) and pd.notnull(row['weight']) else np.nan,
        axis=1
    )
merged_df['vo2'] = merged_df['vo2'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

if 'glucose' in merged_df.columns:
    merged_df['glucose'] = merged_df['glucose'] * 0.0555

for col in ['cholesterol', 'hdl', 'ldl']:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col] / 38.67

if 'triglycerides' in merged_df.columns:
    merged_df['triglycerides'] = merged_df['triglycerides'] / 88.57

merged_df = merged_df.rename(columns={'vo2': 'vo2_max'})

life = life.rename(columns={
    'participant_id': 'id',
    'source_stopped_studies_7': 'stopped_studies',
    'source_started_job_7': 'started_job',
    'source_years_unemployed_7': 'years_unemployed',
    'source_live_own_7': 'live_own',
    'source_house_owner_7': 'house_owner'
})

life.columns = life.columns.str.strip()

life = life[[
    'id',
    'stopped_studies',
    'started_job',
    'years_unemployed',
    'live_own',
    'house_owner'
]].copy()

for col in ['stopped_studies', 'started_job', 'years_unemployed', 'live_own', 'house_owner']:
    life[col] = (
        life[col]
        .replace({98: np.nan, 99: np.nan, '': np.nan})
        .pipe(pd.to_numeric, errors='coerce')
    )

zmap = {
    'stopped_studies': 'z_score_studies',
    'started_job': 'z_score_job',
    'years_unemployed': 'z_score_unemployed',
    'live_own': 'z_score_own',
    'house_owner': 'z_score_house'
}

for raw, zcol in zmap.items():
    life[zcol] = (life[raw] - life[raw].mean()) / life[raw].std()
    life[zcol] *= -1

life['z_score_affluence'] = life[list(zmap.values())].mean(axis=1)

merged_df = merged_df.merge(
    life[['id'] + list(zmap.keys()) + list(zmap.values()) + ['z_score_affluence']],
    on='id',
    how='left'
)

def format_numeric(x):
    if pd.isnull(x):
        return x
    try:
        if np.isfinite(x) and x == int(x):
            return int(x)
        else:
            return round(x, 2)
    except Exception:
        return x

numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    merged_df[col] = merged_df[col].apply(format_numeric)

desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
    'diastolic_bp', 'systolic_bp',
    'smoking', 'alcohol', 'sugar', 'protein', 'fat',
    'vigorous_act', 'moderate_act', 'mvpa', 'sleep',
    'dairy',
    'resting_heart_rate', 'sit_and_reach',
    'family_diabetes', 'vo2_max', 'medication',
    'stopped_studies', 'started_job', 'years_unemployed', 'live_own', 'house_owner',
    'z_score_studies', 'z_score_job', 'z_score_unemployed', 'z_score_own', 'z_score_house',
    'z_score_affluence'
]
final_columns = [col for col in desired_columns if col in merged_df.columns]
merged_df = merged_df[final_columns]

merged_df['id'] = 'BE-KUL-' + merged_df['id'].astype(str)

merged_df = merged_df.replace('', np.nan)
merged_df = merged_df.dropna(thresh=6)

print('\nAlcohol (g/day) summary:')
print('Min:', merged_df['alcohol'].min())
print('Max:', merged_df['alcohol'].max())

print(merged_df['alcohol'].describe())

os.makedirs(results_folder, exist_ok=True)
out_path = os.path.join(results_folder, 'Phase234_edited.csv')
merged_df.to_csv(out_path, index=False)
print(f"Merged dataset saved to '{out_path}' with the desired column order.")
