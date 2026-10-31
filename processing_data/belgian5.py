# public version of the belgian wave 5 preprocessing script
# source file names and source variable names have been replaced with generic names
# the original datasets are not included in this repository
# input files must use the generic column names defined in this script

import os
import pandas as pd
import numpy as np

input_folder = os.path.join('DATA', 'belgian_wave5')
output_folder = 'RESULTS'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'Phase5_edited.csv')

# load the five input tables
df1 = pd.read_csv(os.path.join(input_folder, 'anthropometry_diet.csv'))
df2 = pd.read_csv(os.path.join(input_folder, 'psychosocial.csv'))
df3 = pd.read_csv(os.path.join(input_folder, 'physical_activity.csv'))
df4 = pd.read_csv(os.path.join(input_folder, 'medical.csv'))
df5 = pd.read_csv(os.path.join(input_folder, 'wellbeing.csv'))

# use the same participant key across tables
for table in [df1, df2, df3, df4, df5]:
    if 'participant_id_alt' in table.columns:
        table.rename(columns={'participant_id_alt': 'participant_id'}, inplace=True)

def map_sex(value):
    return 'Male' if value == 1 else 'Female'

# recode smoking status
for col in ['smoking_current', 'smoking_ever']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce').replace(999, np.nan)

def process_smoking(row):
    current = row['smoking_current']
    ever = row['smoking_ever']
    if current in [1, 2]:
        return 2
    if current == 0:
        if ever == 1:
            return 1
        if ever == 0:
            return 0
    return np.nan

# rename source fields to the output schema
df1 = df1.rename(columns={
    'sex_code': 'sex',
    'bmi_value': 'bmi',
    'waist_value': 'waist_circ',
    'height_value': 'height',
    'weight_value': 'weight',
    'age_value': 'age',
    'glucose_value': 'glucose',
    'cholesterol_value': 'cholesterol',
    'triglycerides_value': 'triglycerides',
    'hdl_value': 'hdl',
    'ldl_value': 'ldl',
    'alcohol_energy_percent': 'alcohol',
    'total_energy_kcal': 'energy_kcal',
    'sugar_energy_percent': 'sugar',
    'protein_energy_percent': 'protein',
    'fat_energy_percent': 'fat',
    'vo2max_value': 'vo2_max',
    'diastolic_pressure': 'diastolic_bp',
    'systolic_pressure': 'systolic_bp',
    'shuttle_run_value': 'shuttle_run_speed',
})
df1['sex'] = df1['sex'].apply(map_sex)

# add smoking and stress
df2['smoking'] = df2.apply(process_smoking, axis=1)
df2['stress'] = df2['stress_category'] if 'stress_category' in df2.columns else np.nan

# calculate activity and sleep averages
for col in ['vigorous_activity_total', 'moderate_activity_total']:
    df3[col] = pd.to_numeric(df3[col], errors='coerce')
if 'sleep_total' in df3.columns:
    df3['sleep_total'] = pd.to_numeric(df3['sleep_total'], errors='coerce')
df3['vigorous_act'] = df3['vigorous_activity_total'] / 7
df3['moderate_act'] = df3['moderate_activity_total'] / 7
df3['sleep'] = df3['sleep_total'] / 7

# rename medical variables
df4 = df4.rename(columns={
    'diabetes_status': 'diabetes',
    'family_diabetes_codes': 'family_diabetes',
})

# calculate loneliness score
if 'loneliness_response' in df5.columns:
    df5['loneliness_response'] = pd.to_numeric(df5['loneliness_response'], errors='coerce')
    df5['lonely'] = df5['loneliness_response'] - 1

# convert alcohol energy percentage to grams per day
if {'alcohol', 'energy_kcal'}.issubset(df1.columns):
    df1['alcohol'] = pd.to_numeric(df1['alcohol'], errors='coerce')
    df1['energy_kcal'] = pd.to_numeric(df1['energy_kcal'], errors='coerce')
    df1['alcohol'] = ((df1['alcohol'] / 100) * df1['energy_kcal'] / 7).round(2)

# merge the input tables
merged = pd.merge(df1, df2[['participant_id', 'smoking', 'stress']], on='participant_id', how='outer')
merged = pd.merge(merged, df3[['participant_id', 'vigorous_act', 'moderate_act', 'sleep']], on='participant_id', how='outer')
merged = pd.merge(merged, df4[['participant_id', 'diabetes', 'family_diabetes']], on='participant_id', how='outer')
merged = pd.merge(merged, df5[['participant_id', 'lonely']], on='participant_id', how='outer')
merged['origin'] = 'LGS_Wave_5'
merged['country'] = 'Belgium'
merged = merged.rename(columns={'participant_id': 'id'})
merged['mvpa'] = merged[['vigorous_act', 'moderate_act']].sum(axis=1, min_count=1)

# recode family diabetes
merged['family_diabetes'] = pd.to_numeric(merged['family_diabetes'], errors='coerce')
keep_codes = {1, 2, 3, 4, 12, 23, 123}
merged['family_diabetes'] = np.where(
    merged['family_diabetes'].isin(keep_codes),
    1,
    np.where(merged['family_diabetes'] == 0, 0, np.nan),
)

# select output columns
desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl', 'diastolic_bp', 'systolic_bp',
    'smoking', 'alcohol', 'sugar', 'protein', 'fat',
    'vigorous_act', 'moderate_act', 'mvpa', 'sleep', 'lonely', 'stress',
    'diabetes', 'family_diabetes',
    'vegetables', 'fruits', 'dairy', 'resting_heart_rate', 'sit_and_reach',
    'shuttle_run_speed', 'vo2_max',
]
final_columns = [col for col in desired_columns if col in merged.columns]
merged = merged.reindex(columns=final_columns)

# convert values and laboratory units
numeric_cols = [col for col in merged.columns if col not in ['origin', 'sex']]
for col in numeric_cols:
    try:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    except Exception:
        continue
numeric_cols = merged.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in ['cholesterol', 'hdl', 'ldl']:
    if col in merged.columns:
        merged[col] = merged[col] / 38.67
if 'triglycerides' in merged.columns:
    merged['triglycerides'] = merged['triglycerides'] / 88.57
if 'glucose' in merged.columns:
    merged['glucose'] = merged['glucose'] * 0.0555
for col in numeric_cols:
    merged[col] = merged[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    merged[col] = merged[col].apply(
        lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.') if pd.notnull(x) else ''
    )

# save and add the cohort prefix to ids
merged.to_csv(output_path, index=False)
final_df = pd.read_csv(output_path)
final_df['id'] = final_df['id'].apply(
    lambda x: f"BE-KUL-{str(x).split('BE-KUL-')[-1]}" if pd.notnull(x) else ''
)
final_df.to_csv(output_path, index=False)
print(f"Processed dataset saved to '{output_path}'")
