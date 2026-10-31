# public version of the ask preprocessing script
# source column names and the original input filename have been replaced
# this public version uses generic input column names
# # no original dataset files are included

import os
import pandas as pd
import numpy as np

input_file = "DATA/ask/input.csv"
output_file = "RESULTS/ASK_edited.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

ask_data = pd.read_csv(input_file)

# rename the generic input columns to the harmonized names
ask_data = ask_data.rename(columns={
    'participant_id': 'id',
    'participant_sex': 'sex',
    'participant_age': 'age',
    'measured_height': 'height',
    'measured_weight': 'weight',
    'body_mass_index': 'bmi',
    'waist_measurement': 'waist_circ',
    'blood_glucose': 'glucose',
    'total_cholesterol': 'cholesterol',
    'blood_triglycerides': 'triglycerides',
    'hdl_level': 'hdl',
    'ldl_level': 'ldl',
    'diastolic_pressure': 'diastolic_bp',
    'systolic_pressure': 'systolic_bp',
    'moderate_activity_minutes': 'moderate_act',
    'vigorous_activity_minutes': 'vigorous_act',
    'maternal_education_category': 'mother_educ',
    'paternal_education_category': 'father_educ',
    'andersen_test_distance': 'andersen_meters'
})

# add cohort information
ask_data['origin'] = 'ASK'
ask_data['country'] = 'Norway'
ask_data['id'] = 'ASK-' + ask_data['id'].astype(str)
ask_data['sex'] = ask_data['sex'].replace({'GIRL': 'Female', 'BOY': 'Male'})

def format_float(x):
    if pd.notnull(x):
        return ('{:.2f}'.format(x).rstrip('0').rstrip('.'))
    return ''

# format age and bmi
for col in ['age', 'bmi']:
    ask_data[col] = pd.to_numeric(ask_data[col], errors='coerce').round(2)
    ask_data[col] = ask_data[col].apply(format_float)

# convert activity minutes to hours
ask_data['moderate_act'] = pd.to_numeric(ask_data['moderate_act'], errors='coerce') / 60
ask_data['vigorous_act'] = pd.to_numeric(ask_data['vigorous_act'], errors='coerce') / 60
for col in ['moderate_act', 'vigorous_act']:
    ask_data[col] = ask_data[col].round(2)
    ask_data[col] = ask_data[col].apply(format_float)

# sum moderate and vigorous activity
def compute_mvpa(row):
    m = pd.to_numeric(row['moderate_act'], errors='coerce')
    v = pd.to_numeric(row['vigorous_act'], errors='coerce')
    if pd.isna(m) and pd.isna(v):
        return ''
    total = (m if not pd.isna(m) else 0) + (v if not pd.isna(v) else 0)
    return format_float(round(total, 2))

ask_data['mvpa'] = ask_data.apply(compute_mvpa, axis=1)

# format numeric measurements
numeric_cols = [
    'height', 'weight', 'glucose', 'cholesterol', 'triglycerides',
    'hdl', 'ldl', 'diastolic_bp', 'systolic_bp'
]
for col in numeric_cols:
    ask_data[col] = pd.to_numeric(ask_data[col], errors='coerce').round(2)
    ask_data[col] = ask_data[col].apply(format_float)

# recode parental education
education_mapping = {
    "Not university": 12,
    "University 4 and more y": 19,
    "University <4 y": 15
}
ask_data['mother_educ'] = ask_data['mother_educ'].map(education_mapping)
ask_data['father_educ'] = ask_data['father_educ'].map(education_mapping)

# calculate vo2peak from the andersen test
ask_data['andersen_meters'] = pd.to_numeric(ask_data['andersen_meters'], errors='coerce')
ask_data['sex_num'] = ask_data['sex'].map({'Male': 0, 'Female': 1})
ask_data['vo2peak_andersen'] = (
    18.38 + 0.03301 * ask_data['andersen_meters'] - 5.92 * ask_data['sex_num']
)
ask_data['vo2peak_andersen'] = ask_data['vo2peak_andersen'].round(2).apply(format_float)
ask_data.drop(columns=['sex_num'], inplace=True)

# calculate sex-specific andersen z-scores
ask_data['vo2peak_andersen_num'] = pd.to_numeric(ask_data['vo2peak_andersen'], errors='coerce')
ask_data['mean_andersen'] = ask_data.groupby('sex')['vo2peak_andersen_num'].transform('mean')
ask_data['std_andersen'] = ask_data.groupby('sex')['vo2peak_andersen_num'].transform(
    lambda x: x.std(ddof=0)
)
ask_data['z_score_andersen'] = (
    (ask_data['vo2peak_andersen_num'] - ask_data['mean_andersen']) / ask_data['std_andersen']
).round(2).apply(format_float)
ask_data.drop(columns=['vo2peak_andersen_num', 'mean_andersen', 'std_andersen'], inplace=True)

# calculate parental education and affluence z-scores
edu_cols = ['mother_educ', 'father_educ']
for col in edu_cols:
    mean = ask_data[col].mean()
    std = ask_data[col].std(ddof=0)
    z_col = f'z_score_{col}'
    ask_data[z_col] = ask_data[col].apply(
        lambda x: (x - mean) / std if pd.notnull(x) else np.nan
    )

ask_data['z_score_affluence'] = ask_data[
    [f'z_score_{col}' for col in edu_cols]
].mean(axis=1, skipna=True)
z_cols_all = [f'z_score_{col}' for col in edu_cols] + ['z_score_affluence']
for col in z_cols_all:
    ask_data[col] = ask_data[col].round(2).apply(format_float)

# select and order output columns
desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
    'diastolic_bp', 'systolic_bp', 'moderate_act', 'vigorous_act', 'mvpa',
    'mother_educ', 'father_educ', 'andersen_meters', 'vo2peak_andersen',
    'z_score_andersen', 'z_score_affluence', 'z_score_mother_educ',
    'z_score_father_educ'
]
final_columns = [col for col in desired_columns if col in ask_data.columns]
ask_data = ask_data[final_columns]

# save processed data
ask_data.to_csv(output_file, index=False)
print('ASK dataset processed and saved successfully!')
