# public version of the helena preprocessing script
# source variable names and the input filename have been replaced with generic names
# the original dataset is not included

import os
import pandas as pd
import numpy as np

input_file = 'DATA/helena/input.csv'
output_file = 'RESULTS/Helena_edited.csv'

# load the generic input dataset
data = pd.read_csv(input_file)

# identify the cohort and map study centers to countries
data['origin'] = 'HELENA'
country_map = {
    'Athens in Greece': 'Greece', 'Dortmund in Germany': 'Germany',
    'Gent in Belgium': 'Belgium', 'Heraklion in Crete': 'Greece',
    'Lille in France': 'France', 'Pecs in Hungary': 'Hungary',
    'Roma in Italy': 'Italy', 'Stockholm in Sweden': 'Sweden',
    'Vienna in Austria': 'Austria', 'Zaragoza in Spain': 'Spain'
}
data['country'] = data['study_center'].map(country_map)

# rename generic input columns to harmonized output names
rename_columns = {
    'participant_id': 'id', 'participant_sex': 'sex', 'participant_age': 'age',
    'measured_height': 'height', 'measured_weight': 'weight', 'measured_waist': 'waist_circ',
    'blood_glucose': 'glucose', 'total_cholesterol': 'cholesterol',
    'blood_triglycerides': 'triglycerides', 'hdl_cholesterol': 'hdl',
    'ldl_cholesterol': 'ldl', 'diastolic_pressure': 'diastolic_bp',
    'systolic_pressure': 'systolic_bp', 'ever_smoked': 'gq_smoke_ever',
    'current_smoking_response': 'gq_smoke_present', 'alcohol_intake': 'alcohol',
    'vigorous_minutes': 'vigorous_act', 'moderate_minutes': 'moderate_act',
    'resting_pulse': 'resting_heart_rate', 'flexibility_test': 'sit_and_reach',
    'shuttle_test_speed': 'shuttle_run_speed', 'estimated_vo2': 'vo2_max'
}

data.rename(columns=rename_columns, inplace=True)
# standardize sex labels and add the cohort id prefix
data['sex'] = data['sex'].str.capitalize()
data['id'] = 'HELENA-' + data['id'].astype(str)

# convert activity from minutes to hours per day
data['vigorous_act'] = data['vigorous_act'] / 60
data['moderate_act'] = data['moderate_act'] / 60

# calculate combined moderate-to-vigorous activity
data['mvpa'] = data['vigorous_act'] + data['moderate_act']

# convert biomarkers to mmol/l
for col in ['cholesterol', 'hdl', 'ldl']:
    data[col] = data[col].apply(lambda x: round(x / 38.67, 2) if pd.notnull(x) else '')
data['triglycerides'] = data['triglycerides'].apply(lambda x: round(x / 88.57, 2) if pd.notnull(x) else '')
data['glucose'] = data['glucose'].apply(lambda x: round(x * 0.0555, 2) if pd.notnull(x) else '')


# calculate bmi from weight and height
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

# smoking: 0 never, 1 former, 2 current; missing stays missing
def compute_smoking(row):
    if pd.isna(row['gq_smoke_ever']):
        return np.nan
    if str(row['gq_smoke_ever']).strip().lower() == 'no':
        return 0
    if str(row['gq_smoke_ever']).strip().lower() == 'yes':
        if str(row['gq_smoke_present']).strip().lower() == "don't smoke":
            return 1
        return 2
    return np.nan

if {'gq_smoke_ever', 'gq_smoke_present'}.issubset(data.columns):
    data['smoking'] = data.apply(compute_smoking, axis=1)
else:
    print('warning: smoking-related columns not found')

# group stress responses into three categories
stress_map = {'Completely disagree': 1, 'Disagree': 1, 'Agree/disagree': 2, 'Agree': 3, 'Strongly agree': 3}
data['stress'] = data['stress_response'].map(stress_map)
education_map = {
    'Lower education': 6, 'Lower secondary education': 9,
    'Higher secondary education': 12, 'Higher education or university degree': 16
}

# convert parental education categories into years
data['mother_educ'] = data['mother_education_response'].map(education_map)
data['father_educ'] = data['father_education_response'].map(education_map)

# score parental employment and occupation responses
status_income = {
    'passed away': 0, 'never see him': 0, 'does not work': 0,
    'looking for work': 1, 'trainee / student': 1, 'retired or ill': 1,
    'temporarily on employment leave': 1, 'househusband': 1,
    'working part-time': 2, 'working full-time': 4,
    'do not know': np.nan, 'NA': np.nan
}

isco_income = {
    'legislators, senior officials or managers': 4, 'professionals': 4,
    'technicians and associate professionals': 3, 'clerks': 3,
    'service workers and shop and market sales workers': 2,
    'craft and related workers': 2, 'plant and machine operators and assemblers': 2,
    'armed forces': 2, 'skilled agricultural and fishery worker': 1,
    'Unskilled occupations': 1, 'other namely (describe as detailed as possible)': 1,
    'does not work': 0, 'NA': np.nan
}

data['mother_status_score'] = data['mother_employment_response'].map(status_income)
data['mother_isco_score'] = data['mother_occupation_response'].map(isco_income)
data['father_status_score'] = data['father_employment_response'].map(status_income)
data['father_isco_score'] = data['father_occupation_response'].map(isco_income)

# combine each parent's employment and occupation scores
data['mother_occup'] = data[['mother_status_score', 'mother_isco_score']].sum(axis=1, min_count=1)
data['father_occup'] = data[['father_status_score', 'father_isco_score']].sum(axis=1, min_count=1)

# remove intermediate scoring columns
data.drop(columns=['mother_status_score', 'mother_isco_score', 'father_status_score', 'father_isco_score'], inplace=True)

# calculate z-scores using each variable's mean and sample standard deviation
for var in ['mother_educ', 'father_educ', 'mother_occup', 'father_occup']:
    mu = data[var].mean(skipna=True)
    sigma = data[var].std(skipna=True)
    data[f'z_score_{var}'] = data[var].apply(lambda x: (x - mu) / sigma if pd.notnull(x) else np.nan)

# average available z-scores into a composite affluence score
data['z_score_affluence'] = data[['z_score_mother_educ', 'z_score_father_educ', 'z_score_mother_occup', 'z_score_father_occup']].mean(axis=1, skipna=True)

# round numeric values to two decimals and remove trailing zeros
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].apply(lambda col: col.map(lambda x: round(x, 2) if not pd.isnull(x) else ''))
data = data.apply(lambda col: col.map(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.') if isinstance(x, (float, np.float64)) else x))

# select harmonized variables in the required order
ordered_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl', 'diastolic_bp', 'systolic_bp',
    'smoking', 'alcohol', 'vigorous_act', 'moderate_act', 'mvpa', 'resting_heart_rate',
    'sit_and_reach', 'shuttle_run_speed', 'vo2_max', 'stress',
    'mother_educ', 'father_educ', 'mother_occup', 'father_occup',
    'z_score_mother_educ', 'z_score_father_educ', 'z_score_mother_occup',
    'z_score_father_occup', 'z_score_affluence'
]
data = data[[col for col in ordered_columns if col in data.columns]]

# create the output folder and save the harmonized csv
os.makedirs('RESULTS', exist_ok=True)
data.to_csv(output_file, index=False)
print(f'dataset processed and saved to {output_file}')