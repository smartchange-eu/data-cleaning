# afine-te (uporto) data preprocessing and harmonization
# this script processes cohort data into the standardized smartchange format
# original input filenames and source column names have been replaced with generic names
# the input file must use the generic column names defined in this script
# no original participant data are included

import os
import pandas as pd
import numpy as np

# define file paths
input_folder = "DATA/uporto"
output_folder = "RESULTS"
input_file = "input.csv"
output_file = "UPorto_edited.csv"

os.makedirs(output_folder, exist_ok=True)
# load the datasetuporto_data = pd.read_csv(os.path.join(input_folder, input_file))# replace the dataset's missing-value codeuporto_data = uporto_data.replace(99999, np.nan)

# rename generic source columns to the harmonized output names
uporto_data = uporto_data.rename(columns={
    'participant_id': 'id',
    'participant_sex': 'sex',
    'participant_age': 'age',
    'body_mass_index': 'bmi',
    'body_height': 'height',
    'body_weight': 'weight',
    'waist_measurement': 'waist_circ',
    'blood_glucose': 'glucose',
    'total_cholesterol': 'cholesterol',
    'blood_triglycerides': 'triglycerides',
    'hdl_cholesterol': 'hdl',
    'ldl_cholesterol': 'ldl',
    'diastolic_pressure_first': 'diastolic_bp',
    'systolic_pressure_first': 'systolic_bp',
    'vigorous_minutes_daily_mean': 'vigorous_act',
    'moderate_minutes_daily_mean': 'moderate_act',
    'shuttle_run_lap_count': 'shuttle_run'
})

# map sex values
uporto_data['sex'] = uporto_data['sex'].map({0: 'Female', 1: 'Male'})

# convert daily activity minutes into hours
uporto_data['vigorous_act'] = uporto_data['vigorous_act'].apply(
    lambda x: round(x / 60, 2) if pd.notnull(x) else ''
)
uporto_data['moderate_act'] = uporto_data['moderate_act'].apply(
    lambda x: round(x / 60, 2) if pd.notnull(x) else ''
)

# calculate total moderate-to-vigorous activity
uporto_data['mvpa'] = (
    pd.to_numeric(uporto_data['vigorous_act'], errors='coerce')
    + pd.to_numeric(uporto_data['moderate_act'], errors='coerce')
).round(2)

# keep the mean step count as an integer
uporto_data['steps'] = uporto_data['daily_step_count_mean'].apply(
    lambda x: int(x) if pd.notnull(x) else ''
)

# assign caregiver schooling to both parental education fields
uporto_data['mother_educ'] = uporto_data['caregiver_schooling']
uporto_data['father_educ'] = uporto_data['caregiver_schooling']

# recode household motor vehicle ownership
uporto_data['family_cars'] = uporto_data['household_motor_vehicles']

def recode_family_cars(x):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return 1
    elif x == 1:
        return 2
    else:
        return 3

uporto_data['family_cars'] = uporto_data['family_cars'].apply(recode_family_cars)

# calculate education and vehicle ownership z-scores
affluence_vars = ['mother_educ', 'father_educ', 'family_cars']

for var in affluence_vars:
    mean_value = uporto_data[var].mean(skipna=True)
    std_value = uporto_data[var].std(skipna=True)
    zname = f'z_score_{var}'
    uporto_data[zname] = uporto_data[var].apply(
        lambda x: (x - mean_value) / std_value if pd.notnull(x) else np.nan
    )

# calculate overall affluence score
z_cols = [f'z_score_{v}' for v in affluence_vars]
uporto_data['z_score_affluence'] = uporto_data[z_cols].mean(axis=1, skipna=True)

# convert blood markers from mg/dl to mmol/l
uporto_data['cholesterol'] = uporto_data['cholesterol'].apply(
    lambda x: round(x / 38.67, 2) if pd.notnull(x) else ''
)
uporto_data['hdl'] = uporto_data['hdl'].apply(
    lambda x: round(x / 38.67, 2) if pd.notnull(x) else ''
)
uporto_data['ldl'] = uporto_data['ldl'].apply(
    lambda x: round(x / 38.67, 2) if pd.notnull(x) else ''
)
uporto_data['triglycerides'] = uporto_data['triglycerides'].apply(
    lambda x: round(x / 88.57, 2) if pd.notnull(x) and x >= 0 else ''
)
uporto_data['glucose'] = uporto_data['glucose'].apply(
    lambda x: round(x * 0.0555, 2) if pd.notnull(x) else ''
)

# preserve the shuttle-run lap count
uporto_data['shuttle_run_laps'] = uporto_data['shuttle_run']

# convert shuttle-run laps to test level and speed
shuttle_run_levels = [7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16]

def calculate_shuttle_level(laps):
    total_laps = 0
    for level, laps_needed in enumerate(shuttle_run_levels, start=1):
        total_laps += laps_needed
        if laps < total_laps:
            return level
    return len(shuttle_run_levels)

uporto_data['shuttle_run_speed'] = uporto_data['shuttle_run'].apply(
    lambda x: round(8.5 + 0.5 * (calculate_shuttle_level(x) - 1), 2) if pd.notnull(x) else ''
)

# add cohort information and participant id prefix
uporto_data['origin'] = 'AFINE-TE'
uporto_data['country'] = 'Portugal'
uporto_data['id'] = 'AFINE-TE-' + uporto_data['id'].astype(str)

# arrange the harmonized output columns
desired_columns = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
    'diastolic_bp', 'systolic_bp',
    'vigorous_act', 'moderate_act', 'mvpa',
    'shuttle_run_laps', 'shuttle_run_speed',
    'steps', 'mother_educ', 'father_educ', 'family_cars',
    'z_score_mother_educ', 'z_score_father_educ',
    'z_score_family_cars', 'z_score_affluence'
]

final_columns = [col for col in desired_columns if col in uporto_data.columns]
uporto_data = uporto_data.reindex(columns=final_columns)

# format numeric values for export
numeric_cols = [c for c in uporto_data.columns if c not in ['origin', 'country', 'sex', 'id']]

for col in numeric_cols:
    uporto_data[col] = pd.to_numeric(uporto_data[col], errors='coerce')

for col in numeric_cols:
    if col == 'steps':
        uporto_data[col] = uporto_data[col].apply(lambda x: int(x) if pd.notnull(x) else '')
        uporto_data[col] = uporto_data[col].astype(str)
    else:
        uporto_data[col] = uporto_data[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
        uporto_data[col] = uporto_data[col].apply(
            lambda x: ('{:.2f}'.format(x).rstrip('0').rstrip('.')) if pd.notnull(x) else ''
        )

# save the harmonized dataset
uporto_data.to_csv(os.path.join(output_folder, output_file), index=False)
print('AFINE-TE dataset processed and saved successfully!')