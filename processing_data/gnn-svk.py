# public version of the gnn preprocessing script
# original input filenames and source column names have been replaced
# input files must use the generic column names expected below
# participant-level data are not included

import os
import pandas as pd
import numpy as np

input_folder = 'DATA/gnn'
output_file = 'RESULTS/gnn_edited.csv'
os.makedirs('RESULTS', exist_ok=True)
svk_df = pd.read_excel(os.path.join(input_folder, 'physical_measurements.xlsx'))
gnn_df = pd.read_excel(os.path.join(input_folder, 'questionnaire.xlsx'))

# keep records with height and weight
initial_len = len(svk_df)
svk_df = svk_df.dropna(subset=['height_cm', 'weight_kg']).copy()
print(f'Dropped {initial_len - len(svk_df)} rows due to missing height or weight.')

# select matching participants and questionnaire responses
common_ids = set(svk_df['participant_id']).intersection(gnn_df['participant_id'])
gnn_df = gnn_df[gnn_df['participant_id'].isin(common_ids) & (gnn_df['recent_illness_or_injury'] != 2)].copy()
gnn_df['questionnaire_date'] = pd.to_datetime(gnn_df['questionnaire_date'], errors='coerce', dayfirst=True)
gnn_df['year'] = gnn_df['questionnaire_date'].dt.year
svk_df['measurement_date'] = pd.to_datetime(svk_df['measurement_date'], errors='coerce', dayfirst=True)
svk_df['year'] = svk_df['measurement_date'].dt.year

# use the most complete questionnaire per participant and year
gnn_df['completeness'] = gnn_df.notna().sum(axis=1)
gnn_df = gnn_df.sort_values(['participant_id', 'year', 'completeness'], ascending=[True, True, False])
gnn_df = gnn_df.drop_duplicates(['participant_id', 'year'], keep='first')
svk_df['vigorous_act'] = np.nan
svk_df['moderate_act'] = np.nan
svk_df['sleep'] = np.nan
svk_df['origin'] = 'GNN'
svk_df['used'] = False

hour_map = {1: 0, 2: 60, 3: 120, 4: 180}
minute_map = {1: 0, 2: 15, 3: 30, 4: 45}
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
vig_cols = [(f'vigorous_hours_{day}', f'vigorous_minutes_{day}') for day in days]
mod_cols = [(f'moderate_hours_{day}', f'moderate_minutes_{day}') for day in days]
weekday_wake_map = {i + 1: 240 + 15 * i for i in range(29)}
weekday_sleep_map = {i + 1: 1200 + 15 * i for i in range(25)}
weekend_wake_map = {i + 1: 240 + 15 * i for i in range(41)}
weekend_sleep_map = {i + 1: 1200 + 15 * i for i in range(37)}

# calculate activity and sleep, then match to the nearest measurement
for _, gnn_row in gnn_df.iterrows():
    uid = gnn_row['participant_id']
    created = gnn_row['questionnaire_date']
    gnn_year = gnn_row['year']
    vig_total = sum(hour_map.get(gnn_row.get(h), 0) + minute_map.get(gnn_row.get(m), 0) for h, m in vig_cols)
    mod_total = sum(hour_map.get(gnn_row.get(h), 0) + minute_map.get(gnn_row.get(m), 0) for h, m in mod_cols)
    vig_avg_hr = round(vig_total / 7 / 60, 2)
    mod_avg_hr = round(mod_total / 7 / 60, 2)
    wk_wake = weekday_wake_map.get(gnn_row.get('weekday_wake_time'), np.nan)
    wk_sleep = weekday_sleep_map.get(gnn_row.get('weekday_bedtime'), np.nan)
    we_wake = weekend_wake_map.get(gnn_row.get('weekend_wake_time'), np.nan)
    we_sleep = weekend_sleep_map.get(gnn_row.get('weekend_bedtime'), np.nan)
    if np.isnan([wk_wake, wk_sleep, we_wake, we_sleep]).any():
        continue
    wk_duration = ((1440 - wk_sleep) + wk_wake) % 1440
    we_duration = ((1440 - we_sleep) + we_wake) % 1440
    sleep_hr = round((wk_duration * 5 + we_duration * 2) / 7 / 60, 2)
    svk_user_rows = svk_df[(svk_df['participant_id'] == uid) & (~svk_df['used'])]
    if svk_user_rows.empty:
        continue
    same_year_rows = svk_user_rows[svk_user_rows['year'] == gnn_year]
    if not same_year_rows.empty:
        best_idx = (same_year_rows['measurement_date'] - created).abs().idxmin()
    else:
        best_idx = (svk_user_rows['measurement_date'] - created).abs().idxmin()
    svk_df.loc[best_idx, 'vigorous_act'] = vig_avg_hr
    svk_df.loc[best_idx, 'moderate_act'] = mod_avg_hr
    svk_df.loc[best_idx, 'sleep'] = sleep_hr
    svk_df.loc[best_idx, 'origin'] = 'GNN'
    svk_df.loc[best_idx, 'used'] = True

# build the harmonized output
final = pd.DataFrame(index=svk_df.index)
final['origin'] = svk_df['origin']
final['country'] = 'Slovenia'
final['id'] = 'GNN-' + svk_df['participant_id'].astype(str)
final['sex'] = svk_df['sex_code'].map({1: 'Male', 2: 'Female'})
final['age'] = svk_df['age_years'].round(2)
final['bmi'] = svk_df['bmi_value'].round(2)
final['height'] = svk_df['height_cm']
final['weight'] = svk_df['weight_kg']
final['vigorous_act'] = svk_df['vigorous_act']
final['moderate_act'] = svk_df['moderate_act']
final['mvpa'] = final[['vigorous_act', 'moderate_act']].apply(
    lambda row: round(row.sum(), 2) if not (pd.isna(row['vigorous_act']) and pd.isna(row['moderate_act'])) else np.nan,
    axis=1
)
final['sleep'] = svk_df['sleep']

# estimate shuttle speed and vo2max from the 600 m test
def estimate_vo2max(age, sex, height, weight, t600):
    bmi = weight / (height / 100) ** 2
    age2, age3 = age ** 2, age ** 3
    if sex == 1:
        srt_speed = 12.47157 - 0.01532 * t600 - 0.65344 * age + 0.07034 * age2 - 0.00192 * age3 + 0.02428 * height - 0.02831 * weight + 0.00363 * bmi
    elif sex == 2:
        srt_speed = 12.13319 - 0.01203 * t600 - 0.23114 * age + 0.03633 * age2 - 0.00118 * age3 + 0.00813 * height - 0.01158 * weight - 0.01690 * bmi
    else:
        return np.nan, np.nan
    level = max(1, int((srt_speed - 8.0) / 0.5) + 1)
    estimated_laps = level * 9
    vo2max = 41.77 + 0.49 * estimated_laps - 0.0029 * (estimated_laps ** 2) - 0.62 * bmi
    return round(srt_speed, 2), round(vo2max, 2)

shuttle_speeds = []
vo2maxes = []
for idx, row in final.iterrows():
    try:
        t600 = svk_df.loc[idx, 'run_600m_time']
        if pd.notna(t600):
            srt, vo2 = estimate_vo2max(row['age'], 1 if row['sex'] == 'Male' else 2, row['height'], row['weight'], t600)
            shuttle_speeds.append(srt)
            vo2maxes.append(vo2)
        else:
            shuttle_speeds.append(np.nan)
            vo2maxes.append(np.nan)
    except Exception:
        shuttle_speeds.append(np.nan)
        vo2maxes.append(np.nan)
final['t600'] = svk_df['run_600m_time']
final['shuttle_run_speed'] = shuttle_speeds
final['vo2_max'] = vo2maxes
final.to_csv(output_file, index=False, sep=',')
print(f'Final dataset saved to {output_file}')
