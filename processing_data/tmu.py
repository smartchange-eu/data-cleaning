# tmu data preprocessing
# this public version uses generic input filenames and source column names
# input files must use the generic names defined in this script
# no original participant data are included

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import sys

# generate the diabetes code list before preprocessing
script_dir = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(script_dir / "tmu_diabetes_codes.py")], check=True)

# define file paths and suffixes
Path("RESULTS").mkdir(parents=True, exist_ok=True)

# load basic participant files
basic_files = {
    'W': 'DATA/tmu/basic_w.csv',
    'S': 'DATA/tmu/basic_s.csv',
    'T': 'DATA/tmu/basic_t.csv'
}

# load biometric files
bio_files = {
    'W': 'DATA/tmu/biometrics_w.csv',
    'S': 'DATA/tmu/biometrics_s.csv',
    'T': 'DATA/tmu/biometrics_t.csv'
}

# load bp data
bp_df = pd.read_csv('DATA/tmu/blood_pressure.csv', encoding='big5hkscs')

# load figure dataset
fig_df = pd.read_csv('DATA/tmu/body_measurements.csv', encoding='big5hkscs')

# load epd history files
epd_files = [
    'DATA/tmu/history_s.csv',
    'DATA/tmu/history_t.csv',
    'DATA/tmu/history_w.csv'
]

# load icd datasets
icd_files = [
    'DATA/tmu/diagnoses_s.csv',
    'DATA/tmu/diagnoses_t.csv'
]

# load ipd datasets
ipd_files = [
    'DATA/tmu/inpatient_s.csv',
    'DATA/tmu/inpatient_t.csv',
    'DATA/tmu/inpatient_w.csv'
]

# load opd datasets
opd_files = [
    'DATA/tmu/outpatient_s.csv',
    'DATA/tmu/outpatient_t.csv',
    'DATA/tmu/outpatient_w.csv'
]

# load diabetes code list
diab_codes_df = pd.read_csv('RESULTS/TMU_Diabetes_codes.csv')

# function for converting minguo date to gregorian
def convert_minguo_to_gregorian(minguo_date):
    try:
        date_str = str(int(minguo_date)).zfill(7)
        year = int(date_str[:3]) + 1911
        month = int(date_str[3:5])
        day = int(date_str[5:])
        return datetime(year, month, day)
    except:
        return np.nan

# list to collect processed basic dataframes
processed_dfs = []

for suffix, file_path in basic_files.items():
    df = pd.read_csv(file_path, encoding='big5hkscs')

    # origin and id (still uppercase in main_df)
    df['ORIGIN'] = 'TMU'
    df['COUNTRY'] = 'Taiwan'
    df['ID'] = 'TMU_' + suffix + '_' + df['participant_id'].astype(str)

    # sex determination

    # df['sex'] = np.where(df['sex_category_0'] == true, 'female', np.where(df['sex_category_1'] == true, 'male', np.nan))
    df['SEX'] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df['sex_category_0'] == True, 'SEX'] = 'Female'
    df.loc[df['sex_category_1'] == True, 'SEX'] = 'Male'

    # dob conversion
    df['DOB_HELPER'] = df['birth_date_source'].apply(convert_minguo_to_gregorian)

    processed_dfs.append(df)

# combine all basic records
main_df = pd.concat(processed_dfs, ignore_index=True)

# extract participant_id for matching later
main_df['participant_id'] = main_df['ID'].str.replace(r'^TMU_[WST]_', '', regex=True)

# keep only required columns (still uppercase in main_df)
main_df = main_df[['ORIGIN', 'COUNTRY', 'ID', 'SEX', 'DOB_HELPER', 'participant_id']]

print("Basic TMU data processed.")

# store processed bio records and stats
all_bio_records = []
stats = {}

# age calculation function with 2 decimal precision
def calculate_age(dob, update_date):
    try:
        return round((update_date - dob).days / 365.25, 2)
    except:
        return np.nan

for suffix, path in bio_files.items():
    bio_df = pd.read_csv(path, encoding='big5hkscs')
    total = len(bio_df)

    # merge on participant_id
    merged = bio_df.merge(main_df, on='participant_id', how='left', indicator=True)

    # filter matches
    matched = merged[merged['_merge'] == 'both'].copy()
    matched['measurement_date'] = matched['measurement_date'].apply(convert_minguo_to_gregorian)
    matched['AGE'] = matched.apply(lambda row: calculate_age(row['DOB_HELPER'], row['measurement_date']), axis=1)
    matched['RESTING HEART RATE'] = pd.to_numeric(matched['pulse_source'], errors='coerce')
    matched.loc[matched['RESTING HEART RATE'].round(2) == 0.0, 'RESTING HEART RATE'] = np.nan

    # drop rows without heart rate
    matched = matched.dropna(subset=['RESTING HEART RATE'])

    # final selection -> rename here to lower_snake_case for our working/final dataset
    final_bio = matched[['ORIGIN', 'COUNTRY', 'participant_id', 'ID', 'SEX', 'DOB_HELPER', 'AGE', 'RESTING HEART RATE']].rename(
    columns={
        'ORIGIN': 'origin',
        'COUNTRY': 'country',
        'participant_id': 'chr_no',
        'ID': 'id',
        'SEX': 'sex',
        'DOB_HELPER': 'dob_helper',
        'AGE': 'age',
        'RESTING HEART RATE': 'resting_heart_rate'
    }
)
    all_bio_records.append(final_bio)

    # stats
    used = len(final_bio)
    stats[suffix] = {'used': used, 'not_used': total - used}

# combine all bio-enriched rows
final_df = pd.concat(all_bio_records, ignore_index=True)

# drop duplicates based on id + age (same person, same age), keep first
final_df = final_df.drop_duplicates(subset=['id', 'age'], keep='first')

# save initial enriched dataset
final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("\nInitial TMU dataset (heart rate) saved to RESULTS/TMU_edited.csv")

# print statistics
for suffix, stat in stats.items():
    print(f"Biometrics source {suffix} -> Used: {stat['used']} | Not used: {stat['not_used']}")

# editing blood pressure
print("\nProcessing blood pressure data from blood_pressure.csv...")

# parse record_date
try:
    bp_df['record_date'] = pd.to_datetime(bp_df['record_date'], errors='raise')
except:
    bp_df['record_date'] = bp_df['record_date'].apply(convert_minguo_to_gregorian)

# filter only people present in final_df (final_df uses chr_no in lowercase)
valid_chr_nos = set(final_df['chr_no'].dropna())
bp_df = bp_df[bp_df['participant_id'].isin(valid_chr_nos)].copy()

# drop records where systolic_source is zero or null
bp_df['systolic_source'] = pd.to_numeric(bp_df['systolic_source'], errors='coerce')
bp_df = bp_df[bp_df['systolic_source'].round(2) != 0.0]

# extract year and keep one measurement per participant_id + year
bp_df['YEAR'] = bp_df['record_date'].dt.year
bp_df = bp_df.sort_values(by=['participant_id', 'YEAR', 'record_date'])
bp_df = bp_df.drop_duplicates(subset=['participant_id', 'YEAR'], keep='first').drop(columns='YEAR')

print(f"Filtered BP records to one per person per year. Remaining: {len(bp_df)}")

# merge to get person info from final_df (includes dob_helper)
bp_merged = bp_df.merge(
    final_df[['chr_no', 'id', 'origin', 'country', 'sex', 'dob_helper']],
    left_on='participant_id',
    right_on='chr_no',
    how='left'
)
bp_merged['age_bp'] = bp_merged.apply(lambda row: calculate_age(row['dob_helper'], row['record_date']), axis=1)

# tracking
used_count = 0
not_used_count = 0
matched_ids = set()
new_bp_rows = []

# ensure final_df has bp columns in snake_case
if 'systolic_bp' not in final_df.columns:
    final_df['systolic_bp'] = np.nan
if 'diastolic_bp' not in final_df.columns:
    final_df['diastolic_bp'] = np.nan

# process each row in bp dataset
for idx, row in bp_merged.iterrows():
    if pd.isna(row['id']) or pd.isna(row['age_bp']):
        continue  # skip invalid/missing person

    matched_ids.add(row['id'])
    person_rows = final_df[final_df['id'] == row['id']]
    if not person_rows.empty:
        person_rows = person_rows.copy()
        person_rows['age_diff'] = abs(person_rows['age'] - row['age_bp'])
        min_diff_idx = person_rows['age_diff'].idxmin()
        min_diff = person_rows.loc[min_diff_idx, 'age_diff']

        if min_diff <= 1.0:

            # update existing row
            final_df.loc[min_diff_idx, 'systolic_bp'] = row['systolic_source']
            final_df.loc[min_diff_idx, 'diastolic_bp'] = row['diastolic_source']
            used_count += 1
            continue

    # no close age found -> create new row
    new_row = {
        'origin': row['origin'],
        'chr_no': row['participant_id'],
        'id': row['id'],
        'sex': row['sex'],
        'country': row['country'],
        'dob_helper': row['dob_helper'],
        'age': row['age_bp'],
        'resting_heart_rate': np.nan,
        'systolic_bp': row['systolic_source'],
        'diastolic_bp': row['diastolic_source'],
        'height': np.nan,
        'weight': np.nan
    }
    new_bp_rows.append(new_row)
    not_used_count += 1

# append any new bp rows
if new_bp_rows:
    bp_add_df = pd.DataFrame(new_bp_rows)
    final_df = pd.concat([final_df, bp_add_df], ignore_index=True)

# sort final_df by id then age
final_df = final_df.sort_values(by=['id', 'age']).reset_index(drop=True)

# save dataset with bp
final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Updated TMU_edited.csv with blood pressure data.\n")

print(f"Total individuals with BP in both datasets: {len(matched_ids)}")
print(f"BP records used (matched with close age): {used_count}")
print(f"BP records added as new rows: {not_used_count}")

print("\nProcessing body figure data from body_measurements.csv...")

# parse record_date for figure
try:
    fig_df['record_date'] = pd.to_datetime(fig_df['record_date'], errors='raise')
except:
    fig_df['record_date'] = fig_df['record_date'].apply(convert_minguo_to_gregorian)

# filter to participant_id values present in final_df
fig_df = fig_df[fig_df['participant_id'].isin(final_df['chr_no'])].copy()

# convert and clean height_source and weight_source (source names)
fig_df['height_source'] = pd.to_numeric(fig_df['height_source'], errors='coerce')
fig_df['weight_source'] = pd.to_numeric(fig_df['weight_source'], errors='coerce')

# drop rows where height or weight are missing or zero
fig_df = fig_df.dropna(subset=['height_source', 'weight_source'])
fig_df = fig_df[(fig_df['height_source'].round(2) != 0.0) & (fig_df['weight_source'].round(2) != 0.0)]

# extract year
fig_df['YEAR'] = fig_df['record_date'].dt.year

# sort to keep earliest measurement per person per year
fig_df = fig_df.sort_values(by=['participant_id', 'YEAR', 'record_date'])

# drop duplicates per participant_id + year
fig_df = fig_df.drop_duplicates(subset=['participant_id', 'YEAR'], keep='first').drop(columns='YEAR')

# merge to get person info (joining to final_df)
fig_merged = fig_df.merge(
    final_df[['chr_no', 'id', 'origin', 'country', 'sex', 'dob_helper']],
    left_on='participant_id',
    right_on='chr_no',
    how='left'
)

# calculate age at figure measurement
fig_merged['age_fig'] = fig_merged.apply(lambda row: calculate_age(row['dob_helper'], row['record_date']), axis=1)

# ensure final_df has height/weight columns in snake_case
if 'height' not in final_df.columns:
    final_df['height'] = np.nan
if 'weight' not in final_df.columns:
    final_df['weight'] = np.nan

# tracking
used_count = 0
not_used_count = 0
matched_ids = set()
new_fig_rows = []

# prevent creating near-identical records per person-age
for idx, row in fig_merged.iterrows():
    if pd.isna(row['id']) or pd.isna(row['age_fig']):
        continue

    matched_ids.add(row['id'])

    person_rows = final_df[final_df['id'] == row['id']]
    if not person_rows.empty:
        person_rows = person_rows.copy()
        person_rows['age_diff'] = abs(person_rows['age'] - row['age_fig'])
        close_matches = person_rows[person_rows['age_diff'] <= 1.0]

        if not close_matches.empty:
            best_idx = close_matches['age_diff'].idxmin()
            final_df.loc[best_idx, 'height'] = row['height_source']
            final_df.loc[best_idx, 'weight'] = row['weight_source']
            used_count += 1
            continue

    # otherwise, create a new unique entry
    new_row = {
        'origin': row['origin'],
        'chr_no': row['participant_id'],
        'id': row['id'],
        'sex': row['sex'],
        'country': row['country'],
        'dob_helper': row['dob_helper'],
        'age': row['age_fig'],
        'resting_heart_rate': np.nan,
        'systolic_bp': np.nan,
        'diastolic_bp': np.nan,
        'height': row['height_source'],
        'weight': row['weight_source']
    }
    new_fig_rows.append(new_row)
    not_used_count += 1

# add unmatched rows
if new_fig_rows:
    fig_add_df = pd.DataFrame(new_fig_rows)
    final_df = pd.concat([final_df, fig_add_df], ignore_index=True)

# drop duplicate id + age combos again
final_df = final_df.drop_duplicates(subset=['id', 'age'], keep='first')

print("\nMerging closely timed partial records...")

# threshold in years (~4 months)
merge_threshold = 0.34

merged_rows = []

# sort final_df by id and age
final_df = final_df.sort_values(by=['id', 'age']).reset_index(drop=True)

# columns to check for merging
value_cols = ['resting_heart_rate', 'systolic_bp', 'diastolic_bp', 'height', 'weight']

# process per individual
for person_id, group in final_df.groupby('id'):
    group = group.sort_values('age').reset_index(drop=True)
    used = [False] * len(group)

    for i in range(len(group)):
        if used[i]:
            continue
        base = group.loc[i].copy()
        base_age = base['age']
        for j in range(i + 1, len(group)):
            if used[j]:
                continue
            candidate = group.loc[j]
            if abs(candidate['age'] - base_age) <= merge_threshold:
                for col in value_cols:
                    if pd.isna(base[col]) and not pd.isna(candidate[col]):
                        base[col] = candidate[col]
                used[j] = True
        merged_rows.append(base)
        used[i] = True

# final combined dataframe
final_df = pd.DataFrame(merged_rows)

# sort again before saving
final_df = final_df.sort_values(by=['id', 'age']).reset_index(drop=True)

print(f"Merged records: total rows now {len(final_df)}.")

# save updated dataset (with hr, bp, height/weight)
final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Updated TMU_edited.csv with heart rate, blood pressure, height and weight.\n")

print(f"Total individuals with figure data in both datasets: {len(matched_ids)}")
print(f"Figure records used (matched with close age): {used_count}")
print(f"Figure records added as new rows: {not_used_count}")

print("\nProcessing diabetes diagnosis...")

# get diabetes code list
known_diab_codes = set(diab_codes_df['code_icd9'].dropna().astype(str)) | \
                   set(diab_codes_df['code_icd10'].dropna().astype(str))

# initialize diabetes indicator column as empty (snake_case)
final_df['diabetes'] = np.nan

# process each icd dataset
for file in icd_files:
    print(f"Checking {file}...")
    icd_df = pd.read_csv(file, encoding='big5hkscs', dtype=str)

    # only consider people already in final_df
    icd_df = icd_df[icd_df['participant_id'].isin(final_df['chr_no'])]

    diag_cols = [col for col in ['primary_diagnosis', 'diagnosis_icd10_10', 'diagnosis_icd10_10_out'] if col in icd_df]
    icd_df = icd_df.dropna(subset=diag_cols, how='all')

    icd_df['HAS_DIABETES'] = icd_df[diag_cols].apply(
        lambda row: any(code.strip() in known_diab_codes for code in row if pd.notna(code)),
        axis=1
    )

    diab_chr_nos = icd_df.loc[icd_df['HAS_DIABETES'], 'participant_id'].unique()
    final_df.loc[final_df['chr_no'].isin(diab_chr_nos), 'diabetes'] = 1

print(f"Diabetes flag assigned to {final_df['diabetes'].eq(1).sum()} individuals.")

final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Saved TMU_edited.csv with initial DIABETES flag.")

print("\nProcessing EPD history datasets for diabetes codes...")

epd_diag_cols = ['diagnosis_icd9_1', 'diagnosis_icd9_2']

for file in epd_files:
    print(f"Checking {file}...")
    epd_df = pd.read_csv(file, encoding='big5hkscs', dtype=str)
    epd_df = epd_df[epd_df['participant_id'].isin(final_df['chr_no'])]
    epd_df = epd_df.dropna(subset=epd_diag_cols, how='all')

    epd_df['HAS_DIABETES'] = epd_df[epd_diag_cols].apply(
        lambda row: any(code.strip() in known_diab_codes for code in row if pd.notna(code)),
        axis=1
    )

    flagged = epd_df.loc[epd_df['HAS_DIABETES'], 'participant_id'].unique()
    final_df.loc[final_df['chr_no'].isin(flagged), 'diabetes'] = 1

print(f"After EPD history, total flagged diabetes individuals: {final_df['diabetes'].eq(1).sum()}")

final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Saved TMU_edited.csv with EPD diabetes flags.")

print("\nProcessing IPD basic datasets for diabetes codes...")

ipd_diag_cols = [
    'admission_diagnosis',
    'admission_secondary_1', 'admission_secondary_2', 'admission_secondary_3', 'admission_secondary_4',
    'primary_diagnosis',
    'secondary_diagnosis_1', 'secondary_diagnosis_2', 'secondary_diagnosis_3', 'secondary_diagnosis_4', 'secondary_diagnosis_5',
    'secondary_diagnosis_6', 'secondary_diagnosis_7',
    'diagnosis_icd10_1', 'diagnosis_icd10_2', 'diagnosis_icd10_3', 'diagnosis_icd10_4', 'diagnosis_icd10_5',
    'diagnosis_icd10_6', 'diagnosis_icd10_7',
    'diagnosis_icd10_1_out', 'diagnosis_icd10_2_out', 'diagnosis_icd10_3_out', 'diagnosis_icd10_4_out',
    'diagnosis_icd10_5_out', 'diagnosis_icd10_6_out', 'diagnosis_icd10_7_out', 'diagnosis_icd10_8_out'
]

for file in ipd_files:
    print(f"Checking {file}...")
    ipd_df = pd.read_csv(file, encoding='big5hkscs', dtype=str)
    ipd_df = ipd_df[ipd_df['participant_id'].isin(final_df['chr_no'])]
    ipd_df = ipd_df.dropna(subset=ipd_diag_cols, how='all')

    ipd_df['HAS_DIABETES'] = ipd_df[ipd_diag_cols].apply(
        lambda row: any(code.strip() in known_diab_codes for code in row if pd.notna(code)),
        axis=1
    )

    flagged = ipd_df.loc[ipd_df['HAS_DIABETES'], 'participant_id'].unique()
    final_df.loc[final_df['chr_no'].isin(flagged), 'diabetes'] = 1

print(f"After IPD files, total flagged diabetes individuals: {final_df['diabetes'].eq(1).sum()}")

final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Saved TMU_edited.csv with IPD diabetes flags applied.")

print("\nProcessing OPD basic datasets for diabetes codes...")

opd_diag_cols = [
    'diagnosis_icd9_1', 'diagnosis_icd9_2', 'diagnosis_icd9_3', 'diagnosis_icd9_4', 'diagnosis_icd9_5',
    'diagnosis_icd10_1', 'diagnosis_icd10_2', 'diagnosis_icd10_3', 'diagnosis_icd10_4', 'diagnosis_icd10_5'
]

for file in opd_files:
    print(f"Checking {file}...")
    opd_df = pd.read_csv(file, encoding='big5hkscs', dtype=str)
    opd_df = opd_df[opd_df['participant_id'].isin(final_df['chr_no'])]
    opd_df = opd_df.dropna(subset=opd_diag_cols, how='all')

    opd_df['HAS_DIABETES'] = opd_df[opd_diag_cols].apply(
        lambda row: any(code.strip() in known_diab_codes for code in row if pd.notna(code)),
        axis=1
    )

    flagged = opd_df.loc[opd_df['HAS_DIABETES'], 'participant_id'].unique()
    final_df.loc[final_df['chr_no'].isin(flagged), 'diabetes'] = 1

print(f"After OPD files, total flagged diabetes individuals: {final_df['diabetes'].eq(1).sum()}")

# calculate bmi from height (cm) and weight (kg)
final_df['height'] = pd.to_numeric(final_df['height'], errors='coerce')
final_df['weight'] = pd.to_numeric(final_df['weight'], errors='coerce')

def calc_bmi(row):
    h = row['height']
    w = row['weight']
    if pd.notnull(h) and pd.notnull(w) and h > 0:
        m = h / 100.0
        return round(w / (m * m), 2)
    return np.nan

final_df['bmi'] = final_df.apply(calc_bmi, axis=1)

# save the fully updated dataset
final_df.to_csv('RESULTS/TMU_edited.csv', index=False)
print("Saved TMU_edited.csv with final lowercase/underscore variables and BMI.")
