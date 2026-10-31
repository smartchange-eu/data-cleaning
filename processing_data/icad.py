# icad data preprocessing and harmonization
# original input filenames and source column names have been replaced with generic names
# the input file must use the generic column names defined in this script
# daily vigorous activity columns must start with vigorous_minutes_day_
# daily moderate activity columns must start with moderate_minutes_day_
# no original participant data are included

import os
import pandas as pd
import numpy as np

# paths for the public version
INPUT_PATH = 'DATA/icad/input.csv'
OUTPUT_PATH = 'RESULTS/ICAD_edited.csv'

# read the comma-delimited input file
df = pd.read_csv(INPUT_PATH, sep=',', engine='python', on_bad_lines='warn')

# clean column names and remove any byte order mark
df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

# add cohort origin and check the participant identifier
if 'participant_id' not in df.columns:
    raise KeyError('participant_id column not found. Columns: ' + ', '.join(df.columns))
df['origin'] = 'ICAD'

# map country codes to country names
country_map = {
    1: 'United Kingdom',
    2: 'Switzerland',
    3: 'Belgium',
    4: 'United States',
    5: 'Australia',
    6: 'Denmark',
    7: 'Estonia',
    8: 'Norway',
    9: 'Portugal',
    10: 'Brazil'
}

df['country'] = pd.to_numeric(df['country_code'], errors='coerce').map(country_map)
df.rename(columns={'participant_id': 'id'}, inplace=True)

# map sex codes to harmonized categories
sex_source = df.get('sex_code', df.get('sex_code_alt', pd.Series(index=df.index, dtype='float64')))
df['sex'] = pd.to_numeric(sex_source, errors='coerce').map({0: 'Male', 1: 'Female'})

# convert age to numeric and round to two decimals
if 'age_value' not in df.columns:
    raise KeyError('age_value column not found')
df['age'] = pd.to_numeric(df['age_value'], errors='coerce').round(2)

# convert height, weight, and waist circumference to numeric
for src, dst in [
    ('height_value', 'height'),
    ('weight_value', 'weight'),
    ('waist_value', 'waist_circ')
]:
    df[dst] = pd.to_numeric(df.get(src, pd.Series(index=df.index, dtype='float64')), errors='coerce').round(2)

# calculate body mass index in kg/m2
height_m = df['height'] / 100
df['bmi'] = (df['weight'] / (height_m ** 2)).round(2)

# rename laboratory measurements and blood pressure variables
labs = {
    'glucose_value': 'glucose',
    'triglycerides_value': 'triglycerides',
    'hdl_value': 'hdl',
    'ldl_value': 'ldl',
    'diastolic_value': 'diastolic_bp',
    'systolic_value': 'systolic_bp'
}
for old, new in labs.items():
    if old in df.columns:
        df.rename(columns={old: new}, inplace=True)

# calculate average daily vigorous and moderate activity in hours
vig = [c for c in df.columns if c.startswith('vigorous_minutes_day_')]
mod = [c for c in df.columns if c.startswith('moderate_minutes_day_')]
df[vig] = df[vig].apply(pd.to_numeric, errors='coerce')
df[mod] = df[mod].apply(pd.to_numeric, errors='coerce')
df['vigorous_act'] = (df[vig].sum(axis=1) / 7 / 60).round(2)
df['moderate_act'] = (df[mod].sum(axis=1) / 7 / 60).round(2)

# calculate total moderate-to-vigorous physical activity
df['mvpa'] = (df['vigorous_act'] + df['moderate_act']).round(2)

# combine parental education responses using the higher value
map1 = {999: np.nan, 0: 10, 1: 12}
map2 = {999: np.nan, 0: 10, 1: 14, 2: 20}
df['moth_ed1'] = df.get('mother_education_1', pd.Series(index=df.index, dtype='float64')).map(map1)
df['moth_ed2'] = df.get('mother_education_2', pd.Series(index=df.index, dtype='float64')).map(map2)
df['mother_educ'] = df[['moth_ed1', 'moth_ed2']].max(axis=1)
df['fath_ed1'] = df.get('father_education_1', pd.Series(index=df.index, dtype='float64')).map(map1)
df['fath_ed2'] = df.get('father_education_2', pd.Series(index=df.index, dtype='float64')).map(map2)
df['father_educ'] = df[['fath_ed1', 'fath_ed2']].max(axis=1)

# map family car responses and take the highest score
c1 = df.get('family_cars_1', pd.Series(index=df.index, dtype='float64')).map({0: 1, 1: 2, 999: np.nan})
c2 = df.get('family_cars_2', pd.Series(index=df.index, dtype='float64')).map({0: 1, 1: 2, 2: 3, 3: 3, 999: np.nan})
c3 = df.get('family_cars_3', pd.Series(index=df.index, dtype='float64')).map({0: 2, 1: 3, 999: np.nan})
df['family_cars'] = pd.concat([c1, c2, c3], axis=1).max(axis=1)

# select and order the harmonized columns
keep = [
    'origin', 'country', 'id', 'sex', 'age', 'bmi', 'height', 'weight', 'waist_circ',
    'glucose', 'triglycerides', 'hdl', 'ldl', 'diastolic_bp', 'systolic_bp',
    'vigorous_act', 'moderate_act', 'mvpa',
    'mother_educ', 'father_educ', 'family_cars'
]
cols = [c for c in keep if c in df.columns]

# round floating-point values to two decimals
for col in df.select_dtypes(include='float').columns:
    df[col] = df[col].round(2)

# remove participants with four or fewer non-missing selected fields
initial_count = len(df)
completed_counts = df[cols].notnull().sum(axis=1)
df = df[completed_counts > 4]
removed = initial_count - len(df)
print(f'{removed} participants removed due to <=4 completed fields.')

# calculate means and standard deviations after participant filtering
means = {
    'mother_educ': df['mother_educ'].mean(),
    'father_educ': df['father_educ'].mean(),
    'family_cars': df['family_cars'].mean()
}
stds = {
    'mother_educ': df['mother_educ'].std(),
    'father_educ': df['father_educ'].std(),
    'family_cars': df['family_cars'].std()
}

# calculate the individual affluence z-scores
zdf = pd.DataFrame({
    'z_score_mother_educ': ((df['mother_educ'] - means['mother_educ']) / stds['mother_educ']).round(2),
    'z_score_father_educ': ((df['father_educ'] - means['father_educ']) / stds['father_educ']).round(2),
    'z_score_family_cars': ((df['family_cars'] - means['family_cars']) / stds['family_cars']).round(2)
})

# calculate the combined affluence score
zdf['z_score_affluence'] = zdf.mean(axis=1).round(2)

# add the z-scores to the processed dataset
df = pd.concat([df, zdf], axis=1).copy()
keep += [
    'z_score_mother_educ',
    'z_score_father_educ',
    'z_score_family_cars',
    'z_score_affluence'
]
cols = [c for c in keep if c in df.columns]

# save the harmonized dataset
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame(df[cols]).to_csv(OUTPUT_PATH, index=False)
print(f'Preprocessed data saved to {OUTPUT_PATH}')
