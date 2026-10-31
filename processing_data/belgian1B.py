# public version of the belgian wave 1b preprocessing script
# source file and column names have been replaced with generic names
# original participant id mappings were removed for confidentiality
# the id mapping below is a fictional example only
# the original dataset is not included

import pandas as pd
import numpy as np
import os

# education codes converted to years
belgian_mapping = {
    0: 0, 1: 8, 2: 10, 3: 9, 4: 10, 5: 11,
    6: 12, 7: 13, 8: 16, 9: np.nan
}

# occupation codes converted to affluence ranks
occup_affluence = {
    1: 8, 2: 7, 3: 6, 5: 5, 6: 4,
    7: 3, 4: 2, 8: 1, 9: 0, 0: np.nan
}

# fictional example; the original id mappings are not public
id_mapping = {
    1001: 2001
}

# read the input file with generic column names
input_file = os.path.join("DATA", "belgian_1b", "input.xlsx")
df = pd.read_excel(input_file)
result = pd.DataFrame(index=df.index)
result["origin"] = "LGS_Wave_1B"
result["country"] = "Belgium"
result["id"] = df["participant_id"]
result["sex"] = "Female"
result["age"] = df["age"]
result["height"] = df["height"]
result["weight"] = df["weight"]

# calculate bmi and convert blood pressure units
result["bmi"] = result["weight"] / ((result["height"] / 100) ** 2)
result["diastolic_bp"] = df["diastolic_pressure"] * 10
result["systolic_bp"] = df["systolic_pressure"] * 10

# recode smoking status
result["smoking"] = df["smoking_code"].apply(lambda x: 0 if x == 4 else 1)
result["resting_heart_rate"] = df["resting_heart_rate"]
result["sit_and_reach"] = df["sit_and_reach"]

# convert yearly ten-minute sport units to average hours per week
result["sports_participation"] = pd.to_numeric(df["sports_total"], errors="coerce") / 6 / 52

# recode parental education and occupation
result["mother_educ"] = df["mother_education_code"].map(belgian_mapping)
result["father_educ"] = df["father_education_code"].map(belgian_mapping)
result["id"] = result["id"].apply(lambda x: f"BE-KUL-{id_mapping.get(x, x)}")
result["mother_occup"] = df["mother_occupation_code"].map(occup_affluence)
result["father_occup"] = df["father_occupation_code"].map(occup_affluence)

# convert numeric columns before calculating z-scores
to_numeric_cols = [
    "mother_educ", "father_educ", "mother_occup", "father_occup",
    "sports_participation", "age", "bmi", "height", "weight",
    "diastolic_bp", "systolic_bp", "resting_heart_rate", "sit_and_reach"
]
for col in to_numeric_cols:
    if col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

# calculate population-based z-scores
z_vars = ["mother_educ", "father_educ", "mother_occup", "father_occup"]
for var in z_vars:
    zcol = f"z_score_{var}"
    mu = result[var].mean(skipna=True)
    sigma = result[var].std(ddof=0, skipna=True)
    if pd.isna(sigma) or sigma == 0:
        result[zcol] = np.nan
    else:
        result[zcol] = (result[var] - mu) / sigma

# average the available parental z-scores
z_cols = [f"z_score_{var}" for var in z_vars]
result["z_score_affluence"] = result[z_cols].mean(axis=1, skipna=True)

# keep the harmonized output columns
final_columns = [
    "origin", "country", "id", "sex", "age", "bmi", "height", "weight",
    "diastolic_bp", "systolic_bp", "smoking", "resting_heart_rate",
    "sit_and_reach", "sports_participation", "mother_educ", "father_educ",
    "mother_occup", "father_occup", "z_score_mother_educ",
    "z_score_father_educ", "z_score_mother_occup", "z_score_father_occup",
    "z_score_affluence"
]
final_columns = [col for col in final_columns if col in result.columns]
result = result[final_columns]
result = result.round(2)

# save the processed dataset
output_folder = "RESULTS"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "phase1b_edited.csv")
result.to_csv(output_file, index=False, float_format="%.2f")
print(f"Processed dataset saved to '{output_file}'")
