# public version of the belgian 1a preprocessing script
# original input file names and source variable names have been replaced
# with generic names. the original datasets are not included.
# input files must use the generic column names defined below.

import os
import pandas as pd
import numpy as np

input_folder = "DATA/belgian_1a"
output_folder = "RESULTS"
output_file = "Phase1A_edited.csv"
os.makedirs(output_folder, exist_ok=True)


def load_and_normalize(fname):
    df = pd.read_csv(os.path.join(input_folder, fname))
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


anthro = load_and_normalize("anthropometry.csv")
motor = load_and_normalize("motor.csv")
age_data = load_and_normalize("age.csv")
sport = load_and_normalize("sport.csv")

# reshape measurements into long format
anthro_long = pd.wide_to_long(
    anthro, stubnames=["bmi", "height", "weight"],
    i="participant_id", j="measurement", sep="_", suffix=r"\d+"
).reset_index()

age_long = pd.wide_to_long(
    age_data, stubnames=["age_value"],
    i="participant_id", j="measurement", sep="_", suffix=r"\d+"
).reset_index().rename(columns={"age_value": "age"})

motor_long = pd.wide_to_long(
    motor, stubnames=["flexibility", "pulse"],
    i="participant_id", j="measurement", sep="_", suffix=r"\d+"
).reset_index().rename(columns={
    "flexibility": "sit_and_reach",
    "pulse": "resting_heart_rate"
})

sport_long = pd.wide_to_long(
    sport, stubnames=["sport_total"],
    i="participant_id", j="measurement", sep="_", suffix=r"\d+"
).reset_index().rename(columns={"sport_total": "sports_participation"})

# convert yearly 10-minute units to average hours per week
sport_long["sports_participation"] = (
    pd.to_numeric(sport_long["sports_participation"], errors="coerce") / 6 / 52
)

# merge measurements by participant and measurement number
merged = (
    anthro_long
    .merge(age_long, on=["participant_id", "measurement"], how="outer")
    .merge(motor_long, on=["participant_id", "measurement"], how="outer")
    .merge(
        sport_long[["participant_id", "measurement", "sports_participation"]],
        on=["participant_id", "measurement"], how="outer"
    )
)

# add cohort information
merged["sex"] = "Male"
merged["origin"] = "LGS_Wave1A"
merged["country"] = "Belgium"
merged["id"] = "BE-KUL-" + merged["participant_id"].astype(str)
merged = merged.drop(columns="participant_id")

final_columns = [
    "origin", "country", "id", "sex", "age", "bmi", "height", "weight",
    "resting_heart_rate", "sit_and_reach", "sports_participation"
]
final_df = merged[final_columns].copy()

# convert numeric columns and save
num_cols = [
    "age", "bmi", "height", "weight", "resting_heart_rate",
    "sit_and_reach", "sports_participation"
]
final_df[num_cols] = final_df[num_cols].apply(pd.to_numeric, errors="coerce")
final_df = final_df.sort_values(by=["id", "age"])
final_df.to_csv(os.path.join(output_folder, output_file), index=False, float_format="%.2f")
print(f"Merged dataset saved to '{os.path.join(output_folder, output_file)}'")
