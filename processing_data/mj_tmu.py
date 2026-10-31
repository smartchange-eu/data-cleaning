# tmu mj data preprocessing and harmonization
# this script converts cohort data into the standardized smartchange format
# original input filenames and source column names have been replaced with generic names
# the input file must use the generic column names defined in this script
# no original participant data are included

import pandas as pd
import numpy as np
import os

# paths
input_path = "DATA/tmu/input.csv"
output_dir = "RESULTS"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "MJ_edited.csv")

# helper functions
def safe_bool_series(df, col):
    if col in df.columns:
        return df[col].astype(bool)
    else:
        return pd.Series(False, index=df.index)

def mgdl_to_mmol_glucose(series):
    return pd.to_numeric(series, errors="coerce") / 18.0

def mgdl_to_mmol_chol(series):
    return pd.to_numeric(series, errors="coerce") / 38.67

def mgdl_to_mmol_trig(series):
    return pd.to_numeric(series, errors="coerce") / 88.57

# load data
df = pd.read_csv(input_path)

# participant information and anthropometric measurements
df["origin"] = "TMU MJ"
df["country"] = "Taiwan"
df["id"] = "TMU_MJ_" + df["participant_id"].astype(str)

df["sex"] = pd.Series(np.nan, index=df.index, dtype="object")
df.loc[safe_bool_series(df, "sex_male"), "sex"] = "Male"
df.loc[safe_bool_series(df, "sex_female"), "sex"] = "Female"

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["height"] = pd.to_numeric(df.get("height_source"), errors="coerce")
df["weight"] = pd.to_numeric(df.get("weight_source"), errors="coerce")
df["bmi"] = pd.to_numeric(df.get("bmi_source"), errors="coerce")
df["waist_circ"] = pd.to_numeric(df.get("waist_source"), errors="coerce")

# average available fasting glucose measurements and convert to mmol/l
glucose_1 = pd.to_numeric(df.get("fasting_glucose_1"), errors="coerce")
glucose_2 = pd.to_numeric(df.get("fasting_glucose_2"), errors="coerce") if "fasting_glucose_2" in df.columns else pd.Series(np.nan, index=df.index)
glucose_mean = pd.concat([glucose_1, glucose_2], axis=1).mean(axis=1, skipna=True)
df["glucose"] = mgdl_to_mmol_glucose(glucose_mean)

# convert lipid measurements from mg/dl to mmol/l
df["cholesterol"] = mgdl_to_mmol_chol(df.get("cholesterol_source"))
df["triglycerides"] = mgdl_to_mmol_trig(df.get("triglycerides_source"))
df["hdl"] = mgdl_to_mmol_chol(df.get("hdl_source"))
df["ldl"] = mgdl_to_mmol_chol(df.get("ldl_source"))

# use right-side blood pressure, falling back to left-side measurements
dbp_right = pd.to_numeric(df.get("diastolic_right"), errors="coerce") if "diastolic_right" in df.columns else pd.Series(np.nan, index=df.index)
dbp_left = pd.to_numeric(df.get("diastolic_left"), errors="coerce") if "diastolic_left" in df.columns else pd.Series(np.nan, index=df.index)
df["diastolic_bp"] = dbp_right.where(~dbp_right.isna(), dbp_left)

sbp_right = pd.to_numeric(df.get("systolic_right"), errors="coerce") if "systolic_right" in df.columns else pd.Series(np.nan, index=df.index)
sbp_left = pd.to_numeric(df.get("systolic_left"), errors="coerce") if "systolic_left" in df.columns else pd.Series(np.nan, index=df.index)
df["systolic_bp"] = sbp_right.where(~sbp_right.isna(), sbp_left)
df["resting_heart_rate"] = pd.to_numeric(df.get("resting_pulse_source"), errors="coerce")

# recode old and new weekly sport frequency categories
freq_old = pd.Series(np.nan, index=df.index, dtype="float")
freq_old[safe_bool_series(df, "sport_frequency_old_1")] = 17.5
freq_old[safe_bool_series(df, "sport_frequency_old_2")] = 7.0
freq_old[safe_bool_series(df, "sport_frequency_old_3")] = 3.0
freq_old[safe_bool_series(df, "sport_frequency_old_4")] = 1.0
freq_old[safe_bool_series(df, "sport_frequency_old_5")] = 0.0

freq_new = pd.Series(np.nan, index=df.index, dtype="float")
freq_new[safe_bool_series(df, "sport_frequency_new_1")] = 10.0
freq_new[safe_bool_series(df, "sport_frequency_new_2")] = 5.0
freq_new[safe_bool_series(df, "sport_frequency_new_3")] = 2.5
freq_new[safe_bool_series(df, "sport_frequency_new_4")] = 1.0
freq_new[safe_bool_series(df, "sport_frequency_new_5")] = 0.0
freq_week = freq_new.where(~freq_new.isna(), freq_old)

# estimate hours per sport session
dur_hours = pd.Series(np.nan, index=df.index, dtype="float")
dur_hours[safe_bool_series(df, "sport_duration_1")] = 0.25
dur_hours[safe_bool_series(df, "sport_duration_2")] = 0.75
dur_hours[safe_bool_series(df, "sport_duration_3")] = 1.5
dur_hours[safe_bool_series(df, "sport_duration_4")] = 2.5
H = freq_week * dur_hours

# classify activity by sport intensity category
fs1 = safe_bool_series(df, "sport_type_1")
fs2 = safe_bool_series(df, "sport_type_2")
fs3 = safe_bool_series(df, "sport_type_3")
fs4 = safe_bool_series(df, "sport_type_4")
fs5 = safe_bool_series(df, "sport_type_5")

moderate = pd.Series(np.nan, index=df.index, dtype="float")
moderate[fs2] = H[fs2]
moderate[fs1 | fs3 | fs4 | fs5] = 0.0
df["moderate_act"] = moderate

vigorous = pd.Series(np.nan, index=df.index, dtype="float")
vigorous[fs3 | fs4] = H[fs3 | fs4]
vigorous[fs1 | fs2 | fs5] = 0.0
df["vigorous_act"] = vigorous

# smoking: 0 = never, 1 = current smoker, 2 = former smoker
smoking = pd.Series(np.nan, index=df.index, dtype="float")
mask_never = safe_bool_series(df, "smoking_status_1")
mask_ex = safe_bool_series(df, "smoking_status_2")
mask_current = (
    safe_bool_series(df, "smoking_status_3") |
    safe_bool_series(df, "smoking_status_4") |
    safe_bool_series(df, "smoking_status_5")
)
smoking[mask_never] = 0.0
smoking[mask_current] = 1.0
smoking[mask_ex] = 2.0
df["smoking"] = smoking

# recode alcohol consumption from one-hot response categories
alcohol = pd.Series(np.nan, index=df.index, dtype="float")
mask_0 = (
    safe_bool_series(df, "alcohol_old_1") |
    safe_bool_series(df, "alcohol_new_1") |
    safe_bool_series(df, "alcohol_old_2") |
    safe_bool_series(df, "alcohol_new_2")
)
alcohol[mask_0] = 0.0
mask_429 = safe_bool_series(df, "alcohol_old_3") | safe_bool_series(df, "alcohol_new_3")
alcohol[mask_429] = 4.29
mask_630 = safe_bool_series(df, "alcohol_old_4") | safe_bool_series(df, "alcohol_new_4")
alcohol[mask_630] = 6.30
alcohol[safe_bool_series(df, "alcohol_new_5")] = 8.5
mask_10 = safe_bool_series(df, "alcohol_old_5") | safe_bool_series(df, "alcohol_new_6")
alcohol[mask_10] = 10.0
df["alcohol"] = alcohol

# cardiovascular disease: 0 = no, 1 = yes
cvd = pd.Series(np.nan, index=df.index, dtype="float")
cvd[safe_bool_series(df, "cvd_no")] = 0.0
cvd[safe_bool_series(df, "cvd_yes")] = 1.0
df["cvd"] = cvd

# family cardiovascular disease: 0 = no, 1 = yes
family_cvd = pd.Series(np.nan, index=df.index, dtype="float")
family_cvd[safe_bool_series(df, "family_cvd_no")] = 0.0
family_cvd[safe_bool_series(df, "family_cvd_yes")] = 1.0
df["family_cvd"] = family_cvd

# sleep duration categories in hours
sleep = pd.Series(np.nan, index=df.index, dtype="float")
sleep[safe_bool_series(df, "sleep_duration_1")] = 3.5
sleep[safe_bool_series(df, "sleep_duration_2")] = 5.0
sleep[safe_bool_series(df, "sleep_duration_3")] = 6.5
sleep[safe_bool_series(df, "sleep_duration_4")] = 7.5
sleep[safe_bool_series(df, "sleep_duration_5")] = 8.5
sleep[safe_bool_series(df, "sleep_duration_6")] = 9.5
df["sleep"] = sleep

# diabetes: 0 = no, 1 = yes
diab = pd.Series(np.nan, index=df.index, dtype="float")
diab[safe_bool_series(df, "diabetes_no")] = 0.0
diab[safe_bool_series(df, "diabetes_yes")] = 1.0
df["diabetes"] = diab

# family diabetes: 0 = no, 1 = yes
fam_diab = pd.Series(np.nan, index=df.index, dtype="float")
fam_diab[safe_bool_series(df, "family_diabetes_no")] = 0.0
fam_diab[safe_bool_series(df, "family_diabetes_yes")] = 1.0
df["family_diabetes"] = fam_diab

# select and order harmonized output columns
cols_order = [
    "origin", "country", "id", "sex", "age",
    "height", "weight", "bmi", "waist_circ",
    "glucose", "cholesterol", "triglycerides", "hdl", "ldl",
    "systolic_bp", "diastolic_bp", "resting_heart_rate",
    "moderate_act", "vigorous_act", "smoking", "alcohol",
    "cvd", "family_cvd", "sleep", "diabetes", "family_diabetes"
]
final_cols = [c for c in cols_order if c in df.columns]
final_df = df[final_cols].copy()

# round numeric columns to two decimals
num_cols = final_df.select_dtypes(include="number").columns
final_df[num_cols] = final_df[num_cols].round(2)

# save processed data
final_df.to_csv(output_path, index=False)
print(f"Saved processed MJ dataset to {output_path}")