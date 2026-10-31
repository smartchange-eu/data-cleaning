# uk biobank data preprocessing and harmonization
# this script processes uk biobank data into the standardized smartchange format
# input filenames and source column names are retained from the original script
# uk biobank participant-level data are not included and require authorized access

import pandas as pd
import numpy as np
import os

# paths
input_file = "DATA/uk-biobank/uk_biobank.csv"
output_folder = "RESULTS"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "uk_biobank_edited.csv")

# load data
df = pd.read_csv(input_file, low_memory=False)

# basic variables: origin, id, sex
df["origin"] = "UK Biobank"
df["country"] = "United Kingdom"
df["id"] = "uk-biobank-" + df["eid"].astype(str)

sex_map = {
    0: "Female",
    1: "Male",
    "0": "Female",
    "1": "Male",
    "Female": "Female",
    "Male": "Male"
}
df["sex"] = df["p31"].map(sex_map).fillna(df["p31"])

# helper: compute age from a measurement date + birth year/month
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

df["birth_year"] = pd.to_numeric(df["p34"], errors="coerce")
birth_month_num = df["p52"].map(month_map)
birth_month_num = birth_month_num.fillna(pd.to_numeric(df["p52"], errors="coerce"))
df["birth_month"] = birth_month_num

def compute_age_vec(date_series, birth_year, birth_month):
    """Vectorised integer age from date + birth year/month."""
    age = date_series.dt.year - birth_year
    adjust = (date_series.dt.month < birth_month).astype("int")
    age = age - adjust
    age[date_series.isna() | birth_year.isna() | birth_month.isna()] = np.nan
    return age.astype("float")

# parse date columns
timepoints_04 = ["i0", "i1", "i2", "i3"]
timepoints_05 = ["i0", "i1", "i2", "i3", "i4"]

for tp in timepoints_04:
    col = f"p53_{tp}"
    if col in df.columns:
        df[f"date_main_{tp}"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    else:
        df[f"date_main_{tp}"] = pd.NaT

lab_date_specs = {
    "glucose":      "p30741",
    "cholesterol":  "p30691",
    "triglycerides": "p30871",
    "hdl":          "p30761",
    "ldl":          "p30781"
}
for var_name, prefix in lab_date_specs.items():
    for tp in ["i0", "i1"]:
        col = f"{prefix}_{tp}"
        if col in df.columns:
            df[f"date_{var_name}_{tp}"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        else:
            df[f"date_{var_name}_{tp}"] = pd.NaT

for tp in timepoints_05:
    col = f"p105010_{tp}"
    if col in df.columns:
        df[f"date_sb_{tp}"] = pd.to_datetime(df[col], errors="coerce")
    else:
        df[f"date_sb_{tp}"] = pd.NaT

# ages per event type
for tp in ["i0", "i1", "i2", "i3"]:
    date_col = f"date_main_{tp}"
    if date_col in df.columns:
        df[f"age_main_{tp}"] = compute_age_vec(df[date_col], df["birth_year"], df["birth_month"])
    else:
        df[f"age_main_{tp}"] = np.nan

for var_name in ["glucose", "cholesterol", "triglycerides", "hdl", "ldl"]:
    for tp in ["i0", "i1"]:
        date_col = f"date_{var_name}_{tp}"
        if date_col in df.columns:
            df[f"age_{var_name}_{tp}"] = compute_age_vec(df[date_col], df["birth_year"], df["birth_month"])
        else:
            df[f"age_{var_name}_{tp}"] = np.nan

for tp in ["i0", "i1", "i2", "i3", "i4"]:
    date_col = f"date_sb_{tp}"
    if date_col in df.columns:
        df[f"age_sb_{tp}"] = compute_age_vec(df[date_col], df["birth_year"], df["birth_month"])
    else:
        df[f"age_sb_{tp}"] = np.nan

# anthropometry
for tp in timepoints_04:
    col = f"p50_{tp}"
    if col in df.columns:
        df[f"height_{tp}"] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[f"height_{tp}"] = np.nan

    col = f"p21002_{tp}"
    if col in df.columns:
        df[f"weight_{tp}"] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[f"weight_{tp}"] = np.nan

    col = f"p21001_{tp}"
    if col in df.columns:
        df[f"bmi_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"bmi_{tp}"] = np.nan

    col = f"p48_{tp}"
    if col in df.columns:
        df[f"waist_circ_{tp}"] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[f"waist_circ_{tp}"] = np.nan

# blood markers
for tp in ["i0", "i1"]:
    col = f"p30740_{tp}"
    if col in df.columns:
        df[f"glucose_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"glucose_{tp}"] = np.nan

    col = f"p30690_{tp}"
    if col in df.columns:
        df[f"cholesterol_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"cholesterol_{tp}"] = np.nan

    col = f"p30870_{tp}"
    if col in df.columns:
        df[f"triglycerides_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"triglycerides_{tp}"] = np.nan

    col = f"p30760_{tp}"
    if col in df.columns:
        df[f"hdl_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"hdl_{tp}"] = np.nan

    col = f"p30780_{tp}"
    if col in df.columns:
        df[f"ldl_{tp}"] = pd.to_numeric(df[col], errors="coerce").round(2)
    else:
        df[f"ldl_{tp}"] = np.nan

# blood pressure
for tp in timepoints_04:
    col_a0 = f"p4079_{tp}_a0"
    col_a1 = f"p4079_{tp}_a1"
    a0 = pd.to_numeric(df[col_a0], errors="coerce") if col_a0 in df.columns else np.nan
    a1 = pd.to_numeric(df[col_a1], errors="coerce") if col_a1 in df.columns else np.nan
    avg = (a0 + a1) / 2
    df[f"diastolic_bp_{tp}"] = np.round(avg).astype("float")

    col_a0 = f"p4080_{tp}_a0"
    col_a1 = f"p4080_{tp}_a1"
    a0 = pd.to_numeric(df[col_a0], errors="coerce") if col_a0 in df.columns else np.nan
    a1 = pd.to_numeric(df[col_a1], errors="coerce") if col_a1 in df.columns else np.nan
    avg = (a0 + a1) / 2
    df[f"systolic_bp_{tp}"] = np.round(avg).astype("float")

# smoking
def recode_smoking(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if v.startswith("never"):
        return 0
    if v.startswith("previous"):
        return 1
    if v.startswith("current"):
        return 2
    if "prefer not" in v:
        return 2
    return np.nan

for tp in timepoints_04:
    col = f"p20116_{tp}"
    if col in df.columns:
        df[f"smoking_{tp}"] = df[col].apply(recode_smoking)
    else:
        df[f"smoking_{tp}"] = np.nan

for idx in df.index:
    ever_smoker_before = False
    for tp in timepoints_04:
        col = f"smoking_{tp}"
        val = df.at[idx, col]
        if pd.isnull(val):
            continue
        if val in [1, 2]:
            ever_smoker_before = True
        elif val == 0 and ever_smoker_before:
            df.at[idx, col] = 1

# alcohol
alcohol_map = {
    "daily or almost daily": 12.0,
    "three or four times a week": 10.0,
    "once or twice a week": 4.5,
    "one to three times a month": 2.5,
    "special occasions only": 1.0,
    "never": 0.0
}
def recode_alcohol(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "prefer not" in v:
        return np.nan
    return alcohol_map.get(v, np.nan)

for tp in timepoints_04:
    col = f"p1558_{tp}"
    if col in df.columns:
        df[f"alcohol_{tp}"] = df[col].apply(recode_alcohol)
    else:
        df[f"alcohol_{tp}"] = np.nan

# moderate & vigorous pa
def minutes_day_to_hours_day(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v:
        return np.nan
    num = pd.to_numeric(val, errors="coerce")
    if pd.isnull(num):
        return np.nan
    return round(num / 60.0, 2)

for tp in timepoints_04:
    col = f"p894_{tp}"
    if col in df.columns:
        df[f"moderate_act_{tp}"] = df[col].apply(minutes_day_to_hours_day)
    else:
        df[f"moderate_act_{tp}"] = np.nan

    col = f"p914_{tp}"
    if col in df.columns:
        df[f"vigorous_act_{tp}"] = df[col].apply(minutes_day_to_hours_day)
    else:
        df[f"vigorous_act_{tp}"] = np.nan

    df[f"mvpa_{tp}"] = (
        pd.to_numeric(df[f"moderate_act_{tp}"], errors="coerce") +
        pd.to_numeric(df[f"vigorous_act_{tp}"], errors="coerce")
    ).round(2)

# resting heart rate
for tp in timepoints_04:
    col_a0 = f"p102_{tp}_a0"
    col_a1 = f"p102_{tp}_a1"
    a0 = pd.to_numeric(df[col_a0], errors="coerce") if col_a0 in df.columns else np.nan
    a1 = pd.to_numeric(df[col_a1], errors="coerce") if col_a1 in df.columns else np.nan
    avg = (a0 + a1) / 2
    df[f"resting_hr_{tp}"] = avg.round(2)

# vo2max
for tp in ["i0", "i1"]:
    col = f"p30038_{tp}"
    if col in df.columns:
        df[f"vo2_max_{tp}"] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[f"vo2_max_{tp}"] = np.nan
for tp in ["i2", "i3"]:
    df[f"vo2_max_{tp}"] = np.nan

# fruits
def fruits_to_portions(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if "less than one" in v:
        return 0.5
    num = pd.to_numeric(val, errors="coerce")
    return float(num) if pd.notnull(num) else np.nan

for tp in timepoints_04:
    col = f"p1309_{tp}"
    if col in df.columns:
        df[f"fruits_{tp}"] = df[col].apply(fruits_to_portions)
    else:
        df[f"fruits_{tp}"] = np.nan

# vegetables
def tbsp_to_portions(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if "less than one" in v:
        tbsp = 0.5
    else:
        tbsp = pd.to_numeric(val, errors="coerce")
    if pd.isnull(tbsp):
        return np.nan
    portions = tbsp / 3.0
    return round(portions, 2)

for tp in timepoints_04:
    col_cooked = f"p1289_{tp}"
    col_raw = f"p1299_{tp}"

    cooked = df[col_cooked].apply(tbsp_to_portions) if col_cooked in df.columns else pd.Series(np.nan, index=df.index)
    raw = df[col_raw].apply(tbsp_to_portions) if col_raw in df.columns else pd.Series(np.nan, index=df.index)

    veg_avg = pd.concat([cooked, raw], axis=1).mean(axis=1)
    df[f"vegetables_{tp}"] = veg_avg.round(2)

# sweetened beverages
def beverages_to_portions(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if "less than one" in v:
        grams = 0.5 * 330.0
    else:
        grams = pd.to_numeric(val, errors="coerce")
    if pd.isnull(grams):
        return np.nan
    portions = grams / 330.0
    return round(portions, 2)

for tp in timepoints_05:
    col = f"p26127_{tp}"
    if col in df.columns:
        df[f"sweet_bev_{tp}"] = df[col].apply(beverages_to_portions)
    else:
        df[f"sweet_bev_{tp}"] = np.nan

# red meats
freq_map = {
    "never": 0.0,
    "less than once a week": 0.5 / 7.0,
    "once a week": 1.0 / 7.0,
    "2–4 times a week": 3.0 / 7.0,
    "2-4 times a week": 3.0 / 7.0,
    "5–6 times a week": 5.5 / 7.0,
    "5-6 times a week": 5.5 / 7.0,
    "once or more daily": 1.0
}
def freq_to_perday(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower().replace("–", "-")
    if "do not know" in v or "prefer not" in v:
        return np.nan
    return freq_map.get(v, np.nan)

for tp in timepoints_04:
    meats_cols = {
        "beef": f"p1369_{tp}",
        "lamb": f"p1379_{tp}",
        "pork": f"p1389_{tp}"
    }
    daily_parts = []
    for meat, col in meats_cols.items():
        if col in df.columns:
            daily_parts.append(df[col].apply(freq_to_perday))
        else:
            daily_parts.append(pd.Series(np.nan, index=df.index))

    meat_sum = pd.concat(daily_parts, axis=1).sum(axis=1)
    df[f"red_meats_{tp}"] = meat_sum.round(3)

# processed meats
for tp in timepoints_04:
    col = f"p1349_{tp}"
    if col in df.columns:
        df[f"processed_meats_{tp}"] = df[col].apply(freq_to_perday)
    else:
        df[f"processed_meats_{tp}"] = np.nan

# cereals
def cereals_per_day(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if "less than one" in v:
        bowls = 0.5
    else:
        bowls = pd.to_numeric(val, errors="coerce")
    if pd.isnull(bowls):
        return np.nan
    return round(bowls / 7.0, 3)

for tp in timepoints_04:
    col = f"p1458_{tp}"
    if col in df.columns:
        df[f"cereals_{tp}"] = df[col].apply(cereals_per_day)
    else:
        df[f"cereals_{tp}"] = np.nan

# sleep
def clean_sleep(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    return val

for tp in timepoints_04:
    col = f"p1160_{tp}"
    if col in df.columns:
        df[f"sleep_{tp}"] = df[col].apply(clean_sleep)
    else:
        df[f"sleep_{tp}"] = np.nan

# family diabetes & cvd
def has_keyword(val, keywords):
    if pd.isnull(val) or str(val).strip() == "":
        return False
    text = str(val)
    return any(k in text for k in keywords)

for tp in timepoints_04:
    m_col = f"p20110_{tp}"
    s_col = f"p20111_{tp}"
    f_col = f"p20107_{tp}"

    m = df[m_col] if m_col in df.columns else pd.Series("", index=df.index)
    s = df[s_col] if s_col in df.columns else pd.Series("", index=df.index)
    f = df[f_col] if f_col in df.columns else pd.Series("", index=df.index)

    diabetes_any = (
        m.apply(has_keyword, keywords=["Diabetes"]) |
        s.apply(has_keyword, keywords=["Diabetes"]) |
        f.apply(has_keyword, keywords=["Diabetes"])
    )
    df[f"family_diabetes_{tp}"] = np.where(diabetes_any, "Yes", "No")

    cvd_any = (
        m.apply(has_keyword, keywords=["Heart disease", "Stroke"]) |
        s.apply(has_keyword, keywords=["Heart disease", "Stroke"]) |
        f.apply(has_keyword, keywords=["Heart disease", "Stroke"])
    )
    df[f"family_cvd_{tp}"] = np.where(cvd_any, "Yes", "No")

# own diabetes p2443_*
def clean_diabetes_self(val):
    if pd.isnull(val) or str(val).strip() == "":
        return ""
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return ""
    return val

for tp in timepoints_04:
    col = f"p2443_{tp}"
    if col in df.columns:
        df[f"diabetes_{tp}"] = df[col].apply(clean_diabetes_self)
    else:
        df[f"diabetes_{tp}"] = ""

# medication: p6153_* and p6177_*
def bp_med_flag(val):
    if pd.isnull(val) or str(val).strip() == "":
        return False
    return "Blood pressure medication" in str(val)

for tp in timepoints_04:
    col1 = f"p6153_{tp}"
    col2 = f"p6177_{tp}"
    v1 = df[col1] if col1 in df.columns else pd.Series("", index=df.index)
    v2 = df[col2] if col2 in df.columns else pd.Series("", index=df.index)

    any_bp_med = v1.apply(bp_med_flag) | v2.apply(bp_med_flag)

    both_empty = (v1.fillna("").str.strip() == "") & (v2.fillna("").str.strip() == "")
    med = np.where(both_empty, "", np.where(any_bp_med, "Yes", "No"))
    df[f"medication_{tp}"] = med

# family cars
def recode_family_cars(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if v.startswith("none"):
        return 1
    if v.startswith("one"):
        return 2
    if v.startswith("two") or v.startswith("three") or "four or more" in v:
        return 3
    return np.nan

for tp in timepoints_04:
    col = f"p728_{tp}"
    if col in df.columns:
        df[f"family_cars_{tp}"] = df[col].apply(recode_family_cars)
    else:
        df[f"family_cars_{tp}"] = np.nan

# income
income_prefix = "p738"
has_p738 = any(f"{income_prefix}_i{t}" in df.columns for t in range(4))
income_prefix = "p738" if has_p738 else "p728"

def recode_income(val):
    if pd.isnull(val) or str(val).strip() == "":
        return np.nan
    v = str(val).strip().lower()
    if "do not know" in v or "prefer not" in v:
        return np.nan
    if v.startswith("less than 18,000"):
        return 1
    if v.startswith("18,000 to 30,999"):
        return 2
    if v.startswith("31,000 to 51,999"):
        return 3
    if v.startswith("52,000 to 100,000"):
        return 4
    if v.startswith("greater than 100,000"):
        return 5
    return np.nan

for tp in timepoints_04:
    col = f"{income_prefix}_{tp}"
    if col in df.columns:
        df[f"income_{tp}"] = df[col].apply(recode_income)
    else:
        df[f"income_{tp}"] = np.nan

# build event-level table, then collapse by
base_id_cols = ["origin", "id", "sex"]

measure_cols_all = [
    "bmi", "height", "weight", "waist_circ",
    "glucose", "cholesterol", "triglycerides", "hdl", "ldl",
    "diastolic_bp", "systolic_bp",
    "smoking", "alcohol",
    "moderate_act", "vigorous_act", "mvpa",
    "resting_heart_rate", "vo2_max",
    "fruits", "vegetables", "sweetened_beverages",
    "red_meats", "processed_meats", "cereals",
    "sleep",
    "family_diabetes", "family_cvd",
    "diabetes", "medication",
    "family_cars", "income"
]

event_dfs = []

for tp in ["i0", "i1", "i2", "i3"]:
    age_col = f"age_main_{tp}"
    if age_col not in df.columns:
        continue

    colmap = {
        f"bmi_{tp}": "bmi",
        f"height_{tp}": "height",
        f"weight_{tp}": "weight",
        f"waist_circ_{tp}": "waist_circ",
        f"diastolic_bp_{tp}": "diastolic_bp",
        f"systolic_bp_{tp}": "systolic_bp",
        f"smoking_{tp}": "smoking",
        f"alcohol_{tp}": "alcohol",
        f"moderate_act_{tp}": "moderate_act",
        f"vigorous_act_{tp}": "vigorous_act",
        f"mvpa_{tp}": "mvpa",
        f"resting_hr_{tp}": "resting_heart_rate",
        f"vo2_max_{tp}": "vo2_max",
        f"fruits_{tp}": "fruits",
        f"vegetables_{tp}": "vegetables",
        f"red_meats_{tp}": "red_meats",
        f"processed_meats_{tp}": "processed_meats",
        f"cereals_{tp}": "cereals",
        f"sleep_{tp}": "sleep",
        f"family_diabetes_{tp}": "family_diabetes",
        f"family_cvd_{tp}": "family_cvd",
        f"diabetes_{tp}": "diabetes",
        f"medication_{tp}": "medication",
        f"family_cars_{tp}": "family_cars",
        f"income_{tp}": "income",
    }

    used_input_cols = [c for c in colmap.keys() if c in df.columns]
    if not used_input_cols and df[age_col].isna().all():
        continue

    cols = base_id_cols + [age_col] + used_input_cols
    chunk = df[cols].copy()
    chunk = chunk.rename(columns={age_col: "age", **colmap})

    visit_vars = list(colmap.values())
    mask_all_missing = chunk[visit_vars].apply(
        lambda row: all((pd.isnull(x) or x == "") for x in row),
        axis=1
    )
    chunk = chunk[~(chunk["age"].isna() & mask_all_missing)].copy()

    event_dfs.append(chunk)

lab_value_map = {
    "glucose": "glucose",
    "cholesterol": "cholesterol",
    "triglycerides": "triglycerides",
    "hdl": "hdl",
    "ldl": "ldl"
}

for var_name in lab_value_map.keys():
    for tp in ["i0", "i1"]:
        age_col = f"age_{var_name}_{tp}"
        val_col = f"{var_name}_{tp}"
        if age_col not in df.columns or val_col not in df.columns:
            continue

        mask = df[age_col].notna() & df[val_col].notna()
        if not mask.any():
            continue

        chunk = df.loc[mask, base_id_cols + [age_col, val_col]].copy()
        chunk = chunk.rename(columns={
            age_col: "age",
            val_col: lab_value_map[var_name]
        })
        event_dfs.append(chunk)

for tp in ["i0", "i1", "i2", "i3", "i4"]:
    age_col = f"age_sb_{tp}"
    val_col = f"sweet_bev_{tp}"
    if age_col not in df.columns or val_col not in df.columns:
        continue

    mask = df[age_col].notna() & df[val_col].notna()
    if not mask.any():
        continue

    chunk = df.loc[mask, base_id_cols + [age_col, val_col]].copy()
    chunk = chunk.rename(columns={age_col: "age", val_col: "sweetened_beverages"})
    event_dfs.append(chunk)

# concatenate all events
events_df = pd.concat(event_dfs, ignore_index=True)

for c in measure_cols_all:
    if c not in events_df.columns:
        events_df[c] = np.nan

# collapse to one row per
group_cols = ["origin", "id", "sex", "age"]
agg_dict = {c: "first" for c in measure_cols_all}

final_df = events_df.groupby(group_cols, as_index=False).agg(agg_dict)
final_df["country"] = "United Kingdom"

# binary coding: family_diabetes, family_cvd, diabetes, medication

for col in ["family_diabetes", "family_cvd"]:
    if col in final_df.columns:
        final_df[col] = final_df[col].map({"Yes": 1, "No": 0})

if "diabetes" in final_df.columns:
    final_df["diabetes"] = np.where(
        final_df["diabetes"].astype(str).str.strip() == "",
        0,
        1
    )

if "medication" in final_df.columns:
    final_df["medication"] = np.where(final_df["medication"] == "Yes", 1, 0)

# drop rows that only have
id_cols = group_cols
measurement_cols = [c for c in final_df.columns if c not in id_cols]
simple_cols = {"sweetened_beverages", "red_meats", "family_diabetes", "family_cvd", "diabetes", "medication"}
simple_cols_present = [c for c in measurement_cols if c in simple_cols]

def row_has_only_simple(row):
    non_missing = []
    for c in measurement_cols:
        val = row[c]
        if pd.isnull(val):
            continue
        if isinstance(val, str) and val == "":
            continue
        non_missing.append(c)
    if not non_missing:
        return False
    return all(c in simple_cols_present for c in non_missing)

if measurement_cols:
    mask_simple_only = final_df.apply(row_has_only_simple, axis=1)
    n_removed = int(mask_simple_only.sum())
    final_df = final_df.loc[~mask_simple_only].reset_index(drop=True)
    print(f"Removed {n_removed} rows with only sweetened_beverages/red_meats/family_diabetes/family_cvd (plus id/sex/age).")

# sort by id and age
final_df = final_df.sort_values(by=["id", "age"], kind="mergesort").reset_index(drop=True)

# final formatting
for col in [
    "bmi", "glucose", "cholesterol", "triglycerides",
    "hdl", "ldl",
    "moderate_act", "vigorous_act", "mvpa",
    "resting_heart_rate", "fruits", "vegetables",
    "sweetened_beverages", "red_meats", "processed_meats",
    "cereals"
]:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce").round(2)

for col in ["diastolic_bp", "systolic_bp", "family_cars", "income", "smoking"]:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

# column order
desired_cols = [
    "origin", "id", "sex", "country", "age",
    "bmi", "height", "weight", "waist_circ",
    "glucose", "cholesterol", "triglycerides", "hdl", "ldl",
    "diastolic_bp", "systolic_bp",
    "smoking", "alcohol",
    "moderate_act", "vigorous_act", "mvpa",
    "resting_heart_rate", "vo2_max",
    "fruits", "vegetables", "sweetened_beverages",
    "red_meats", "processed_meats", "cereals",
    "sleep",
    "family_diabetes", "family_cvd",
    "diabetes", "medication",
    "family_cars", "income"
]

final_cols = [c for c in desired_cols if c in final_df.columns]
final_df = final_df[final_cols]

# save
final_df.to_csv(output_file, index=False)
print("UK Biobank dataset processed and saved successfully to:", output_file)