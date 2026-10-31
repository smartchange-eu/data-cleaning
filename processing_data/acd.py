import pandas as pd
import numpy as np
import os

# Define file paths
folder_path = "DATA/acd"
data_file = "input.csv"
data_path = os.path.join(folder_path, data_file)
results_folder = "RESULTS"
os.makedirs(results_folder, exist_ok=True)
output_path = os.path.join(results_folder, "ACD_edited.csv")

# Load the dataset
data = pd.read_csv(data_path, delimiter=';', decimal=',')

# Initialize processed data
data_processed = pd.DataFrame()

# Country
data_processed["country"] = "Slovenia"

# Add origin column
data_processed["origin"] = ["ACD.Si"] * len(data)

# Generate id column with new format "ACD-<number>"
data_processed["id"] = [f"ACD-{i}" for i in range(1, len(data) + 1)]

# Map sex column
data_processed["sex"] = data["gender"].str.capitalize().replace({"Men": "Male", "Female": "Female"})

# Process age column
data_processed["age"] = pd.to_numeric(data["AgeRand"].str.replace(',', '.'), errors="coerce").round(1)

# Convert height and weight to numeric
data["BH"] = pd.to_numeric(data["BH"].str.replace(',', '.'), errors="coerce")
data["BM"] = pd.to_numeric(data["BM"].str.replace(',', '.'), errors="coerce")

data_processed["height"] = data["BH"]
data_processed["weight"] = data["BM"]

# Calculate BMI
data_processed["bmi"] = data.apply(
    lambda row: round(row["BM"] / (row["BH"] / 100) ** 2, 1)
    if pd.notna(row["BH"]) and pd.notna(row["BM"]) and row["BH"] > 0 else None,
    axis=1
)

# Add waist_circ column
data_processed["waist_circ"] = pd.to_numeric(data["WC"].str.replace(',', '.'), errors="coerce")

# Calculate physical activity (average hours per day)
moderate_minutes = (
    pd.to_numeric(data["Q22zmeTeden"], errors="coerce") +
    pd.to_numeric(data["Q22zmeWeekend"], errors="coerce")
)
data_processed["moderate_act"] = moderate_minutes.apply(
    lambda x: round((x / 60) / 7, 2) if pd.notna(x) and x > 0 else None
)

vigorous_minutes = (
    pd.to_numeric(data["Q21zahWeek"], errors="coerce") +
    pd.to_numeric(data["Q21zahWeekend"], errors="coerce")
)
data_processed["vigorous_act"] = vigorous_minutes.apply(
    lambda x: round((x / 60) / 7, 2) if pd.notna(x) and x > 0 else None
)

# MVPA (sum of vigorous + moderate)
data_processed["mvpa"] = (
    data_processed[["vigorous_act", "moderate_act"]]
    .sum(axis=1, min_count=1)
    .round(2)
)

# Mapping for portions per day
portions_mapping = {
    "1-2 times a day": 1.5,
    "1-3 times a month": 0.07,
    "1x a week": 0.14,
    "2-3 times a week": 0.36,
    "3x a day or more": 3.00,
    "4-6 times a week": 0.71,
    "Interrupted": 0.00,
    "Interrupted/Never": 0.00,
    "Never": 0.00
}

# Fruits & Veg
data_processed["vegetables"] = data["QVEG"].map(portions_mapping) if "QVEG" in data.columns else None
data_processed["fruits"]     = data["QFRUIT"].map(portions_mapping) if "QFRUIT" in data.columns else None

# Lonely mapping
lonely_mapping = {
    "Not at all/Never": 0,
    "Somewhat/Rarely":    1,
    "Moderate/Frequent":  2,
    "Very often":         3,
    "Absolutely/Always":  4,
    "Interrupted":        0
}
data_processed["lonely"] = data["QFL"].map(lonely_mapping) if "QFL" in data.columns else None

# Sweetened beverages
data_processed["sweetened_beverages"] = (
    data["QSODA"].map(portions_mapping) if "QSODA" in data.columns else None
)

# Bedroom
bedroom_mapping = {"Yes": 1, "No": 0, "Interrupted": 0}
data_processed["bedroom"] = data["QR"].map(bedroom_mapping) if "QR" in data.columns else None

# Family vacations
vacation_mapping = {
    "We didn't even go to": 0,
    "Once":    1,
    "Twice":   2,
    "More than twice": 3,
    "Interrupted":      0
}
data_processed["family_vacations"] = (
    data["QV"].map(vacation_mapping) if "QV" in data.columns else None
)

# Family computers 
computer_mapping = {
    "None":           0,
    "One":            1,
    "Two":            2,
    "More than two":  3,
    "Interrupted":    0
}
data_processed["family_computers"] = (
    data["QCOMP"].map(computer_mapping) if "QCOMP" in data.columns else None
)

# --- REMAPPING STEPS ---

# bedroom: 0 → 1, 1 → 2
if "bedroom" in data_processed:
    data_processed["bedroom"] = data_processed["bedroom"].map({0: 1, 1: 2})

# family_vacations: 0 → 1, 1 → 2, 2 → 2, 3 → 4
if "family_vacations" in data_processed:
    data_processed["family_vacations"] = data_processed["family_vacations"].map({
        0: 1,
        1: 2,
        2: 2,
        3: 4
    })

# family_computers: 0 → 1, 1 → 2, 2 → 3, 3 → 4
if "family_computers" in data_processed:
    data_processed["family_computers"] = data_processed["family_computers"].map({
        0: 1,
        1: 2,
        2: 3,
        3: 4
    })

# Calculate average daily sleep (rounded to 2 decimals)
data["QLSWeek"]    = pd.to_numeric(data["QLSWeek"].str.replace(',', '.'), errors="coerce")
data["QLSWeekend"] = pd.to_numeric(data["QLSWeekend"].str.replace(',', '.'), errors="coerce")
data_processed["sleep"] = (((data["QLSWeek"] * 5) + (data["QLSWeekend"] * 2)) / 7).round(2)

# Shuttle run speed
data_processed["shuttle_run_speed"] = pd.to_numeric(
    data["MSRT_20Speed"].str.replace(',', '.'), errors="coerce"
)

# VO2 max
data_processed["vo2_max"] = pd.to_numeric(
    data["VO2maxMaharA"].str.replace(',', '.'), errors="coerce"
).round(2)

# standing_lj
data_processed["standing_lj"] = pd.to_numeric(
    data["MSLJ"].str.replace(',', '.'), errors="coerce"
)
data_processed.loc[data_processed["standing_lj"] == 0, "standing_lj"] = np.nan

# Z-scores for affluence variables
aff_cols = ["bedroom", "family_vacations", "family_computers"]
z_cols = []
for col in aff_cols:
    mean = data_processed[col].mean()
    std  = data_processed[col].std()
    zcol = f"z_score_{col}"
    data_processed[zcol] = (
        (data_processed[col] - mean) / std
    ).round(2)
    z_cols.append(zcol)

# Aggregate to a single affluence z-score
data_processed["z_score_affluence"] = (
    data_processed[z_cols].mean(axis=1)
    .round(2)
)

# Save the processed dataset
data_processed.to_csv(output_path, index=False)
print(f"Processed dataset saved to '{output_path}'.")

# Reload, filter incomplete rows, and overwrite
df = pd.read_csv(output_path)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
filtered_df = df.dropna(thresh=5)
filtered_df.to_csv(output_path, index=False)
print(f"Filtered data saved back to '{output_path}'.")
