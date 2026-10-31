# tmu diabetes code extraction
# this public version uses generic input filenames and source column names
# input files must use the generic names defined in this script
# no original participant data are included

import os
import pandas as pd

# file paths and encoding
DATA_FOLDER = "DATA/tmu"
RESULT_FOLDER = "RESULTS"
FILE_ENCODING = "big5hkscs"
OUTPUT_FILE = os.path.join(RESULT_FOLDER, "TMU_Diabetes_codes.csv")
os.makedirs(RESULT_FOLDER, exist_ok=True)

# icd9 and icd10 reference files
files = [
    "icd9_s.csv", "icd9_t.csv", "icd9_w.csv",
    "icd10_s.csv", "icd10_t.csv", "icd10_w.csv"
]

# collect rows describing diabetes diagnoses
all_diabetes_rows = []
for filename in files:
    file_path = os.path.join(DATA_FOLDER, filename)
    try:
        df = pd.read_csv(file_path, encoding=FILE_ENCODING)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue

    if "diagnosis_description" not in df.columns:
        print(f"diagnosis_description not found in {filename}")
        continue

    diabetes_df = df[df["diagnosis_description"].str.contains("diabetes", case=False, na=False)].copy()

    # retain the original code fields for each classification
    if filename.startswith("icd9"):
        required = ["code_icd9", "code_alternate", "diagnosis_description"]
        if all(col in diabetes_df.columns for col in required):
            diabetes_df = diabetes_df[required]
            diabetes_df["source_type"] = "ICD9"
            all_diabetes_rows.append(diabetes_df)
        else:
            print(f"Missing columns in {filename} for ICD9")
    elif filename.startswith("icd10"):
        required = ["code_icd10", "diagnosis_description"]
        if all(col in diabetes_df.columns for col in required):
            diabetes_df = diabetes_df[required]
            diabetes_df["code_alternate"] = None
            diabetes_df["source_type"] = "ICD10"
            diabetes_df = diabetes_df[["code_icd10", "code_alternate", "diagnosis_description", "source_type"]]
            all_diabetes_rows.append(diabetes_df)
        else:
            print(f"Missing columns in {filename} for ICD10")

# combine and save the code list
if all_diabetes_rows:
    final_df = pd.concat(all_diabetes_rows, ignore_index=True).drop_duplicates()
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved combined diabetes data to {OUTPUT_FILE}")
else:
    print("No diabetes entries found.")
