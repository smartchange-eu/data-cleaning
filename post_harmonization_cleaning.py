# final longitudinal data cleaning

import numpy as np
import pandas as pd

def to_numeric_if_exists(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def remove_bad_bp_rows(df):
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        mask_bad_bp = (
            df["systolic_bp"].notna()
            & df["diastolic_bp"].notna()
            & (df["systolic_bp"] <= df["diastolic_bp"])
        )
        n_removed = int(mask_bad_bp.sum())
        df = df.loc[~mask_bad_bp].copy()
    else:
        n_removed = 0

    return df, n_removed


def keep_most_complete_origin_id_age(df):
    if not {"origin", "id", "age"}.issubset(df.columns):
        return df, 0

    before = len(df)

    df["_n_non_missing"] = df.notna().sum(axis=1)

    df = (
        df.sort_values(
            ["origin", "id", "age", "_n_non_missing"],
            ascending=[True, True, True, False]
        )
        .drop_duplicates(subset=["origin", "id", "age"], keep="first")
        .drop(columns=["_n_non_missing"])
        .copy()
    )

    removed = before - len(df)
    return df, removed


def fix_smoking_history(df):
    """
    Final smoking coding:
    0 = never
    1 = smoker
    2 = ex-smoker

    Rule:
    If a person was smoker/ex-smoker before, later never becomes ex-smoker.
    """
    if not {"origin", "id", "age", "smoking"}.issubset(df.columns):
        return df, 0

    df = df.sort_values(["origin", "id", "age"]).copy()
    changed = 0

    for _, idx in df.groupby(["origin", "id"]).groups.items():
        idx = list(idx)
        ever_smoked_before = False

        for i in idx:
            val = df.at[i, "smoking"]

            if pd.isna(val):
                continue

            if val in [1, 2]:
                ever_smoked_before = True

            elif val == 0 and ever_smoked_before:
                df.at[i, "smoking"] = 2
                changed += 1

    return df, changed


def clean_binary_sequence(values):
    """
    For diabetes, family_diabetes, cvd, family_cvd.

    Rules:
    - Isolated value between same opposite values is corrected:
      0 0 1 0 0 -> 0 0 0 0 0
      1 1 0 1 1 -> 1 1 1 1 1

    - Sustained changes are kept:
      0 0 1 1 1 -> keep

    - Complex switchers:
      use majority value across participant;
      if tie, use 1.
    """
    s = pd.Series(values).copy()
    nonmiss = s.dropna()

    if len(nonmiss) < 3:
        return s

    vals = nonmiss.astype(int).tolist()
    positions = nonmiss.index.tolist()

    n_switches = sum(vals[i] != vals[i - 1] for i in range(1, len(vals)))

    # sustained 0 -> 1 or 1 -> 0 with only one switch: keep
    if n_switches <= 1:
        return s

    # isolated middle value correction
    corrected = vals.copy()
    changed_any = False

    for i in range(1, len(vals) - 1):
        if vals[i - 1] == vals[i + 1] and vals[i] != vals[i - 1]:
            corrected[i] = vals[i - 1]
            changed_any = True

    corrected_switches = sum(
        corrected[i] != corrected[i - 1]
        for i in range(1, len(corrected))
    )

    # If isolated correction resolves the problem, use it
    if changed_any and corrected_switches <= 1:
        for pos, val in zip(positions, corrected):
            s.loc[pos] = val
        return s

    # Complex switcher: majority rule, tie -> yes
    n_yes = sum(v == 1 for v in vals)
    n_no = sum(v == 0 for v in vals)

    final_val = 1 if n_yes >= n_no else 0

    for pos in positions:
        s.loc[pos] = final_val

    return s

def fix_binary_health_histories(df):
    binary_cols = ["diabetes", "family_diabetes", "cvd", "family_cvd"]
    binary_cols = [c for c in binary_cols if c in df.columns]

    if not binary_cols or not {"origin", "id", "age"}.issubset(df.columns):
        return df, {}

    df = df.sort_values(["origin", "id", "age"]).copy()
    changes = {}

    for col in binary_cols:
        before = df[col].copy()

        df[col] = (
            df.groupby(["origin", "id"])[col]
              .transform(clean_binary_sequence)
        )

        # Count real changes only, not NaN -> NaN
        compare_mask = before.notna() | df[col].notna()

        changes[col] = int(
            (before[compare_mask] != df.loc[compare_mask, col]).sum()
        )

    return df, changes


def impossible_height_mask(df):
    """
    Conservative absolute height rules.
    These are not meant to catch all outliers, only clearly impossible ones.
    """
    h = df["height"]
    age = df["age"]

    return (
        (h < 80)
        | (h > 250)
        | ((age < 5) & (h > 140))
        | ((age < 10) & (h > 180))
        | ((age < 12) & (h > 190))
        | ((age < 15) & (h > 210))
    )


def fit_height_prediction_model(df):
    """
    Predict height from age, sex, and weight.
    Used only when no usable neighbouring height exists.
    """
    needed = ["age", "sex", "weight", "height"]
    if not all(c in df.columns for c in needed):
        return None

    train = df[needed].copy()
    train = train.dropna()

    if len(train) < 100:
        return None

    train = train[
        (train["height"] >= 80)
        & (train["height"] <= 250)
        & (train["weight"] > 0)
    ].copy()

    if len(train) < 100:
        return None

    X = pd.get_dummies(train[["age", "sex", "weight"]], columns=["sex"], drop_first=True)
    y = train["height"]

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns.tolist()


def predict_height_from_model(df, idx, model_info):
    if model_info is None:
        return np.nan

    model, feature_cols = model_info

    row = df.loc[[idx], ["age", "sex", "weight"]].copy()

    if row[["age", "weight"]].isna().any(axis=None):
        return np.nan

    X = pd.get_dummies(row, columns=["sex"], drop_first=True)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0

    X = X[feature_cols]

    pred = model.predict(X)[0]

    if pred < 80 or pred > 250:
        return np.nan

    return pred


def clean_heights(df):
    """
    Height correction logic:

    1. Flag impossible absolute heights.
    2. For suspicious values:
       - previous + next available: use interpolation
       - only previous or only next: use nearest valid height
       - no neighbours: predict from age + sex + weight
       - still impossible: set height and BMI to missing
    3. For trajectory deviations:
       - if deviation from expected is 10-20 cm: repair
       - if deviation >=20 cm: height/BMI missing
    4. Recalculate BMI after height changes.
    """
    needed = {"origin", "id", "age", "height"}
    if not needed.issubset(df.columns):
        return df, {}

    df = to_numeric_if_exists(df, ["age", "height", "weight", "bmi"])
    df = df.sort_values(["origin", "id", "age"]).copy()

    stats = {
        "height_impossible_absolute": 0,
        "height_repaired_interpolation": 0,
        "height_repaired_nearest": 0,
        "height_repaired_model": 0,
        "height_set_missing": 0,
        "bmi_recalculated": 0
    }

    model_info = fit_height_prediction_model(df)

    # helper columns
    g = df.groupby(["origin", "id"])

    df["_prev_age"] = g["age"].shift(1)
    df["_next_age"] = g["age"].shift(-1)
    df["_prev_height"] = g["height"].shift(1)
    df["_next_height"] = g["height"].shift(-1)

    # expected height from previous and next
    denom = df["_next_age"] - df["_prev_age"]

    df["_expected_height"] = np.where(
        df["_prev_height"].notna()
        & df["_next_height"].notna()
        & denom.notna()
        & (denom != 0),
        df["_prev_height"]
        + (
            (df["age"] - df["_prev_age"])
            / denom
        ) * (df["_next_height"] - df["_prev_height"]),
        np.nan
    )

    df["_height_deviation"] = (df["height"] - df["_expected_height"]).abs()

    abs_bad = impossible_height_mask(df)
    stats["height_impossible_absolute"] = int(abs_bad.sum())

    trajectory_repair = (
        df["_expected_height"].notna()
        & df["_height_deviation"].between(10, 20, inclusive="left")
    )

    trajectory_remove = (
        df["_expected_height"].notna()
        & (df["_height_deviation"] >= 20)
    )

    to_fix = abs_bad | trajectory_repair | trajectory_remove

    for idx in df.index[to_fix]:
        current_height = df.at[idx, "height"]
        expected = df.at[idx, "_expected_height"]
        prev_h = df.at[idx, "_prev_height"]
        next_h = df.at[idx, "_next_height"]

        new_height = np.nan
        method = None

        # If extremely far from trajectory, do not repair
        if pd.notna(expected) and abs(current_height - expected) >= 20:
            new_height = np.nan
            method = "missing"

        # Previous + next available: interpolate
        elif pd.notna(expected):
            new_height = expected
            method = "interpolation"

        # Only previous or only next available: use nearest valid height
        elif pd.notna(prev_h):
            new_height = prev_h
            method = "nearest"

        elif pd.notna(next_h):
            new_height = next_h
            method = "nearest"

        # No neighbour: model prediction
        else:
            pred = predict_height_from_model(df, idx, model_info)
            if pd.notna(pred):
                new_height = pred
                method = "model"
            else:
                new_height = np.nan
                method = "missing"

        # final plausibility check
        temp_age = df.at[idx, "age"]
        if pd.notna(new_height):
            test = pd.DataFrame({
                "age": [temp_age],
                "height": [new_height]
            })
            if (
                (new_height < 80)
                or (new_height > 250)
                or ((temp_age < 5) and (new_height > 140))
                or ((temp_age < 10) and (new_height > 180))
                or ((temp_age < 12) and (new_height > 190))
                or ((temp_age < 15) and (new_height > 210))
            ):
                new_height = np.nan
                method = "missing"

        df.at[idx, "height"] = round(new_height, 2) if pd.notna(new_height) else np.nan

        if method == "interpolation":
            stats["height_repaired_interpolation"] += 1
        elif method == "nearest":
            stats["height_repaired_nearest"] += 1
        elif method == "model":
            stats["height_repaired_model"] += 1
        elif method == "missing":
            stats["height_set_missing"] += 1

    # Recalculate BMI
    if {"height", "weight", "bmi"}.issubset(df.columns):
        valid_bmi = df["height"].notna() & df["weight"].notna() & (df["height"] > 0)
        stats["bmi_recalculated"] = int(valid_bmi.sum())

        df.loc[valid_bmi, "bmi"] = (
            df.loc[valid_bmi, "weight"]
            / ((df.loc[valid_bmi, "height"] / 100) ** 2)
        ).round(2)

        df.loc[df["height"].isna(), "bmi"] = np.nan

    df = df.drop(columns=[
        "_prev_age", "_next_age",
        "_prev_height", "_next_height",
        "_expected_height", "_height_deviation"
    ], errors="ignore")

    return df, stats


def clean_dataset(df):
    df = to_numeric_if_exists(df, [
        "age", "height", "weight", "bmi",
        "systolic_bp", "diastolic_bp",
        "smoking",
        "diabetes", "family_diabetes", "cvd", "family_cvd"
    ])

    summary = {}

    df, n_bp_removed = remove_bad_bp_rows(df)
    summary["rows_removed_bad_bp"] = n_bp_removed

    df, n_duplicates_removed = keep_most_complete_origin_id_age(df)
    summary["rows_removed_duplicate_origin_id_age"] = n_duplicates_removed

    df, height_stats = clean_heights(df)
    summary.update(height_stats)

    df, n_smoking_changed = fix_smoking_history(df)
    summary["smoking_never_after_smoker_changed_to_ex"] = n_smoking_changed

    df, binary_changes = fix_binary_health_histories(df)
    for k, v in binary_changes.items():
        summary[f"{k}_values_changed"] = v

    return df, summary
