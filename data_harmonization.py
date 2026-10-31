# smartchange data merging and harmonization
# this script combines preprocessed cohort files into the standardized smartchange format
# participant-level input data and generated results require authorized access

import os
import pandas as pd
import numpy as np

# define file paths
input_folder  = "RESULTS"
output_folder = "RESULTS/FULL-VFinal"
final_path = os.path.join(output_folder, "merged_dataset_vfinal_with_zscores.csv")
os.makedirs(output_folder, exist_ok=True)

cols_to_never_save = [
    "source_dataset",
    "resting_heart_rate_imputed",
    "donor_rhr_row_id",
    "vo2max_from_6mwt_rhr_donor",
    "sit_and_reach",
    "standing_lj",
    "row_id"
]

def save_without_hidden_cols(df, path, **kwargs):
    df_to_save = df.drop(columns=[c for c in cols_to_never_save if c in df.columns])
    df_to_save.to_csv(path, index=False, **kwargs)

# constants
PA_INTENSITY_ORIGINS = {
    "ACD.Si", "AFINE-TE", "ASK", "DADOS", "GNN", "HELENA", "ICAD",
    "LGS_Wave234", "LGS_Wave_234", "LGS_Wave_5", "LGS_Wave_6",
    "TMU", "MJ", "UK Biobank"
}

PA_INTENSITY_SOURCES = {
    "ACD_edited.csv", "ASK_edited.csv", "DADOS_edited.csv", "GNN_edited.csv",
    "HELENA_edited.csv", "ICAD_edited.csv", "Phase234_edited.csv",
    "Phase5_edited.csv", "Phase6_edited.csv", "TMU_edited.csv",
    "MJ_edited.csv", "uk_biobank_edited.csv"
}

YFS_SOURCE = "YFS_edited.csv"
YFS_ORIGIN = "YFS"

SPORTS_PA_TARGET_ORIGINS = {"LGS_Wave1A", "LGS_Wave_1A", "LGS_Wave1B", "LGS_Wave_1B"}
SPORTS_PA_TARGET_SOURCES = {"Phase1A_edited.csv", "Phase1B_edited.csv"}

SPARSE_ROW_PATTERNS = [
    ("origin", "id", "sex", "age", "height", "weight", "bmi"),
    ("origin", "id", "sex", "age"),
    ("origin", "id", "sex", "age", "height"),
    ("origin", "id", "sex", "age", "weight"),
    ("origin", "id", "sex", "age", "height", "weight", "bmi", "waist_circ"),
    ("origin", "id", "sex", "age", "height", "weight"),
    ("origin", "id", "sex", "age", "waist_circ"),
    ("origin", "id", "sex", "age", "height", "waist_circ"),
]

def cap_pa_components_with_mean(df, moderate_col="moderate_act", vigorous_col="vigorous_act",
                                threshold=7.0, upper=8.29):
    # replace values above the threshold with the mean of the specified range
    df = df.copy()
    replacement = (threshold + upper) / 2

    for col in [moderate_col, vigorous_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            mask = df[col] > threshold
            df.loc[mask, col] = replacement
            print(f"Capped {int(mask.sum())} rows of {col} > {threshold} to {replacement}")

    return df

# outlier detection
def detect_outliers_by_age(df, features, contamination=0.01, min_group_size=30):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df['outlier'] = False

    for age_int, group in df.groupby('age_int'):
        if len(group) < min_group_size:
            continue

        X = group[features].dropna()
        if len(X) < min_group_size:
            continue

        Xs = StandardScaler().fit_transform(X)
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        preds = model.fit_predict(Xs)
        out_idx = X.index[preds == -1]
        df.loc[out_idx, 'outlier'] = True

    report = df.groupby('age_int')['outlier'].agg(total='count', n_outliers='sum')
    report['pct_outliers'] = 100 * report.n_outliers / report.total
    print("\nOutlier Summary by age_int:\n", report)
    return df

# helpers
def sex_to_female_indicator(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        s = x.strip().lower()
        if s in ['m', 'male', 'man', 'boy']:
            return 0
        if s in ['f', 'female', 'woman', 'girl']:
            return 1

    try:
        v = float(x)
        if v == 0:
            return 0
        if v == 1:
            return 1
        if v == 2:
            return 1
    except:
        pass

    return np.nan

def normalize_sex_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "man", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "girl"}:
        return "Female"
    s = str(x).strip()
    return s if s != "" else np.nan

def harmonize_sex_within_id(df, id_col="id", sex_col="sex"):
    df = df.copy()
    if id_col not in df.columns or sex_col not in df.columns:
        return df

    df[sex_col] = df[sex_col].apply(normalize_sex_value)

    def choose_sex(series):
        vals = series.dropna()
        if vals.empty:
            return np.nan
        counts = vals.value_counts()
        max_count = counts.max()
        tied = counts[counts == max_count].index.tolist()
        if "Female" in tied:
            return "Female"
        return tied[0]

    chosen = df.groupby(id_col)[sex_col].agg(choose_sex)
    df[sex_col] = df[id_col].map(chosen)
    return df

def drop_sparse_rows_by_exact_presence(df, patterns, ignore_cols=("source_dataset",)):
    df = df.copy()
    consider_cols = [c for c in df.columns if c not in ignore_cols]
    present = df[consider_cols].notna()

    drop_mask = pd.Series(False, index=df.index)

    for patt in patterns:
        patt = [c for c in patt if c in consider_cols]
        other = [c for c in consider_cols if c not in patt]
        if len(patt) == 0:
            continue

        mask = present[patt].all(axis=1)
        if len(other) > 0:
            mask &= (~present[other]).all(axis=1)

        drop_mask |= mask

    dropped = int(drop_mask.sum())
    print(f"Dropped {dropped} rows matching sparse exact-presence patterns")
    return df.loc[~drop_mask].copy()

# pa / yfs helpers
def add_yfs_intensity_minutes(df, min_donors=100):
    """
    For YFS:
      intensity_minutes = mvpa * (1 + p)

    p is estimated from non-YFS donors with:
      p = vigorous_act / (moderate_act + vigorous_act)

    By vpa_indicator:
      1 -> p = 0
      2 -> p = median p from 9th to 77th percentile
      3 -> p = median p from 77th to 100th percentile

    Same sex, close age donors are preferred.
    """
    df = df.copy()

    if "intensity_minutes" not in df.columns:
        df["intensity_minutes"] = np.nan
    if "vpa_indicator" not in df.columns or "mvpa" not in df.columns:
        print("YFS intensity_minutes: required columns missing, skipping.")
        return df

    mod = pd.to_numeric(df.get("moderate_act"), errors="coerce")
    vig = pd.to_numeric(df.get("vigorous_act"), errors="coerce")
    donor_mvpa = mod + vig

    donor_source_mask = pd.Series(False, index=df.index)
    if "source_dataset" in df.columns:
        donor_source_mask = df["source_dataset"] != YFS_SOURCE

    donor_origin_mask = pd.Series(False, index=df.index)
    if "origin" in df.columns:
        donor_origin_mask = df["origin"] != YFS_ORIGIN

    donor_mask = (
        donor_source_mask &
        donor_origin_mask &
        df["sex"].notna() &
        df["age"].notna() &
        mod.notna() &
        vig.notna() &
        (donor_mvpa > 0)
    )

    donors = df.loc[donor_mask, ["sex", "age"]].copy()
    donors["vpa_prop"] = (vig[donor_mask] / donor_mvpa[donor_mask]).clip(lower=0, upper=1)

    yfs_mask = pd.Series(False, index=df.index)
    if "source_dataset" in df.columns:
        yfs_mask = df["source_dataset"].eq(YFS_SOURCE)
    elif "origin" in df.columns:
        yfs_mask = df["origin"].eq(YFS_ORIGIN)

    target_mask = (
        yfs_mask &
        df["sex"].notna() &
        df["age"].notna() &
        pd.to_numeric(df["mvpa"], errors="coerce").notna() &
        pd.to_numeric(df["vpa_indicator"], errors="coerce").isin([1, 2, 3])
    )

    targets = df.loc[target_mask, ["sex", "age", "mvpa", "vpa_indicator"]].copy()

    if donors.empty or targets.empty:
        print("YFS intensity_minutes: no eligible donors or targets.")
        return df

    filled = 0
    for idx, row in targets.iterrows():
        sex0 = row["sex"]
        age0 = float(row["age"])
        mvpa0 = float(row["mvpa"])
        ind = int(float(row["vpa_indicator"]))

        donors_sex = donors[donors["sex"] == sex0].copy()
        if donors_sex.empty:
            continue

        chosen = None
        for age_window in [2, 4, 6, 8, 10, 15, 100]:
            subset = donors_sex[(donors_sex["age"] - age0).abs() <= age_window].copy()
            if len(subset) >= min_donors or age_window == 100:
                chosen = subset if not subset.empty else donors_sex.copy()
                break

        if chosen is None or chosen.empty:
            continue

        pvals = chosen["vpa_prop"].dropna()
        if pvals.empty:
            continue

        if ind == 1:
            p_est = 0.0
        else:
            q09 = pvals.quantile(0.09)
            q77 = pvals.quantile(0.77)

            if ind == 2:
                band = pvals[(pvals >= q09) & (pvals <= q77)]
                p_est = float(band.median()) if not band.empty else float(pvals.median())
            else:  # ind == 3
                band = pvals[pvals >= q77]
                p_est = float(band.median()) if not band.empty else float(pvals.quantile(0.885))

        df.at[idx, "intensity_minutes"] = round(mvpa0 * (1 + p_est), 2)
        filled += 1

    print(f"YFS intensity_minutes: filled {filled} rows from mvpa + vpa_indicator donor correction")
    return df

def repair_mvpa_and_add_intensity_minutes(df):
    """
    1) Replace moderate_act and vigorous_act above 7 with the mean of [7, 8.29]
    2) Recompute mvpa = moderate_act + vigorous_act wherever at least one exists
       (missing component treated as 0)
    3) For target datasets with separate MPA/VPA:
       intensity_minutes = moderate_act + 2 * vigorous_act
    4) For YFS:
       intensity_minutes estimated from mvpa + vpa_indicator
    """
    df = df.copy()

    for col in ["moderate_act", "vigorous_act", "mvpa", "intensity_minutes"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = cap_pa_components_with_mean(
        df,
        moderate_col="moderate_act",
        vigorous_col="vigorous_act",
        threshold=7.0,
        upper=8.29
    )

    mod = pd.to_numeric(df["moderate_act"], errors="coerce").astype("float64")
    vig = pd.to_numeric(df["vigorous_act"], errors="coerce").astype("float64")

    has_any = mod.notna() | vig.notna()

    # recompute mvpa everywhere from components
    df.loc[has_any, "mvpa"] = mod.fillna(0) + vig.fillna(0)

    source_mask = pd.Series(False, index=df.index)
    if "source_dataset" in df.columns:
        source_mask = df["source_dataset"].isin(PA_INTENSITY_SOURCES)

    origin_mask = pd.Series(False, index=df.index)
    if "origin" in df.columns:
        origin_mask = df["origin"].isin(PA_INTENSITY_ORIGINS)

    target_mask = has_any & (source_mask | origin_mask)
    df.loc[target_mask, "intensity_minutes"] = mod.fillna(0) + 2 * vig.fillna(0)

    # YFS special handling
    df = add_yfs_intensity_minutes(df)

    return df

# fallback z-score helpers
def _choose_reference_stats_with_fallback(
    df, raw_col, sex_val, age_bin_val,
    sex_col="sex", age_bin_col="age_bin",
    min_n=30, max_fallback_years=6
):
    pool = df[
        (df[sex_col] == sex_val) &
        df[age_bin_col].notna() &
        df[raw_col].notna()
    ].copy()

    if pool.empty:
        return None

    candidates = []
    for cand_bin in sorted(pool[age_bin_col].dropna().unique()):
        diff_years = abs(float(cand_bin) - float(age_bin_val))
        if diff_years > max_fallback_years:
            continue

        vals = pool.loc[pool[age_bin_col] == cand_bin, raw_col]
        n = len(vals)
        if n < min_n:
            continue

        sd = vals.std(ddof=0)
        if sd < 1e-6:
            continue

        mu = vals.mean()
        candidates.append((diff_years, cand_bin, mu, sd, n))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[4]))
    _, chosen_bin, mu, sd, n = candidates[0]

    return {"chosen_bin": chosen_bin, "mu": mu, "sd": sd, "n": n}

def add_groupwise_zscore_with_fallback(
    df, raw_col, z_col,
    age_col='age', sex_col='sex',
    bin_size=2, min_n=30, max_fallback_years=6
):
    df = df.copy()
    df[z_col] = np.nan

    if raw_col not in df.columns:
        return df

    age_nonmissing = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age_nonmissing.empty:
        return df

    max_age = int(age_nonmissing.max())
    bins = np.arange(0, max_age + bin_size, bin_size)

    df["age_bin"] = pd.cut(
        pd.to_numeric(df[age_col], errors="coerce"),
        bins=bins,
        labels=bins[:-1],
        right=False
    ).astype(float)

    for sex_val in df[sex_col].dropna().unique():
        age_bins = sorted(df.loc[
            (df[sex_col] == sex_val) &
            df["age_bin"].notna() &
            df[raw_col].notna(),
            "age_bin"
        ].unique())

        for age_bin_val in age_bins:
            mask = (
                (df[sex_col] == sex_val) &
                (df["age_bin"] == age_bin_val) &
                df[raw_col].notna()
            )
            if not mask.any():
                continue

            ref = _choose_reference_stats_with_fallback(
                df, raw_col, sex_val, age_bin_val,
                sex_col=sex_col,
                age_bin_col="age_bin",
                min_n=min_n,
                max_fallback_years=max_fallback_years
            )
            if ref is None:
                continue

            df.loc[mask, z_col] = (df.loc[mask, raw_col] - ref["mu"]) / ref["sd"]

    df.drop(columns=["age_bin"], inplace=True)
    df[z_col] = df[z_col].round(2)
    return df

def add_groupwise_harmonized_with_fallback(
    df, z_col, raw_col, out_col,
    age_col='age', sex_col='sex',
    bin_size=2, min_n=30, max_fallback_years=6
):
    df = df.copy()
    df[out_col] = np.nan

    if z_col not in df.columns or raw_col not in df.columns:
        return df

    age_nonmissing = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age_nonmissing.empty:
        return df

    max_age = int(age_nonmissing.max())
    bins = np.arange(0, max_age + bin_size, bin_size)

    df["age_bin"] = pd.cut(
        pd.to_numeric(df[age_col], errors="coerce"),
        bins=bins,
        labels=bins[:-1],
        right=False
    ).astype(float)

    for sex_val in df[sex_col].dropna().unique():
        age_bins = sorted(df.loc[
            (df[sex_col] == sex_val) &
            df["age_bin"].notna() &
            df[z_col].notna(),
            "age_bin"
        ].unique())

        for age_bin_val in age_bins:
            apply_mask = (
                (df[sex_col] == sex_val) &
                (df["age_bin"] == age_bin_val) &
                df[z_col].notna()
            )
            if not apply_mask.any():
                continue

            ref = _choose_reference_stats_with_fallback(
                df, raw_col, sex_val, age_bin_val,
                sex_col=sex_col,
                age_bin_col="age_bin",
                min_n=min_n,
                max_fallback_years=max_fallback_years
            )
            if ref is None:
                continue

            df.loc[apply_mask, out_col] = ref["mu"] + df.loc[apply_mask, z_col] * ref["sd"]

    df.drop(columns=["age_bin"], inplace=True)
    df[out_col] = df[out_col].round(2).clip(lower=0)
    return df

# sports -> pa imputation
def impute_harmonized_pa_from_sports_participation(
    df,
    target_origins=SPORTS_PA_TARGET_ORIGINS,
    target_sources=SPORTS_PA_TARGET_SOURCES,
    age_penalty=0.25,
    proxy_penalty=2.0
):
    df = df.copy()

    if "harmonized_pa" not in df.columns:
        df["harmonized_pa"] = np.nan

    if "z_score_sports_participation" not in df.columns or "z_score_pa" not in df.columns:
        print("sports->PA imputation: required columns not found, skipping.")
        return df, 0

    source_mask = pd.Series(False, index=df.index)
    if "source_dataset" in df.columns:
        source_mask = df["source_dataset"].isin(target_sources)

    origin_mask = pd.Series(False, index=df.index)
    if "origin" in df.columns:
        origin_mask = df["origin"].isin(target_origins)

    donor_mask = (
        df["sex"].notna() &
        df["age"].notna() &
        df["z_score_pa"].notna() &
        df["harmonized_pa"].notna()
    )

    target_mask = (
        (source_mask | origin_mask) &
        df["sex"].notna() &
        df["age"].notna() &
        df["z_score_sports_participation"].notna() &
        df["harmonized_pa"].isna()
    )

    donors = df.loc[donor_mask, ["row_id", "sex", "age", "z_score_pa", "harmonized_pa"]].copy()
    targets = df.loc[target_mask, ["row_id", "sex", "age", "z_score_sports_participation"]].copy()

    if donors.empty or targets.empty:
        print("sports->PA imputation: no eligible donors or targets.")
        return df, 0

    filled = 0
    for sex_val in targets["sex"].dropna().unique():
        donors_sex = donors[donors["sex"] == sex_val].copy()
        targets_sex = targets[targets["sex"] == sex_val].copy()

        if donors_sex.empty or targets_sex.empty:
            continue

        mu_age = donors_sex["age"].mean()
        sd_age = donors_sex["age"].std(ddof=0)
        mu_pr  = donors_sex["z_score_pa"].mean()
        sd_pr  = donors_sex["z_score_pa"].std(ddof=0)

        sd_age = sd_age if sd_age > 1e-8 else 1.0
        sd_pr  = sd_pr  if sd_pr  > 1e-8 else 1.0

        donors_sex["age_z"] = (donors_sex["age"] - mu_age) / sd_age
        donors_sex["pr_z"]  = (donors_sex["z_score_pa"] - mu_pr) / sd_pr

        for idx in targets_sex.index:
            row_id = targets_sex.at[idx, "row_id"]
            age_z = (targets_sex.at[idx, "age"] - mu_age) / sd_age
            pr_z  = (targets_sex.at[idx, "z_score_sports_participation"] - mu_pr) / sd_pr

            donors_sex["dist"] = (
                proxy_penalty * (donors_sex["pr_z"] - pr_z).abs() +
                age_penalty   * (donors_sex["age_z"] - age_z).abs()
            )

            best_idx = donors_sex["dist"].idxmin()
            best_val = donors_sex.at[best_idx, "harmonized_pa"]

            if pd.notna(best_val):
                df.loc[df["row_id"] == row_id, "harmonized_pa"] = float(best_val)
                if "z_score_pa" in df.columns:
                    df.loc[df["row_id"] == row_id, "z_score_pa"] = float(targets_sex.at[idx, "z_score_sports_participation"])
                filled += 1

    print(f"sports->PA imputation: filled harmonized_pa for {filled} rows")
    return df, filled

# 6mwt -> vo2max helper
def add_predicted_vo2max_from_6mwt(
    df,
    donor_sources,
    target_sources,
    age_penalty=1.0,
    weight_penalty=1.0,
    height_penalty=1.0
):
    df = df.copy()

    if 'resting_heart_rate_imputed' not in df.columns:
        df['resting_heart_rate_imputed'] = False
    if 'donor_rhr_row_id' not in df.columns:
        df['donor_rhr_row_id'] = np.nan
    if 'vo2max_from_6mwt_rhr_donor' not in df.columns:
        df['vo2max_from_6mwt_rhr_donor'] = False

    df['sex_female'] = df['sex'].apply(sex_to_female_indicator)

    donor_mask = (
        df['source_dataset'].isin(donor_sources) &
        df['age'].notna() &
        df['weight'].notna() &
        df['height'].notna() &
        df['resting_heart_rate'].notna() &
        df['sex_female'].notna()
    )

    target_mask = (
        df['source_dataset'].isin(target_sources) &
        df['6_min_walking_test'].notna() &
        df['vo2_max'].isna() &
        df['age'].notna() &
        df['weight'].notna() &
        df['height'].notna() &
        df['sex_female'].notna()
    )

    donors = df.loc[donor_mask, [
        'row_id', 'age', 'weight', 'height', 'resting_heart_rate', 'sex_female'
    ]].copy()

    targets = df.loc[target_mask, [
        'row_id', 'age', 'weight', 'height', 'sex_female', '6_min_walking_test'
    ]].copy()

    print(f"\nEligible donor rows for RHR matching: {len(donors)}")
    print(f"Eligible target rows for VO2max prediction: {len(targets)}")

    matched_count = 0

    for sex_val in [0, 1]:
        donors_sex = donors[donors['sex_female'] == sex_val].copy()
        targets_sex = targets[targets['sex_female'] == sex_val].copy()

        if donors_sex.empty or targets_sex.empty:
            continue

        mu_age = donors_sex['age'].mean()
        sd_age = donors_sex['age'].std(ddof=0)
        mu_wt  = donors_sex['weight'].mean()
        sd_wt  = donors_sex['weight'].std(ddof=0)
        mu_ht  = donors_sex['height'].mean()
        sd_ht  = donors_sex['height'].std(ddof=0)

        sd_age = sd_age if sd_age > 1e-8 else 1.0
        sd_wt  = sd_wt if sd_wt > 1e-8 else 1.0
        sd_ht  = sd_ht if sd_ht > 1e-8 else 1.0

        donors_sex['age_z'] = (donors_sex['age'] - mu_age) / sd_age
        donors_sex['wt_z']  = (donors_sex['weight'] - mu_wt) / sd_wt
        donors_sex['ht_z']  = (donors_sex['height'] - mu_ht) / sd_ht

        for idx in targets_sex.index:
            row_id = targets_sex.at[idx, 'row_id']
            age_z = (targets_sex.at[idx, 'age'] - mu_age) / sd_age
            wt_z  = (targets_sex.at[idx, 'weight'] - mu_wt) / sd_wt
            ht_z  = (targets_sex.at[idx, 'height'] - mu_ht) / sd_ht

            donors_sex['dist'] = (
                age_penalty    * (donors_sex['age_z'] - age_z).abs() +
                weight_penalty * (donors_sex['wt_z']  - wt_z).abs() +
                height_penalty * (donors_sex['ht_z']  - ht_z).abs()
            )

            best_idx = donors_sex['dist'].idxmin()
            best_rhr = donors_sex.at[best_idx, 'resting_heart_rate']
            best_donor_row_id = donors_sex.at[best_idx, 'row_id']

            if pd.notna(best_rhr):
                df.loc[df['row_id'] == row_id, 'resting_heart_rate'] = best_rhr
                df.loc[df['row_id'] == row_id, 'resting_heart_rate_imputed'] = True
                df.loc[df['row_id'] == row_id, 'donor_rhr_row_id'] = best_donor_row_id
                matched_count += 1

    fill_mask = (
        df['source_dataset'].isin(target_sources) &
        df['6_min_walking_test'].notna() &
        df['vo2_max'].isna() &
        df['age'].notna() &
        df['weight'].notna() &
        df['resting_heart_rate'].notna() &
        df['sex_female'].notna()
    )

    if 'vo2_max' in df.columns:
        df['vo2_max'] = df['vo2_max'].astype('float64')

    vo2_pred = (
        70.161
        + 0.023 * df.loc[fill_mask, '6_min_walking_test'].astype('float64')
        - 0.276 * df.loc[fill_mask, 'weight'].astype('float64')
        - 6.79  * df.loc[fill_mask, 'sex_female'].astype('float64')
        - 0.193 * df.loc[fill_mask, 'resting_heart_rate'].astype('float64')
        - 0.191 * df.loc[fill_mask, 'age'].astype('float64')
    ).round(2)

    df.loc[fill_mask, 'vo2_max'] = vo2_pred
    df.loc[fill_mask, 'vo2max_from_6mwt_rhr_donor'] = True

    print(f"Matched resting heart rate for {matched_count} target rows")
    print(f"Filled vo2_max from 6MWT equation for {int(vo2_pred.notna().sum())} target rows")

    print("\nVO2max fills by source_dataset:")
    print(df.loc[df['vo2max_from_6mwt_rhr_donor']].groupby('source_dataset').size())

    df.drop(columns=['sex_female'], inplace=True)

    return df

# other z-score helpers
def ensure_groupwise_zscore(df, raw_col, z_col, age_col='age', sex_col='sex', bin_size=2, min_n=30):
    df = df.copy()
    df[z_col] = np.nan

    if raw_col not in df.columns:
        return df

    age_nonmissing = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age_nonmissing.empty:
        return df

    max_age = int(age_nonmissing.max())
    bins = np.arange(0, max_age + bin_size, bin_size)

    df['age_bin'] = pd.cut(
        pd.to_numeric(df[age_col], errors="coerce"),
        bins=bins,
        labels=bins[:-1],
        right=False
    ).astype(float)

    for sex in df[sex_col].dropna().unique():
        for age_bin in df['age_bin'].dropna().unique():
            mask = (
                (df[sex_col] == sex) &
                (df['age_bin'] == age_bin) &
                df[raw_col].notna()
            )
            if mask.sum() < min_n:
                continue

            vals = df.loc[mask, raw_col]
            mu, sd = vals.mean(), vals.std(ddof=0)
            if sd < 1e-6:
                continue

            df.loc[mask, z_col] = (vals - mu) / sd

    df.drop(columns='age_bin', inplace=True)
    df[z_col] = df[z_col].round(2)
    return df

def add_groupwise_harmonized(df, z_col, raw_col, out_col, age_col='age', sex_col='sex', bin_size=2, min_n=30):
    df = df.copy()
    df[out_col] = np.nan

    if z_col not in df.columns or raw_col not in df.columns:
        return df

    age_nonmissing = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age_nonmissing.empty:
        return df

    max_age = int(age_nonmissing.max())
    bins = np.arange(0, max_age + bin_size, bin_size)

    df['age_bin'] = pd.cut(
        pd.to_numeric(df[age_col], errors="coerce"),
        bins=bins,
        labels=bins[:-1],
        right=False
    ).astype(float)

    for sex in df[sex_col].dropna().unique():
        for age_bin in df['age_bin'].dropna().unique():
            ref = (
                (df[sex_col] == sex) &
                (df['age_bin'] == age_bin) &
                df[raw_col].notna()
            )
            if ref.sum() < min_n:
                continue

            mu = df.loc[ref, raw_col].mean()
            sd = df.loc[ref, raw_col].std(ddof=0)
            if sd < 1e-6:
                continue

            apply_ = (
                (df[sex_col] == sex) &
                df['age_bin'].eq(age_bin) &
                df[z_col].notna()
            )
            df.loc[apply_, out_col] = mu + df.loc[apply_, z_col] * sd

    df.drop(columns='age_bin', inplace=True)
    df[out_col] = df[out_col].round(2).clip(lower=0)
    return df

def fill_missing_harmonized_from_proxy_z(
    df,
    target_col,
    proxy_z_col,
    reference_raw_col,
    label,
    bin_size=2,
    min_n=30,
    max_fallback_years=6
):
    """
    Fill only missing values of target_col by projecting a proxy z-score
    onto the age/sex-specific distribution of reference_raw_col.
    Existing observed/formula-derived target values are preserved.
    """
    df = df.copy()

    if proxy_z_col not in df.columns or reference_raw_col not in df.columns:
        print(f"{label}: required columns missing, skipping.")
        return df, 0

    if target_col not in df.columns:
        df[target_col] = np.nan

    temp_col = f"__{target_col}_from_{proxy_z_col}"

    projected = add_groupwise_harmonized_with_fallback(
        df,
        z_col=proxy_z_col,
        raw_col=reference_raw_col,
        out_col=temp_col,
        bin_size=bin_size,
        min_n=min_n,
        max_fallback_years=max_fallback_years
    )

    mask = df[target_col].isna() & projected[temp_col].notna()
    n_filled = int(mask.sum())

    if n_filled > 0:
        df.loc[mask, target_col] = projected.loc[mask, temp_col]

    print(f"{label}: filled {n_filled} missing values of {target_col} from {proxy_z_col}")
    return df, n_filled

def add_direct_and_proxy_pa_variables(df):
    """
    Create two PA outputs:
      - harmonized_pa_mvpa: direct MVPA when available; otherwise proxy-z projected to MVPA scale.
      - harmonized_pa_intensity: direct intensity_minutes when available; otherwise proxy-z projected to intensity scale.

    This avoids converting observed MVPA/intensity into z-scores and back again.
    """
    df = df.copy()

    for col in ["mvpa", "intensity_minutes"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["harmonized_pa_mvpa"] = df["mvpa"]
    df["harmonized_pa_intensity"] = df["intensity_minutes"]

    proxy_order = [
        ("z_score_pa_fallback", "YFS PA fallback z-score"),
        ("z_score_sports_participation", "sports participation z-score"),
    ]

    for proxy_col, label in proxy_order:
        if proxy_col not in df.columns:
            continue

        df, _ = fill_missing_harmonized_from_proxy_z(
            df,
            target_col="harmonized_pa_mvpa",
            proxy_z_col=proxy_col,
            reference_raw_col="mvpa",
            label=f"{label} -> harmonized_pa_mvpa"
        )

        df, _ = fill_missing_harmonized_from_proxy_z(
            df,
            target_col="harmonized_pa_intensity",
            proxy_z_col=proxy_col,
            reference_raw_col="intensity_minutes",
            label=f"{label} -> harmonized_pa_intensity"
        )

    return df

def add_direct_proxy_and_average_fitness_variables(df):
    """
    Create:
      - harmonized_fitness: direct/formula-derived shuttle_run_speed when available;
        otherwise filled from proxy z-scores.
      - harmonized_fitness_average: average of all available relevant fitness z-scores,
        projected to the shuttle_run_speed scale.
    """
    df = df.copy()

    if "shuttle_run_speed" not in df.columns:
        df["shuttle_run_speed"] = np.nan
    df["shuttle_run_speed"] = pd.to_numeric(df["shuttle_run_speed"], errors="coerce")

    # Main variable: preserve original/formula-derived shuttle_run_speed.
    df["harmonized_fitness"] = df["shuttle_run_speed"]

    for proxy_col, label in [
        ("z_score_vo2_max", "VO2max z-score"),
        ("z_score_andersen", "Andersen z-score"),
        ("z_score_6_min", "6-minute walk z-score"),
    ]:
        if proxy_col not in df.columns:
            continue
        df, _ = fill_missing_harmonized_from_proxy_z(
            df,
            target_col="harmonized_fitness",
            proxy_z_col=proxy_col,
            reference_raw_col="shuttle_run_speed",
            label=f"{label} -> harmonized_fitness"
        )

    # Secondary variable
    fitness_z_cols = [
        c for c in [
            "z_score_fitness",
            "z_score_vo2_max",
            "z_score_andersen",
            "z_score_6_min",
        ]
        if c in df.columns
    ]

    if fitness_z_cols:
        df["z_score_fitness_average"] = df[fitness_z_cols].mean(axis=1, skipna=True).round(2)
        df = add_groupwise_harmonized(
            df,
            z_col="z_score_fitness_average",
            raw_col="shuttle_run_speed",
            out_col="harmonized_fitness_average",
            bin_size=2,
            min_n=30
        )
    else:
        df["z_score_fitness_average"] = np.nan
        df["harmonized_fitness_average"] = np.nan

    return df

def impute_shuttle_from_proxy_zscore(
    df,
    proxy_z_col,
    proxy_name_for_print,
    age_penalty=1.0,
    height_penalty=1.0,
    weight_penalty=1.0,
    proxy_penalty=2.0
):
    df = df.copy()

    if proxy_z_col not in df.columns:
        print(f"{proxy_name_for_print}: proxy column {proxy_z_col} not found, skipping.")
        return df, 0, 0

    if 'shuttle_run_speed' in df.columns:
        df['shuttle_run_speed'] = df['shuttle_run_speed'].astype('float64')
    if 'shuttle_run_laps' in df.columns:
        df['shuttle_run_laps'] = df['shuttle_run_laps'].astype('float64')

    donor_mask = (
        df['sex'].notna() &
        df['age'].notna() &
        df['height'].notna() &
        df['weight'].notna() &
        df[proxy_z_col].notna() &
        df['shuttle_run_speed'].notna()
    )

    target_mask = (
        df['sex'].notna() &
        df['age'].notna() &
        df['height'].notna() &
        df['weight'].notna() &
        df[proxy_z_col].notna() &
        df['shuttle_run_speed'].isna()
    )

    donors = df.loc[donor_mask, [
        'row_id', 'sex', 'age', 'height', 'weight',
        proxy_z_col, 'shuttle_run_speed', 'shuttle_run_laps'
    ]].copy()

    targets = df.loc[target_mask, [
        'row_id', 'sex', 'age', 'height', 'weight',
        proxy_z_col
    ]].copy()

    if donors.empty or targets.empty:
        print(f"{proxy_name_for_print}: no eligible donors or targets.")
        return df, 0, int(target_mask.sum())

    filled_speed = 0
    filled_laps = 0

    for sex_val in df['sex'].dropna().unique():
        donors_sex = donors[donors['sex'] == sex_val].copy()
        targets_sex = targets[targets['sex'] == sex_val].copy()

        if donors_sex.empty or targets_sex.empty:
            continue

        mu_age = donors_sex['age'].mean()
        sd_age = donors_sex['age'].std(ddof=0)
        mu_ht  = donors_sex['height'].mean()
        sd_ht  = donors_sex['height'].std(ddof=0)
        mu_wt  = donors_sex['weight'].mean()
        sd_wt  = donors_sex['weight'].std(ddof=0)
        mu_pr  = donors_sex[proxy_z_col].mean()
        sd_pr  = donors_sex[proxy_z_col].std(ddof=0)

        sd_age = sd_age if sd_age > 1e-8 else 1.0
        sd_ht  = sd_ht  if sd_ht  > 1e-8 else 1.0
        sd_wt  = sd_wt  if sd_wt  > 1e-8 else 1.0
        sd_pr  = sd_pr  if sd_pr  > 1e-8 else 1.0

        donors_sex['age_z'] = (donors_sex['age'] - mu_age) / sd_age
        donors_sex['ht_z']  = (donors_sex['height'] - mu_ht) / sd_ht
        donors_sex['wt_z']  = (donors_sex['weight'] - mu_wt) / sd_wt
        donors_sex['pr_z']  = (donors_sex[proxy_z_col] - mu_pr) / sd_pr

        for idx in targets_sex.index:
            row_id = targets_sex.at[idx, 'row_id']

            age_z = (targets_sex.at[idx, 'age'] - mu_age) / sd_age
            ht_z  = (targets_sex.at[idx, 'height'] - mu_ht) / sd_ht
            wt_z  = (targets_sex.at[idx, 'weight'] - mu_wt) / sd_wt
            pr_z  = (targets_sex.at[idx, proxy_z_col] - mu_pr) / sd_pr

            donors_sex['dist'] = (
                proxy_penalty  * (donors_sex['pr_z']  - pr_z).abs() +
                age_penalty    * (donors_sex['age_z'] - age_z).abs() +
                height_penalty * (donors_sex['ht_z']  - ht_z).abs() +
                weight_penalty * (donors_sex['wt_z']  - wt_z).abs()
            )

            best_idx = donors_sex['dist'].idxmin()
            best_speed = donors_sex.at[best_idx, 'shuttle_run_speed']
            best_laps  = donors_sex.at[best_idx, 'shuttle_run_laps']

            if pd.notna(best_speed):
                df.loc[df['row_id'] == row_id, 'shuttle_run_speed'] = float(best_speed)
                filled_speed += 1

                if 'shuttle_run_laps' in df.columns and pd.notna(best_laps):
                    current_laps = df.loc[df['row_id'] == row_id, 'shuttle_run_laps']
                    if current_laps.isna().all():
                        df.loc[df['row_id'] == row_id, 'shuttle_run_laps'] = float(best_laps)
                        filled_laps += 1

    remaining_missing = int(df['shuttle_run_speed'].isna().sum())

    print(f"{proxy_name_for_print}: filled shuttle_run_speed for {filled_speed} rows")
    print(f"{proxy_name_for_print}: filled shuttle_run_laps  for {filled_laps} rows")

    return df, filled_speed, remaining_missing

# load & merge datasets
print("Loading and merging datasets...")

files = [
    "Phase1A_edited.csv","Phase1B_edited.csv","Phase234_edited.csv",
    "Phase5_edited.csv","Phase6_edited.csv","ACD_edited.csv",
    "HELENA_edited.csv","SLOfit_lifelong_edited.csv","SLOfit_Students_edited.csv",
    "UPorto_edited.csv","YFS_edited.csv","DADOS_edited.csv",
    "ASK_edited.csv","GNN_edited.csv","TMU_edited.csv","ICAD_edited.csv",
    "MJ_edited.csv","uk_biobank_edited.csv"
]

dfs = []

for fn in files:
    in_path = os.path.join(input_folder, fn)
    df = pd.read_csv(in_path)
    print("  loaded", fn)

    df['source_dataset'] = fn

    for c in df.select_dtypes(include='float64').columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include='int64').columns:
        df[c] = pd.to_numeric(df[c], downcast='unsigned')

    if fn == "Phase6_edited.csv" and {"vegetables","fruits","dairy"}.issubset(df.columns):
        df[["vegetables","fruits","dairy"]] = np.nan

    if fn == "TMU_edited.csv":
        keep_cols = [
            "origin","id","sex","age",
            "resting_heart_rate","systolic_bp","diastolic_bp",
            "height","weight","diabetes","bmi",
            "source_dataset"
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

    dfs.append(df)

for d in dfs:
    for col in ["dairy", "sugar", "protein", "fat"]:
        if col in d.columns:
            d.drop(columns=[col], inplace=True)

merged = pd.concat(dfs, ignore_index=True)

merged = harmonize_sex_within_id(merged, id_col="id", sex_col="sex")

before_drop = len(merged)
merged = merged[merged["sex"].notna() & merged["age"].notna()].copy()
print(f"Dropped {before_drop - len(merged)} rows with missing sex or age")

df_post = merged.copy()
df_post['row_id'] = df_post.index

# repair mvpa and create intensity_minutes
dp = df_post.copy()
dp = repair_mvpa_and_add_intensity_minutes(dp)

# outlier detection
dp['age_int'] = dp.age.fillna(-1).astype(int).clip(lower=0)
exclude = {
    'origin','id','sex','diabetes','family_diabetes','medication',
    'bedroom','family_vacations','family_computers','internet',
    'mother_occup','father_occup','income','stopped_studies',
    'started_job','years_unemployed','live_own','house_owner','education',
    'source_dataset'
}
features = [
    c for c in dp.select_dtypes(include=np.number).columns
    if (c not in exclude and not c.startswith('z_score') and c not in ['row_id','outlier','age_int'])
]

df_flagged = detect_outliers_by_age(dp, features)

mask = df_flagged.outlier & (df_flagged.height.isna() | df_flagged.weight.isna())
df_flagged.loc[mask, 'bmi'] = np.nan

cleaned = df_flagged.loc[~df_flagged.outlier].drop(columns=['outlier','age_int']).copy()

# rows with only a few low-value original variables would require extensive imputation, so they are excluded
cleaned = drop_sparse_rows_by_exact_presence(cleaned, SPARSE_ROW_PATTERNS, ignore_cols=("source_dataset",))

# impute resting hr for target datasets and predict vo2max from 6mwt
donor_sources = {
    "Phase1A_edited.csv",
    "Phase1B_edited.csv",
    "Phase234_edited.csv",
    "TMU_edited.csv",
    "MJ_edited.csv",
    "uk_biobank_edited.csv"
}

target_sources = {
    "SLOfit_lifelong_edited.csv",
    "Phase6_edited.csv"
}

cleaned = add_predicted_vo2max_from_6mwt(
    cleaned,
    donor_sources=donor_sources,
    target_sources=target_sources,
    age_penalty=1.0,
    weight_penalty=1.0,
    height_penalty=1.0
)

# compute z-scores on cleaned
dfz = cleaned.copy()

# preliminary z-scores
if 'vo2_max' in dfz.columns:
    dfz = ensure_groupwise_zscore(dfz, 'vo2_max', 'z_score_vo2_max', bin_size=2, min_n=30)

if 'andersen_meters' in dfz.columns:
    dfz = ensure_groupwise_zscore(dfz, 'andersen_meters', 'z_score_andersen', bin_size=2, min_n=30)

if '6_min_walking_test' in dfz.columns:
    dfz = ensure_groupwise_zscore(dfz, '6_min_walking_test', 'z_score_6_min', bin_size=2, min_n=30)

# pa now based on intensity_minutes
if 'intensity_minutes' in dfz.columns:
    dfz = add_groupwise_zscore_with_fallback(
        dfz, 'intensity_minutes', 'z_score_pa',
        bin_size=2, min_n=30, max_fallback_years=6
    )

if 'sports_participation' in dfz.columns:
    dfz = ensure_groupwise_zscore(dfz, 'sports_participation', 'z_score_sports_participation', bin_size=2, min_n=30)

# impute shuttle_run_speed before z_score_fitness
initial_missing_shuttle = int(dfz['shuttle_run_speed'].isna().sum())
print(f"\nInitially missing shuttle_run_speed: {initial_missing_shuttle}")

filled_total = 0

if 'z_score_vo2_max' in dfz.columns:
    dfz, n_filled_vo2, _ = impute_shuttle_from_proxy_zscore(
        dfz,
        proxy_z_col='z_score_vo2_max',
        proxy_name_for_print='vo2_max-based matching',
        age_penalty=1.0,
        height_penalty=1.0,
        weight_penalty=1.0,
        proxy_penalty=2.0
    )
    filled_total += n_filled_vo2

if 'z_score_andersen' in dfz.columns:
    dfz, n_filled_and, _ = impute_shuttle_from_proxy_zscore(
        dfz,
        proxy_z_col='z_score_andersen',
        proxy_name_for_print='andersen-based matching',
        age_penalty=1.0,
        height_penalty=1.0,
        weight_penalty=1.0,
        proxy_penalty=2.0
    )
    filled_total += n_filled_and

final_missing_shuttle = int(dfz['shuttle_run_speed'].isna().sum())

print("\nShuttle run imputation summary:")
print(f"  Initially missing shuttle_run_speed: {initial_missing_shuttle}")
print(f"  Total assigned shuttle_run_speed:    {filled_total}")
print(f"  Still missing shuttle_run_speed:     {final_missing_shuttle}")

# compute z-scores and final pa/fitness harmonized variables
# z_score_fitness here is the shuttle-run-speed z-score only.
# it is not overwritten by an average of all fitness z-scores.
if 'shuttle_run_speed' in dfz.columns:
    dfz = ensure_groupwise_zscore(dfz, 'shuttle_run_speed', 'z_score_fitness', bin_size=2, min_n=30)

# main fitness output:
# 1) use observed/formula-derived shuttle_run_speed directly
# 2) if missing, fill from proxy z-scores projected to shuttle_run_speed scale
# secondary fitness output:
dfz = add_direct_proxy_and_average_fitness_variables(dfz)

# pa outputs:
dfz = add_direct_and_proxy_pa_variables(dfz)

# fitness bounds
for fit_col in ["harmonized_fitness", "harmonized_fitness_average"]:
    if fit_col in dfz.columns:
        dfz.loc[
            (pd.to_numeric(dfz[fit_col], errors="coerce") < 0) |
            (pd.to_numeric(dfz[fit_col], errors="coerce") > 16.5),
            fit_col
        ] = np.nan

# requested pa invalidation rule
pa_cols = [
    "mvpa", "harmonized_pa_mvpa", "harmonized_pa_intensity",
    "moderate_act", "vigorous_act", "z_score_pa", "intensity_minutes"
]
pa_cols = [c for c in pa_cols if c in dfz.columns]

bad_pa = pd.Series(False, index=dfz.index)
if "mvpa" in dfz.columns:
    bad_pa |= pd.to_numeric(dfz["mvpa"], errors="coerce") > 14
if "harmonized_pa_mvpa" in dfz.columns:
    bad_pa |= pd.to_numeric(dfz["harmonized_pa_mvpa"], errors="coerce") > 14
if "harmonized_pa_intensity" in dfz.columns:
    bad_pa |= pd.to_numeric(dfz["harmonized_pa_intensity"], errors="coerce") > 14

dfz.loc[bad_pa, pa_cols] = np.nan

for pa_col in ["harmonized_pa_mvpa", "harmonized_pa_intensity"]:
    if pa_col in dfz.columns:
        dfz[pa_col] = pd.to_numeric(dfz[pa_col], errors="coerce").round(2).clip(lower=0)

for fit_col in ["harmonized_fitness", "harmonized_fitness_average"]:
    if fit_col in dfz.columns:
        dfz[fit_col] = pd.to_numeric(dfz[fit_col], errors="coerce").round(2).clip(lower=0, upper=16.5)

# fitness proxy back-fill handled above
# harmonized_fitness is already filled using shuttle_run_speed first,
# then proxy z-scores only where shuttle_run_speed is missing.

# soc-dem specials
specials = [
    ("stopped_studies", 98, "z_score_studies"),
    ("live_own", 99, "z_score_own"),
    ("house_owner", 99, "z_score_house"),
]

for raw_col, bad_val, z_col in specials:
    if raw_col in dfz.columns:
        mask = dfz[raw_col] == bad_val
        n = int(mask.sum())
        if n > 0:
            print(f"Blanking {n} rows of {raw_col}=={bad_val} -> NaN, and setting {z_col} to NaN")
            dfz.loc[mask, raw_col] = np.nan
            if z_col in dfz.columns:
                dfz.loc[mask, z_col] = np.nan

# re-standardize affluence score

if "z_score_affluence" in dfz.columns:

    mu = dfz["z_score_affluence"].mean(skipna=True)
    sd = dfz["z_score_affluence"].std(skipna=True)

    dfz["z_score_affluence"] = (
        (dfz["z_score_affluence"] - mu) / sd
    ).round(2)

print("Final z_score_affluence mean:", dfz["z_score_affluence"].mean())
print("Final z_score_affluence SD:", dfz["z_score_affluence"].std())
# reorder columns & save final dataset
desired_columns = [
    "origin","id","sex","age","height","weight","bmi","waist_circ",
    "glucose","cholesterol","triglycerides","hdl","ldl","diastolic_bp","systolic_bp","resting_heart_rate",
    "moderate_act","vigorous_act","mvpa","intensity_minutes","shuttle_run_speed","vo2_max",
    "z_score_fitness","z_score_fitness_average","z_score_pa",
    "harmonized_pa_mvpa","harmonized_pa_intensity",
    "harmonized_fitness","harmonized_fitness_average",
    "steps","smoking","alcohol",

    "vegetables","fruits","sweetened_beverages","red_meats","processed_meats","cereals",
    "mother_educ","father_educ","sleep","lonely","stress","family_diabetes","diabetes","medication",
    "z_score_affluence",

    "family_cars","bedroom","family_vacations","family_computers","internet","mother_occup","father_occup",
    "income","stopped_studies","started_job","years_unemployed","live_own","house_owner","education",
    "t600","sports_participation","shuttle_run_laps","andersen_meters","vo2peak_andersen","6_min_walking_test",
    "vpa_indicator","z_score_pa_fallback",

    "z_score_sports_participation","z_score_vo2_max","z_score_andersen","z_score_6_min",
    "z_score_mother_educ","z_score_father_educ","z_score_bedroom","z_score_family_cars",
    "z_score_family_vacations","z_score_family_computers","z_score_internet",
    "z_score_mother_occup","z_score_father_occup","z_score_income","z_score_studies","z_score_job","z_score_unemployed",
    "z_score_own","z_score_house","z_score_education"
]

all_cols = list(dfz.columns)
final_columns = desired_columns + [c for c in all_cols if c not in desired_columns]
dfz = dfz[final_columns]

# nearest-neighbor imputation on harmonized_fitness
df = dfz.copy()

mask_miss = (
    (df['origin'] == 'SLOfit lifelong') &
    df['z_score_fitness'].notna() &
    df['harmonized_fitness'].isna()
)

df_donors = df[
    df['z_score_fitness'].notna() &
    df['harmonized_fitness'].notna()
].copy()

age_penalty = 0.1
filled = 0

for idx in df.index[mask_miss]:
    z0 = df.at[idx, 'z_score_fitness']
    age0 = df.at[idx, 'age']

    df_donors['dist'] = (
        (df_donors['z_score_fitness'] - z0).abs() +
        age_penalty * (df_donors['age'] - age0).abs()
    )

    best_idx = df_donors['dist'].idxmin()
    best_val = df_donors.at[best_idx, 'harmonized_fitness']

    if not pd.isna(best_val):
        df.at[idx, 'harmonized_fitness'] = best_val
        filled += 1

print(f"Imputed harmonized_fitness for {filled} SLOfit_lifelong rows by nearest-neighbor match")

# final fixes
for pa_col in ["harmonized_pa_mvpa", "harmonized_pa_intensity"]:
    if pa_col in df.columns:
        mask_pa_negzero = (df[pa_col] == 0) & np.signbit(df[pa_col])
        mask_pa_negative = df[pa_col] < 0
        df.loc[mask_pa_negzero | mask_pa_negative, pa_col] = np.nan

for fit_col in ["harmonized_fitness", "harmonized_fitness_average"]:
    if fit_col in df.columns:
        mask_fit_negzero = (df[fit_col] == 0) & np.signbit(df[fit_col])
        mask_fit_negative = df[fit_col] < 0
        df.loc[mask_fit_negzero | mask_fit_negative, fit_col] = np.nan

if 'bmi_bracket' in df.columns:
    df = df.drop(columns=['bmi_bracket'])

from post_harmonization_cleaning import clean_dataset

df, cleaning_summary = clean_dataset(df)
print("Final cleaning summary:")
for name, count in cleaning_summary.items():
    print(f"  {name}: {count}")

save_without_hidden_cols(df, final_path, float_format="%.2f")
print("Saved final merged dataset to", final_path)
