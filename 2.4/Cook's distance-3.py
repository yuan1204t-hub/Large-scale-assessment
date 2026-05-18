# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch read original Excel datasets from a specified folder.
2. By default, the last column is used as the dependent variable y,
   and all preceding columns are used as independent variables X.
3. Calculate Cook's distance for three traditional regression models:
   - M0: Full quadratic regression model
   - M1: Optimal subset model selected by adjusted R-squared / Cr-related criterion
   - M2: Optimal subset model selected by Mallows' Cr-related criterion
4. Automatically handle NaN values in Cook's distance.
5. Output separate sheets for M0, M1, M2, a combined long-format sheet,
   a wide-format comparison sheet, and error records.

Required input files:
1. Original Excel datasets
2. M1 summary file containing Dataset_ID and Best_Variables
3. M2 summary file containing Dataset_ID and Best_Variables

Notes:
- The empirical Cook's distance threshold is set as 4/n.
- Prop_CooksD_gt_4_over_n is the proportion of valid Cook's distance values
  exceeding the threshold 4/n.
"""

import os
import glob
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm


# =========================
# 1. Path settings
# =========================
data_folder = r"Please enter your path here"

m1_result_file = r"Please enter your path here"
m2_result_file = r"Please enter your path here"

output_file = r"Please enter your path here"


# =========================
# 2. Build full quadratic design matrix for M0
# =========================
def build_full_quadratic_matrix(X_raw):
    """
    Build the full quadratic design matrix.

    The full quadratic model includes:
    1. Linear terms
    2. Squared terms
    3. Two-way interaction terms
    """
    X_full = pd.DataFrame(index=X_raw.index)
    cols = list(X_raw.columns)

    # Linear terms
    for col in cols:
        X_full[col] = X_raw[col]

    # Squared terms
    for col in cols:
        X_full[f"{col}^2"] = X_raw[col] ** 2

    # Two-way interaction terms
    for c1, c2 in itertools.combinations(cols, 2):
        X_full[f"{c1} {c2}"] = X_raw[c1] * X_raw[c2]

    return X_full


# =========================
# 3. Parse selected variables for M1 and M2
# =========================
def parse_best_variables(text):
    """
    Parse the Best_Variables field into a list.

    Example:
    'A, B, A^2, A B, B^2'
    ->
    ['A', 'B', 'A^2', 'A B', 'B^2']
    """
    if pd.isna(text):
        return []

    return [x.strip() for x in str(text).split(",") if x.strip()]


# =========================
# 4. Build design matrix from selected terms
# =========================
def build_design_matrix_from_terms(X_raw, terms):
    """
    Build a design matrix according to selected terms.

    Supported terms:
    1. Linear term: A
    2. Squared term: A^2
    3. Interaction term: A B
    """
    X = pd.DataFrame(index=X_raw.index)

    for term in terms:
        # Squared term
        if "^2" in term:
            base_var = term.replace("^2", "").strip()

            if base_var not in X_raw.columns:
                raise ValueError(f"Base variable for squared term not found: {base_var}")

            X[term] = X_raw[base_var] ** 2

        # Interaction term
        elif " " in term:
            parts = term.split()

            if len(parts) != 2:
                raise ValueError(f"Unrecognized interaction term format: {term}")

            v1, v2 = parts

            if v1 not in X_raw.columns or v2 not in X_raw.columns:
                raise ValueError(f"Variables for interaction term not found: {term}")

            X[term] = X_raw[v1] * X_raw[v2]

        # Linear term
        else:
            if term not in X_raw.columns:
                raise ValueError(f"Linear variable not found: {term}")

            X[term] = X_raw[term]

    return X


# =========================
# 5. Read and clean one dataset
# =========================
def read_original_dataset(file_path):
    """
    Read an original Excel dataset and split it into X and y.
    """
    df = pd.read_excel(file_path)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty after removing missing values.")

    if df.shape[1] < 2:
        raise ValueError("At least one X column and one y column are required.")

    y_col = df.columns[-1]
    x_cols = list(df.columns[:-1])

    X_raw = df[x_cols]
    y = df[y_col]

    return df, X_raw, y, y_col, x_cols


# =========================
# 6. Fit OLS and calculate Cook's distance
# =========================
def fit_ols_and_cooks_distance(X, y):
    """
    Fit an OLS model and calculate Cook's distance.
    """
    X_const = sm.add_constant(X, has_constant="add")

    if len(X_const) <= X_const.shape[1]:
        raise ValueError(
            f"Insufficient sample size for OLS fitting: "
            f"n={len(X_const)}, parameters={X_const.shape[1]}"
        )

    model = sm.OLS(y, X_const).fit()

    influence = model.get_influence()
    cooks_d, _ = influence.cooks_distance
    cooks_d = np.asarray(cooks_d, dtype=float)

    n = len(y)
    threshold_4_over_n = 4 / n if n > 0 else np.nan

    nan_count = int(np.isnan(cooks_d).sum())
    valid_cooks_d = cooks_d[~np.isnan(cooks_d)]

    if len(valid_cooks_d) == 0:
        max_cooksd = np.nan
        mean_cooksd = np.nan
        median_cooksd = np.nan
        num_gt_4_over_n = np.nan
        prop_gt_4_over_n = np.nan
    else:
        max_cooksd = float(np.nanmax(valid_cooks_d))
        mean_cooksd = float(np.nanmean(valid_cooks_d))
        median_cooksd = float(np.nanmedian(valid_cooks_d))
        num_gt_4_over_n = int(np.sum(valid_cooks_d > threshold_4_over_n))
        prop_gt_4_over_n = num_gt_4_over_n / len(valid_cooks_d)

    return {
        "Nt": X_const.shape[1] - 1,
        "CooksD_Threshold_4_over_n": threshold_4_over_n,
        "CooksD_NaN_Count": nan_count,
        "Max_CooksD": max_cooksd,
        "Mean_CooksD": mean_cooksd,
        "Median_CooksD": median_cooksd,
        "Num_CooksD_gt_4_over_n": num_gt_4_over_n,
        "Prop_CooksD_gt_4_over_n": prop_gt_4_over_n,
        "R2_refit": model.rsquared,
        "Adj_R2_refit": model.rsquared_adj
    }


# =========================
# 7. Calculate M0 Cook's distance for one dataset
# =========================
def calculate_m0_for_one_dataset(file_path):
    """
    Calculate Cook's distance results for M0.
    """
    df, X_raw, y, y_col, x_cols = read_original_dataset(file_path)

    X_m0 = build_full_quadratic_matrix(X_raw)
    metrics = fit_ols_and_cooks_distance(X_m0, y)

    return {
        "Model": "M0",
        "n_samples": len(df),
        "n_original_predictors": len(x_cols),
        "n_candidate_terms": X_m0.shape[1],
        "Best_Variables": "Full quadratic model",
        **metrics
    }


# =========================
# 8. Calculate M1 or M2 Cook's distance for one dataset
# =========================
def calculate_selected_model_for_one_dataset(file_path, best_variables_text, model_name):
    """
    Calculate Cook's distance results for M1 or M2 according to selected variables.
    """
    df, X_raw, y, y_col, x_cols = read_original_dataset(file_path)

    terms = parse_best_variables(best_variables_text)

    if len(terms) == 0:
        raise ValueError("Best_Variables is empty.")

    X_selected = build_design_matrix_from_terms(X_raw, terms)
    metrics = fit_ols_and_cooks_distance(X_selected, y)

    return {
        "Model": model_name,
        "n_samples": len(df),
        "n_original_predictors": len(x_cols),
        "n_candidate_terms": np.nan,
        "Best_Variables": ", ".join(terms),
        **metrics
    }


# =========================
# 9. Run M0
# =========================
def run_m0(file_list):
    """
    Run Cook's distance calculation for M0.
    """
    results = []
    errors = []

    total = len(file_list)

    print("\n[INFO] Running M0 Cook's distance calculation...")

    for idx, file_path in enumerate(file_list, start=1):
        file_name = os.path.basename(file_path)
        print(f"[M0] [{idx}/{total}] Processing: {file_name}")

        try:
            metrics = calculate_m0_for_one_dataset(file_path)

            results.append({
                "Dataset_ID": file_name,
                **metrics
            })

            print(
                f"  -> Success: Max_CooksD = {metrics['Max_CooksD']}, "
                f"NaN_Count = {metrics['CooksD_NaN_Count']}"
            )

        except Exception as e:
            errors.append({
                "Model": "M0",
                "Dataset_ID": file_name,
                "Best_Variables": "",
                "Error": str(e)
            })

            print(f"  -> Failed: {e}")

    return results, errors


# =========================
# 10. Run M1 or M2
# =========================
def run_selected_model(result_file, model_name):
    """
    Run Cook's distance calculation for M1 or M2 based on the selected variables
    reported in the corresponding summary file.
    """
    results = []
    errors = []

    print(f"\n[INFO] Running {model_name} Cook's distance calculation...")
    print(f"[INFO] Reading {model_name} summary file: {result_file}")

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"{model_name} summary file not found: {result_file}")

    summary_df = pd.read_excel(result_file)

    required_cols = ["Dataset_ID", "Best_Variables"]
    for col in required_cols:
        if col not in summary_df.columns:
            raise ValueError(f"{model_name} summary file is missing required column: {col}")

    total = len(summary_df)

    if total == 0:
        print(f"[WARN] {model_name} summary file is empty.")

    for idx, row in summary_df.iterrows():
        file_name = str(row["Dataset_ID"]).strip()

        # Automatically add .xlsx if the extension is missing
        if not file_name.lower().endswith(".xlsx"):
            file_name += ".xlsx"

        best_variables = row["Best_Variables"]
        file_path = os.path.join(data_folder, file_name)

        print(f"[{model_name}] [{idx + 1}/{total}] Processing: {file_name}")

        if not os.path.exists(file_path):
            errors.append({
                "Model": model_name,
                "Dataset_ID": file_name,
                "Best_Variables": best_variables,
                "Error": "Original dataset file not found."
            })

            print("  -> Failed: Original dataset file not found.")
            continue

        try:
            metrics = calculate_selected_model_for_one_dataset(
                file_path=file_path,
                best_variables_text=best_variables,
                model_name=model_name
            )

            results.append({
                "Dataset_ID": file_name,
                **metrics
            })

            print(
                f"  -> Success: Max_CooksD = {metrics['Max_CooksD']}, "
                f"NaN_Count = {metrics['CooksD_NaN_Count']}"
            )

        except Exception as e:
            errors.append({
                "Model": model_name,
                "Dataset_ID": file_name,
                "Best_Variables": best_variables,
                "Error": str(e)
            })

            print(f"  -> Failed: {e}")

    return results, errors


# =========================
# 11. Main program
# =========================
def main():
    print("[INFO] Starting comprehensive Cook's distance calculation for M0, M1, and M2...")

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    file_list = glob.glob(os.path.join(data_folder, "*.xlsx"))
    file_list = sorted(file_list)

    print(f"[INFO] Found {len(file_list)} original Excel datasets.")

    if len(file_list) == 0:
        raise ValueError("No .xlsx files were found in the data folder.")

    all_results = []
    all_errors = []

    # M0
    m0_results, m0_errors = run_m0(file_list)
    all_results.extend(m0_results)
    all_errors.extend(m0_errors)

    # M1
    m1_results, m1_errors = run_selected_model(m1_result_file, "M1")
    all_results.extend(m1_results)
    all_errors.extend(m1_errors)

    # M2
    m2_results, m2_errors = run_selected_model(m2_result_file, "M2")
    all_results.extend(m2_results)
    all_errors.extend(m2_errors)

    # Convert to DataFrames
    all_results_df = pd.DataFrame(all_results)
    all_errors_df = pd.DataFrame(all_errors)

    # Split results by model
    if not all_results_df.empty:
        m0_df = all_results_df[all_results_df["Model"] == "M0"].copy()
        m1_df = all_results_df[all_results_df["Model"] == "M1"].copy()
        m2_df = all_results_df[all_results_df["Model"] == "M2"].copy()
    else:
        m0_df = pd.DataFrame()
        m1_df = pd.DataFrame()
        m2_df = pd.DataFrame()

    # Create a wide comparison table
    comparison_cols = [
        "CooksD_Threshold_4_over_n",
        "CooksD_NaN_Count",
        "Max_CooksD",
        "Mean_CooksD",
        "Median_CooksD",
        "Num_CooksD_gt_4_over_n",
        "Prop_CooksD_gt_4_over_n",
        "R2_refit",
        "Adj_R2_refit",
        "Nt"
    ]

    if not all_results_df.empty:
        comparison_df = all_results_df[
            ["Dataset_ID", "Model"] + comparison_cols
        ].copy()

        comparison_wide_df = comparison_df.pivot(
            index="Dataset_ID",
            columns="Model",
            values=comparison_cols
        )

        comparison_wide_df.columns = [
            f"{metric}_{model}" for metric, model in comparison_wide_df.columns
        ]

        comparison_wide_df = comparison_wide_df.reset_index()

        # Add a simple best label based on the smallest proportion of influential points
        prop_cols = [
            col for col in comparison_wide_df.columns
            if col.startswith("Prop_CooksD_gt_4_over_n_")
        ]

        if len(prop_cols) > 0:
            comparison_wide_df["Lowest_Influential_Point_Proportion_Model"] = (
                comparison_wide_df[prop_cols].idxmin(axis=1)
            )
            comparison_wide_df["Lowest_Influential_Point_Proportion_Model"] = (
                comparison_wide_df["Lowest_Influential_Point_Proportion_Model"]
                .str.replace("Prop_CooksD_gt_4_over_n_", "", regex=False)
            )
            comparison_wide_df["Lowest_Influential_Point_Proportion"] = (
                comparison_wide_df[prop_cols].min(axis=1)
            )

    else:
        comparison_wide_df = pd.DataFrame()

    # Create output folder
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_results_df.to_excel(writer, sheet_name="All_CooksD_Long", index=False)
        comparison_wide_df.to_excel(writer, sheet_name="CooksD_Comparison_Wide", index=False)
        m0_df.to_excel(writer, sheet_name="M0_CooksD", index=False)
        m1_df.to_excel(writer, sheet_name="M1_CooksD", index=False)
        m2_df.to_excel(writer, sheet_name="M2_CooksD", index=False)
        all_errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] All calculations completed.")
    print(f"[INFO] Total successful records: {len(all_results_df)}")
    print(f"[INFO] Total error records: {len(all_errors_df)}")
    print(f"[INFO] Comprehensive Cook's distance results saved to: {output_file}")


if __name__ == "__main__":
    main()