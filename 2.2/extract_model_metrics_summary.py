# -*- coding: utf-8 -*-
"""
Purpose:
1. Extract model performance metrics from multiple Excel result files.
2. Merge the extracted metrics into one wide-format summary table by dataset ID.
3. The extracted metrics include:
   - R2
   - RMSE
   - MAE
4. Standardize column names so that different source files can be compared
   in a unified format.
5. Save the merged table as an Excel file for later descriptive statistics,
   significance tests, and model comparison analysis.

Applicable scenarios:
- Multiple models have already been fitted separately.
- Each model has its own output summary file.
- The user wants to integrate the main performance metrics from all models.
- The merged table will be used for Chapter 3.1 model performance comparison.
"""

import os
import pandas as pd


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the folder where all model result folders are stored.
base_path = r"Please enter your path here"

# Please enter the output file path.
output_file = r"Please enter your path here"


# =========================================================
# 2. General utility function
# =========================================================
def extract_metrics(file_path, columns_to_extract, model_prefix):
    """
    Read one Excel file, extract selected metric columns, and rename them
    using a unified model-based prefix.

    Parameters
    ----------
    file_path : str
        Path to the source Excel file.

    columns_to_extract : list
        Metric columns to extract from the source file.

    model_prefix : str
        Model name used as the prefix of output columns.

    Returns
    -------
    pandas.DataFrame or None
        A dataframe containing ID and selected metric columns.
        If the file does not exist, None will be returned.
    """

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    df = pd.read_excel(file_path)

    if df.shape[1] == 0:
        print(f"[WARNING] Empty file skipped: {file_path}")
        return None

    # Standardize the first column as dataset ID.
    first_col_name = df.columns[0]
    df = df.rename(columns={first_col_name: "ID"})

    # Keep ID and available metric columns.
    available_cols = [col for col in columns_to_extract if col in df.columns]
    target_cols = ["ID"] + available_cols

    if len(available_cols) == 0:
        print(f"[WARNING] No required metric columns found in: {file_path}")
        return None

    subset = df[target_cols].copy()

    # Standardize column names.
    rename_dict = {}

    for col in subset.columns:
        if col == "ID":
            continue

        clean_name = (
            col.replace("_refit", "")
               .replace("Train_", "")
               .replace("LOOCV_", "")
        )

        rename_dict[col] = f"{model_prefix}_{clean_name}"

    subset = subset.rename(columns=rename_dict)

    return subset


# =========================================================
# 3. Metric extraction settings
# =========================================================
# R2 results from different model summary files.
r2_configs = [
    (
        os.path.join(base_path, "svm_results", "svm_batch_summary.xlsx"),
        ["Train_R2"],
        "SVM"
    ),
    (
        os.path.join(base_path, "quadratic_ridge_results", "quadratic_ridge_batch_summary.xlsx"),
        ["Train_R2"],
        "Ridge"
    ),
    (
        os.path.join(base_path, "pls_results", "pls_batch_summary.xlsx"),
        ["Train_R2"],
        "PLS"
    ),
    (
        os.path.join(base_path, "gpr_results", "gpr_batch_summary.xlsx"),
        ["Train_R2"],
        "GPR"
    ),
    (
        os.path.join(base_path, "M1_Optimization_Summary_NoCV.xlsx"),
        ["R2"],
        "M1"
    ),
    (
        os.path.join(base_path, "Optimal_M2_Cp_Summary.xlsx"),
        ["R2"],
        "M2"
    ),
    (
        os.path.join(base_path, "AIC", "m1_aic_summary.xlsx"),
        ["R2_refit"],
        "M0"
    ),
]

# RMSE and MAE results from different model result files.
rmse_mae_configs = [
    (
        os.path.join(base_path, "metric_results", "M0_RMSE_MAE_by_dataset.xlsx"),
        ["RMSE", "MAE"],
        "M0"
    ),
    (
        os.path.join(base_path, "metric_results", "M1_RMSE_MAE_by_dataset.xlsx"),
        ["RMSE", "MAE"],
        "M1"
    ),
    (
        os.path.join(base_path, "metric_results", "M2_RMSE_MAE_by_dataset.xlsx"),
        ["RMSE", "MAE"],
        "M2"
    ),
    (
        os.path.join(base_path, "RMSECV", "GPR", "GPR_LOOCV_summary_fixed_params.xlsx"),
        ["RMSE_refit", "MAE_refit"],
        "GPR"
    ),
    (
        os.path.join(base_path, "RMSECV", "PLS", "PLS_LOOCV_summary_fixed_params.xlsx"),
        ["RMSE_refit", "MAE_refit"],
        "PLS"
    ),
    (
        os.path.join(base_path, "RMSECV", "Ridge", "Ridge_LOOCV_summary_fixed_params.xlsx"),
        ["RMSE_refit", "MAE_refit"],
        "Ridge"
    ),
    (
        os.path.join(base_path, "RMSECV", "SVM", "SVR_LOOCV_summary_fixed_params.xlsx"),
        ["RMSE_refit", "MAE_refit"],
        "SVM"
    ),
]


# =========================================================
# 4. Merge all extracted metrics
# =========================================================
def main():
    """Run metric extraction and merge all results into one summary table."""

    all_configs = r2_configs + rmse_mae_configs
    final_df = None

    print("[INFO] Starting metric extraction and merging...")

    for file_path, columns_to_extract, model_prefix in all_configs:
        print(f"[INFO] Reading: {file_path}")

        subset = extract_metrics(
            file_path=file_path,
            columns_to_extract=columns_to_extract,
            model_prefix=model_prefix
        )

        if subset is None:
            continue

        if final_df is None:
            final_df = subset
        else:
            final_df = pd.merge(final_df, subset, on="ID", how="outer")

    if final_df is None:
        print("[ERROR] No valid data were extracted. Please check file paths.")
        return

    # Optional: sort by ID for easier checking.
    final_df = final_df.sort_values(by="ID").reset_index(drop=True)

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_df.to_excel(output_file, index=False)

    print("\n[INFO] Metric extraction completed.")
    print(f"[INFO] Number of datasets: {final_df.shape[0]}")
    print(f"[INFO] Number of columns: {final_df.shape[1]}")
    print(f"[INFO] Output saved to: {output_file}")


if __name__ == "__main__":
    main()