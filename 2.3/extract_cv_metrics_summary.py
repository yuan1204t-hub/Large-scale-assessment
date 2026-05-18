# -*- coding: utf-8 -*-
"""
Purpose:
1. Extract cross-validation performance metrics from multiple model result files.
2. Merge the extracted metrics into one wide-format summary table by dataset ID.
3. The extracted metrics include:
   - RMSECV
   - MAECV
   - Q2
4. Standardize column names using model prefixes from M0 to M6.
5. Save one complete summary table containing all metrics and all models.
6. Save three separate metric summary tables for RMSECV, MAECV, and Q2.

Applicable scenarios:
- Multiple models have already been evaluated by cross-validation.
- Each model has its own output summary file.
- The user wants to integrate cross-validation metrics from all models.
- The merged table will be used for Chapter 3.2 predictive performance
  comparison, statistical tests, and visualization.
"""

import os
import pandas as pd


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the output folder path.
output_dir = r"Please enter your path here"

# Please enter the result file paths for each model.
# Model labels can be adjusted according to the study design.
model_file_dict = {
    "M0": r"Please enter your path here",
    "M1": r"Please enter your path here",
    "M2": r"Please enter your path here",
    "M3": r"Please enter your path here",
    "M4": r"Please enter your path here",
    "M5": r"Please enter your path here",
    "M6": r"Please enter your path here",
}


# =========================================================
# 2. Model and metric settings
# =========================================================
model_order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

metric_order = ["RMSECV", "MAECV", "Q2"]


# =========================================================
# 3. Utility functions
# =========================================================
def detect_id_column(df, file_path):
    """
    Detect the dataset ID column automatically.

    Accepted ID column names include:
    - Dataset_ID
    - File_Name
    - dataset_id
    - file_name
    - ID
    """

    candidate_columns = [
        "Dataset_ID",
        "File_Name",
        "dataset_id",
        "file_name",
        "ID"
    ]

    for col in candidate_columns:
        if col in df.columns:
            return col

    raise ValueError(
        f"Dataset ID column could not be detected in file: {file_path}\n"
        f"Existing columns: {list(df.columns)}"
    )


def read_model_metric_file(file_path, model_name, metric_columns=None):
    """
    Read one model result file and extract cross-validation metrics.

    Parameters
    ----------
    file_path : str
        Path to the model result file.

    model_name : str
        Model label, such as M0, M1, M2, M3, M4, M5, or M6.

    metric_columns : list or None
        Metric columns to extract. If None, RMSECV, MAECV, and Q2 are used.

    Returns
    -------
    pandas.DataFrame
        A standardized dataframe containing:
        Dataset_ID, Model_RMSECV, Model_MAECV, Model_Q2
    """

    if metric_columns is None:
        metric_columns = metric_order

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model result file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    id_column = detect_id_column(df, file_path)

    missing_columns = [
        col for col in metric_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following metric columns are missing in {file_path}: "
            f"{missing_columns}"
        )

    output_df = df[[id_column] + metric_columns].copy()
    output_df = output_df.rename(columns={id_column: "Dataset_ID"})

    rename_dict = {
        metric: f"{model_name}_{metric}"
        for metric in metric_columns
    }

    output_df = output_df.rename(columns=rename_dict)

    return output_df


def sort_columns_by_metric_and_model(merged_df):
    """
    Sort columns by metric first and model second.

    Output order:
    Dataset_ID,
    M0_RMSECV, M1_RMSECV, ..., M6_RMSECV,
    M0_MAECV, M1_MAECV, ..., M6_MAECV,
    M0_Q2, M1_Q2, ..., M6_Q2
    """

    ordered_columns = ["Dataset_ID"]

    for metric in metric_order:
        for model in model_order:
            col = f"{model}_{metric}"
            if col in merged_df.columns:
                ordered_columns.append(col)

    remaining_columns = [
        col for col in merged_df.columns
        if col not in ordered_columns
    ]

    ordered_columns.extend(remaining_columns)

    return merged_df[ordered_columns]


def create_single_metric_table(merged_df, metric):
    """
    Create a separate wide-format table for one metric.

    The output columns are:
    Dataset_ID, M0, M1, M2, M3, M4, M5, M6
    """

    selected_columns = ["Dataset_ID"]

    for model in model_order:
        col = f"{model}_{metric}"
        if col in merged_df.columns:
            selected_columns.append(col)

    metric_df = merged_df[selected_columns].copy()

    rename_dict = {
        f"{model}_{metric}": model
        for model in model_order
        if f"{model}_{metric}" in metric_df.columns
    }

    metric_df = metric_df.rename(columns=rename_dict)

    return metric_df


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Extract and merge cross-validation metrics from all model result files."""

    os.makedirs(output_dir, exist_ok=True)

    merged_df = None
    error_records = []

    print("[INFO] Starting cross-validation metric extraction...")

    for model_name in model_order:
        file_path = model_file_dict.get(model_name)

        if file_path is None:
            error_records.append({
                "Model": model_name,
                "File_Path": "",
                "Error": "No file path was provided."
            })
            print(f"[WARNING] No file path was provided for {model_name}.")
            continue

        try:
            print(f"[INFO] Reading {model_name}: {file_path}")

            model_df = read_model_metric_file(
                file_path=file_path,
                model_name=model_name,
                metric_columns=metric_order
            )

            if merged_df is None:
                merged_df = model_df
            else:
                merged_df = pd.merge(
                    merged_df,
                    model_df,
                    on="Dataset_ID",
                    how="outer"
                )

        except Exception as e:
            error_records.append({
                "Model": model_name,
                "File_Path": file_path,
                "Error": str(e)
            })

            print(f"[ERROR] Failed to read {model_name}: {e}")

    if merged_df is None:
        print("[ERROR] No valid model metric files were read.")
        return

    merged_df = sort_columns_by_metric_and_model(merged_df)
    merged_df = merged_df.sort_values(by="Dataset_ID").reset_index(drop=True)

    # =====================================================
    # 5. Save outputs
    # =====================================================
    full_output_path = os.path.join(output_dir, "CV_metrics_all_models.xlsx")
    merged_df.to_excel(full_output_path, index=False)

    print(f"[INFO] Complete summary table saved to: {full_output_path}")

    for metric in metric_order:
        metric_df = create_single_metric_table(
            merged_df=merged_df,
            metric=metric
        )

        metric_output_path = os.path.join(output_dir, f"{metric}_summary.xlsx")
        metric_df.to_excel(metric_output_path, index=False)

        print(f"[INFO] {metric} summary table saved to: {metric_output_path}")

    if error_records:
        error_df = pd.DataFrame(error_records)
        error_output_path = os.path.join(output_dir, "metric_extraction_errors.xlsx")
        error_df.to_excel(error_output_path, index=False)

        print(f"[WARNING] Error records saved to: {error_output_path}")
    else:
        print("[INFO] No errors occurred during metric extraction.")

    print("\n[INFO] Cross-validation metric extraction completed.")
    print(f"[INFO] Number of datasets: {merged_df.shape[0]}")
    print(f"[INFO] Number of columns: {merged_df.shape[1]}")


if __name__ == "__main__":
    main()