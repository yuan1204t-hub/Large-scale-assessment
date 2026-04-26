# -*- coding: utf-8 -*-
"""
Purpose:
1. Read two diagnostic summary tables:
   - A residual diagnostic summary table for all models.
   - A BP and Cook's distance diagnostic summary table for traditional
     regression models.
2. Perform Shapiro-Wilk normality tests for each model-indicator combination.
3. Save the normality test results into one Excel file.

Applicable scenarios:
- Residual diagnostic indicators have already been merged into a wide-format table.
- BP test and Cook's distance indicators have already been merged for traditional
  regression models.
- The user wants to check the distribution normality of diagnostic indicators.
- The outputs will be used for Chapter 3.3 diagnostic analysis.

Notes:
- pn, pm, Max_Abs_Residual, and AE_IQR are tested across all models.
- BP_pvalue, Cooks_Count, and Cooks_Proportion are tested only across
  traditional regression models M0, M1, and M2.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the merged residual diagnostic summary table path.
all_model_diagnostic_file = r"Please enter your path here"

# Please enter the merged BP and Cook's distance summary table path.
traditional_model_diagnostic_file = r"Please enter your path here"

# Please enter the output Excel file path.
output_file = r"Please enter your path here"


# =========================================================
# 2. Model and metric settings
# =========================================================
all_models = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

traditional_models = ["M0", "M1", "M2"]

# Mapping between manuscript model labels and actual column prefixes.
model_prefix_map = {
    "M0": "M0",
    "M1": "M1",
    "M2": "M2",
    "M3": "Ridge",
    "M4": "SVM",
    "M5": "PLS",
    "M6": "GPR",
}

# Diagnostic indicators tested across all models.
all_model_metric_columns = {
    "pn": {
        model: f"{prefix}_pn"
        for model, prefix in model_prefix_map.items()
    },
    "pm": {
        model: f"{prefix}_pm"
        for model, prefix in model_prefix_map.items()
    },
    "Max_Abs_Residual": {
        model: f"{prefix}_Max_Abs_Residual"
        for model, prefix in model_prefix_map.items()
    },
    "AE_IQR": {
        model: f"{prefix}_AE_IQR"
        for model, prefix in model_prefix_map.items()
    },
}

# Diagnostic indicators tested only across traditional regression models.
traditional_model_metric_columns = {
    "BP_pvalue": {
        "M0": "M0_BP_pvalue",
        "M1": "M1_BP_pvalue",
        "M2": "M2_BP_pvalue",
    },
    "Cooks_Count": {
        "M0": "M0_Cooks_Count",
        "M1": "M1_Cooks_Count",
        "M2": "M2_Cooks_Count",
    },
    "Cooks_Proportion": {
        "M0": "M0_Cooks_Proportion",
        "M1": "M1_Cooks_Proportion",
        "M2": "M2_Cooks_Proportion",
    },
}


# =========================================================
# 3. Utility functions
# =========================================================
def read_first_sheet(file_path):
    """Read the first sheet of an Excel file."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    excel_file = pd.ExcelFile(file_path)
    return pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])


def standardize_file_name_column(df):
    """
    Standardize the dataset identifier column as File_Name.

    Accepted column names:
    - File_Name
    - Dataset_ID
    - ID
    """

    if "File_Name" in df.columns:
        df["File_Name"] = df["File_Name"].astype(str).str.strip()
        return df

    if "Dataset_ID" in df.columns:
        df = df.rename(columns={"Dataset_ID": "File_Name"})
        df["File_Name"] = df["File_Name"].astype(str).str.strip()
        return df

    if "ID" in df.columns:
        df = df.rename(columns={"ID": "File_Name"})
        df["File_Name"] = df["File_Name"].astype(str).str.strip()
        return df

    raise ValueError(
        "The dataframe does not contain File_Name, Dataset_ID, or ID column."
    )


def to_numeric_series(values):
    """Convert values to a numeric series and remove missing values."""

    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def run_shapiro_test(values):
    """
    Perform the Shapiro-Wilk normality test.

    If there are fewer than three valid observations, the test is not performed.
    """

    x = to_numeric_series(values)
    n = len(x)

    if n < 3:
        return {
            "n": n,
            "Shapiro_W": np.nan,
            "Shapiro_pvalue": np.nan,
            "Is_Normal": "Not tested",
            "Note": "Fewer than 3 valid observations."
        }

    if x.nunique() < 2:
        return {
            "n": n,
            "Shapiro_W": np.nan,
            "Shapiro_pvalue": 1.0,
            "Is_Normal": "Yes",
            "Note": "All valid values are identical."
        }

    try:
        statistic, p_value = shapiro(x)

        return {
            "n": n,
            "Shapiro_W": float(statistic),
            "Shapiro_pvalue": float(p_value),
            "Is_Normal": "Yes" if p_value > 0.05 else "No",
            "Note": ""
        }

    except Exception as e:
        return {
            "n": n,
            "Shapiro_W": np.nan,
            "Shapiro_pvalue": np.nan,
            "Is_Normal": "Not tested",
            "Note": str(e)
        }


def check_columns_exist(df, metric_column_map, table_name):
    """Check whether all required columns exist in the input dataframe."""

    missing_records = []

    for metric_name, model_column_map in metric_column_map.items():
        for model_name, column_name in model_column_map.items():
            if column_name not in df.columns:
                missing_records.append({
                    "Table": table_name,
                    "Metric": metric_name,
                    "Model": model_name,
                    "Missing_Column": column_name
                })

    return missing_records


def run_shapiro_group(df, metric_column_map, model_order, group_name):
    """Run Shapiro-Wilk tests for one group of diagnostic indicators."""

    result_rows = []

    for metric_name, column_map in metric_column_map.items():
        print(f"[INFO] Processing {group_name}: {metric_name}")

        for model_name in model_order:
            column_name = column_map.get(model_name)

            if column_name is None or column_name not in df.columns:
                continue

            test_result = run_shapiro_test(df[column_name])

            result_rows.append({
                "Group": group_name,
                "Metric": metric_name,
                "Model": model_name,
                "Column_Name": column_name,
                **test_result
            })

    return pd.DataFrame(result_rows)


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Run Shapiro-Wilk normality tests for diagnostic indicators."""

    print("[INFO] Reading input files...")

    df_all = standardize_file_name_column(read_first_sheet(all_model_diagnostic_file))
    df_traditional = standardize_file_name_column(
        read_first_sheet(traditional_model_diagnostic_file)
    )

    missing_records = []

    missing_records.extend(
        check_columns_exist(
            df=df_all,
            metric_column_map=all_model_metric_columns,
            table_name="All_Model_Diagnostics"
        )
    )

    missing_records.extend(
        check_columns_exist(
            df=df_traditional,
            metric_column_map=traditional_model_metric_columns,
            table_name="Traditional_Model_Diagnostics"
        )
    )

    if missing_records:
        missing_df = pd.DataFrame(missing_records)
        print("[WARNING] Some required columns are missing.")
    else:
        missing_df = pd.DataFrame(columns=["Table", "Metric", "Model", "Missing_Column"])

    print("[INFO] Running Shapiro-Wilk tests for all-model indicators...")

    shapiro_all = run_shapiro_group(
        df=df_all,
        metric_column_map=all_model_metric_columns,
        model_order=all_models,
        group_name="All_Models"
    )

    print("[INFO] Running Shapiro-Wilk tests for traditional-model indicators...")

    shapiro_traditional = run_shapiro_group(
        df=df_traditional,
        metric_column_map=traditional_model_metric_columns,
        model_order=traditional_models,
        group_name="Traditional_Models"
    )

    shapiro_df = pd.concat(
        [shapiro_all, shapiro_traditional],
        ignore_index=True
    )

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        shapiro_df.to_excel(writer, sheet_name="Shapiro_Normality", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_Columns", index=False)

    print("\n[INFO] Shapiro-Wilk normality analysis completed.")
    print(f"[INFO] Output saved to: {output_file}")
    print(f"[INFO] Shapiro records: {len(shapiro_df)}")
    print(f"[INFO] Missing column records: {len(missing_df)}")


if __name__ == "__main__":
    main()