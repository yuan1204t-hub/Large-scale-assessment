# -*- coding: utf-8 -*-
"""
Purpose:
1. Read two diagnostic summary tables:
   - A residual diagnostic summary table for all models.
   - A BP and Cook's distance diagnostic summary table for traditional
     regression models.
2. Perform Friedman overall tests for diagnostic indicators.
3. Perform Wilcoxon signed-rank pairwise comparisons between models.
4. Apply Holm correction to pairwise p-values.
5. Save all significance test results into one Excel file.

Applicable scenarios:
- Residual diagnostic indicators have already been merged into a wide-format table.
- BP test and Cook's distance indicators have already been merged for traditional
  regression models.
- The user wants to compare diagnostic performance across models.
- The outputs will be used for Chapter 3.3 diagnostic comparison.

Notes:
- pn, pm, Max_Abs_Residual, and AE_IQR are compared across all models.
- BP_pvalue, Cooks_Count, and Cooks_Proportion are compared only across
  traditional regression models M0, M1, and M2.
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


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

# Diagnostic indicators compared across all models.
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

# Diagnostic indicators compared only across traditional regression models.
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
# 3. General utility functions
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


def significance_label(p_value):
    """Convert a p-value into a significance label."""

    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


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


def holm_correction(p_values):
    """
    Apply Holm correction to a list of p-values.

    Missing p-values are ignored during correction and then restored.
    """

    p_values = np.array(p_values, dtype=float)
    adjusted_p_values = np.full_like(p_values, fill_value=np.nan, dtype=float)

    valid_mask = ~np.isnan(p_values)
    valid_p_values = p_values[valid_mask]

    if len(valid_p_values) == 0:
        return adjusted_p_values

    m = len(valid_p_values)

    sorted_indices = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sorted_indices]

    sorted_adjusted = np.zeros(m)
    previous_adjusted = 0.0

    for i in range(m):
        adjusted_p = sorted_p_values[i] * (m - i)
        adjusted_p = max(adjusted_p, previous_adjusted)
        adjusted_p = min(adjusted_p, 1.0)

        sorted_adjusted[i] = adjusted_p
        previous_adjusted = adjusted_p

    restored_adjusted = np.zeros(m)
    restored_adjusted[sorted_indices] = sorted_adjusted

    adjusted_p_values[valid_mask] = restored_adjusted

    return adjusted_p_values


# =========================================================
# 4. Statistical functions
# =========================================================
def run_friedman_test(df, column_map, model_order, metric_name):
    """
    Perform the Friedman overall test for one diagnostic indicator.

    Only complete cases across all included models are used.
    """

    available_models = [
        model for model in model_order
        if column_map.get(model) in df.columns
    ]

    columns = [column_map[model] for model in available_models]

    if len(columns) < 3:
        return {
            "Metric": metric_name,
            "Models_Compared": ", ".join(available_models),
            "n_datasets_used": 0,
            "k_models": len(columns),
            "df": np.nan,
            "Friedman_chi2": np.nan,
            "Friedman_pvalue": np.nan,
            "Significance": ""
        }

    sub = df[columns].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    n_used = len(sub)
    k_models = len(columns)

    if n_used < 3:
        return {
            "Metric": metric_name,
            "Models_Compared": ", ".join(available_models),
            "n_datasets_used": n_used,
            "k_models": k_models,
            "df": k_models - 1,
            "Friedman_chi2": np.nan,
            "Friedman_pvalue": np.nan,
            "Significance": ""
        }

    arrays = [sub[col].values for col in columns]
    chi2_statistic, p_value = friedmanchisquare(*arrays)

    return {
        "Metric": metric_name,
        "Models_Compared": ", ".join(available_models),
        "n_datasets_used": n_used,
        "k_models": k_models,
        "df": k_models - 1,
        "Friedman_chi2": float(chi2_statistic),
        "Friedman_pvalue": float(p_value),
        "Significance": significance_label(p_value)
    }


def run_pairwise_wilcoxon_tests(df, column_map, model_order, metric_name):
    """
    Perform Wilcoxon signed-rank pairwise comparisons for one indicator.

    Holm correction is applied within each indicator.
    """

    available_models = [
        model for model in model_order
        if column_map.get(model) in df.columns
    ]

    rows = []
    raw_p_values = []

    for model_1, model_2 in itertools.combinations(available_models, 2):
        col_1 = column_map[model_1]
        col_2 = column_map[model_2]

        pair_df = df[[col_1, col_2]].apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna(axis=0, how="any")

        n_used = len(pair_df)

        if n_used < 3:
            wilcoxon_statistic = np.nan
            raw_pvalue = np.nan

        else:
            x = pair_df[col_1].values
            y = pair_df[col_2].values

            if np.allclose(x, y, equal_nan=True):
                wilcoxon_statistic = 0.0
                raw_pvalue = 1.0

            else:
                try:
                    wilcoxon_statistic, raw_pvalue = wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto"
                    )
                except ValueError:
                    wilcoxon_statistic = np.nan
                    raw_pvalue = np.nan

        raw_p_values.append(raw_pvalue)

        rows.append({
            "Metric": metric_name,
            "Model_1": model_1,
            "Model_2": model_2,
            "Column_1": col_1,
            "Column_2": col_2,
            "n_datasets_used": n_used,
            "Wilcoxon_stat": wilcoxon_statistic,
            "Raw_pvalue": raw_pvalue
        })

    pairwise_df = pd.DataFrame(rows)

    if pairwise_df.empty:
        return pairwise_df

    adjusted_p_values = holm_correction(raw_p_values)

    pairwise_df["Holm_pvalue"] = adjusted_p_values
    pairwise_df["Significance"] = pairwise_df["Holm_pvalue"].apply(significance_label)

    return pairwise_df


def run_significance_group(df, metric_column_map, model_order, group_name):
    """Run Friedman and Wilcoxon-Holm tests for one group of diagnostic indicators."""

    friedman_rows = []
    pairwise_tables = []

    for metric_name, column_map in metric_column_map.items():
        print(f"[INFO] Processing {group_name}: {metric_name}")

        friedman_result = run_friedman_test(
            df=df,
            column_map=column_map,
            model_order=model_order,
            metric_name=metric_name
        )

        friedman_result["Group"] = group_name
        friedman_rows.append(friedman_result)

        pairwise_df = run_pairwise_wilcoxon_tests(
            df=df,
            column_map=column_map,
            model_order=model_order,
            metric_name=metric_name
        )

        if not pairwise_df.empty:
            pairwise_df.insert(0, "Group", group_name)
            pairwise_tables.append(pairwise_df)

    friedman_df = pd.DataFrame(friedman_rows)

    if pairwise_tables:
        pairwise_df = pd.concat(pairwise_tables, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    return friedman_df, pairwise_df


# =========================================================
# 5. Main program
# =========================================================
def main():
    """Run Friedman and Wilcoxon-Holm significance tests."""

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

    print("[INFO] Running significance tests for all-model indicators...")

    friedman_all, pairwise_all = run_significance_group(
        df=df_all,
        metric_column_map=all_model_metric_columns,
        model_order=all_models,
        group_name="All_Models"
    )

    print("[INFO] Running significance tests for traditional-model indicators...")

    friedman_traditional, pairwise_traditional = run_significance_group(
        df=df_traditional,
        metric_column_map=traditional_model_metric_columns,
        model_order=traditional_models,
        group_name="Traditional_Models"
    )

    friedman_df = pd.concat(
        [friedman_all, friedman_traditional],
        ignore_index=True
    )

    pairwise_tables = [
        df for df in [pairwise_all, pairwise_traditional]
        if not df.empty
    ]

    if pairwise_tables:
        pairwise_df = pd.concat(pairwise_tables, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        friedman_df.to_excel(writer, sheet_name="Friedman_Test", index=False)
        pairwise_df.to_excel(writer, sheet_name="Wilcoxon_Holm", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_Columns", index=False)

    print("\n[INFO] Diagnostic significance analysis completed.")
    print(f"[INFO] Output saved to: {output_file}")
    print(f"[INFO] Friedman records: {len(friedman_df)}")
    print(f"[INFO] Pairwise records: {len(pairwise_df)}")
    print(f"[INFO] Missing column records: {len(missing_df)}")


if __name__ == "__main__":
    main()