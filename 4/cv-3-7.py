# -*- coding: utf-8 -*-
"""
Purpose:
1. Analyze the consistency of optimization conclusions across multiple models.
2. Extract the optimal predicted response and optimal factor settings from each model.
3. Support two types of model result inputs:
   - Folder-based results: one Excel file for each dataset.
   - Summary-table results: one Excel file containing all datasets.
4. Calculate response-level stability indicators across models.
5. Calculate factor-level stability indicators across models.
6. Generate a global stability summary table.
7. Generate individual stability reports for each dataset.

Applicable scenarios:
- Different models have already produced their optimal predicted responses
  and optimal factor settings.
- The user wants to evaluate whether different models lead to consistent
  optimization conclusions.
- The analysis can be applied to:
  - Three traditional regression models: M0, M1, and M2.
  - Seven models: M0, M1, M2, SVM, Ridge, PLS, and GPR.
- The outputs will be used for optimization stability or optimization
  conclusion consistency analysis.

Main indicators:
- Response_CV:
  Coefficient of variation of model-predicted optimal responses after
  z-score normalization using the original response distribution.

- Avg_Factor_CV:
  Average coefficient of variation of optimal factor settings across models.

- Mean_Predicted_Y:
  Mean of the optimal predicted responses across included models.

- StdDev_Predicted_Y:
  Standard deviation of the optimal predicted responses across included models.

- 95%_Confidence_Interval_CI:
  Confidence interval for the mean predicted optimal response.

- 95%_Prediction_Interval_PI:
  Prediction interval describing the dispersion of model-predicted optimal
  responses.
"""

import os
import re
import numpy as np
import pandas as pd


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the folder containing the original datasets.
raw_data_folder = r"Please enter your path here"

# Folder-based model outputs:
# Each model has one folder, and each dataset has one Excel file in that folder.
# Example:
# folder_model_paths = {
#     "M0": r"...\M0_optimal_conditions",
#     "M1": r"...\M1_optimal_conditions",
#     "M2": r"...\M2_optimal_conditions",
# }
folder_model_paths = {
    "M0": r"Please enter your path here",
    "M1": r"Please enter your path here",
    "M2": r"Please enter your path here",
}

# Summary-table model outputs:
# Each model has one Excel summary file containing all datasets.
# If you only want to analyze M0, M1, and M2, leave this dictionary empty.
# Example:
# summary_model_paths = {
#     "SVM": r"...\svm_optimal_conditions.xlsx",
#     "Ridge": r"...\quadratic_ridge_optimal_conditions.xlsx",
#     "PLS": r"...\pls_optimal_conditions.xlsx",
#     "GPR": r"...\gpr_optimal_conditions.xlsx",
# }
summary_model_paths = {
    "SVM": r"Please enter your path here",
    "Ridge": r"Please enter your path here",
    "PLS": r"Please enter your path here",
    "GPR": r"Please enter your path here",
}

# Please enter the global summary output file path.
global_summary_file = r"Please enter your path here"

# Please enter the folder for individual dataset stability reports.
individual_report_folder = r"Please enter your path here"


# =========================================================
# 2. Column settings
# =========================================================
# Candidate column names for the optimal predicted response in folder-based files.
folder_y_column_candidates = [
    "Max_Predicted_Response",
    "Max_Prediction",
    "Max_Response",
    "Best_Predicted_Y",
    "Predicted_Y"
]

# Candidate column names for dataset ID in summary-table files.
summary_id_column_candidates = [
    "Dataset_ID",
    "File_Name",
    "Filename",
    "file_name",
    "dataset_id",
    "ID"
]

# Candidate column names for the optimal predicted response in summary-table files.
summary_y_column_candidates = [
    "Best_Predicted_Y",
    "Max_Predicted_Response",
    "Max_Prediction",
    "Max_Response",
    "Predicted_Y"
]

# Prefixes used to identify optimal factor-setting columns.
# For example:
# - Optimal_Temperature
# - Best_Temperature
factor_column_prefix_candidates = [
    "Optimal_",
    "Best_"
]

# Columns that should not be treated as factor-setting columns.
excluded_summary_columns = {
    "Dataset_ID",
    "File_Name",
    "Filename",
    "file_name",
    "dataset_id",
    "ID",
    "Y_Name",
    "Model_Type",
    "Max_Predicted_Response",
    "Max_Prediction",
    "Max_Response",
    "Best_Predicted_Y",
    "Predicted_Y",
    "Observed_Max_Response",
    "n_samples",
    "n_original_vars",
    "n_poly_features",
    "n_used_vars",
    "Grid_Size",
    "Best_Variables",
    "Used_Variables",
    "Best_Kernel",
    "Best_C",
    "Best_Gamma",
    "Best_Epsilon",
    "Best_Alpha",
    "Best_Fit_Intercept",
    "Best_n_components",
    "Best_Normalize_Y"
}


# =========================================================
# 3. General utility functions
# =========================================================
def normalize_dataset_id(name):
    """
    Normalize a dataset identifier.

    Examples:
    - 399.9.xlsx -> 399.9
    - 535.xlsx   -> 535
    - 535         -> 535
    """

    dataset_id = str(name).strip()
    dataset_id = re.sub(r"\.xlsx?$", "", dataset_id, flags=re.IGNORECASE)

    return dataset_id


def find_first_existing_column(df, candidate_columns):
    """Find the first existing column from a list of candidate column names."""

    for column in candidate_columns:
        if column in df.columns:
            return column

    return None


def read_excel_first_sheet(file_path):
    """Read the first sheet of an Excel file."""

    excel_file = pd.ExcelFile(file_path)
    return pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])


def safe_numeric(value):
    """Convert a value to numeric. Return NaN if conversion fails."""

    return pd.to_numeric(value, errors="coerce")


def extract_numeric_first_valid_value(row):
    """Extract the first numeric value from a row."""

    for value in row:
        numeric_value = safe_numeric(value)
        if not pd.isna(numeric_value):
            return numeric_value

    return np.nan


def clean_factor_name(column_name):
    """
    Remove common prefixes from factor-setting column names.

    Examples:
    - Optimal_Temperature -> Temperature
    - Best_Time           -> Time
    """

    factor_name = str(column_name)

    for prefix in factor_column_prefix_candidates:
        if factor_name.startswith(prefix):
            factor_name = factor_name.replace(prefix, "", 1)

    return factor_name


def list_original_dataset_files(raw_folder):
    """List original Excel dataset files and return normalized dataset IDs."""

    if not os.path.exists(raw_folder):
        raise FileNotFoundError(f"Original data folder not found: {raw_folder}")

    raw_files = [
        file_name for file_name in os.listdir(raw_folder)
        if file_name.lower().endswith((".xlsx", ".xls"))
        and not file_name.startswith("~$")
    ]

    dataset_id_to_file = {
        normalize_dataset_id(file_name): os.path.join(raw_folder, file_name)
        for file_name in raw_files
    }

    return dataset_id_to_file


def read_original_response(file_path):
    """
    Read the original response column from one dataset.

    By default:
    - All columns except the last one are regarded as X.
    - The last column is regarded as y.
    """

    df = read_excel_first_sheet(file_path)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("At least one X column and one y column are required.")

    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()

    if len(y) < 2:
        raise ValueError("The original response column has fewer than two valid values.")

    y_mean = y.mean()
    y_sd = y.std(ddof=1)

    if pd.isna(y_sd) or y_sd == 0:
        raise ValueError("The original response standard deviation is zero or invalid.")

    return y, y_mean, y_sd


# =========================================================
# 4. Load model optimization results
# =========================================================
def load_folder_model_results(folder_path, model_name):
    """
    Load folder-based model optimization results.

    Expected structure:
    - One folder for one model.
    - One Excel file for one dataset.

    Returns
    -------
    dict
        {
            dataset_id: {
                "Y": optimal_predicted_response,
                "X": {
                    factor_name: optimal_factor_value,
                    ...
                }
            }
        }
    """

    results = {}

    if folder_path is None or str(folder_path).strip() == "":
        print(f"[WARNING] No folder path provided for {model_name}. Skipped.")
        return results

    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found for {model_name}: {folder_path}")
        return results

    file_names = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.lower().endswith((".xlsx", ".xls"))
        and not file_name.startswith("~$")
    ]

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        # Many old outputs use file names such as "399.9_xxx.xlsx".
        dataset_id = normalize_dataset_id(file_name.split("_")[0])

        try:
            df = read_excel_first_sheet(file_path)

            if df.empty:
                continue

            row = df.iloc[0]

            y_value = np.nan

            for y_col in folder_y_column_candidates:
                if y_col in df.columns:
                    y_value = safe_numeric(row[y_col])
                    if not pd.isna(y_value):
                        break

            if pd.isna(y_value):
                y_value = extract_numeric_first_valid_value(row)

            if pd.isna(y_value):
                print(f"[WARNING] No valid predicted response found in: {file_path}")
                continue

            x_values = {}

            for col in df.columns:
                col_name = str(col)

                if any(col_name.startswith(prefix) for prefix in factor_column_prefix_candidates):
                    factor_name = clean_factor_name(col_name)
                    value = safe_numeric(row[col])
                    if not pd.isna(value):
                        x_values[factor_name] = float(value)

            # Fallback for older outputs:
            # If no prefixed factor columns are found, use numeric columns excluding metadata.
            if len(x_values) == 0:
                for col in df.columns:
                    col_name = str(col)

                    if col_name in excluded_summary_columns:
                        continue

                    if pd.api.types.is_numeric_dtype(df[col]):
                        value = safe_numeric(row[col])
                        if not pd.isna(value) and value != y_value:
                            x_values[col_name] = float(value)

            results[dataset_id] = {
                "Y": float(y_value),
                "X": x_values
            }

        except Exception as e:
            print(f"[WARNING] Failed to read {model_name} file: {file_path} | {e}")

    return results


def load_summary_model_results(summary_file, model_name):
    """
    Load summary-table model optimization results.

    Expected structure:
    - One Excel file for one model.
    - The file contains all datasets.

    Returns
    -------
    dict
        {
            dataset_id: {
                "Y": optimal_predicted_response,
                "X": {
                    factor_name: optimal_factor_value,
                    ...
                }
            }
        }
    """

    results = {}

    if summary_file is None or str(summary_file).strip() == "":
        print(f"[WARNING] No summary file path provided for {model_name}. Skipped.")
        return results

    if not os.path.exists(summary_file):
        print(f"[WARNING] Summary file not found for {model_name}: {summary_file}")
        return results

    try:
        df = read_excel_first_sheet(summary_file)
    except Exception as e:
        print(f"[WARNING] Failed to read summary file for {model_name}: {summary_file} | {e}")
        return results

    id_col = find_first_existing_column(df, summary_id_column_candidates)

    if id_col is None:
        print(f"[WARNING] No dataset ID column found in {model_name}: {summary_file}")
        return results

    y_col = find_first_existing_column(df, summary_y_column_candidates)

    if y_col is None:
        print(f"[WARNING] No predicted response column found in {model_name}: {summary_file}")
        return results

    for _, row in df.iterrows():
        dataset_id = normalize_dataset_id(row[id_col])

        y_value = safe_numeric(row[y_col])

        if pd.isna(y_value):
            continue

        x_values = {}

        for col in df.columns:
            col_name = str(col)

            if col_name in excluded_summary_columns:
                continue

            if any(col_name.startswith(prefix) for prefix in factor_column_prefix_candidates):
                factor_name = clean_factor_name(col_name)
                value = safe_numeric(row[col])

                if not pd.isna(value):
                    x_values[factor_name] = float(value)

        results[dataset_id] = {
            "Y": float(y_value),
            "X": x_values
        }

    return results


def load_all_model_results(folder_paths, summary_paths):
    """Load all folder-based and summary-table model results."""

    all_results = {}

    print("[INFO] Loading folder-based model results...")

    for model_name, folder_path in folder_paths.items():
        model_results = load_folder_model_results(
            folder_path=folder_path,
            model_name=model_name
        )

        if model_results:
            all_results[model_name] = model_results

        print(f"[INFO] {model_name}: {len(model_results)} datasets loaded.")

    print("[INFO] Loading summary-table model results...")

    for model_name, summary_file in summary_paths.items():
        model_results = load_summary_model_results(
            summary_file=summary_file,
            model_name=model_name
        )

        if model_results:
            all_results[model_name] = model_results

        print(f"[INFO] {model_name}: {len(model_results)} datasets loaded.")

    return all_results


# =========================================================
# 5. Stability calculation
# =========================================================
def calculate_response_stability(y_predictions, original_y_mean, original_y_sd):
    """
    Calculate response-level stability across models.

    The response CV is calculated after normalizing predicted optimal responses
    with the original response mean and standard deviation.
    """

    y_array = pd.to_numeric(pd.Series(y_predictions), errors="coerce").dropna().values

    if len(y_array) < 2:
        return {
            "Response_CV": np.nan,
            "Mean_Predicted_Y": np.nan,
            "StdDev_Predicted_Y": np.nan,
            "CI_Low": np.nan,
            "CI_High": np.nan,
            "PI_Low": np.nan,
            "PI_High": np.nan
        }

    z_scores = (y_array - original_y_mean) / original_y_sd

    z_mean = np.mean(z_scores)
    z_sd = np.std(z_scores, ddof=1)

    response_cv = z_sd / abs(z_mean) if z_mean != 0 else np.nan

    mean_y = np.mean(y_array)
    sd_y = np.std(y_array, ddof=1)
    n_models = len(y_array)

    ci_margin = 1.96 * (sd_y / np.sqrt(n_models))
    ci_low = mean_y - ci_margin
    ci_high = mean_y + ci_margin

    pi_margin = 1.96 * sd_y * np.sqrt(1 + (1 / n_models))
    pi_low = mean_y - pi_margin
    pi_high = mean_y + pi_margin

    return {
        "Response_CV": response_cv,
        "Mean_Predicted_Y": mean_y,
        "StdDev_Predicted_Y": sd_y,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "PI_Low": pi_low,
        "PI_High": pi_high
    }


def calculate_factor_stability(factor_values_by_name):
    """
    Calculate factor-level stability across models.

    For each factor:
    CV = standard deviation / absolute mean

    The average of valid factor CVs is used as Avg_Factor_CV.
    """

    factor_cv_rows = []
    valid_factor_cvs = []

    for factor_name, values in factor_values_by_name.items():
        values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values

        if len(values) < 2:
            factor_cv = np.nan
            factor_mean = np.nan
            factor_sd = np.nan
            n_values = len(values)
        else:
            factor_mean = np.mean(values)
            factor_sd = np.std(values, ddof=1)
            factor_cv = factor_sd / abs(factor_mean) if factor_mean != 0 else np.nan
            n_values = len(values)

        if not pd.isna(factor_cv):
            valid_factor_cvs.append(factor_cv)

        factor_cv_rows.append({
            "Factor": factor_name,
            "n_models_used_for_factor": n_values,
            "Factor_Mean": factor_mean,
            "Factor_SD": factor_sd,
            "Factor_CV": factor_cv
        })

    avg_factor_cv = np.mean(valid_factor_cvs) if valid_factor_cvs else np.nan

    return avg_factor_cv, factor_cv_rows


def analyze_one_dataset(dataset_id, original_file, all_model_results):
    """
    Analyze optimization stability for one dataset.

    Returns
    -------
    summary_row : dict
        One row for the global summary table.

    individual_report_df : pandas.DataFrame
        Detailed stability report for this dataset.
    """

    _, original_y_mean, original_y_sd = read_original_response(original_file)

    y_predictions = []
    factor_values_by_name = {}
    included_models = []

    for model_name, model_result_map in all_model_results.items():
        if dataset_id not in model_result_map:
            continue

        model_result = model_result_map[dataset_id]

        y_predictions.append(model_result["Y"])
        included_models.append(model_name)

        for factor_name, value in model_result["X"].items():
            if factor_name not in factor_values_by_name:
                factor_values_by_name[factor_name] = []

            factor_values_by_name[factor_name].append(value)

    if len(included_models) == 0:
        raise ValueError("No model results matched this dataset.")

    response_stability = calculate_response_stability(
        y_predictions=y_predictions,
        original_y_mean=original_y_mean,
        original_y_sd=original_y_sd
    )

    avg_factor_cv, factor_cv_rows = calculate_factor_stability(
        factor_values_by_name=factor_values_by_name
    )

    summary_row = {
        "Dataset_ID": dataset_id,
        "n_models_used": len(included_models),
        "Models_Used": ", ".join(included_models),
        "Response_CV": response_stability["Response_CV"],
        "Avg_Factor_CV": avg_factor_cv,
        "Mean_Predicted_Y": response_stability["Mean_Predicted_Y"],
        "StdDev_Predicted_Y": response_stability["StdDev_Predicted_Y"],
        "95%_Confidence_Interval_CI": (
            f"[{response_stability['CI_Low']:.4f}, {response_stability['CI_High']:.4f}]"
            if not pd.isna(response_stability["CI_Low"]) else "N/A"
        ),
        "95%_Prediction_Interval_PI": (
            f"[{response_stability['PI_Low']:.4f}, {response_stability['PI_High']:.4f}]"
            if not pd.isna(response_stability["PI_Low"]) else "N/A"
        )
    }

    report_rows = [
        {
            "Parameter": "n_models_used",
            "Value": len(included_models)
        },
        {
            "Parameter": "Models_Used",
            "Value": ", ".join(included_models)
        },
        {
            "Parameter": "Response_CV",
            "Value": response_stability["Response_CV"]
        },
        {
            "Parameter": "Avg_Factor_CV",
            "Value": avg_factor_cv
        },
        {
            "Parameter": "Mean_Predicted_Y",
            "Value": response_stability["Mean_Predicted_Y"]
        },
        {
            "Parameter": "StdDev_Predicted_Y",
            "Value": response_stability["StdDev_Predicted_Y"]
        },
        {
            "Parameter": "95%_Confidence_Interval_CI",
            "Value": summary_row["95%_Confidence_Interval_CI"]
        },
        {
            "Parameter": "95%_Prediction_Interval_PI",
            "Value": summary_row["95%_Prediction_Interval_PI"]
        }
    ]

    for factor_row in factor_cv_rows:
        report_rows.append({
            "Parameter": f"{factor_row['Factor']}_CV",
            "Value": factor_row["Factor_CV"]
        })

    individual_report_df = pd.DataFrame(report_rows)

    return summary_row, individual_report_df


# =========================================================
# 6. Main program
# =========================================================
def main():
    """Run optimization stability analysis."""

    print("[INFO] Starting optimization stability analysis...")

    dataset_id_to_file = list_original_dataset_files(raw_data_folder)

    if len(dataset_id_to_file) == 0:
        raise ValueError("No original dataset files were found.")

    print(f"[INFO] Original datasets found: {len(dataset_id_to_file)}")

    all_model_results = load_all_model_results(
        folder_paths=folder_model_paths,
        summary_paths=summary_model_paths
    )

    if len(all_model_results) == 0:
        raise ValueError("No valid model optimization results were loaded.")

    os.makedirs(individual_report_folder, exist_ok=True)

    summary_rows = []
    error_records = []

    for dataset_id, original_file in dataset_id_to_file.items():
        try:
            summary_row, individual_report_df = analyze_one_dataset(
                dataset_id=dataset_id,
                original_file=original_file,
                all_model_results=all_model_results
            )

            summary_rows.append(summary_row)

            individual_report_path = os.path.join(
                individual_report_folder,
                f"{dataset_id}_Stability_Report.xlsx"
            )

            individual_report_df.to_excel(individual_report_path, index=False)

            print(
                f"[INFO] {dataset_id} processed | "
                f"n_models = {summary_row['n_models_used']} | "
                f"Response_CV = {summary_row['Response_CV']}"
            )

        except Exception as e:
            error_records.append({
                "Dataset_ID": dataset_id,
                "Error": str(e)
            })

            print(f"[WARNING] Failed to process {dataset_id}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    error_df = pd.DataFrame(error_records)

    output_dir = os.path.dirname(global_summary_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(global_summary_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Stability_Summary", index=False)
        error_df.to_excel(writer, sheet_name="Errors", index=False)

    print("\n[INFO] Optimization stability analysis completed.")
    print(f"[INFO] Successful datasets: {len(summary_df)}")
    print(f"[INFO] Error records: {len(error_df)}")
    print(f"[INFO] Global summary saved to: {global_summary_file}")
    print(f"[INFO] Individual reports saved to: {individual_report_folder}")


if __name__ == "__main__":
    main()