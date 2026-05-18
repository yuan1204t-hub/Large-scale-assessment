# -*- coding: utf-8 -*-
"""
Purpose:
1. Read residual diagnostic summary files from seven models.
2. Extract common diagnostic indicators from each model result file.
3. The extracted indicators include:
   - pn
   - pm
   - Residual_Mean
   - Residual_SD
   - Max_Abs_Residual
   - AE_IQR
4. Standardize column names using model prefixes.
5. Merge all model diagnostic results into one wide-format table by file name.
6. Save the merged residual diagnostic summary as an Excel file.

Applicable scenarios:
- Residual diagnostic results have already been calculated for multiple models.
- The models include traditional regression models and machine learning models.
- The user wants to compare diagnostic performance across seven models.
- The merged table will be used for Chapter 3.3 diagnostic comparison,
  descriptive statistics, normality tests, and significance tests.
"""

import os
import pandas as pd


# =========================================================
# 1. Path settings
# =========================================================
# Please enter the residual diagnostic result file paths for each model.
model_file_dict = {
    "M0": r"Please enter your path here",
    "M1": r"Please enter your path here",
    "M2": r"Please enter your path here",
    "SVM": r"Please enter your path here",
    "Ridge": r"Please enter your path here",
    "PLS": r"Please enter your path here",
    "GPR": r"Please enter your path here",
}

# Please enter the output Excel file path.
output_file = r"Please enter your path here"


# =========================================================
# 2. Model and metric settings
# =========================================================
model_order = ["M0", "M1", "M2", "SVM", "Ridge", "PLS", "GPR"]

target_metrics = [
    "pn",
    "pm",
    "Residual_Mean",
    "Residual_SD",
    "Max_Abs_Residual",
    "AE_IQR"
]

# If some regression files contain multiple sheets, specify the sheet name here.
# If the file contains only one sheet or the first sheet should be used, set it to None.
model_sheet_dict = {
    "M0": "M0_Residual_Diagnostics",
    "M1": "M1_Residual_Diagnostics",
    "M2": "M2_Residual_Diagnostics",
    "SVM": None,
    "Ridge": None,
    "PLS": None,
    "GPR": None,
}


# =========================================================
# 3. Utility functions
# =========================================================
def read_excel_file(file_path, sheet_name=None):
    """
    Read an Excel file.

    If sheet_name is None, the first sheet will be read by default.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if sheet_name is None:
        return pd.read_excel(file_path)

    return pd.read_excel(file_path, sheet_name=sheet_name)


def standardize_file_name_column(df, model_name):
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
        f"{model_name} file does not contain File_Name, Dataset_ID, or ID column. "
        f"Existing columns: {list(df.columns)}"
    )


def read_model_diagnostic_summary(file_path, model_name, sheet_name=None):
    """
    Read one model residual diagnostic summary file and standardize column names.

    Output column format:
    File_Name, Model_pn, Model_pm, Model_Residual_Mean, ...
    """

    df = read_excel_file(file_path=file_path, sheet_name=sheet_name)
    df = standardize_file_name_column(df, model_name=model_name)

    missing_metrics = [
        metric for metric in target_metrics
        if metric not in df.columns
    ]

    if missing_metrics:
        raise ValueError(
            f"{model_name} file is missing required columns: {missing_metrics}\n"
            f"Existing columns: {list(df.columns)}"
        )

    keep_columns = ["File_Name"] + target_metrics
    output_df = df[keep_columns].copy()

    for metric in target_metrics:
        output_df[metric] = pd.to_numeric(output_df[metric], errors="coerce")

    rename_dict = {
        metric: f"{model_name}_{metric}"
        for metric in target_metrics
    }

    output_df = output_df.rename(columns=rename_dict)

    return output_df


def sort_columns(merged_df):
    """
    Sort columns by model first and metric second.

    Output order:
    File_Name,
    M0_pn, M0_pm, ...,
    M1_pn, M1_pm, ...,
    ...
    GPR_pn, GPR_pm, ...
    """

    ordered_columns = ["File_Name"]

    for model in model_order:
        for metric in target_metrics:
            col_name = f"{model}_{metric}"
            if col_name in merged_df.columns:
                ordered_columns.append(col_name)

    remaining_columns = [
        col for col in merged_df.columns
        if col not in ordered_columns
    ]

    ordered_columns.extend(remaining_columns)

    return merged_df[ordered_columns]


# =========================================================
# 4. Main program
# =========================================================
def main():
    """Merge residual diagnostic indicators from all models."""

    merged_df = None
    error_records = []

    print("[INFO] Starting residual diagnostic data extraction...")

    for model_name in model_order:
        file_path = model_file_dict.get(model_name)
        sheet_name = model_sheet_dict.get(model_name)

        if file_path is None or file_path.strip() == "":
            error_records.append({
                "Model": model_name,
                "File_Path": "",
                "Sheet_Name": sheet_name,
                "Error": "No file path was provided."
            })
            print(f"[WARNING] No file path was provided for {model_name}.")
            continue

        try:
            print(f"[INFO] Reading {model_name}: {file_path}")

            model_df = read_model_diagnostic_summary(
                file_path=file_path,
                model_name=model_name,
                sheet_name=sheet_name
            )

            if merged_df is None:
                merged_df = model_df
            else:
                merged_df = pd.merge(
                    merged_df,
                    model_df,
                    on="File_Name",
                    how="outer"
                )

        except Exception as e:
            error_records.append({
                "Model": model_name,
                "File_Path": file_path,
                "Sheet_Name": sheet_name,
                "Error": str(e)
            })

            print(f"[ERROR] Failed to process {model_name}: {e}")

    if merged_df is None:
        print("[ERROR] No valid residual diagnostic files were read.")
        return

    merged_df = sort_columns(merged_df)
    merged_df = merged_df.sort_values(by="File_Name").reset_index(drop=True)

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        merged_df.to_excel(
            writer,
            sheet_name="Merged_Wide_Table",
            index=False
        )

        if error_records:
            error_df = pd.DataFrame(error_records)
            error_df.to_excel(
                writer,
                sheet_name="Errors",
                index=False
            )

    print("\n[INFO] Residual diagnostic data extraction completed.")
    print(f"[INFO] Number of datasets: {merged_df.shape[0]}")
    print(f"[INFO] Number of columns: {merged_df.shape[1]}")
    print(f"[INFO] Output saved to: {output_file}")

    if error_records:
        print(f"[WARNING] Number of error records: {len(error_records)}")
    else:
        print("[INFO] No errors occurred during data extraction.")


if __name__ == "__main__":
    main()