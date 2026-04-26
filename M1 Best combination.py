# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch read Excel files from a specified folder.
2. By default, all columns except the last one are used as independent variables X,
   and the last column is used as the dependent variable y.
3. Generate a full quadratic candidate feature pool, including linear terms,
   squared terms, and two-way interaction terms.
4. Perform exhaustive subset selection for each file using the maximum adjusted
   R-squared as the selection criterion.
5. Fit the optimal full-data M1 model for each dataset and calculate fitting
   and diagnostic indicators, including R2, adjusted R2, RMSE, MAE, and the
   maximum p-value of selected predictors.
6. Perform nested leave-one-out cross-validation, where variable selection is
   repeated within each training fold before predicting the left-out sample.
7. Output the selected variables, fitting results, LOOCV prediction results,
   fold-level errors, and batch-level summary results.

Applicable scenarios:
- A folder contains many Excel files.
- In each file, the first n-1 columns are independent variables.
- In each file, the last column is the dependent variable.
- The aim is to compare and evaluate M1 regression models based on exhaustive
  subset selection from a full quadratic candidate space.
- The prediction performance needs to be evaluated using nested LOOCV, so that
  each predicted value is obtained from a model trained without that sample.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def evaluate_ols_model(X, y):
    """Fit an OLS model and return key performance metrics."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    return (
        model.rsquared,
        model.rsquared_adj,
        max(model.pvalues[1:] if len(model.pvalues) > 1 else [0]),
        model
    )


def find_best_subset_by_adj_r2(X_pool, y):
    """
    Find the best variable subset according to the maximum adjusted R-squared.

    This function performs exhaustive subset search over the predefined
    full quadratic candidate space.
    """
    best_model = None
    best_adj_r2 = -float("inf")
    best_combo = None
    best_p_max = np.nan

    num_features = len(X_pool.columns)

    for k in range(1, num_features + 1):
        for combo in combinations(X_pool.columns, k):
            subset_X = X_pool[list(combo)]

            try:
                r2, adj_r2, p_max, model = evaluate_ols_model(subset_X, y)

                if np.isfinite(adj_r2) and adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_model = model
                    best_combo = list(combo)
                    best_p_max = p_max

            except Exception:
                continue

    return best_model, best_combo, best_p_max


def run_nested_loocv_m1(X_all, y):
    """
    Run nested LOOCV for M1.

    For each LOOCV fold:
    1. Leave one sample out.
    2. Use the remaining samples to perform exhaustive subset selection.
    3. Fit the selected model on the training samples.
    4. Predict the left-out sample.

    This procedure matches the manuscript statement that the predicted value
    of each sample is obtained from the model trained after excluding that sample.
    """
    loo = LeaveOneOut()

    y_true_list = []
    y_pred_list = []
    selected_vars_list = []
    fold_error_list = []

    X_all = X_all.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold_idx, (train_index, test_index) in enumerate(loo.split(X_all), start=1):
        X_train_pool = X_all.iloc[train_index].reset_index(drop=True)
        X_test_pool = X_all.iloc[test_index].reset_index(drop=True)

        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

        try:
            best_model, best_vars, p_max_val = find_best_subset_by_adj_r2(
                X_train_pool,
                y_train
            )

            if best_model is None or best_vars is None:
                raise ValueError("No valid model found in this LOOCV fold.")

            X_test_selected = sm.add_constant(
                X_test_pool[list(best_vars)],
                has_constant="add"
            )

            y_pred = best_model.predict(X_test_selected).iloc[0]

            y_true_list.append(y_test.iloc[0])
            y_pred_list.append(y_pred)
            selected_vars_list.append(", ".join(best_vars))

            fold_error_list.append({
                "Fold": fold_idx,
                "Error_Message": ""
            })

        except Exception as e:
            y_true_list.append(y_test.iloc[0])
            y_pred_list.append(np.nan)
            selected_vars_list.append("")

            fold_error_list.append({
                "Fold": fold_idx,
                "Error_Message": str(e)
            })

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)

    valid_mask = ~np.isnan(y_pred_arr)

    if valid_mask.sum() == 0:
        return {
            "PRESS": np.nan,
            "LOOCV_R2": np.nan,
            "LOOCV_RMSE": np.nan,
            "LOOCV_MAE": np.nan,
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "selected_vars": selected_vars_list,
            "fold_errors": fold_error_list
        }

    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]

    press = np.sum((y_true_valid - y_pred_valid) ** 2)
    loocv_rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    loocv_mae = mean_absolute_error(y_true_valid, y_pred_valid)

    if len(y_true_valid) > 1:
        loocv_r2 = r2_score(y_true_valid, y_pred_valid)
    else:
        loocv_r2 = np.nan

    return {
        "PRESS": press,
        "LOOCV_R2": loocv_r2,
        "LOOCV_RMSE": loocv_rmse,
        "LOOCV_MAE": loocv_mae,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "selected_vars": selected_vars_list,
        "fold_errors": fold_error_list
    }


def run_m1_optimization(input_folder, output_summary_path):
    """
    Run M1 optimization.

    The full-data model is used for overall fitting and diagnostic metrics.
    Nested LOOCV is used for predictive performance metrics.
    """

    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".xlsx") and not f.startswith("~$")
    ]

    total_files = len(files)

    if total_files == 0:
        print(f"[WARN] No valid .xlsx files found in: {input_folder}")
        return

    results_summary = []
    predictions_list = []
    error_files_list = []
    fold_error_records = []

    global_start_time = time.time()

    print("[INFO] Initializing M1 optimization with nested LOOCV...")
    print(f"[EXEC] Target: {total_files} datasets.")

    for idx, filename in enumerate(files):
        file_start_time = time.time()
        file_path = os.path.join(input_folder, filename)

        try:
            # 1. Load data
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

            if df.shape[1] < 2:
                raise ValueError(
                    "Insufficient number of columns. "
                    "At least one X column and one y column are required."
                )

            # By default, all columns except the last one are independent variables,
            # and the last column is the dependent variable.
            X_orig = df.iloc[:, :-1]
            y_orig = df.iloc[:, -1].astype(float).reset_index(drop=True)

            if X_orig.isnull().any().any() or y_orig.isnull().any():
                raise ValueError("Missing values exist in the data.")

            # 2. Generate the full quadratic candidate feature pool
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = poly.fit_transform(X_orig)
            feature_names = poly.get_feature_names_out(X_orig.columns)
            X_all = pd.DataFrame(X_poly_raw, columns=feature_names)

            # 3. Full-data model selection for fitting and diagnostic metrics
            full_best_model, full_best_vars, full_p_max = find_best_subset_by_adj_r2(
                X_all,
                y_orig
            )

            if full_best_model is None or full_best_vars is None:
                raise ValueError("No valid full-data M1 model found.")

            full_best_vars_str = ", ".join(full_best_vars)

            X_full_selected = sm.add_constant(
                X_all[list(full_best_vars)],
                has_constant="add"
            )
            y_refit_pred = full_best_model.predict(X_full_selected)

            r2_refit = full_best_model.rsquared
            adj_r2_refit = full_best_model.rsquared_adj
            rmse_refit = np.sqrt(mean_squared_error(y_orig, y_refit_pred))
            mae_refit = mean_absolute_error(y_orig, y_refit_pred)

            # 4. Nested LOOCV
            loocv_results = run_nested_loocv_m1(X_all, y_orig)

            press = loocv_results["PRESS"]
            loocv_r2 = loocv_results["LOOCV_R2"]
            loocv_rmse = loocv_results["LOOCV_RMSE"]
            loocv_mae = loocv_results["LOOCV_MAE"]

            duration = time.time() - file_start_time

            results_summary.append({
                "Dataset_ID": filename,
                "n_samples": df.shape[0],
                "n_features_original": X_orig.shape[1],
                "n_features_candidate": X_all.shape[1],
                "n_selected_variables_full_data": len(full_best_vars),
                "Best_Variables_full_data": full_best_vars_str,

                "R2_refit": round(r2_refit, 4),
                "Adjusted_R2_refit": round(adj_r2_refit, 4),
                "RMSE_refit": round(rmse_refit, 4),
                "MAE_refit": round(mae_refit, 4),
                "Max_P_Value_full_data": round(full_p_max, 4),

                "PRESS": round(press, 4) if not pd.isna(press) else np.nan,
                "LOOCV_R2": round(loocv_r2, 4) if not pd.isna(loocv_r2) else np.nan,
                "LOOCV_RMSE": round(loocv_rmse, 4) if not pd.isna(loocv_rmse) else np.nan,
                "LOOCV_MAE": round(loocv_mae, 4) if not pd.isna(loocv_mae) else np.nan,

                "Compute_Time_Sec": round(duration, 2)
            })

            # 5. Save LOOCV prediction details
            for sample_idx, (actual_val, pred_val, selected_vars) in enumerate(
                zip(
                    loocv_results["y_true"],
                    loocv_results["y_pred"],
                    loocv_results["selected_vars"]
                ),
                start=1
            ):
                predictions_list.append({
                    "Dataset_ID": filename,
                    "Sample_Index": sample_idx,
                    "Actual": actual_val,
                    "LOOCV_Predicted": pred_val,
                    "LOOCV_Residual": actual_val - pred_val if not pd.isna(pred_val) else np.nan,
                    "Selected_Variables_in_Fold": selected_vars
                })

            # 6. Save fold-level error records
            for fold_error in loocv_results["fold_errors"]:
                if fold_error["Error_Message"]:
                    fold_error_records.append({
                        "Dataset_ID": filename,
                        "Fold": fold_error["Fold"],
                        "Error_Message": fold_error["Error_Message"]
                    })

            print(
                f"[STATUS] {filename} processed | "
                f"Nested LOOCV RMSE = {loocv_rmse:.6f} | "
                f"Time: {duration:.2f}s"
            )

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            error_files_list.append({
                "Filename": filename,
                "Error_Message": str(e)
            })

        # Report progress
        if (idx + 1) % 5 == 0 or (idx + 1) == total_files:
            elapsed = time.time() - global_start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (total_files - (idx + 1))

            print(
                f"[PROGRESS] {idx + 1}/{total_files} completed | "
                f"Estimated remaining time: {remaining / 60:.1f} mins"
            )

    # ==========================================================================
    # Final export
    # ==========================================================================
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)

    if results_summary:
        pd.DataFrame(results_summary).to_excel(output_summary_path, index=False)
        print(f"[INFO] Summary report saved to: {output_summary_path}")

    if predictions_list:
        output_pred_path = output_summary_path.replace("Summary", "Predictions")
        pd.DataFrame(predictions_list).to_excel(output_pred_path, index=False)
        print(f"[INFO] LOOCV prediction details saved to: {output_pred_path}")

    if error_files_list:
        output_error_path = output_summary_path.replace("Summary", "Error_Files")
        pd.DataFrame(error_files_list).to_excel(output_error_path, index=False)
        print(f"[INFO] Error file list saved to: {output_error_path}")
    else:
        print("[INFO] No file-level errors occurred during processing.")

    if fold_error_records:
        output_fold_error_path = output_summary_path.replace("Summary", "Fold_Error_Files")
        pd.DataFrame(fold_error_records).to_excel(output_fold_error_path, index=False)
        print(f"[INFO] Fold-level error records saved to: {output_fold_error_path}")
    else:
        print("[INFO] No fold-level errors occurred during LOOCV.")


if __name__ == "__main__":
    # Please enter your path here
    INPUT_DIR = r"Please enter your path here"
    OUTPUT_FILE = r"Please enter your path here"

    run_m1_optimization(INPUT_DIR, OUTPUT_FILE)