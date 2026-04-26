# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch read Excel files from a specified folder.
2. By default, all columns except the last one are treated as independent
   variables X, and the last column is treated as the dependent variable y.
3. Construct a full quadratic candidate feature space based on the original
   independent variables, including main effects, squared terms, and two-way
   interaction terms.
4. For each dataset, fit the full quadratic model to obtain the residual mean
   squared error, which is used as the error variance estimate for Mallows' Cp.
5. Perform exhaustive subset selection and identify the M2 model whose Mallows'
   Cp value is closest to the number of model parameters p.
6. Use the selected full-data M2 model to calculate model fitting and diagnostic
   indicators, including R2, adjusted R2, RMSE, MAE, maximum p-value, Cp value,
   and Cp distance to p.
7. Evaluate predictive performance using nested leave-one-out cross-validation
   nested LOOCV. In each fold, the full model is first fitted on the training
   samples to estimate mse_full, then subset selection is repeated based on
   Mallows' Cp, and the selected model is used to predict the left-out sample.
8. Save batch summary results, LOOCV prediction details, file-level error
   records, and fold-level error records.

Applicable scenarios:
- A folder contains multiple Excel files.
- Each Excel file follows the structure that the first n-1 columns are
  independent variables and the last column is the dependent variable.
- The analysis focuses on M2 model construction based on Mallows' Cp from a
  full quadratic regression candidate space.
- The study requires both full-data fitting results and nested LOOCV-based
  predictive performance indicators.
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

def calculate_cp(model_rss, p, n, mse_full):
    """Calculate Mallows' Cp statistic."""
    return (model_rss / mse_full) - n + (2 * p)


def evaluate_subset_cp(X_subset, y, mse_full):
    """Fit a subset model and calculate the Cp distance |Cp - p|."""
    X_const = sm.add_constant(X_subset, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    rss = np.sum(model.resid ** 2)
    p = X_const.shape[1]
    n = len(y)

    cp_value = calculate_cp(rss, p, n, mse_full)
    cp_distance = abs(cp_value - p)

    p_max = max(model.pvalues[1:] if len(model.pvalues) > 1 else [0])

    return {
        "cp": cp_value,
        "cp_dist": cp_distance,
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "max_p": p_max,
        "model": model
    }


def find_best_subset_by_cp(X_pool, y, mse_full):
    """
    Search all variable combinations and select the subset with the minimum |Cp - p|.

    This function performs exhaustive subset search over the predefined
    full quadratic candidate space.
    """
    best_res = None
    min_dist = float("inf")
    best_combo = None

    num_features = len(X_pool.columns)

    for k in range(1, num_features):
        for combo in combinations(X_pool.columns, k):
            subset_X = X_pool[list(combo)]

            try:
                res = evaluate_subset_cp(subset_X, y, mse_full)

                if np.isfinite(res["cp_dist"]) and res["cp_dist"] < min_dist:
                    min_dist = res["cp_dist"]
                    best_res = res
                    best_combo = list(combo)

            except Exception:
                continue

    return best_res, best_combo


def fit_full_model_and_get_mse(X_all, y):
    """
    Fit the full quadratic model and return the residual mean squared error.

    This MSE is used as the error variance estimate in Mallows' Cp.
    """
    X_full_const = sm.add_constant(X_all, has_constant="add")
    full_model = sm.OLS(y, X_full_const).fit()
    mse_full = full_model.mse_resid

    return full_model, mse_full


def run_nested_loocv_m2(X_all, y):
    """
    Run nested LOOCV for M2.

    For each LOOCV fold:
    1. Leave one sample out.
    2. Fit the full model on the training samples to obtain mse_full.
    3. Use the training samples to perform exhaustive subset selection based on Mallows' Cp.
    4. Fit the selected model on the training samples.
    5. Predict the left-out sample.

    This procedure matches the manuscript statement that the predicted value
    of each sample is obtained from the model trained after excluding that sample.
    """
    loo = LeaveOneOut()

    y_true_list = []
    y_pred_list = []
    selected_vars_list = []
    cp_value_list = []
    cp_dist_list = []
    fold_error_list = []

    X_all = X_all.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold_idx, (train_index, test_index) in enumerate(loo.split(X_all), start=1):
        X_train_pool = X_all.iloc[train_index].reset_index(drop=True)
        X_test_pool = X_all.iloc[test_index].reset_index(drop=True)

        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

        try:
            # Fit the full model within the current training fold
            _, mse_full_train = fit_full_model_and_get_mse(X_train_pool, y_train)

            # Search the best Cp subset within the current training fold
            best_res, best_vars = find_best_subset_by_cp(
                X_pool=X_train_pool,
                y=y_train,
                mse_full=mse_full_train
            )

            if best_res is None or best_vars is None:
                raise ValueError("No valid model found in this LOOCV fold.")

            # Predict the left-out sample using the fold-specific selected model
            X_test_selected = sm.add_constant(
                X_test_pool[list(best_vars)],
                has_constant="add"
            )

            y_pred = best_res["model"].predict(X_test_selected).iloc[0]

            y_true_list.append(y_test.iloc[0])
            y_pred_list.append(y_pred)
            selected_vars_list.append(", ".join(best_vars))
            cp_value_list.append(best_res["cp"])
            cp_dist_list.append(best_res["cp_dist"])

            fold_error_list.append({
                "Fold": fold_idx,
                "Error_Message": ""
            })

        except Exception as e:
            y_true_list.append(y_test.iloc[0])
            y_pred_list.append(np.nan)
            selected_vars_list.append("")
            cp_value_list.append(np.nan)
            cp_dist_list.append(np.nan)

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
            "cp_values": cp_value_list,
            "cp_distances": cp_dist_list,
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
        "cp_values": cp_value_list,
        "cp_distances": cp_dist_list,
        "fold_errors": fold_error_list
    }


def process_cp_optimization(input_dir, output_file):
    """
    Run M2 optimization.

    The full-data selected model is used for overall fitting and diagnostic metrics.
    Nested LOOCV is used for predictive performance metrics.
    """

    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return

    files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".xlsx") and not f.startswith("~$")
    ]

    total_files = len(files)

    if total_files == 0:
        print(f"[WARN] No valid .xlsx files found in: {input_dir}")
        return

    summary_results = []
    predictions_list = []
    error_files_list = []
    fold_error_records = []

    global_start_time = time.time()

    print("[INFO] Initializing M2 optimization based on Mallows' Cp with nested LOOCV...")
    print(f"[EXEC] Target: {total_files} datasets.")

    for idx, filename in enumerate(files):
        file_start_time = time.time()
        file_path = os.path.join(input_dir, filename)

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
            y = df.iloc[:, -1].astype(float).reset_index(drop=True)

            if X_orig.isnull().any().any() or y.isnull().any():
                raise ValueError("Missing values exist in the data.")

            # 2. Generate the full quadratic candidate feature pool
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = poly.fit_transform(X_orig)
            feature_names = poly.get_feature_names_out(X_orig.columns)
            X_all = pd.DataFrame(X_poly_raw, columns=feature_names)

            # 3. Full-data model selection for fitting and diagnostic metrics
            _, mse_full = fit_full_model_and_get_mse(X_all, y)

            full_best_res, full_best_vars = find_best_subset_by_cp(
                X_pool=X_all,
                y=y,
                mse_full=mse_full
            )

            if full_best_res is None or full_best_vars is None:
                raise ValueError("No valid full-data M2 model found.")

            full_best_vars_str = ", ".join(full_best_vars)

            # Full-data refit metrics
            full_best_model = full_best_res["model"]
            y_refit_pred = full_best_model.fittedvalues

            r2_refit = full_best_res["r2"]
            adj_r2_refit = full_best_res["adj_r2"]
            rmse_refit = np.sqrt(mean_squared_error(y, y_refit_pred))
            mae_refit = mean_absolute_error(y, y_refit_pred)

            # 4. Nested LOOCV
            loocv_results = run_nested_loocv_m2(X_all, y)

            press = loocv_results["PRESS"]
            loocv_r2 = loocv_results["LOOCV_R2"]
            loocv_rmse = loocv_results["LOOCV_RMSE"]
            loocv_mae = loocv_results["LOOCV_MAE"]

            duration = time.time() - file_start_time

            summary_results.append({
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
                "Max_P_Value_full_data": round(full_best_res["max_p"], 4),

                "Cp_Value_full_data": round(full_best_res["cp"], 4),
                "Cp_Distance_to_p_full_data": round(full_best_res["cp_dist"], 4),

                "PRESS": round(press, 4) if not pd.isna(press) else np.nan,
                "LOOCV_R2": round(loocv_r2, 4) if not pd.isna(loocv_r2) else np.nan,
                "LOOCV_RMSE": round(loocv_rmse, 4) if not pd.isna(loocv_rmse) else np.nan,
                "LOOCV_MAE": round(loocv_mae, 4) if not pd.isna(loocv_mae) else np.nan,

                "Compute_Time_Sec": round(duration, 2)
            })

            # 5. Save LOOCV prediction details
            for sample_idx, (actual_val, pred_val, selected_vars, cp_val, cp_dist) in enumerate(
                zip(
                    loocv_results["y_true"],
                    loocv_results["y_pred"],
                    loocv_results["selected_vars"],
                    loocv_results["cp_values"],
                    loocv_results["cp_distances"]
                ),
                start=1
            ):
                predictions_list.append({
                    "Dataset_ID": filename,
                    "Sample_Index": sample_idx,
                    "Actual": actual_val,
                    "LOOCV_Predicted": pred_val,
                    "LOOCV_Residual": actual_val - pred_val if not pd.isna(pred_val) else np.nan,
                    "Selected_Variables_in_Fold": selected_vars,
                    "Cp_Value_in_Fold": cp_val,
                    "Cp_Distance_to_p_in_Fold": cp_dist
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if summary_results:
        pd.DataFrame(summary_results).to_excel(output_file, index=False)
        print(f"[INFO] Summary report saved to: {output_file}")

    if predictions_list:
        output_pred_path = output_file.replace("Summary", "Predictions")
        pd.DataFrame(predictions_list).to_excel(output_pred_path, index=False)
        print(f"[INFO] LOOCV prediction details saved to: {output_pred_path}")

    if error_files_list:
        output_error_path = output_file.replace("Summary", "Error_Files")
        pd.DataFrame(error_files_list).to_excel(output_error_path, index=False)
        print(f"[INFO] Error file list saved to: {output_error_path}")
    else:
        print("[INFO] No file-level errors occurred during processing.")

    if fold_error_records:
        output_fold_error_path = output_file.replace("Summary", "Fold_Error_Files")
        pd.DataFrame(fold_error_records).to_excel(output_fold_error_path, index=False)
        print(f"[INFO] Fold-level error records saved to: {output_fold_error_path}")
    else:
        print("[INFO] No fold-level errors occurred during nested LOOCV.")


if __name__ == "__main__":
    # Please enter your path here
    INPUT_FOLDER = r"Please enter your path here"
    OUTPUT_PATH = r"Please enter your path here"

    process_cp_optimization(INPUT_FOLDER, OUTPUT_PATH)