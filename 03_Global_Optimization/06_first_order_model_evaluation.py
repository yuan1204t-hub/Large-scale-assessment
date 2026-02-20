import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut

# ==============================================================================
# Script: 06_first_order_model_evaluation.py
# Description: Evaluates the performance of the Full First-order Model (Linear).
#              Calculates goodness-of-fit (R2, Adj.R2), significance (max p-value),
#              and predictive stability via Leave-One-Out Cross-Validation (LOOCV).
# ==============================================================================

def evaluate_linear_models(folder_path, output_path):
    """
    Iterates through datasets to fit OLS linear models and perform LOOCV.
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Input directory not found: {folder_path}")
        return

    print(f"[INFO] Initializing Batch Evaluation for Full First-order (Linear) Models...")

    results = []
    # Fetch valid Excel files
    files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and not f.startswith("~$")]
    
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {folder_path}")
        return

    print(f"[INFO] Found {len(files)} files. Starting analysis...")

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            # Load and clean data (Expecting 'Before' sheet for raw/standardized input)
            df = pd.read_excel(file_path)
            
            # Ensure numeric types and drop NaNs for robust OLS
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            # Separate Features (X) and Response (y)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_const = sm.add_constant(X)

            # 1. Fit the Standard OLS Model
            model = sm.OLS(y, X_const).fit()
            
            # Extract Metrics
            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            max_p = model.pvalues[1:].max() if len(model.pvalues) > 1 else np.nan
            y_pred = model.predict(X_const)

            # 2. Leave-One-Out Cross-Validation (LOOCV)
            loo = LeaveOneOut()
            r2_loo_list = []
            adj_r2_loo_list = []

            for train_idx, test_idx in loo.split(X_const):
                X_train, y_train = X_const.iloc[train_idx], y.iloc[train_idx]
                
                model_loo = sm.OLS(y_train, X_train).fit()
                y_pred_train = model_loo.predict(X_train)
                
                # Manual calculation of LOOCV Metrics
                ss_res = ((y_train - y_pred_train) ** 2).sum()
                ss_tot = ((y_train - y_train.mean()) ** 2).sum()

                if ss_tot != 0:
                    r2_loo = 1 - (ss_res / ss_tot)
                else:
                    r2_loo = np.nan

                # Calculate Adjusted R2 for LOOCV
                n_train = len(y_train)
                p_train = X_train.shape[1] - 1
                if n_train - p_train - 1 > 0:
                    adj_r2_loo = 1 - (1 - r2_loo) * (n_train - 1) / (n_train - p_train - 1)
                else:
                    adj_r2_loo = np.nan
                
                r2_loo_list.append(r2_loo)
                adj_r2_loo_list.append(adj_r2_loo)

            # Aggregate LOOCV results
            mean_r2_loo = np.mean([val for val in r2_loo_list if not np.isnan(val)])
            mean_adj_r2_loo = np.mean([val for val in adj_r2_loo_list if not np.isnan(val)])

            # Append structured results using professional headers
            results.append({
                "Dataset_ID": file,
                "R2": round(r2, 4),
                "Adjusted_R2": round(adj_r2, 4),
                "Max_P_Value": round(max_p, 4),
                "Mean_LOOCV_R2": round(mean_r2_loo, 4),
                "Mean_LOOCV_AdjR2": round(mean_adj_r2_loo, 4),
                "Predicted_Sequence": ','.join(map(str, y_pred.round(6))),
                "Actual_Sequence": ','.join(map(str, y.values.round(6)))
            })
            print(f"[STATUS] Evaluated: {file}")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # Export to Excel
    if results:
        result_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_excel(output_path, index=False)
        print("-" * 60)
        print(f"[COMPLETE] First-order evaluation finished.")
        print(f"[INFO] Final report saved to: {output_path}")
    else:
        print("[WARN] No evaluation results generated.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for first-order datasets
    INPUT_DIR = r"YOUR_INPUT_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the linear evaluation report
    OUTPUT_FILE_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\First_Order_Evaluation_Summary.xlsx"
    
    # -------------------------------------------------------------------------
    evaluate_linear_models(INPUT_DIR, OUTPUT_FILE_PATH)