import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import time

# ==============================================================================
# Script: 01_model1_r2_max_loocv.py
# Description: Implements Model 1 (M1) using All-subset selection combined with 
#              Leave-One-Out Cross-Validation (LOOCV). The optimization criterion 
#              is the maximization of Adjusted R-squared (Rc^2).
# ==============================================================================

def evaluate_ols_model(X, y):
    """Fits an OLS model and returns key performance metrics."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model.rsquared, model.rsquared_adj, max(model.pvalues[1:] if len(model.pvalues) > 1 else [0]), model

def find_best_subset_by_adj_r2(X_pool, y):
    """
    Performs an exhaustive search (all-subset) to find the combination of 
    variables that maximizes the Adjusted R-squared.
    """
    best_model = None
    best_adj_r2 = -float('inf')
    best_combo = None
    
    num_features = len(X_pool.columns)
    for k in range(1, num_features + 1):
        for combo in combinations(X_pool.columns, k):
            subset_X = X_pool[list(combo)]
            try:
                r2, adj_r2, p_max, model = evaluate_ols_model(subset_X, y)
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_model = model
                    best_combo = list(combo)
            except:
                continue
    return best_model, best_combo

def run_m1_optimization(input_folder, output_path):
    """Main execution loop for M1: All-subset LOOCV optimization."""
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx") and not f.startswith("~$")]
    total_files = len(files)
    
    if total_files == 0:
        print(f"[WARN] No valid .xlsx files found in: {input_folder}")
        return

    results_summary = []
    global_start_time = time.time()

    print(f"[INFO] Initializing Model 1 (M1) global optimization...")
    print(f"[EXEC] Target: {total_files} datasets. Criterion: Max Adjusted R-squared.")

    for idx, filename in enumerate(files):
        file_start_time = time.time()
        file_path = os.path.join(input_folder, filename)
        
        try:
            # 1. Load data (Expecting 'Before' sheet for raw encoding)
            df = pd.read_excel(file_path, sheet_name='Before')
            X_orig = df.iloc[:, :-1]
            y_orig = df.iloc[:, -1].reset_index(drop=True)
            
            # Polynomial expansion
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = poly.fit_transform(X_orig)
            feature_names = poly.get_feature_names_out(X_orig.columns)
            X_all = pd.DataFrame(X_poly_raw, columns=feature_names)
            
            n_samples = len(y_orig)
            cv_preds = []
            cv_adj_r2 = []
            
            # 2. Leave-One-Out Cross-Validation (LOOCV) loop
            for i in range(n_samples):
                X_train = X_all.drop(index=i).reset_index(drop=True)
                y_train = y_orig.drop(index=i).reset_index(drop=True)
                X_test_single = X_all.iloc[[i]].reset_index(drop=True)
                
                # Search for best subset on training fold
                best_model, best_vars = find_best_subset_by_adj_r2(X_train, y_train)
                
                if best_model:
                    X_test_prepared = sm.add_constant(X_test_single[best_vars], has_constant='add')
                    X_test_prepared = X_test_prepared[best_model.model.exog_names]
                    
                    pred_val = best_model.predict(X_test_prepared).iloc[0]
                    cv_preds.append(pred_val)
                    cv_adj_r2.append(best_model.rsquared_adj)
            
            duration = time.time() - file_start_time
            
            # 3. Aggregate metrics
            if cv_preds:
                avg_adj_r2 = np.mean(cv_adj_r2)
                corr_matrix = np.corrcoef(cv_preds, y_orig)
                q2_cv = corr_matrix[0, 1]**2 if not np.isnan(corr_matrix[0, 1]) else 0
                
                results_summary.append({
                    'Dataset': filename,
                    'Avg_Adjusted_R2': round(avg_adj_r2, 4),
                    'LOOCV_Q2': round(q2_cv, 4),
                    'Compute_Time_Sec': round(duration, 2)
                })
                print(f"[STATUS] {filename} processed | Q2: {q2_cv:.4f} | Time: {duration:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

        # Progress reporting
        if (idx + 1) % 5 == 0 or (idx + 1) == total_files:
            elapsed = time.time() - global_start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (total_files - (idx + 1))
            print(f"[PROGRESS] {idx+1}/{total_files} completed | Est. Remaining: {remaining/60:.1f} mins")

    # Final export
    if results_summary:
        output_df = pd.DataFrame(results_summary)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_excel(output_path, index=False)
        print("-" * 60)
        print(f"[COMPLETE] M1 Optimization finished.")
        print(f"[INFO] Report generated at: {output_path}")
    else:
        print("[WARN] No results generated.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for input datasets
    INPUT_DIR = r"YOUR_INPUT_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the summary report
    OUTPUT_FILE = r"YOUR_OUTPUT_SUMMARY_PATH_HERE\M1_Optimization_Summary.xlsx"
    
    # -------------------------------------------------------------------------
    run_m1_optimization(INPUT_DIR, OUTPUT_FILE)