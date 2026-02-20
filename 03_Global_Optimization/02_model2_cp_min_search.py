import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

# ==============================================================================
# Script: 02_model2_cp_min_search.py
# Description: Implements Model 2 (M2) using All-subset selection. 
#              The optimization criterion is based on Mallows' Cp statistic.
#              The goal is to select a model where Cp is closest to p (number of 
#              parameters), minimizing |Cp - p| to balance bias and variance.
# ==============================================================================

def calculate_cp(model_rss, p, n, mse_full):
    """
    Calculates Mallows' Cp statistic.
    Formula: Cp = (RSS / MSE_full) - n + 2p
    """
    return (model_rss / mse_full) - n + (2 * p)

def evaluate_subset_cp(X_subset, y, mse_full):
    """
    Fits a subset model and calculates the Cp distance (|Cp - p|).
    """
    # Always include the constant term (intercept) in the model
    X_const = sm.add_constant(X_subset, has_constant='add')
    model = sm.OLS(y, X_const).fit()
    
    rss = sum(model.resid ** 2)
    p = X_const.shape[1] # Number of parameters including intercept
    n = len(y)
    
    cp_value = calculate_cp(rss, p, n, mse_full)
    # The criterion for M2: minimize the absolute difference between Cp and p
    cp_distance = abs(cp_value - p)
    
    return {
        'cp': cp_value,
        'cp_dist': cp_distance,
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'max_p': max(model.pvalues),
        'model': model
    }

def find_best_subset_by_cp(X_pool, y, mse_full):
    """
    Exhaustively searches all possible variable combinations (excluding full set)
    to find the one that minimizes |Cp - p|.
    """
    best_res = None
    min_dist = float('inf')
    best_combo = None
    
    num_features = len(X_pool.columns)
    # Iterate through subset sizes from 1 up to m-1 (excluding the full model)
    for k in range(1, num_features):
        for combo in combinations(X_pool.columns, k):
            subset_X = X_pool[list(combo)]
            try:
                res = evaluate_subset_cp(subset_X, y, mse_full)
                if res['cp_dist'] < min_dist:
                    min_dist = res['cp_dist']
                    best_res = res
                    best_combo = combo
            except:
                continue
    return best_res, best_combo

def process_cp_optimization(input_dir, output_file):
    """Batch processes datasets to find the optimal M2 model structure."""
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {input_dir}")
        return

    summary_results = []
    print(f"[INFO] Initializing Model 2 (M2) optimization via Mallows' Cp criterion...")
    print(f"[INFO] Batch processing {len(files)} files...")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        try:
            # Load experimental data: Note the use of 'Before' sheet as raw input
            df = pd.read_excel(file_path, sheet_name='Before')
            
            # Ensure proper separation of factors and response
            X_orig = df.iloc[:, :-1]
            y = df.iloc[:, -1].astype(float)

            # 1. Polynomial expansion (degree 2, no bias to avoid redundant constants)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = poly.fit_transform(X_orig)
            f_names = poly.get_feature_names_out(X_orig.columns)
            X_all = pd.DataFrame(X_poly_raw, columns=f_names)

            # 2. Fit Full Model to obtain MSE_full (Benchmark for Cp)
            X_full_const = sm.add_constant(X_all, has_constant='add')
            full_model = sm.OLS(y, X_full_const).fit()
            mse_full = full_model.mse_resid # Estimate of the true error variance

            # 3. Perform exhaustive search for optimal M2 subset
            best_res, best_vars = find_best_subset_by_cp(X_all, y, mse_full)

            if best_res:
                summary_results.append({
                    'Dataset_ID': filename,
                    'Best_Combination': ', '.join(best_vars),
                    'Cp_Value': round(best_res['cp'], 4),
                    'Cp_Distance_to_p': round(best_res['cp_dist'], 4),
                    'R2': round(best_res['r2'], 4),
                    'Adj_R2': round(best_res['adj_r2'], 4),
                    'Max_P_Value': round(best_res['max_p'], 4)
                })
                print(f"[STATUS] Optimized: {filename}")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    # Save summary table
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        summary_df.to_excel(output_file, index=False)
        print("-" * 60)
        print(f"[COMPLETE] M2 optimization finished. Results saved at:")
        print(f" -> {output_file}")
    else:
        print("[WARN] No optimization results were generated.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with your absolute directory path for input datasets
    INPUT_FOLDER = r"YOUR_INPUT_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your desired output path for the Cp summary report
    OUTPUT_PATH = r"YOUR_OUTPUT_REPORT_PATH_HERE\Optimal_M2_Cp_Summary.xlsx"
    
    # -------------------------------------------------------------------------
    process_cp_optimization(INPUT_FOLDER, OUTPUT_PATH)