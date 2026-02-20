import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import itertools
import ast

# ==============================================================================
# Script: 01_m1_global_optimization_grid_search.py
# Section: 3.3 Stability Assessment
# Description: Performs a global grid search for each optimized M1 model to 
#              find the theoretical maximum response (Y_max) and the 
#              corresponding optimal factor settings.
# ==============================================================================

def execute_process_optimization(input_folder, combo_file, output_folder):
    """
    Reads optimized model structures and raw data to locate the global peak
    on the response surface via exhaustive grid search.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 1. Load the Best Combination mapping for M1
    print("[INFO] Loading optimized model structures (Best Subsets)...")
    try:
        combo_df = pd.read_excel(combo_file)
        
        # Identify key columns (Dataset Name and Best Combination)
        id_col = next((c for c in combo_df.columns if 'Dataset' in str(c) or '数据集' in str(c)), None)
        combo_col = next((c for c in combo_df.columns if 'Combination' in str(c) or '最佳组合' in str(c)), None)

        if not id_col or not combo_col:
            print(f"[ERROR] Required columns not found. Headers: {combo_df.columns.tolist()}")
            return

        # Standardize the combination column into list format
        combo_df[combo_col] = combo_df[combo_col].apply(
            lambda x: ast.literal_eval(str(x)) if str(x).startswith('[') else str(x).split(',')
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model structure file: {e}")
        return

    # 2. Process each dataset file
    xlsx_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    total_files = len(xlsx_files)

    print(f"[INFO] Found {total_files} datasets. Initializing grid search...")

    for idx, file in enumerate(xlsx_files, start=1):
        file_path = os.path.join(input_folder, file)
        data = pd.read_excel(file_path) # Reads the first worksheet by default
        
        # Identify variables and target
        raw_vars = data.columns[:-1].tolist()
        target_name = data.columns[-1]

        # Match optimized subset for this specific dataset
        match_row = combo_df[combo_df[id_col] == file]
        if match_row.empty:
            print(f"[STATUS] Skipping {file}: No optimized structure found.")
            continue

        best_combo = [v.strip() for v in match_row.iloc[0][combo_col]]

        # Identify unique base variables involved in the terms
        base_var_candidates = set()
        for var in best_combo:
            parts = var.replace('^2', '').split(' ')
            base_var_candidates.update([p.strip() for p in parts if p.strip()])

        used_vars = sorted([v for v in base_var_candidates if v in raw_vars])

        if not used_vars:
            print(f"[ERROR] {file}: Base variables in subset do not match raw data columns.")
            continue

        # 3. Re-fit the optimized regression model
        try:
            y = data[target_name]
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly.fit(data[used_vars])
            
            all_features = poly.get_feature_names_out(used_vars)
            selected_idx = [i for i, name in enumerate(all_features) if name in best_combo]

            if not selected_idx:
                print(f"[ERROR] {file}: Polynomial expansion failed to match selected terms.")
                continue

            X_train_poly = poly.transform(data[used_vars])[:, selected_idx]
            model = LinearRegression().fit(X_train_poly, y)

            # 4. Construct Variable Ranges for Grid Search (100 steps per dimension)
            var_ranges = []
            for var in used_vars:
                vmin, vmax = data[var].min(), data[var].max()
                step = max((vmax - vmin) / 100, 0.01)
                values = np.arange(vmin, vmax + step / 2, step)
                var_ranges.append(values)

            # 5. Global Exhaustive Search (Logic Preserved)
            max_pred = -np.inf
            best_settings = None
            total_comb = np.prod([len(r) for r in var_ranges])

            print(f"[EXEC] Searching {file}: {total_comb} combinations...")
            
            for comb in itertools.product(*var_ranges):
                comb_df = pd.DataFrame([comb], columns=used_vars)
                comb_poly = poly.transform(comb_df)[:, selected_idx]
                y_pred = model.predict(comb_poly)[0]
                
                if y_pred > max_pred:
                    max_pred = y_pred
                    best_settings = comb

            # 6. Save Optimization results
            output_df = pd.DataFrame([['Max_Response'] + used_vars, [max_pred] + list(best_settings)])
            output_file = f"{os.path.splitext(file)[0]}_optimized_result.xlsx"
            output_df.to_excel(os.path.join(output_folder, output_file), header=False, index=False)

            print(f"[STATUS] Completed {idx}/{total_files}: {file} | Max Y: {max_pred:.4f}")

        except Exception as e:
            print(f"[ERROR] Failure in processing {file}: {e}")

    print(f"\n[COMPLETE] Global grid search finished. Results in: {output_folder}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Directory containing raw experimental datasets
    INPUT_DATA_DIR = r"YOUR_INPUT_DATA_DIRECTORY"
    
    # [TODO] Path to the M1 optimization results (Best Combination summary)
    M1_STRUCTURE_FILE = r"YOUR_OPTIMIZED_COMBO_FILE.xlsx"
    
    # [TODO] Destination folder for the grid search peak results
    OUTPUT_RESULTS_DIR = r"YOUR_OUTPUT_RESULTS_DIRECTORY"

    # -------------------------------------------------------------------------
    execute_process_optimization(INPUT_DATA_DIR, M1_STRUCTURE_FILE, OUTPUT_RESULTS_DIR)