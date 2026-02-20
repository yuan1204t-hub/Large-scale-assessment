import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import itertools

# ==============================================================================
# Script: 02_m2_global_optimization_grid_search.py
# Section: 3.3 Stability Assessment
# Description: Performs a global grid search based on the best subsets found 
#              via the Mallows' Cp criterion (Model 2). It identifies the 
#              theoretical peak (Y_max) and optimal factor levels.
# ==============================================================================

def execute_m2_optimization(input_folder, combo_file, output_folder):
    """
    Re-fits Model 2 structures and executes a high-resolution grid search
    to find global maximum response settings.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 1. Load the Best Combination mapping for M2 (Cp-based)
    print("[INFO] Loading M2 model structures (Cp-criterion subsets)...")
    try:
        combo_df = pd.read_excel(combo_file)
        # Identify key columns (Dataset ID and Best Combination)
        id_col = next((c for c in combo_df.columns if 'Dataset' in str(c) or '数据集' in str(c)), None)
        combo_col = next((c for c in combo_df.columns if 'Combination' in str(c) or '最佳组合' in str(c)), None)

        if not id_col or not combo_col:
            print(f"[ERROR] Required headers not found in {combo_file}")
            return

        # Standardize combination strings into lists
        combo_df[combo_col] = combo_df[combo_col].apply(
            lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model structure file: {e}")
        return

    # 2. Iterate through experimental datasets
    xlsx_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    total_files = len(xlsx_files)

    print(f"[INFO] Found {total_files} datasets. Initializing M2 optimization...")

    for idx, file in enumerate(xlsx_files, start=1):
        file_path = os.path.join(input_folder, file)
        data = pd.read_excel(file_path) # Reads the first worksheet by default
        
        # Match dataset with its pre-selected M2 subset
        match_row = combo_df[combo_df[id_col] == file]
        if match_row.empty:
            print(f"[STATUS] Skipping {file}: No M2 subset found.")
            continue

        best_combo = match_row.iloc[0][combo_col]
        raw_vars = data.columns[:-1].tolist()
        target_name = data.columns[-1]

        # 3. Identify base variables for polynomial expansion
        base_var_candidates = set()
        for var in best_combo:
            var_clean = var.replace('^2', '')
            parts = var_clean.split(' ')
            base_var_candidates.update([p.strip() for p in parts if p.strip()])

        used_vars = sorted([v for v in base_var_candidates if v in raw_vars])

        if not used_vars:
            print(f"[ERROR] {file}: Variables in subset do not match raw data.")
            continue

        # 4. Model Re-fitting with M2 variables
        try:
            y = data[target_name]
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly.fit(data[used_vars])
            
            all_feature_names = poly.get_feature_names_out(used_vars)
            selected_indices = [i for i, name in enumerate(all_feature_names) if name in best_combo]

            if not selected_indices:
                print(f"[ERROR] {file}: Polynomial feature mismatch.")
                continue

            X_poly_train = poly.transform(data[used_vars])[:, selected_indices]
            model = LinearRegression().fit(X_poly_train, y)

            # 5. Grid Search Initialization (100 steps per factor)
            var_ranges = []
            for var in used_vars:
                vmin, vmax = data[var].min(), data[var].max()
                step = max((vmax - vmin) / 100, 0.1)
                values = np.arange(vmin, vmax + step / 2, step)
                var_ranges.append(values)

            max_pred = -np.inf
            best_values = None
            total_comb = np.prod([len(r) for r in var_ranges])

            print(f"[EXEC] Optimization Search for {file}: {total_comb} combinations...")

            # 6. Global Search Loop (Logic Preserved)
            for i, comb in enumerate(itertools.product(*var_ranges), start=1):
                comb_df = pd.DataFrame([comb], columns=used_vars)
                comb_poly = poly.transform(comb_df)[:, selected_indices]
                y_pred = model.predict(comb_poly)[0]
                
                if y_pred > max_pred:
                    max_pred = y_pred
                    best_values = comb

                # Progress feedback for computationally intensive datasets
                if i % 100000 == 0 or i == total_comb:
                    progress = (i / total_comb) * 100
                    print(f"    Progress: {progress:.1f}% | Current Peak Y: {max_pred:.4f}")

            # 7. Output Result Storage
            output_df = pd.DataFrame([['Max_Response'] + used_vars, [max_pred] + list(best_values)])
            output_file_name = f"{os.path.splitext(file)[0]}_m2_optimized.xlsx"
            output_df.to_excel(os.path.join(output_folder, output_file_name), header=False, index=False)

            print(f"[STATUS] Completed {idx}/{total_files}: {file} | Optimal Y: {max_pred:.4f}")

        except Exception as e:
            print(f"[ERROR] Processing failure for {file}: {e}")

    print(f"\n[COMPLETE] M2 Global optimization finished. Results in: {output_folder}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Directory containing expanded experimental datasets
    INPUT_DATA_DIR = r"YOUR_INPUT_DIRECTORY_HERE"
    
    # [TODO] Path to the M2 optimization results (Cp-criterion summary)
    M2_SUBSET_FILE = r"YOUR_M2_COMBO_FILE_HERE.xlsx"
    
    # [TODO] Destination folder for M2 peak results
    M2_OUTPUT_DIR = r"YOUR_OUTPUT_DIRECTORY_HERE"

    # -------------------------------------------------------------------------
    execute_m2_optimization(INPUT_DATA_DIR, M2_SUBSET_FILE, M2_OUTPUT_DIR)