import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import itertools

# ==============================================================================
# Script: 03_full_quadratic_global_optimization_search.py
# Section: 3.3 Stability Assessment
# Description: Performs a global grid search based on the Full Quadratic Model (M0).
#              This serves as the "Stability Baseline." It identifies the 
#              optimal process settings for the unoptimized RSM model.
# ==============================================================================

def execute_full_quadratic_optimization(input_folder, output_folder):
    """
    Fits a Full Quadratic (M0) model and performs a high-resolution grid search
    to find the global maximum response settings.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # --- Fetch valid files ---
    xlsx_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    total_files = len(xlsx_files)

    print(f"[INFO] Initializing Full Quadratic (M0) Grid Search for {total_files} datasets...")

    for idx, file in enumerate(xlsx_files, start=1):
        file_path = os.path.join(input_folder, file)
        
        try:
            # Load raw experimental data
            data = pd.read_excel(file_path)
            
            # Separation of factors (X) and response (y)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            feature_names = X.columns.tolist()

            # 1. Fit Full Quadratic Model (M0)
            # Logic: Fit all main effects, interactions, and squared terms
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            # 2. Construct Search Grid (Logic Preserved)
            # Create a 100-step resolution for each factor range
            var_ranges = []
            for var in feature_names:
                vmin, vmax = X[var].min(), X[var].max()
                if vmax == vmin:
                    values = [vmin]
                else:
                    # Maintain 100-step granularity
                    step = max((vmax - vmin) / 100, 0.1)
                    values = np.arange(vmin, vmax + step / 2, step)
                var_ranges.append(values)

            # 3. Exhaustive Global Search (Logic Preserved)
            max_pred = -np.inf
            best_values = None
            total_comb = np.prod([len(r) for r in var_ranges])

            print(f"[EXEC] Searching {file}: {total_comb} combinations...")

            # Traversing the high-dimensional response surface
            for i, comb in enumerate(itertools.product(*var_ranges), start=1):
                comb_df = pd.DataFrame([comb], columns=feature_names)
                comb_poly = poly.transform(comb_df)
                y_pred = model.predict(comb_poly)[0]
                
                if y_pred > max_pred:
                    max_pred = y_pred
                    best_values = comb

                # Progress feedback for computationally heavy tasks
                if i % 100000 == 0 or i == total_comb:
                    percentage = (i / total_comb) * 100
                    print(f"    Progress: {percentage:.1f}% | Current Max Y: {max_pred:.4f}")

            # 4. Save Optimization Result
            output_file_name = f"{os.path.splitext(file)[0]}_M0_optimal.xlsx"
            output_path = os.path.join(output_folder, output_file_name)
            
            ordered_columns = ['Max_Prediction'] + feature_names
            result_values = [max_pred] + list(best_values)
            result_df = pd.DataFrame([result_values], columns=ordered_columns)
            result_df.to_excel(output_path, index=False)

            print(f"[STATUS] Completed {idx}/{total_files}: {file} | Optimal Y: {max_pred:.4f}")

        except Exception as e:
            print(f"[ERROR] Critical failure processing {file}: {e}")

    print(f"\n[COMPLETE] Global M0 optimization finished. Results in: {output_folder}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Replace with directory containing your raw experimental tables
    INPUT_DATA_DIR = r"YOUR_INPUT_DIRECTORY_PATH"
    
    # [TODO] Replace with destination directory for M0 optimization results
    OUTPUT_DATA_DIR = r"YOUR_OUTPUT_DIRECTORY_PATH"

    # -------------------------------------------------------------------------
    execute_full_quadratic_optimization(INPUT_DATA_DIR, OUTPUT_DATA_DIR)