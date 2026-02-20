import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# ==============================================================================
# Script: 10_stepwise_result_validation_MATLAB.py
# Description: Validates the statistical significance of variable combinations 
#              selected by MATLAB's stepwiselm algorithm.
# ==============================================================================

def validate_matlab_stepwise_results(data_folder, mat_result_folder, output_dir):
    """
    Reads original data and MATLAB stepwise results, then re-fits models in Python
    to evaluate the significance of selected terms across platforms.
    """
    if not os.path.exists(data_folder):
        print(f"[ERROR] Source data directory not found: {data_folder}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all Excel files in the experimental data folder (Expecting 'After' encoding)
    datasets = [f for f in os.listdir(data_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not datasets:
        print(f"[WARN] No valid .xlsx files found in: {data_folder}")
        return

    # Lists to collect summary statistics for the final report
    summary_max_p = []
    summary_insignificant = []

    print(f"[INFO] Initializing significance validation for {len(datasets)} MATLAB-selected models...")

    for dataset in datasets:
        dataset_name = os.path.splitext(dataset)[0]
        dataset_path = os.path.join(data_folder, dataset)
        # Expected MATLAB result filename format: result_MATLAB_filename.xlsx
        mat_file_path = os.path.join(mat_result_folder, f'result_MATLAB_{dataset_name}.xlsx')

        if not os.path.exists(mat_file_path):
            print(f"[SKIP] {dataset_name}: Corresponding MATLAB-selection result missing.")
            continue

        try:
            # 1. Load original experimental data (After encoding)
            df_data = pd.read_excel(dataset_path)
            
            if df_data.empty:
                print(f"[SKIP] {dataset_name}: Dataset is empty.")
                continue

            y = df_data.iloc[:, -1].astype(float)
            X_raw = df_data.iloc[:, :-1]

            # 2. Generate full quadratic features
            pf = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = pf.fit_transform(X_raw)
            # Add constant manually for statsmodels OLS compatibility
            X_with_const = sm.add_constant(X_poly_raw, has_constant='add')

            # 3. Read variable indices selected by MATLAB
            df_mat = pd.read_excel(mat_file_path)
            
            # MATLAB output typically lists variables starting from Row 2
            # Extract names/indices (adapting to the list-based output from the MATLAB script)
            mat_variables = df_mat.iloc[1:, 0].dropna().tolist()
            
            selected_indices = []
            for item in mat_variables:
                item_str = str(item)
                # Check for linear terms x1, x2... or interaction/quadratic terms
                if item_str.startswith('x'):
                    try:
                        # Extract the integer part (e.g., 'x5' -> 5)
                        # Note: This logic assumes MATLAB variable names align with PolynomialFeatures indices
                        idx = int(item_str.split('^')[0].replace('x', ''))
                        selected_indices.append(idx)
                    except ValueError:
                        continue

            if not selected_indices:
                print(f"[WARN] {dataset_name}: No valid indices extracted from MATLAB result.")
                continue

            # 4. Fit OLS Model based on MATLAB's selection
            X_selected = X_with_const[:, selected_indices]
            model = sm.OLS(y, X_selected).fit()

            # 5. Calculate Significance Metrics
            p_values = model.pvalues
            p_max = p_values.max()
            p_gt_0_05 = (p_values > 0.05).sum()

            # 6. Save individual detailed results
            df_pvalues = pd.DataFrame({"Variable_Index": selected_indices, "p_value": p_values.values})
            df_summary = pd.DataFrame({"p_max": [p_max], "Insignificant_Count": [p_gt_0_05]})

            individual_output = os.path.join(output_dir, f"{dataset_name}_MATLAB_validation.xlsx")
            with pd.ExcelWriter(individual_output) as writer:
                df_pvalues.to_excel(writer, sheet_name="P_Values", index=False)
                df_summary.to_excel(writer, sheet_name="Summary_Stats", index=False)

            # 7. Append data for final global reports
            summary_max_p.append({"Dataset": dataset_name, "p_max": p_max})
            if p_gt_0_05 > 0:
                summary_insignificant.append({"Dataset": dataset_name, "Count_p_gt_0.05": p_gt_0_05})

            print(f"[STATUS] Validated: {dataset_name}")

        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_name}: {e}")

    # Save final aggregated summary reports
    if summary_max_p:
        pd.DataFrame(summary_max_p).to_excel(os.path.join(output_dir, "Summary_Max_P_Values_MATLAB.xlsx"), index=False)
        pd.DataFrame(summary_insignificant).to_excel(os.path.join(output_dir, "Summary_Insignificant_Terms_MATLAB.xlsx"), index=False)
        
        print("-" * 60)
        print(f"[COMPLETE] MATLAB-stepwise validation task finished.")
        print(f"[INFO] Summary reports generated in: {output_dir}")
    else:
        print("[WARN] No valid results generated. Please verify directory paths.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Input: Directory containing quadratic expansion data (After encoding)
    DATA_IN_PATH = r"YOUR_EXPANDED_DATA_PATH_HERE"
    
    # [TODO] Input: Directory containing MATLAB-stepwise selection results
    MATLAB_RESULT_DIR = r"YOUR_MATLAB_RESULTS_PATH_HERE"
    
    # [TODO] Output: Directory for MATLAB significance validation reports
    VALIDATION_OUTPUT = r"YOUR_MATLAB_P_VALUE_OUTPUT_PATH_HERE"

    # -------------------------------------------------------------------------
    validate_matlab_stepwise_results(DATA_IN_PATH, MATLAB_RESULT_DIR, VALIDATION_OUTPUT)