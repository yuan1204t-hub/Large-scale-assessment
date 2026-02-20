import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import toad

# ==============================================================================
# Script: 08_stepwise_result_validation_Python.py
# Description: Validates the statistical significance of variable combinations 
#              selected by Python's toad-stepwise algorithm.
# ==============================================================================

def validate_python_stepwise_results(data_folder, output_dir):
    """
    Performs Python-based stepwise selection and immediately validates 
    the resulting model's statistical rigorousness.
    """
    if not os.path.exists(data_folder):
        print(f"[ERROR] Directory not found: {data_folder}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all Excel files in the experimental data folder (After encoding)
    datasets = [f for f in os.listdir(data_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not datasets:
        print(f"[WARN] No valid .xlsx files found in: {data_folder}")
        return

    # Lists to collect summary results for the final report
    summary_max_p = []
    summary_insignificant = []

    print(f"[INFO] Initializing Python-stepwise significance validation for {len(datasets)} datasets...")

    for dataset in datasets:
        dataset_name = os.path.splitext(dataset)[0]
        dataset_path = os.path.join(data_folder, dataset)

        try:
            # Load encoded experimental data and ensure numeric types
            # Expected sheet: 'After' (as per standardized workflow)
            df = pd.read_excel(dataset_path).astype(float)
            
            if df.empty:
                print(f"[SKIP] {dataset_name}: Dataset is empty.")
                continue

            # 1. Perform Stepwise Selection (Python - toad implementation)
            # This follows the bidirectional AIC criteria
            selected_df = toad.selection.stepwise(
                df, 
                target='Y', 
                estimator='ols', 
                direction='both', 
                criterion='aic', 
                intercept=True
            )

            # 2. Extract selected features and target variable
            X_selected = selected_df.drop(columns='Y')
            y = selected_df['Y']

            if X_selected.empty:
                print(f"[WARN] {dataset_name}: No variables retained after toad selection.")
                continue

            # 3. Fit OLS Model for statistical validation
            # Add constant manually to ensure the intercept is correctly estimated
            X_with_const = sm.add_constant(X_selected, has_constant='add')
            model = sm.OLS(y, X_with_const).fit()
            
            p_vals = model.pvalues
            p_max = p_vals.max()  # The largest p-value among coefficients
            p_gt_0_05 = (p_vals > 0.05).sum() # Count of non-significant terms

            # 4. Save detailed individual results
            df_pvalues = pd.DataFrame({"Variable": p_vals.index, "p_value": p_vals.values})
            df_summary = pd.DataFrame({"p_max": [p_max], "Insignificant_Count": [p_gt_0_05]})

            output_file_path = os.path.join(output_dir, f"{dataset_name}_Py_validation.xlsx")
            with pd.ExcelWriter(output_file_path) as writer:
                df_pvalues.to_excel(writer, sheet_name="P_Values", index=False)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

            # 5. Collect data for global summary
            summary_max_p.append({"Dataset": dataset_name, "p_max": p_max})
            if p_gt_0_05 > 0:
                summary_insignificant.append({"Dataset": dataset_name, "Count_p_gt_0.05": p_gt_0_05})

            print(f"[STATUS] Validated: {dataset_name}")

        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_name}: {e}")

    # Save final aggregated reports
    if summary_max_p:
        pd.DataFrame(summary_max_p).to_excel(os.path.join(output_dir, "Summary_Max_P_Values_Py.xlsx"), index=False)
        pd.DataFrame(summary_insignificant).to_excel(os.path.join(output_dir, "Summary_Insignificant_Terms_Py.xlsx"), index=False)
        
        print("-" * 60)
        print(f"[COMPLETE] Python-stepwise validation task finished.")
        print(f"[INFO] Reports saved in: {output_dir}")
    else:
        print("[WARN] No valid results generated. Please verify data inputs.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Input: Directory for expanded experimental data (After encoding)
    INPUT_DATA_PATH = r"YOUR_EXPANDED_DATA_PATH_HERE"
    
    # [TODO] Output: Directory for Python significance validation reports
    OUTPUT_REPORT_DIR = r"YOUR_PYTHON_P_VALUE_OUTPUT_PATH_HERE"

    # -------------------------------------------------------------------------
    validate_python_stepwise_results(INPUT_DATA_PATH, OUTPUT_REPORT_DIR)