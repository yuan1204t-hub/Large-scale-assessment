import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import toad
from sklearn.metrics import r2_score

# ==============================================================================
# Script: 07_stepwise_regression_Python.py
# Description: Performs bidirectional stepwise regression using the 'toad' library 
#              based on the Akaike Information Criterion (AIC). 
# ==============================================================================

def perform_python_stepwise_regression(input_folder, output_folder):
    """
    Executes Python-based stepwise selection (toad implementation) on quadratic 
    expansion datasets and saves the goodness-of-fit (R2) and selected variables.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input directory not found: {input_folder}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Fetch all Excel files in the target directory (Standardized 'After' datasets)
    files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {input_folder}")
        return

    print(f"[INFO] Initializing Python Stepwise Analysis (toad + AIC) for {len(files)} files...")

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        
        try:
            # Read experimental data (Expecting 'After' encoded format)
            df = pd.read_excel(filepath)
            
            if df.empty:
                print(f"[SKIP] {filename}: Dataset is empty.")
                continue

            # Ensure numeric precision for the estimator
            df = df.astype('float')

            # 1. Perform Stepwise Selection using 'toad'
            # direction='both': bidirectional selection
            # criterion='aic': optimization based on Akaike Information Criterion
            # intercept=True: explicitly includes the constant term
            final_data = toad.selection.stepwise(
                df,
                target='Y',
                estimator='ols',
                direction='both',
                criterion='aic',
                intercept=True
            )

            # 2. Extract selected features and target
            X_selected = final_data.drop(columns=['Y'])
            y = final_data['Y']

            # Validate selection result
            if X_selected.empty:
                print(f"[WARN] {filename}: No variables retained after selection.")
                continue

            # 3. Fit Final OLS Model via Statsmodels for refined metrics
            # Re-fit to extract standard R-squared and model attributes
            X_with_const = sm.add_constant(X_selected, has_constant='add')
            model_fit = sm.OLS(y, X_with_const).fit()
            
            # 4. Record Results (Goodness-of-fit and Model Structure)
            r2 = model_fit.rsquared
            selected_vars = list(X_selected.columns)

            # Construct result DataFrame for cross-platform comparison
            # Row 0: R-squared value | Row 1: Names of selected regression terms
            results_df = pd.DataFrame(columns=range(max(len(selected_vars), 1)))
            results_df.loc[0, 0] = r2
            for idx, var_name in enumerate(selected_vars):
                results_df.loc[1, idx] = var_name

            # Save to Excel with 're_' prefix for distinction
            output_path = os.path.join(output_folder, f"re_{filename}")
            results_df.to_excel(output_path, index=False)
            print(f"[STATUS] Processed: {filename}")

        except Exception as e:
            print(f"[ERROR] Failure in processing {filename}: {e}")

    print("-" * 60)
    print(f"[COMPLETE] Python Stepwise Analysis finalized.")
    print(f"[INFO] Results saved to: {output_folder}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Input: Directory containing quadratic expansion data (After encoding)
    INPUT_DIR_PATH = r"YOUR_EXPANDED_DATA_PATH_HERE"
    
    # [TODO] Output: Directory for Python-based stepwise selection results
    OUTPUT_DIR_PATH = r"YOUR_PYTHON_STEPWISE_OUTPUT_PATH_HERE"
    
    # -------------------------------------------------------------------------
    perform_python_stepwise_regression(INPUT_DIR_PATH, OUTPUT_DIR_PATH)