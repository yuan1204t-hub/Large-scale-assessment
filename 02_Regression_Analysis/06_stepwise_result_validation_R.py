import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

def validate_r_stepwise_results(data_folder, r_result_folder, output_dir):
    """
    Validates the statistical significance of variable combinations selected by 
    R's stepAIC algorithm. It fits OLS models based on the R-selected indices 
    and calculates p_max and the number of non-significant terms.
    """
    if not os.path.exists(data_folder):
        print(f"[ERROR] Source data directory not found: {data_folder}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all Excel files in the experimental data folder (After encoding)
    datasets = [f for f in os.listdir(data_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not datasets:
        print(f"[WARN] No valid .xlsx files found in: {data_folder}")
        return

    # Lists to collect summary statistics for final reporting
    summary_max_p = []
    summary_insignificant = []

    print(f"[INFO] Initializing significance validation for {len(datasets)} R-selected models...")

    for dataset in datasets:
        dataset_name = os.path.splitext(dataset)[0]
        dataset_path = os.path.join(data_folder, dataset)
        # Note: Mapping to R selection results (e.g., 'dataset_name_result.xlsx')
        r_file_path = os.path.join(r_result_folder, f'{dataset_name}_result.xlsx')

        # Check if the corresponding R selection result exists
        if not os.path.exists(r_file_path):
            print(f"[SKIP] {dataset_name}: R-selection result file missing.")
            continue

        try:
            # Load experimental data (standardized 'After' sheet)
            df_data = pd.read_excel(dataset_path)
            
            if df_data.empty:
                print(f"[SKIP] {dataset_name}: Dataset is empty.")
                continue

            y = df_data.iloc[:, -1].astype(float)
            X_raw = df_data.iloc[:, :-1]

            # Generate full quadratic terms (X_poly)
            pf = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_raw = pf.fit_transform(X_raw)
            
            # Add constant term manually for statsmodels compatibility
            X_with_const = sm.add_constant(X_poly_raw, has_constant='add')

            # Read the indices selected by R's stepAIC
            df_r_selection = pd.read_excel(r_file_path)
            
            # Extract indices from the selection metadata
            if len(df_r_selection) > 1:
                selected_indices = df_r_selection.iloc[1:, 1].dropna().astype(int).tolist()
            else:
                selected_indices = []

            if not selected_indices:
                print(f"[WARN] {dataset_name}: No variables retained by R-stepwise.")
                continue

            # Fit OLS model using R's selected variable subset
            X_selected = X_with_const[:, selected_indices]
            model = sm.OLS(y, X_selected).fit()

            # Calculate significance metrics
            p_values = model.pvalues
            p_max = p_values.max()
            insignificant_count = (p_values > 0.05).sum()

            # Prepare individual result DataFrames
            df_pvalues = pd.DataFrame({
                "Variable_Index": selected_indices,
                "p_value": p_values.values
            })

            df_stats = pd.DataFrame({
                "p_max": [p_max], 
                "insignificant_count": [insignificant_count]
            })

            # Save detailed p-values for each dataset
            individual_output = os.path.join(output_dir, f"{dataset_name}_R_validation.xlsx")
            with pd.ExcelWriter(individual_output) as writer:
                df_pvalues.to_excel(writer, sheet_name="P_Values_Details", index=False)
                df_stats.to_excel(writer, sheet_name="Summary_Stats", index=False)

            # Append to summary lists
            summary_max_p.append({"Dataset": dataset_name, "p_max": p_max})
            if insignificant_count > 0:
                summary_insignificant.append({"Dataset": dataset_name, "Count_p_gt_0.05": insignificant_count})
            
            print(f"[STATUS] Validated: {dataset_name}")

        except Exception as e:
            print(f"[ERROR] Failure in validating {dataset_name}: {e}")

    # Save final summary reports
    if summary_max_p:
        pd.DataFrame(summary_max_p).to_excel(os.path.join(output_dir, "Summary_Max_P_Values_R.xlsx"), index=False)
        pd.DataFrame(summary_insignificant).to_excel(os.path.join(output_dir, "Summary_Insignificant_Terms_R.xlsx"), index=False)

    print("-" * 60)
    print(f"[COMPLETE] R-stepwise significance validation finished.")
    print(f"[INFO] Reports generated in: {output_dir}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Input directory for expanded experimental data (After encoding)
    DATA_PATH_IN = r"YOUR_EXPANDED_DATA_PATH_HERE"
    
    # [TODO] Input directory for R-stepwise selection results
    R_SELECTION_DIR = r"YOUR_R_STEPWISE_RESULTS_PATH_HERE"
    
    # [TODO] Output directory for validation reports
    VALIDATION_OUTPUT = r"YOUR_VALIDATION_OUTPUT_PATH_HERE"
    
    # -------------------------------------------------------------------------
    validate_r_stepwise_results(DATA_PATH_IN, R_SELECTION_DIR, VALIDATION_OUTPUT)