import os
import pandas as pd
import numpy as np

# ==============================================================================
# Script: 04_stability_cv_analysis_z_normalized.py
# Section: 3.3 Stability Assessment
# Description: Aggregates optimization results from M0, M1, and M2. 
#              Calculates the Coefficient of Variation (CV) using Z-score 
#              normalized responses to ensure inter-dataset comparability.
# Formulas: 
#   1. Z-normalization: Z = (Y_pred - mu_raw) / sigma_raw
#   2. CV: CV = sigma_Z / |mu_Z|
# ==============================================================================

def execute_stability_cv_analysis(raw_folder, input_folders, summary_path, individual_dir):
    """
    Analyzes the stability of process advice across modeling strategies.
    Applies Z-normalization to the response and calculates CV for both 
    response (Y) and process factors (X).
    """
    if not os.path.exists(raw_folder):
        print(f"[ERROR] Raw data folder not found: {raw_folder}")
        return

    os.makedirs(individual_dir, exist_ok=True)

    # Helper function to extract dataset IDs
    def get_key(filename):
        return filename.split('_')[0]

    # --- 1. Map Optimization Files ---
    print("[INFO] Mapping optimization result files across M0, M1, and M2...")
    file_maps = []
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"[WARN] Folder missing: {folder}")
            file_maps.append({})
            continue
        mapping = {get_key(f): os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')}
        file_maps.append(mapping)

    # Determine datasets present in the primary result folder
    dataset_keys = list(file_maps[0].keys())
    summary_list = []

    print(f"[INFO] Starting Stability CV Analysis with Z-Normalization for {len(dataset_keys)} datasets...")

    # --- 2. Main Calculation Loop ---
    for key in dataset_keys:
        # Check for original raw experimental data
        orig_file = os.path.join(raw_folder, f"{key}.xlsx")
        if not os.path.exists(orig_file):
            print(f"[STATUS] Skipping {key}: Raw data file not found.")
            continue

        try:
            # Step A: Extract Z-score normalization parameters from raw data
            orig_df = pd.read_excel(orig_file)
            orig_y = orig_df.iloc[:, -1]
            raw_mean = orig_y.mean()
            raw_sd = orig_y.std(ddof=1)

            if raw_sd == 0:
                print(f"[STATUS] Skipping {key}: Zero standard deviation in raw data.")
                continue

            # Step B: Aggregate predictions from the 3 modeling strategies
            df_base = pd.read_excel(file_maps[0][key])
            var_names = df_base.columns[1:] # Col 0: Response, Others: Factors
            
            y_preds = []
            x_vals_accum = {vn: [] for vn in var_names}

            for mapping in file_maps:
                if key in mapping:
                    df_res = pd.read_excel(mapping[key])
                    y_preds.append(df_res.iloc[0, 0])
                    for vn in var_names:
                        if vn in df_res.columns:
                            x_vals_accum[vn].append(df_res.iloc[0][vn])

            # Step C: Calculate Normalized Response CV (Y_CV)
            if len(y_preds) >= 2:
                # Logic: Scale the prediction relative to the original raw distribution
                z_scores = [(y - raw_mean) / raw_sd for y in y_preds]
                z_mean = np.mean(z_scores)
                z_sd = np.std(z_scores, ddof=1)
                y_cv = z_sd / abs(z_mean) if z_mean != 0 else np.nan
            else:
                y_cv = np.nan

            # Step D: Calculate Process Factor CV (X_CV)
            x_cv_list = []
            for vn in var_names:
                vals = x_vals_accum[vn]
                if len(vals) >= 2:
                    arr = np.array(vals)
                    m, s = np.mean(arr), np.std(arr, ddof=1)
                    cv = s / abs(m) if m != 0 else np.nan
                else:
                    cv = np.nan
                x_cv_list.append(cv)

            valid_x_cvs = [c for c in x_cv_list if not np.isnan(c)]
            avg_x_cv = np.mean(valid_x_cvs) if valid_x_cvs else np.nan

            summary_list.append([key, y_cv, avg_x_cv])

            # Save individual dataset stability report
            cv_report = pd.DataFrame({
                'Parameter': ['Response_CV_Z_Norm'] + [f'{vn}_CV' for vn in var_names],
                'CV_Value': [y_cv] + x_cv_list
            })
            cv_report.to_excel(os.path.join(individual_dir, f"{key}_Stability_Report.xlsx"), index=False)
            
            print(f"[CALC] Processed {key} | Response CV: {y_cv:.4f} | Avg Factor CV: {avg_x_cv:.4f}")

        except Exception as e:
            print(f"[ERROR] Failure in stability calculation for {key}: {e}")

    # --- 3. Final Summary Export ---
    if summary_list:
        summary_df = pd.DataFrame(summary_list, columns=['Dataset_ID', 'Response_CV', 'Avg_Factor_CV'])
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_excel(summary_path, index=False)
        print("-" * 60)
        print(f"[COMPLETE] Stability analysis finished.")
        print(f"[INFO] Global summary saved to: {summary_path}")
        print("-" * 60)
    else:
        print("[WARN] No stability results were generated.")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    
    # [TODO] Directory containing original experimental data (for Z-score stats)
    RAW_DATA_PATH = r"YOUR_RAW_EXPERIMENTAL_DATA_DIRECTORY"

    # [TODO] List of folders containing optimization results for M0, M1, and M2
    INPUT_STRATEGY_FOLDERS = [
        r"YOUR_M0_OPTIMIZATION_RESULTS",
        r"YOUR_M1_OPTIMIZATION_RESULTS",
        r"YOUR_M2_OPTIMIZATION_RESULTS"
    ]

    # [TODO] Output path for the aggregated summary file
    GLOBAL_SUMMARY_XLS = r"YOUR_SUMMARY_DIRECTORY\Stability_CV_Summary_Z_Normalized.xlsx"
    
    # [TODO] Destination directory for individual dataset stability reports
    INDIVIDUAL_REPORTS_DIR = r"YOUR_SUMMARY_DIRECTORY\Individual_Dataset_CVs"

    # -------------------------------------------------------------------------
    execute_stability_cv_analysis(
        RAW_DATA_PATH, 
        INPUT_STRATEGY_FOLDERS, 
        GLOBAL_SUMMARY_XLS, 
        INDIVIDUAL_REPORTS_DIR
    )