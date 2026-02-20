import os
import re
import pandas as pd

# ==============================================================================
# Script: 11_cross_platform_aggregation.py
# Description: Aggregates R2 values, p_max values, and Scott's Pi coefficients
#              across MATLAB, Python, and R platforms for consistency analysis.
# ==============================================================================

def extract_number(filename, pattern):
    """Extracts numeric identifiers (including decimals) from filenames."""
    match = re.search(pattern, str(filename))
    return match.group(1) if match else None

def load_p_value_summary(file_path, is_python=False):
    """Loads p_max summary files and maps them to dataset IDs."""
    if not os.path.exists(file_path):
        print(f"[WARN] p-value summary not found: {file_path}")
        return {}
    
    data = pd.read_excel(file_path)
    p_map = {}
    
    # Standard format: Column 0 is dataset name/ID, Column 1 is max_p
    for _, row in data.iterrows():
        raw_id = str(row.iloc[0])
        node_id = extract_number(raw_id, r'(\d+(\.\d+)?)')
        if node_id:
            p_map[node_id] = row.iloc[1]
    return p_map

def aggregate_platform_results(config):
    """Main function to merge data from all three platforms into a master sheet."""
    
    # --- Create Output Directory ---
    output_dir = os.path.dirname(config['output_path'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- Load Supporting Data ---
    print("[INFO] Loading significance metrics and Scott's Pi coefficients...")
    p_values_r = load_p_value_summary(config['p_files']['r'])
    p_values_mat = load_p_value_summary(config['p_files']['matlab'])
    p_values_py = load_p_value_summary(config['p_files']['python'], is_python=True)

    # Load Scott's Pi (sp)
    sp_dict = {}
    if os.path.exists(config['sp_file']):
        sp_df = pd.read_excel(config['sp_file'], header=None)
        for _, row in sp_df.iterrows():
            node_id = extract_number(row[0], r'(\d+(\.\d+)?)')
            if node_id:
                sp_dict[node_id] = row[1]
    else:
        print(f"[WARN] Scott's Pi file missing at: {config['sp_file']}")

    # --- Identify Common Datasets ---
    patterns = {
        'matlab': r'result_MATLAB_(\d+(\.\d+)?)\.xlsx', # Adjusted to match MATLAB script output
        'python': r're_(\d+(\.\d+)?)\.xlsx',         # Adjusted to match Python script output
        'r': r'(\d+(\.\d+)?)_R_stepwise\.xlsx'       # Adjusted to match R script output
    }

    platform_files = {plat: {} for plat in config['folders']}
    for plat, folder in config['folders'].items():
        if os.path.exists(folder):
            for f in os.listdir(folder):
                node_id = extract_number(f, patterns[plat])
                if node_id:
                    platform_files[plat][node_id] = os.path.join(folder, f)
        else:
            print(f"[ERROR] Platform folder missing: {folder}")

    # Intersection of all three platforms
    common_keys = set(platform_files['matlab']) & set(platform_files['python']) & set(platform_files['r'])
    
    # --- Data Merging ---
    results = []
    if common_keys:
        print(f"[INFO] Merging results for {len(common_keys)} shared datasets...")
    else:
        print("[ERROR] No common datasets found across all platforms.")
        return

    for key in sorted(common_keys, key=float):
        try:
            # Extract R2 from individual model files
            # Note: iloc indices correspond to the specific output structure of previous scripts
            r2_mat = pd.read_excel(platform_files['matlab'][key]).iloc[0, 0] 
            r2_py = pd.read_excel(platform_files['python'][key]).iloc[0, 0]
            r2_r = pd.read_excel(platform_files['r'][key]).iloc[0, 1]

            results.append({
                'Dataset_ID': f"{key}.xlsx",
                'MATLAB_R2': r2_mat,
                'Python_R2': r2_py,
                'R_R2': r2_r,
                'Scott_Pi': sp_dict.get(key, None),
                'MATLAB_max_p': p_values_mat.get(key, "pass"),
                'Python_max_p': p_values_py.get(key, "pass"),
                'R_max_p': p_values_r.get(key, "pass")
            })
            print(f"[STATUS] Aggregated: {key}")
        except Exception as e:
            print(f"[ERROR] Failed to merge Dataset {key}: {e}")

    # --- Save Master Summary ---
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_excel(config['output_path'], index=False)
        print("-" * 60)
        print(f"[COMPLETE] Master summary generated successfully.")
        print(f"[INFO] Location: {config['output_path']}")
        print("-" * 60)

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Fill in the absolute paths to your platform-specific result folders
    CONFIG = {
        'folders': {
            'matlab': r"YOUR_MATLAB_STEPWISE_RESULTS_FOLDER",
            'python': r"YOUR_PYTHON_STEPWISE_RESULTS_FOLDER",
            'r': r"YOUR_R_STEPWISE_RESULTS_FOLDER"
        },
        'p_files': {
            'r': r"YOUR_MAX_P_VALUES_R_XLSX",
            'matlab': r"YOUR_MAX_P_VALUES_MATLAB_XLSX",
            'python': r"YOUR_MAX_P_VALUES_PYTHON_XLSX"
        },
        'sp_file': r"YOUR_SCOTTS_PI_FILE_XLSX",
        'output_path': r"YOUR_FINAL_AGGREGATED_SUMMARY_XLSX"
    }
    
    # -------------------------------------------------------------------------
    aggregate_platform_results(CONFIG)