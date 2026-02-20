import os
import pandas as pd
import string

def standardize_excel_columns(directory):
    """
    Standardizes column nomenclature for all Excel datasets in the target directory.
    Renames experimental factors as 'A', 'B', 'C'... and the response variable as 'Y'.
    """
    if not os.path.exists(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {directory}")
        return

    print(f"[INFO] Batch standardizing {len(files)} files...")

    for file in files:
        file_path = os.path.join(directory, file)
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                all_sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in all_sheets.items():
                    if df.empty: 
                        continue
                    
                    num_cols = len(df.columns)
                    if num_cols > 26: 
                        print(f"[SKIP] {file} - {sheet_name}: Column count exceeds A-Z limit.")
                        continue
                    
                    standard_names = list(string.ascii_uppercase[:num_cols-1]) + ['Y']
                    df.columns = standard_names
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            print(f"[STATUS] Processed: {file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify This Path)
    # =========================================================================
    # [TODO] Replace the string below with your absolute directory path
    TARGET_DIR = r"YOUR_RAW_DATA_DIRECTORY_PATH_HERE"
    
    # -------------------------------------------------------------------------
    standardize_excel_columns(TARGET_DIR)