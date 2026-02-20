import os
import pandas as pd

def process_experimental_data(data_dir):
    """
    Standardize experimental data by averaging dependent variables 
    for identical independent variable combinations.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    # Scan for valid Excel files
    files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {data_dir}")
        return

    print(f"[INFO] Batch processing {len(files)} files for data standardization...")

    for filename in files:
        filepath = os.path.join(data_dir, filename)

        try:
            # Load raw data from specific worksheet
            df = pd.read_excel(filepath, sheet_name="编码前")
            
            if df.empty:
                print(f"[SKIP] {filename}: Dataset is empty.")
                continue

            df.columns = df.columns.str.strip() 

            # Define factor (X) and response (y) indices
            x_cols = df.columns[:-1]
            y_col = df.columns[-1]

            # Data type coercion and precision rounding
            for col in x_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].round(6) 
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

            # Aggregate duplicates by calculating the arithmetic mean
            duplicate_mask = df.duplicated(subset=x_cols)
            if duplicate_mask.any():
                df = df.groupby(list(x_cols), as_index=False)[y_col].mean()
            
            # Export cleaned dataset (overwrite mode)
            df.to_excel(filepath, index=False)
            print(f"[STATUS] Standardized: {filename}")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify This Path)
    # =========================================================================
    # [TODO] Replace the string below with your absolute directory path
    DATA_PATH = r"YOUR_RAW_DATA_DIRECTORY_PATH_HERE"
    
    # -------------------------------------------------------------------------
    print(f"[EXEC] Initializing data preprocessing in: {DATA_PATH}")
    process_experimental_data(DATA_PATH)
    print("[COMPLETE] Data preparation finalized.")