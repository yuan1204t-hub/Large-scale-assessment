import os
import pandas as pd
import shutil

def filter_datasets_by_sample_size(source_dir, target_dir):
    """
    Filters experimental datasets based on the sample size requirement for a full quadratic model.
    The number of data rows (N) must be strictly greater than the number of regression parameters (p)
    to maintain degrees of freedom for residuals (N >= p + 1).
    """
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        return

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Map the number of columns (factors + 1 response) to the required number of data rows
    # Formula for parameters p: 1 (intercept) + 2k (linear + squared) + k(k-1)/2 (interactions)
    # Required data rows N >= p + 1
    requirement_map = {
        3: 7,   # 2 factors: p=6  -> needs >= 7 rows
        4: 11,  # 3 factors: p=10 -> needs >= 11 rows
        5: 16,  # 4 factors: p=15 -> needs >= 16 rows
        6: 22,  # 5 factors: p=21 -> needs >= 22 rows
        7: 29   # 6 factors: p=28 -> needs >= 29 rows
    }

    files = [f for f in os.listdir(source_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print(f"[WARN] No valid .xlsx files found in: {source_dir}")
        return

    valid_files = []
    print(f"[INFO] Initializing sample size validation for {len(files)} datasets...")

    # Iterate through all Excel files in the source directory
    for file in files:
        file_path = os.path.join(source_dir, file)

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # df.shape[1] is the number of columns, df.shape[0] is the number of data rows
            num_columns = df.shape[1]
            num_rows = df.shape[0]

            # Check if the column count is within our expected range (2 to 6 factors)
            if num_columns in requirement_map:
                required_rows = requirement_map[num_columns]
                
                # Verify if sample size exceeds the number of regression terms
                if num_rows >= required_rows:
                    valid_files.append(file)
                else:
                    print(f"[SKIP] {file}: Insufficient sample size (Required: {required_rows}, Found: {num_rows})")
            else:
                print(f"[WARN] {file}: Out of defined factor range (Columns: {num_columns})")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # Copy valid files to the target directory
    print(f"\n[INFO] Validation complete. {len(valid_files)} files met the DOF criteria.")
    
    for valid_file in valid_files:
        source_file_path = os.path.join(source_dir, valid_file)
        target_file_path = os.path.join(target_dir, valid_file)
        try:
            shutil.copy(source_file_path, target_file_path)
            print(f"[STATUS] Copied: {valid_file}")
        except Exception as e:
            print(f"[ERROR] Copy failed for {valid_file}: {e}")
        
    print("-" * 30)
    print(f"[COMPLETE] Filtered datasets are located in: {target_dir}")

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace the strings below with your absolute directory paths
    SOURCE_PATH = r"YOUR_SOURCE_DIRECTORY_PATH_HERE"
    TARGET_PATH = r"YOUR_TARGET_DIRECTORY_PATH_HERE"
    
    # -------------------------------------------------------------------------
    filter_datasets_by_sample_size(SOURCE_PATH, TARGET_PATH)