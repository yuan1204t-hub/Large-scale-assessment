import os
import pandas as pd
import pingouin as pg

def run_alternative_tests(file_path, factors_count=5, dv='Y', test_type='welch'):
    """
    Executes robust statistical evaluations (Welch's ANOVA or Kruskal-Wallis)
    to validate factor significance under non-ideal data distributions.
    """
    try:
        # Load dataset from the standardized 'After' worksheet
        # Ensuring the file and sheet exist before processing
        data = pd.read_excel(file_path, sheet_name='After')
        
        if data.empty:
            print(f"[WARN] {os.path.basename(file_path)}: Sheet 'After' is empty.")
            return None

        results = []
        
        # Systematic analysis across independent experimental variables
        for i in range(factors_count):
            factor_name = data.columns[i]
            
            if test_type == 'kruskal':
                # Kruskal-Wallis: Rank-based non-parametric test
                aov = pg.kruskal(data, dv=dv, between=factor_name)
                results.append({
                    "Factor": factor_name,
                    "H_Statistic": round(aov.iloc[0, 2], 4),
                    "p_value": round(aov.iloc[0, 3], 4),
                    "Significance": "p < 0.05" if aov.iloc[0, 3] < 0.05 else "n.s."
                })
            else:
                # Welch's ANOVA: Robust to heteroscedasticity (unequal variances)
                aov = pg.welch_anova(data, dv=dv, between=factor_name)
                results.append({
                    "Factor": factor_name,
                    "F_Statistic": round(aov.iloc[0, 3], 4),
                    "p_value": round(aov.iloc[0, 4], 4),
                    "Significance": "p < 0.05" if aov.iloc[0, 4] < 0.05 else "n.s."
                })

        return pd.DataFrame(results)

    except Exception as e:
        print(f"[ERROR] Statistical failure for {os.path.basename(file_path)}: {e}")
        return None

if __name__ == "__main__":
    # =========================================================================
    # PATH CONFIGURATION (User Must Modify These Paths)
    # =========================================================================
    # [TODO] Replace with the absolute path to your source data directory
    DATA_DIR = r"YOUR_DATA_DIRECTORY_PATH_HERE"
    
    # [TODO] Replace with your specific target filename
    TARGET_FILE = "271.2.xlsx"
    
    # [TODO] Replace with your desired output directory for statistical reports
    OUTPUT_FOLDER = r"YOUR_OUTPUT_DIRECTORY_PATH_HERE"
    
    # -------------------------------------------------------------------------
    FILE_PATH = os.path.join(DATA_DIR, TARGET_FILE)

    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] Target file not found: {FILE_PATH}")
    else:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"[INFO] Initializing comparative statistical analysis...")
        print(f"[EXEC] Target: {TARGET_FILE}")

        # 1. Kruskal-Wallis Test
        kw_results = run_alternative_tests(FILE_PATH, test_type='kruskal')
        if kw_results is not None:
            kw_path = os.path.join(OUTPUT_FOLDER, f"kruskal_results_{TARGET_FILE}")
            kw_results.to_excel(kw_path, index=False)
            print(f"[STATUS] Kruskal-Wallis report generated: {kw_path}")

        # 2. Welch's ANOVA
        welch_results = run_alternative_tests(FILE_PATH, test_type='welch')
        if welch_results is not None:
            welch_path = os.path.join(OUTPUT_FOLDER, f"welch_results_{TARGET_FILE}")
            welch_results.to_excel(welch_path, index=False)
            print(f"[STATUS] Welch's ANOVA report generated: {welch_path}")

        print("-" * 30)
        print("[COMPLETE] Robust statistical evaluations finalized.")