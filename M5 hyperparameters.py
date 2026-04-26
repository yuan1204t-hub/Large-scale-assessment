# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch read Excel files from a specified folder.
2. By default, all columns except the last one are used as independent variables X,
   and the last column is used as the dependent variable y.
3. Perform grid search for PLSRegression on each file.
4. Output the best hyperparameters, cross-validation results, and training fitting results.
5. Save prediction results for each file and summarize all batch results.

Applicable scenarios:
- A folder contains many Excel files.
- In each file, the first n-1 columns are independent variables.
- In each file, the last column is the dependent variable.
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. Input and output paths
# =========================
input_folder = r"Please enter your path here"
output_folder = os.path.join(input_folder, "pls_results")
os.makedirs(output_folder, exist_ok=True)

# =========================
# 2. Search for all Excel files
# =========================
excel_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + \
              glob.glob(os.path.join(input_folder, "*.xls"))

print(f"Found {len(excel_files)} Excel files.")

if len(excel_files) == 0:
    raise ValueError("No Excel files were found in the specified folder. Please check the input path.")

# =========================
# 3. Cross-validation setting
# =========================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# If the computation is too slow, you can use:
# cv = KFold(n_splits=3, shuffle=True, random_state=42)

# =========================
# 4. Initialize result containers
# =========================
summary_results = []
failed_files = []

# =========================
# 5. Process each file
# =========================
for i, file_path in enumerate(excel_files, start=1):
    file_name = os.path.basename(file_path)
    file_stem = os.path.splitext(file_name)[0]

    print(f"\n[{i}/{len(excel_files)}] Processing: {file_name}")

    try:
        # Read data
        df = pd.read_excel(file_path)
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # Check the number of columns
        if df.shape[1] < 2:
            raise ValueError("Insufficient number of columns. At least one X column and one y column are required.")

        # By default, all columns except the last one are X, and the last column is y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Check missing values
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Missing values exist in the data.")

        n_features = X.shape[1]
        n_samples = X.shape[0]

        # The number of PLS components cannot exceed the number of features
        # or the number of samples minus one.
        max_components = min(n_features, n_samples - 1)

        if max_components < 1:
            raise ValueError("The sample size is too small for PLS modeling.")

        # Parameter grid: search from 1 to max_components
        param_grid = {
            "pls__n_components": list(range(1, max_components + 1))
        }

        # Build the modeling pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression())
        ])

        # Grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )

        # Fit the model
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_neg_rmse = grid_search.best_score_
        best_cv_rmse = -best_cv_neg_rmse

        # Training set prediction
        y_pred = best_model.predict(X).ravel()
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Save prediction results for the current file
        pred_df = df.copy()
        pred_df["Observed_Y"] = y
        pred_df["Predicted_Y"] = y_pred
        pred_df["Residual"] = y - y_pred

        pred_output_file = os.path.join(output_folder, f"{file_stem}_predictions.xlsx")
        pred_df.to_excel(pred_output_file, index=False)

        # Summarize results
        summary_results.append({
            "File_Name": file_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "Y_Name": y.name,
            "Best_n_components": best_params.get("pls__n_components"),
            "Best_CV_RMSE": best_cv_rmse,
            "Train_R2": train_r2,
            "Train_RMSE": train_rmse
        })

        print(
            f"Completed: {file_name} | "
            f"n_components = {best_params.get('pls__n_components')} | "
            f"CV RMSE = {best_cv_rmse:.6f}"
        )

    except Exception as e:
        print(f"Failed: {file_name} | Reason: {e}")
        failed_files.append({
            "File_Name": file_name,
            "Error": str(e)
        })

# =========================
# 6. Save summary results
# =========================
summary_df = pd.DataFrame(summary_results)
summary_file = os.path.join(output_folder, "pls_batch_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

# Save failed file records
if failed_files:
    failed_df = pd.DataFrame(failed_files)
    failed_file = os.path.join(output_folder, "pls_failed_files.xlsx")
    failed_df.to_excel(failed_file, index=False)
    print(f"\n{len(failed_files)} files failed. The failure records have been saved.")

print("\nAll files have been processed.")
print(f"Summary results saved to: {summary_file}")
print(f"Prediction results for individual files saved to: {output_folder}")