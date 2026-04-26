# -*- coding: utf-8 -*-
"""
Purpose:
1. Batch read Excel files from a specified folder.
2. By default, all columns except the last one are used as independent variables X,
   and the last column is used as the dependent variable y.
3. Perform grid search for quadratic Ridge regression on each file.
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. Input and output paths
# =========================
input_folder = r"Please enter your path here"
output_folder = os.path.join(input_folder, "quadratic_ridge_results")
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
# 3. Ridge parameter grid
# =========================
param_grid = {
    "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "ridge__fit_intercept": [True, False]
}

# Total number of combinations = 7 × 2 = 14
# With 5-fold cross-validation, this means 14 × 5 = 70 fits per file.

# =========================
# 4. Cross-validation setting
# =========================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# If the computation is too slow, you can use:
# cv = KFold(n_splits=3, shuffle=True, random_state=42)

# =========================
# 5. Build the modeling pipeline
# =========================
pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# =========================
# 6. Initialize result containers
# =========================
summary_results = []
failed_files = []

# =========================
# 7. Process each file
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
        y_pred = best_model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Number of features after quadratic polynomial expansion
        n_poly_features = best_model.named_steps["poly"].n_output_features_

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
            "n_samples": df.shape[0],
            "n_original_features": X.shape[1],
            "n_poly_features": n_poly_features,
            "Y_Name": y.name,
            "Best_Alpha": best_params.get("ridge__alpha"),
            "Best_Fit_Intercept": best_params.get("ridge__fit_intercept"),
            "Best_CV_RMSE": best_cv_rmse,
            "Train_R2": train_r2,
            "Train_RMSE": train_rmse
        })

        print(
            f"Completed: {file_name} | "
            f"Poly Features = {n_poly_features} | "
            f"CV RMSE = {best_cv_rmse:.6f}"
        )

    except Exception as e:
        print(f"Failed: {file_name} | Reason: {e}")
        failed_files.append({
            "File_Name": file_name,
            "Error": str(e)
        })

# =========================
# 8. Save summary results
# =========================
summary_df = pd.DataFrame(summary_results)
summary_file = os.path.join(output_folder, "quadratic_ridge_batch_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

# Save failed file records
if failed_files:
    failed_df = pd.DataFrame(failed_files)
    failed_file = os.path.join(output_folder, "quadratic_ridge_failed_files.xlsx")
    failed_df.to_excel(failed_file, index=False)
    print(f"\n{len(failed_files)} files failed. The failure records have been saved.")

print("\nAll files have been processed.")
print(f"Summary results saved to: {summary_file}")
print(f"Prediction results for individual files saved to: {output_folder}")