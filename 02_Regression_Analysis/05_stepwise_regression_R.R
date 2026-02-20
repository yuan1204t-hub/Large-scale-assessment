# ==============================================================================
# Script: 05_stepwise_regression_R.R
# Description: Performs bidirectional stepwise regression based on the 
#              Akaike Information Criterion (AIC) using the MASS package.
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(MASS)      # For stepAIC bidirectional selection
  library(readxl)    # For reading Excel files
  library(dplyr)     # For data manipulation
  library(openxlsx)  # For creating formatted Excel outputs
})

# ==============================================================================
# PATH CONFIGURATION (User Must Modify These Paths)
# ==============================================================================
# [TODO] Replace with your absolute directory path for expanded datasets
INPUT_DIR <- "YOUR_EXPANDED_DATA_DIRECTORY_PATH_HERE"

# [TODO] Replace with your desired output directory for R-stepwise results
OUTPUT_DIR <- "YOUR_OUTPUT_DIRECTORY_PATH_HERE"

# ------------------------------------------------------------------------------

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# Fetch all Excel files in the target directory
excel_files <- list.files(INPUT_DIR, pattern = "\\.xlsx$", full.names = TRUE)
excel_files <- excel_files[!grepl("^~\\$", basename(excel_files))] # Exclude temp files

if (length(excel_files) == 0) {
  cat(paste0("[WARN] No valid .xlsx files found in: ", INPUT_DIR, "\n"))
} else {
  cat(paste0("[INFO] Batch processing ", length(excel_files), " files for R-Stepwise analysis...\n"))
  
  # Main loop: process each experimental dataset
  for (file in excel_files) {
    file_name <- basename(file)
    
    tryCatch({
      # Read experimental data from 'After'  worksheet
      # Note: Standardizing to 'After' as per Python workflow consistency
      data <- read_excel(file, sheet = "After") 
      
      if (nrow(data) == 0) {
        cat(paste0("[SKIP] ", file_name, ": Dataset is empty.\n"))
        next
      }

      # Pre-processing: Identify independent (X) and dependent (y) variables
      X <- data[, -ncol(data)]
      y <- data[[ncol(data)]]
      df <- cbind(X, y = y)
      
      # 1. Fit the Initial Full Model (M0)
      full_model <- lm(y ~ ., data = df)
      
      # 2. Perform Stepwise Regression (Bidirectional AIC)
      # This corresponds to the R algorithmic pathway described in the manuscript
      step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
      
      # 3. Extract Model Performance Metrics
      r_squared <- summary(step_model)$r.squared
      selected_vars <- names(step_model$coefficients)[-1] # Exclude (Intercept)
      
      # Construct a structured results table
      result_df <- data.frame(
        Metric = "R-squared",
        Value = as.character(round(r_squared, 4)),
        stringsAsFactors = FALSE
      )
      
      # Append the names of variables selected by the AIC criterion
      if (length(selected_vars) > 0) {
        var_df <- data.frame(
          Metric = rep("Selected_Variable", length(selected_vars)),
          Value = selected_vars,
          stringsAsFactors = FALSE
        )
        result_df <- rbind(result_df, var_df)
      }
      
      # Define the unique result file name
      result_file_path <- file.path(OUTPUT_DIR, paste0(tools::file_path_sans_ext(file_name), "_R_stepwise.xlsx"))
      
      # Save results
      write.xlsx(result_df, result_file_path, sheetName = "Stepwise_Results")
      cat(paste0("[STATUS] Processed: ", file_name, "\n"))
      
    }, error = function(e) {
      cat(paste0("[ERROR] Failed to analyze ", file_name, ": ", e$message, "\n"))
    })
  }
  
  cat("-" * 30, "\n")
  cat(paste0("[COMPLETE] R-Stepwise analysis finalized. Results: ", OUTPUT_DIR, "\n"))
}