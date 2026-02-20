% ==============================================================================
% Script: 09_stepwise_regression_MATLAB.m
% Description: Performs stepwise linear regression using MATLAB's built-in 
%              'stepwiselm' function, which relies on iterative F-tests.
% ==============================================================================

clear; clc;

% ==============================================================================
% PATH CONFIGURATION (User Must Modify These Paths)
% ==============================================================================
% [TODO] Replace with your absolute directory path for expanded datasets
folder_path = 'YOUR_EXPANDED_DATA_DIRECTORY_PATH_HERE';

% [TODO] Replace with your desired output directory for MATLAB results
output_folder = 'YOUR_OUTPUT_DIRECTORY_PATH_HERE';

% ------------------------------------------------------------------------------

% Ensure the output directory exists
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% List all Excel files in the directory
files = dir(fullfile(folder_path, '*.xlsx'));
% Filter out temporary/hidden files
excel_files = {files(~startsWith({files.name}, '~$')).name};

if isempty(excel_files)
    fprintf('[WARN] No valid .xlsx files found in: %s\n', folder_path);
else
    fprintf('[INFO] Batch processing %d files for MATLAB-Stepwise analysis...\n', length(excel_files));

    % --- Main Loop: Process each dataset ---
    for i = 1:length(excel_files)
        current_file = excel_files{i};
        file_path = fullfile(folder_path, current_file);
        
        try
            % Read experimental data from 'After' worksheet
            % readtable is used to preserve column names for mdl.VariableNames
            raw_table = readtable(file_path, 'Sheet', 'After');
            
            if isempty(raw_table)
                fprintf('[SKIP] %s: Dataset is empty.\n', current_file);
                continue;
            end

            % 1. Execute Stepwise Linear Regression
            % MATLAB's stepwiselm uses p-values of F-statistics for entry/removal
            % This corresponds to the MATLAB algorithmic pathway in the manuscript
            mdl = stepwiselm(raw_table, 'Verbose', 0);
            
            % 2. Extract Performance Metrics
            R_squared = mdl.Rsquared.Ordinary;
            % CoefficientNames captures the names of retained terms (including intercept)
            retained_terms = mdl.CoefficientNames;
            
            % 3. Format and Export Results
            % Constructing a results table compatible with cross-platform comparison
            % Row 1: R-squared | Row 2+: Selected Variables
            max_len = max(length(retained_terms), 1);
            result_data = cell(max_len + 1, 1);
            result_data{1} = R_squared;
            result_data(2:end) = retained_terms';
            
            result_final_table = cell2table(result_data, 'VariableNames', {'Stepwise_Results'});
            
            % Construct output filename
            [~, name_only, ~] = fileparts(current_file);
            output_file_name = fullfile(output_folder, sprintf('result_MATLAB_%s.xlsx', name_only));
            
            % Write to Excel
            writetable(result_final_table, output_file_name);
            fprintf('[STATUS] Processed: %s\n', current_file);
            
        catch ME
            fprintf('[ERROR] Failed to analyze %s: %s\n', current_file, ME.message);
        end
    end

    fprintf('------------------------------------------------------------\n');
    fprintf('[COMPLETE] MATLAB Stepwise analysis finalized. Results: %s\n', output_folder);
end