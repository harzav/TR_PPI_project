import pandas as pd
import joblib
from sklearn.impute import KNNImputer
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

#provide output path for the results
path=''
#provide the test dataset
test_data_path = "./example_datasets/AFFINITY_DATASET/dg_test_ds.csv"
test_data = pd.read_csv(test_data_path)


#---------------------------------------------------------------------------------------------------#
# best model predictions
#---------------------------------------------------------------------------------------------------#

# Load the compressed pickle (pkl.z) regression model for the BEST MODEL TESTING

# EXAMPLE: model_results/regression_results/Output_dg_dataset_NEW_2

model_path = './Output_dg_dataset_NEW_2/classification_models/00001finalSingleModel.pkl.z'
with open(model_path, 'rb') as file:
    regression_model = joblib.load(file)

# CSV file containing features (headers included) with '1' in the 12th row
features_file_path = './Output_dg_dataset_NEW_2/feature_selection/features_FinalFront1.csv'

# Load the CSV file, skipping the first 11 rows
features_1 = pd.read_csv(features_file_path)


# Identify columns with '1' in the 12th row
selected_columns = features_1.iloc[0] == 1

# Select only the columns with '1' in the 12th row
selected_feature_names = features_1.columns[selected_columns]

# Upload test data

samples=test_data[['Sample Names']]

test_data = test_data.drop(columns=['Sample Names', 'Predicted Classes','Probability Score'], axis=1)

# Extract only the selected features from the test dataset
filtered_test_data = test_data[selected_feature_names]


# Make predictions using the loaded regression model
predictions = regression_model.predict(filtered_test_data)

# Create a DataFrame with the predictions and original data
results_df = pd.DataFrame({'DG': predictions})

column_name='DG'
y_true = samples[column_name]
y_pred = results_df[column_name]

# Ensure the lengths are the same
if len(y_true) != len(y_pred):
    raise ValueError("The target columns in the two datasets must have the same length")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate RRSE (Root Relative Squared Error)
mean_y_true = np.mean(y_true)
rrse = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mean_y_true) ** 2))

# Calculate RAE (Relative Absolute Error)
rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - mean_y_true))

# Calculate Spearman correlation and p-value
spearman_corr, spearman_p_value = spearmanr(y_true, y_pred)

# Print the metrics
print(f"RMSE: {rmse}")
print(f"RRSE: {rrse}")
print(f"RAE: {rae}")
print(f"Spearman Correlation: {spearman_corr}")
print(f"Spearman p-value: {spearman_p_value}")

# save the metrics to a CSV file
metrics = pd.DataFrame({
    'Metric': ['RMSE', 'RRSE', 'RAE', 'Spearman Correlation', 'Spearman p-value'],
    'Value': [rmse, rrse, rae, spearman_corr, spearman_p_value]
})
metrics.to_csv(path + 'dg_test_NEW_bm_run_2.csv', index=False)
# Save the results to a new CSV file
results_df.to_csv(path + 'predictions_test_NEW_bm_run_2.csv', index=False)

#---------------------------------------------------------------------------------------------------#
# ensemble predictions
#---------------------------------------------------------------------------------------------------#

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return joblib.load(file)

# List of paths to model files and corresponding rows for feature selection
model_info = [
    {'path': f'/Output_dg_dataset_NEW_2/classification_models/000{str(i).zfill(2)}finalSingleModel.pkl.z', 'feature_row': i - 1}
    # insert the number of final Pareto front models in the form of range, in this case 24 models
    for i in range(1, 25) if i not in [12] #exclude some models that do not exceed a threshold, in this case ex. 00012finalSingleModel.pkl.z
]

# Load the models
models = [load_model(info['path']) for info in model_info]

# CSV file containing features (headers included)
features_file_path = './Output_dg_dataset_NEW_2/feature_selection/features_FinalFront1.csv'
features_1 = pd.read_csv(features_file_path)

samples=test_data[['Sample Names']]

test_data = test_data.drop(columns=['Sample Names', 'Predicted Classes','Probability Score'], axis=1)
def convert_to_float(value):
    if isinstance(value, str):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return value  # Return the original value if conversion is not possible
    else:
        return value  # Return the original value if it's not a string

# Apply the conversion function to the entire DataFrame
test_data = test_data.applymap(convert_to_float)
print(test_data)

# Make predictions using the loaded regression models and average them for ensemble
ensemble_predictions = np.zeros(len(test_data))
for i, info in enumerate(model_info):
    # Identify columns with '1' in the specified row
    selected_columns = features_1.iloc[info['feature_row']] == 1

    # Select only the columns with '1' in the specified row
    selected_feature_names = features_1.columns[selected_columns]

    # Extract only the selected features from the test dataset
    filtered_test_data = test_data[selected_feature_names]

    # Make predictions using the current model
    predictions = models[i].predict(filtered_test_data)
    ensemble_predictions += predictions

ensemble_predictions /= len(models)

# Create a DataFrame with the ensemble predictions and original data
results_df = pd.DataFrame({'TR_aff': ensemble_predictions})

# Compute the metrics
column_name = 'DG'
y_true = samples[column_name]
y_pred = results_df[column_name]

# Ensure the lengths are the same
if len(y_true) != len(y_pred):
    raise ValueError("The target columns in the two datasets must have the same length")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate RRSE (Root Relative Squared Error)
mean_y_true = np.mean(y_true)
rrse = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mean_y_true) ** 2))

# Calculate RAE (Relative Absolute Error)
rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - mean_y_true))

# Calculate Spearman correlation and p-value
spearman_corr, spearman_p_value = spearmanr(y_true, y_pred)

# Print the metrics
print(f"RMSE: {rmse}")
print(f"RRSE: {rrse}")
print(f"RAE: {rae}")
print(f"Spearman Correlation: {spearman_corr}")
print(f"Spearman p-value: {spearman_p_value}")

# Optionally, save the metrics to a CSV file
metrics = pd.DataFrame({
    'Metric': ['RMSE', 'RRSE', 'RAE', 'Spearman Correlation', 'Spearman p-value'],
    'Value': [rmse, rrse, rae, spearman_corr, spearman_p_value]
})
metrics.to_csv(path+ 'dg_test_NEW_ens_run_2.csv', index=False)

# Save the results to a new CSV file
results_df.to_csv(path+ 'predictions_test_NEW_ens_run_2.csv', index=False)

