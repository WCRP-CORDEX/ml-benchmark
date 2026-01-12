"""
Simplified submission script for the CORDEX Benchmark. 

This script generates predictions for all available test files and formats 
them according to the required directory structure.
"""

import os
import sys
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import zipfile
import glob

# Import format_predictions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set paths
DATA_PATH = '../data/Bench-data'
MODELS_PATH = '../training/models'
OUTPUT_BASE = './submission_files'
TEMPLATES_PATH = '../format_predictions/templates'

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MMap the different domains to training GCMs and spatial dimensions
DOMAIN_INFO = {'ALPS': {'train_gcm': 'CNRM-CM5', 'spatial_dims': ('x', 'y')},
               'NZ': {'train_gcm': 'ACCESS-CM2', 'spatial_dims': ('lat', 'lon')},
               'SA': {'train_gcm': 'ACCESS-CM2', 'spatial_dims': ('lat', 'lon')}}

# Set the experiments to run
EXPERIMENTS = ['ESD_pseudo_reality', 'Emulator_hist_future']

# DeepESD model
class DeepESD(nn.Module):
    def __init__(self, x_shape, y_shape, filters_last_conv, device):
        super(DeepESD, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

        self.conv_1 = nn.Conv2d(in_channels=self.x_shape[1], out_channels=50, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=25, out_channels=self.filters_last_conv, kernel_size=3, padding=1)

        input_size_linear = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv

        self.out = nn.Linear(in_features=input_size_linear, out_features=self.y_shape[1])

    def forward(self, x):
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)

        return self.out(x)

# Re-calculates mean and std from training data to standardize test predictors
def get_training_stats(domain, experiment):
    gcm = DOMAIN_INFO[domain]['train_gcm']
    period = '1961-1980' if experiment == 'ESD_pseudo_reality' else '1961-1980_2080-2099'
    path = f'{DATA_PATH}/{domain}/{domain}_domain/train/{experiment}/predictors/{gcm}_{period}.nc'
    
    ds = xr.open_dataset(path)
    if domain == 'SA':
        ds = ds.drop_vars('time_bnds', errors='ignore')

    return ds.mean('time'), ds.std('time')

# Computes the predictions for a specific predictor file
def run_prediction(domain, experiment, predictor_path):
    ds_test = xr.open_dataset(predictor_path)
    if domain == 'SA': ds_test = ds_test.drop_vars('time_bnds', errors='ignore')
    
    # Standardize the predictors
    mean_train, std_train = get_training_stats(domain, experiment)
    ds_test_stand = (ds_test - mean_train) / std_train
    x_test_arr = ds_test_stand.to_array().transpose("time", "variable", "lat", "lon").values
    x_test_tensor = torch.from_numpy(x_test_arr).float().to(DEVICE)

    # Predict for both pr and tasmax
    ds_out = xr.Dataset(coords={'time': ds_test.time})
    spatial_dims = DOMAIN_INFO[domain]['spatial_dims']

    # Iterate over the variables to predict
    for var in ['pr', 'tasmax']:
        # Load template for coordinates and variable attributes
        template_path = os.path.join(TEMPLATES_PATH, f'{var}_{domain}.nc')
        ds_template = xr.open_dataset(template_path)
        n_gridpoints = ds_template[spatial_dims[0]].size * ds_template[spatial_dims[1]].size

        # Set the model name
        model_name = f'DeepESD_{experiment}_{domain}_{var}.pt'
        model_path = os.path.join(MODELS_PATH, model_name)
        
        # Initialize and load model
        model = DeepESD(x_shape=x_test_arr.shape, y_shape=(None, n_gridpoints), 
                        filters_last_conv=1, device=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        model.eval()
        
        # Compute the predictions
        with torch.no_grad():
            preds = model(x_test_tensor).cpu().numpy()
        
        # Unstack back to 2D grid using template dimensions
        # It is necessary to set the order due to differences in lat/lon and x/y spatial dimensions
        if domain == 'NZ' or domain == 'SA':
            order = 'C'
        elif domain == 'ALPS':
            order = 'F'
            
        preds_reshaped = preds.reshape(len(ds_test.time), ds_template[spatial_dims[0]].size, ds_template[spatial_dims[1]].size,
                                       order=order)
        
        # Create DataArray with template's spatial coords and attributes
        da = xr.DataArray(preds_reshaped,
                          coords={**ds_template.coords, 'time': ds_test.time},
                          dims=('time',) + spatial_dims,
                          name=var,
                          attrs=ds_template[var].attrs)
        ds_out[var] = da

    return ds_out

# Main execution
if __name__ == "__main__":
    # Create the output base directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Iterate over the domains
    for domain in DOMAIN_INFO:
        print(f"Processing Domain: {domain}")
        
        # Find all test predictor files for this domain
        test_dir = f'{DATA_PATH}/{domain}/{domain}_domain/test'
        predictor_files = glob.glob(f'{test_dir}/**/*.nc', recursive=True)
        
        for pred_path in predictor_files:
            # Parse path parts: .../test/{period}/predictors/{condition}/{filename}.nc
            parts = pred_path.split(os.sep)
            # Find the index of 'test' to extract the period folder
            test_idx = parts.index('test')
            # Extract the period folder
            period_folder = parts[test_idx + 1]
            # Extract the condition
            condition = parts[test_idx + 3]
            # Extract the filename
            filename = parts[-1]
            
            # Iterate over the experiments
            for experiment in EXPERIMENTS:
                print(f"Predicting {filename} for {experiment}...")
                
                ds_preds = run_prediction(domain, experiment, pred_path)
                
                # Build output path: Domain_Domain/Experiment/period/condition/
                out_dir = os.path.join(OUTPUT_BASE, f"{domain}_Domain", experiment, period_folder, condition)
                os.makedirs(out_dir, exist_ok=True)
                
                out_filename = f"Predictions_pr_tasmax_{filename}"
                ds_preds.to_netcdf(os.path.join(out_dir, out_filename))

    # ZIP the submission
    zip_filename = "submission.zip"
    zip_path = os.path.join(OUTPUT_BASE, zip_filename)

    print(f"Creating submission package: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_BASE):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, OUTPUT_BASE)
                zipf.write(abs_path, rel_path)

    print("Done!")