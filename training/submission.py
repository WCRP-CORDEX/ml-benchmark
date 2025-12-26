"""
Simplified submission script for the CORDEX Benchmark. 

This script generates predictions for all available test files and formats 
them according to the required directory structure.
"""

import os
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import zipfile
import glob

# Set paths
DATA_PATH = '../data/Bench-data'
MODELS_PATH = './models'
OUTPUT_BASE = './submission_files'

# Set the name of the contribution
EMULATOR_ID = "DeepESD-v1"
INSTITUTION_ID = "IFCA"
TRAINING_ID = "m1"

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MMap the different domains to training GCMs and spatial dimensions
DOMAIN_INFO = {'ALPS': {'train_gcm': 'CNRM-CM5', 'spatial_dims': ('x', 'y')},
               'NZ': {'train_gcm': 'ACCESS-CM2', 'spatial_dims': ('lat', 'lon')},
               'SA': {'train_gcm': 'ACCESS-CM2', 'spatial_dims': ('lat', 'lon')}}

# Set the experiments to run
EXPERIMENTS = ['ESD_pseudo_reality', 'Emulator_hist_future']

# Set the orography options
OROG_OPTIONS = [True, False]

# DeepESD model (accepts orography as input)
class DeepESD(nn.Module):
    def __init__(self, x_shape, y_shape, filters_last_conv, device, orog_data=None):
        super(DeepESD, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.orog_data = orog_data
        self.orog_embed_dim = 128

        if self.orog_data is not None:
            self.orog_data = torch.from_numpy(orog_data).float().to(device)
            if len(self.orog_data.shape) == 1:
                self.orog_data = self.orog_data.unsqueeze(0)

        self.conv_1 = nn.Conv2d(in_channels=self.x_shape[1], out_channels=50, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=25, out_channels=self.filters_last_conv, kernel_size=3, padding=1)

        input_size_linear = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv
        if self.orog_data is not None:
            self.orog_embed = nn.Linear(in_features=self.orog_data.shape[1], out_features=self.orog_embed_dim)
            input_size_linear += self.orog_embed_dim

        self.out = nn.Linear(in_features=input_size_linear, out_features=self.y_shape[1])

    def forward(self, x):
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)

        if self.orog_data is not None:
            orog_embed = torch.relu(self.orog_embed(self.orog_data))
            orog_embed = orog_embed.repeat(x.size(0), 1)
            x = torch.cat((x, orog_embed), dim=1)

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
def run_prediction(domain, experiment, use_orog, predictor_path):
    ds_test = xr.open_dataset(predictor_path)
    if domain == 'SA': ds_test = ds_test.drop_vars('time_bnds', errors='ignore')
    
    # Standardize the predictors
    mean_train, std_train = get_training_stats(domain, experiment)
    ds_test_stand = (ds_test - mean_train) / std_train
    x_test_arr = ds_test_stand.to_array().transpose("time", "variable", "lat", "lon").values
    x_test_tensor = torch.from_numpy(x_test_arr).float().to(DEVICE)

    # 2. Orography
    orog_arr = None
    if use_orog:
        orog_path = f'{DATA_PATH}/{domain}/{domain}_domain/train/{experiment}/predictors/Static_fields.nc'
        orog = xr.open_dataset(orog_path)['orog']
        orog_arr = orog.values.flatten()
        orog_arr = orog_arr / np.max(orog_arr)

    # 3. Predict for both pr and tasmax
    results = {}
    
    # Load training target to get the correct number of gridpoints in the predictand
    gcm_train = DOMAIN_INFO[domain]['train_gcm']
    spatial_dims = DOMAIN_INFO[domain]['spatial_dims']

    period_train = '1961-1980' if experiment == 'ESD_pseudo_reality' else '1961-1980_2080-2099'
    target_path = f'{DATA_PATH}/{domain}/{domain}_domain/train/{experiment}/target/pr_tasmax_{gcm_train}_{period_train}.nc'
    ds_target = xr.open_dataset(target_path)
    n_gridpoints = ds_target[spatial_dims[0]].size * ds_target[spatial_dims[1]].size

    # Iterate over the variables to predict
    for var in ['pr', 'tasmax']:
        # Set the model name
        orog_suffix = "orog" if use_orog else "no_orog"
        model_name = f'DeepESD_{experiment}_{domain}_{var}_{orog_suffix}.pt'
        model_path = os.path.join(MODELS_PATH, model_name)
        
        # Initialize and load model
        model = DeepESD(x_shape=x_test_arr.shape, y_shape=(None, n_gridpoints), 
                        filters_last_conv=1, device=DEVICE, orog_data=orog_arr).to(DEVICE)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        model.eval()
        
        # Compute the predictions
        with torch.no_grad():
            preds = model(x_test_tensor).cpu().numpy()
        
        # Unstack back to 2D grid using target dimensions
        preds_reshaped = preds.reshape(len(ds_test.time), ds_target[spatial_dims[0]].size, ds_target[spatial_dims[1]].size)
        results[var] = (('time',) + spatial_dims, preds_reshaped)

    # Create output dataset with time from test and spatial coords from target
    out_coords = {'time': ds_test.time}
    for dim in spatial_dims:
        out_coords[dim] = ds_target[dim]
    
    ds_out = xr.Dataset(results, coords=out_coords)
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
            
            # Iterate over the experiments and orography options
            for experiment in EXPERIMENTS:
                for use_orog in OROG_OPTIONS:
                    orog_label = "OROG" if use_orog else "NO_OROG"
                    exp_folder = f"{experiment}_{orog_label}"
                    
                    print(f"  Predicting {filename} for {exp_folder}...")
                    
                    ds_preds = run_prediction(domain, experiment, use_orog, pred_path)
                    
                    # Build output path: Domain_Domain/Exp_OROG/period/condition/
                    out_dir = os.path.join(OUTPUT_BASE, f"{domain}_Domain", exp_folder, period_folder, condition)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    out_filename = f"Predictions_pr_tasmax_{filename}"
                    ds_preds.to_netcdf(os.path.join(out_dir, out_filename))

    # ZIP the submission
    zip_filename = f"{EMULATOR_ID}_{INSTITUTION_ID}_{TRAINING_ID}.zip"
    zip_path = os.path.join(OUTPUT_BASE, zip_filename)

    print(f"\nCreating submission package: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_BASE):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, OUTPUT_BASE)
                zipf.write(abs_path, rel_path)

    print("\nDone!")
