"""
Training script for the DeepESD model. It trains the model on the CORDEX Benchmark dataset for
all the required given domains, experiments, variables and orography options.

This script is just for showcase how to train a deep learning model on the CORDEX Benchmark dataset.
It is not intended to produce robust results.
"""

import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Training dataset
class EmulationTrainingDataset(Dataset):
    def __init__(self, x_data, y_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.from_numpy(x_data).float()
        if not isinstance(y_data, torch.Tensor):
            y_data = torch.from_numpy(y_data).float()
        self.x_data, self.y_data = x_data, y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

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

# Paths to the data and models
DATA_PATH = '../data/Bench-data'

MODELS_PATH = './models'
os.makedirs(MODELS_PATH, exist_ok=True)

# Device to use for training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Var target to train on
var_target = ['tasmax', 'pr']

# Domains to train on
domains = ['ALPS', 'NZ', 'SA']

# Experiments to train on
experiments = ['ESD_pseudo_reality', 'Emulator_hist_future']
use_orography_options = [True, False]

# Loop through domains, experiments, and orography options
for var_target in var_target:
    for domain in domains:
        for training_experiment in experiments:
            for use_orography in use_orography_options:
                
                # Set the period
                if training_experiment == 'ESD_pseudo_reality':
                    period_training = '1961-1980'
                else:
                    period_training = '1961-1980_2080-2099'

                # Set the GCM
                gcm_name = 'CNRM-CM5' if domain == 'ALPS' else 'ACCESS-CM2'

                # Set the spatial dimensions
                spatial_dims = ('x', 'y') if domain == 'ALPS' else ('lat', 'lon')

                # Load predictors
                predictor_filename = f'{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
                predictor = xr.open_dataset(predictor_filename)
                if domain == 'SA':
                    predictor = predictor.drop_vars('time_bnds', errors='ignore')

                # Load predictand
                predictand_filename = f'{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
                predictand = xr.open_dataset(predictand_filename)[[var_target]]

                # Standardize predictors
                mean_train = predictor.mean('time')
                std_train = predictor.std('time')
                x_train_stand = (predictor - mean_train) / std_train

                # Stack predictand
                y_train_stack = predictand.stack(gridpoint=spatial_dims)

                # Convert to NumPy array
                x_train_arr = x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values
                y_train_arr = y_train_stack[var_target].values

                # Orography data
                orog_arr = None
                if use_orography:
                    orog_path = f'{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/predictors/Static_fields.nc'
                    orog = xr.open_dataset(orog_path)['orog']
                    orog_arr = orog.values.flatten()
                    orog_arr = orog_arr / np.max(orog_arr) # Normalize orography to 0-1

                # Create dataset and dataloader
                dataset = EmulationTrainingDataset(x_data=x_train_arr, y_data=y_train_arr)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                # Create model
                model = DeepESD(x_shape=x_train_arr.shape, y_shape=y_train_arr.shape, 
                                filters_last_conv=1, device=DEVICE, orog_data=orog_arr)
                model = model.to(DEVICE)
                
                # Create optimizer and loss function
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                loss_function = nn.MSELoss()

                # Set number of epochs
                num_epochs = 20

                # Train model
                print(f"Training: {domain} | {training_experiment} | Orography: {use_orography} | Variable: {var_target}")

                # Compact training loop
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    for batch_x, batch_y in dataloader:
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = loss_function(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * batch_x.size(0)
                    
                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataset):.6f}')

                # Save model
                orog_suffix = "orog" if use_orography else "no_orog"
                model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}_{orog_suffix}.pt'
                torch.save(model.state_dict(), f'{MODELS_PATH}/{model_name}')