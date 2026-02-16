"""
Training script for the DeepESD model. It trains the model on the CORDEX Benchmark dataset for
all the required given domains, experiments and variables.

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

# Loop through domains and experiments
for var_target in var_target:
    for domain in domains:
        for training_experiment in experiments:
                
            # Set the period
            if training_experiment == 'ESD_pseudo_reality':
                period_training = '1961-1980'
            else:
                period_training = '1961-1980_2080-2099'

            # Set the GCM
            gcm_name = 'CNRM-CM5' if domain == 'ALPS' else 'ACCESS-CM2'

            # Set the spatial dimensions
            spatial_dims = ('y', 'x') if domain == 'ALPS' else ('lat', 'lon')

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

            # Create dataset and dataloader
            dataset = EmulationTrainingDataset(x_data=x_train_arr, y_data=y_train_arr)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Create model
            model = DeepESD(x_shape=x_train_arr.shape, y_shape=y_train_arr.shape, 
                            filters_last_conv=1, device=DEVICE)
            model = model.to(DEVICE)
            
            # Create optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            loss_function = nn.MSELoss()

            # Set number of epochs
            num_epochs = 10 # This is a small number of epochs for demonstration purposes

            # Train model
            print(f"Training: {domain} | {training_experiment} | Variable: {var_target}")

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
            model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}.pt'
            torch.save(model.state_dict(), f'{MODELS_PATH}/{model_name}')