"""
DataLoader module for RCM (Regional Climate Model) emulation training.

This module provides PyTorch Dataset classes and data loading utilities for training
deep learning emulators on the CORDEX Benchmark dataset. The workflow handles:
- Loading predictor (GCM) and predictand (RCM target) data from NetCDF files
- Splitting data into training and test sets based on year ranges
- Normalizing/standardizing input features
- Creating PyTorch DataLoaders for training and evaluation

For more details on the CORDEX Benchmark dataset properties, see the data notebooks.
"""

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class EmulationTrainingDataset(Dataset):
    """
    PyTorch Dataset for RCM emulation training.

    This dataset outputs both inputs (GCM predictors) and targets (RCM outputs)
    for supervised training of the emulator.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
        y_data: Target predictand data (RCM output). Can be numpy array or torch tensor.
    """

    def __init__(self, x_data, y_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        if not isinstance(y_data, torch.Tensor):
            y_data = torch.tensor(y_data)
        self.x_data, self.y_data = x_data, y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample, y_sample = self.x_data[idx, :], self.y_data[idx, :]
        return x_sample, y_sample

    def _get_shapes(self):
        return self.x_data.shape[1:], self.y_data.shape[1:]


class EmulationTestDataset(Dataset):
    """
    PyTorch Dataset for making predictions with trained RCM emulator.

    This dataset outputs only inputs (GCM predictors) and is used for inference
    when ground-truth target data is not available.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
    """

    def __init__(self, x_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx, :]

    def _get_shapes(self):
        return self.x_data.shape[1:]


def build_dataloaders(cf):
    """
    Build training and test DataLoaders from CORDEX Benchmark dataset.

    This function handles the complete data loading pipeline:
    1. Loads predictor (GCM) and predictand (RCM target) NetCDF files
    2. Splits data into train/test sets based on year ranges
    3. Applies normalization (standardization, minmax, or none)
    4. Flattens spatial dimensions for models with fully connected output layers
    5. Creates PyTorch DataLoaders for training and evaluation

    The benchmark provides two training experiments:
    - 'ESD_pseudo_reality': Train on 1961-1974, test on 1975-1980
    - 'Emulator_hist_future': Train on 1961-1980 + 2080-2089, test on 2090-2099

    For predictands, both daily maximum temperature ('tasmax') and daily accumulated
    precipitation ('pr') are available.

    Note: The benchmark does not yet provide ground-truth RCM data for evaluation
    experiments, so a test set is generated from the training data to provide an
    initial indication of emulator performance.

    Args:
        cf: Configuration object containing data settings:
            - cf.data.training_experiment: 'ESD_pseudo_reality' or 'Emulator_hist_future'
            - cf.data.domain: 'ALPS', 'NZ', or 'SA'
            - cf.data.var_target: Target variable ('tasmax' or 'pr')
            - cf.data.data_path: Path to CORDEX benchmark data
            - cf.data.normalization: 'standardization', 'minmax', 'log', or None
            - cf.data.num_workers: Number of workers for DataLoader
            - cf.training.batch_size: Batch size for training

    Returns:
        tuple: (dataloader_train, test_dataloader, cf)
            - dataloader_train: DataLoader for training data
            - test_dataloader: DataLoader for test data
            - cf: Updated configuration with gcm_name added
    """
    if cf.data.training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif cf.data.training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError("Provide a valid date")

    # Set the GCM
    if cf.data.domain == "ALPS":
        cf.data.gcm_name = "CNRM-CM5"
    elif (cf.data.domain == "NZ") or (cf.data.domain == "SA"):
        cf.data.gcm_name = "ACCESS-CM2"

    predictor_filename = f"{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/predictors/{cf.data.gcm_name}_{period_training}.nc"
    predictor = xr.open_dataset(predictor_filename)
    if cf.data.domain == "SA":
        predictor = predictor.drop_vars("time_bnds")

    predictand_filename = f"{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/target/pr_tasmax_{cf.data.gcm_name}_{period_training}.nc"
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[cf.data.var_target]]

    if cf.data.training_experiment == "ESD_pseudo_reality":
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1980 + 1))
    elif cf.data.training_experiment == "Emulator_hist_future":
        years_train = list(range(1961, 1980 + 1)) + list(range(2080, 2090))
        years_test = list(range(2090, 2099 + 1))

    x_train = predictor.sel(time=np.isin(predictor["time"].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand["time"].dt.year, years_train))

    x_test = predictor.sel(time=np.isin(predictor["time"].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand["time"].dt.year, years_test))

    if cf.data.normalization == "standardization":
        mean_train = x_train.mean("time")
        std_train = x_train.std("time")

        x_train_stand = (x_train - mean_train) / std_train
        x_test_stand = (x_test - mean_train) / std_train
    elif cf.data.normalization == "minmax":
        min_train = x_train.min("time")
        max_train = x_train.max("time")

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_test_stand = (x_test - min_train) / (max_train - min_train)
    elif cf.data.normalization == "log":
        raise NotImplementedError("Log normalization not implemented yet")
    else:
        x_train_stand = x_train
        x_test_stand = x_test

    if cf.data.domain == "ALPS":
        spatial_dims = ("x", "y")
    elif (cf.data.domain == "NZ") or (cf.data.domain == "SA"):
        spatial_dims = ("lat", "lon")

    y_train_stack = y_train.stack(gridpoint=spatial_dims)
    y_test_stack = y_test.stack(gridpoint=spatial_dims)

    x_train_stand_array = torch.from_numpy(
        x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values
    )
    y_train_stack_array = torch.from_numpy(y_train_stack.to_array()[0, :].values)

    x_test_stand_array = torch.from_numpy(
        x_test_stand.to_array().transpose("time", "variable", "lat", "lon").values
    )
    y_test_stack_array = torch.from_numpy(y_test_stack.to_array()[0, :].values)

    dataset_training = EmulationTrainingDataset(
        x_data=x_train_stand_array, y_data=y_train_stack_array
    )
    dataloader_train = DataLoader(
        dataset=dataset_training,
        batch_size=cf.training.batch_size,
        shuffle=True,
        num_workers=cf.data.num_workers,
    )

    dataset_test = EmulationTrainingDataset(
        x_data=x_test_stand_array, y_data=y_test_stack_array
    )
    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=cf.training.batch_size,
        shuffle=False,
        num_workers=cf.data.num_workers,
    )

    return dataloader_train, test_dataloader, cf


def build_dummy_dataloaders(
    batch_size=2, num_samples_train=100, num_samples_test=20, num_workers=0
):
    """
    Build dummy DataLoaders with random data for debugging and testing.
    Creates synthetic data with the expected shapes for RCM emulation:
    - Input (x): (batch, 15, 16, 16)
    - Target (y): (batch, 128, 128) flattened to (batch, 16384)
    """
    # Generate random training data
    x_train = torch.randn(num_samples_train, 15, 16, 16, dtype=torch.float32)
    y_train = torch.randn(num_samples_train, 128 * 128, dtype=torch.float32)

    # Generate random test data
    x_test = torch.randn(num_samples_test, 15, 16, 16, dtype=torch.float32)
    y_test = torch.randn(num_samples_test, 128 * 128, dtype=torch.float32)

    # Create datasets
    dataset_train = EmulationTrainingDataset(x_data=x_train, y_data=y_train)
    dataset_test = EmulationTrainingDataset(x_data=x_test, y_data=y_test)

    # Create dataloaders
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader_train, test_dataloader
