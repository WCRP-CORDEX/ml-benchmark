# CORDEX ML-Bench: A benchmark for Machine Learning-Based Regional Climate Downscaling

CORDEX ML-Bench is a benchmark designed to evaluate the performance of machine learning–based climate downscaling models across different domains. It provides standardized training and evaluation experiments, along with the corresponding datasets. Example notebooks are included to facilitate ease of use and ensure reproducibility. This repository contains all the materials, instructions, and datasets required to train and evaluate climate downscaling models using the CORDEX ML-Bench framework.

## Dataset Overview

The benchmark covers three geographic domains: New Zealand (NZ), Europe (ALPS) and South Africa (SA) with 0.11º target resolution. Training and evaluation datasets (NetCDF files, approximately 5 GB per domain) for each region are publicly available on [Zenodo](https://zenodo.org/records/17517423). These include predictor (Z, U, V, T, Q at 850, 700, 500, and 300 hPa) and predictand (daily temperature and precipitation) derived from different Regional Climate Models (RCMs) driven by renalysis data (ERA5) and Global Climate Models (GCMs), enabling systematic evaluation under both historical and future climate conditions. More information in [./data](/data)

## Experiments

Two core experiments have been designed to evaluate the performance of Empirical Statistical Downscaling (ESD) methods and emulators using the same dataset:
- ***ESD Pseudo-Reality*** (1961–1980, historical): Models are trained over a 20-year historical period using RCM temperature and precipitation outputs as pseudo-observations. Testing includes future climate conditions (2081–2100) to assess extrapolation.
- ***Emulator*** (1961–1980 historical + 2081–2100 future): Models are trained on a 40-year combined dataset encompassing both historical and future climates. Testing evaluates model transferability across different scenarios (soft transferability) and across GCMs (hard transferability).

## Evaluation

A comprehensive set of evaluation metrics and accompanying code for assessing ML-based downscaling methods within the CORDEX ML-Bench framework is provided in this repository. Additional information and illustrative notebooks demonstrating their use are available in [./evaluation](/evaluation)


## Contributing Models

CORDEX ML-Bench includes a collection of state-of-the-art ML-based methods, with reproducible code provided in some cases. This allows users to benchmark the performance of their own models against well-established reference approaches. The table below summarizes the available models and provides links to their respective implementation repositories.

| Model       | Description | Reference | Implementation |
|-------------|-------------|-----------|----------------|
| DeepESD     | Convolutional neural network  | [Baño-Medina et al., 2024](https://gmd.copernicus.org/articles/15/6747/2022/) | [GitHub repository]() |
| ****        | | | |

## Scoreboard

The following scoreboard presents basic evaluation results for all contributing models (see *** for contribution instructions).

| Model               | RMSE (°C) | MAE (°C) | R²    | Training Time (hrs) | Inference Speed (samples/sec) |
|--------------------|------------|-----------|-------|-------------------|-------------------------------|
| DeepESD            | 1.12       | 0.85      | 0.91  | 5.2               | 120                           |
| LSTM               | 1.05       | 0.80      | 0.93  | 12.4              | 85                            |
| Transformer        | 0.98       | 0.75      | 0.95  | 18.7              | 65                            |
| U-Net              | 1.00       | 0.78      | 0.94  | 10.5              | 90                            |
| GraphConvNet       | 1.15       | 0.88      | 0.89  | 14.0              | 70                            |

## Requirements

The [./requirements](./requirements) directory contains an `environment.yaml` file that allows users to easily recreate the Conda environment required to run all scripts in this repository. To create the environment, run the following command:

```bash
conda env create -f environment.yaml
```
Alternatively, the basic requirements to run these scripts are:

```
os
requests
zipfile
xarray
netcdf4
matplotlib
cartopy
numpy
torch
```
These packages can be installed using any package management tool.

## Contributing to the benchmark

The [./format_predictions](./format_predictions) directory provides utilities and templates to help users structure their model outputs in the required NetCDF format for CORDEX ML-Bench evaluation. It includes ready-to-use NetCDF templates and helper functions for setting the appropriate metadata attributes.
Please note that this directory is currently intended for internal use and may be subject to modification prior to public release.

## Citation

Link to the pre-print

## Contact

Include an issue in this repository

















