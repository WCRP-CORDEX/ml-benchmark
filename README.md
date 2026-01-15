# CORDEX ML-Bench: A benchmark for Machine Learning-Based Regional Climate Downscaling

CORDEX ML-Bench is a benchmark designed to evaluate the performance of machine learning–based climate downscaling models across different regions covering both the standard (perfect prognosis ESD) and emulation climate downscaling approaches. It defines standardized training and test experiments assessing various downscaling challenges along with the corresponding datasets from Regional Climate Models (RCMs) driven by different Global Climate Models (GCMs). 

This repository contains all the materials, instructions, and datasets required to run the different experiments, as well as notebooks facilitating the process. It also incldues instructions for registering and submitting contributions to participate in the online benchmaking, with results publicly reported in this page. 

**Development Status Notice**  
*This repository is currently under active development. As a result, the structure, documentation, datasets and experimental protocols may change in the near future. Users should be aware that updates may require adjustments to existing workflows. We recommend regularly checking for updates.*

## Dataset Overview

The benchmark covers three geographic regions: New Zealand (NZ), Europe (ALPS) and South Africa (SA) with ~10km target resolution. Training and test datasets (NetCDF files, approximately 5 GB per domain) are provided for each region, including common predictors (Z, U, V, T, Q at 850, 700, 500, and 300 hPa) and predictands (daily temperature and precipitation). More information in [./data](/data)

## Experiments

The benchmark covers two experiments with different tests focusing on the standard (ESD) and emulation climate downscaling approaches. For each region data is obtained from a single RCM driven by two different GCMs, one used both for training and testing (denoted below as `same GCM`) and the other only used to test transferability (denoted as `different GCM`). 

For both experiments, training is based on perfect (upscaled) predictors from the RCM while test experiments explore both perfect of imperfect (from the driving GCM) predictors. Predictands (target for training) correspond to the RCM highres 10km temperature and precipitation output.

- **Experiment 1: ESD Pseudo-Reality**: A 20-year (1961–1980) training period in present climate conditions, designed to mimic the standard statistical climate downscaling approach and test extrapolation capabilities of the methods. 

| Test   | Test Period | Predictor type | Eval | 
|----------------|---------------|----------------|------|
| Test1: Perfect Cross-Validation | Historical (1981–2000) | Perfect (from RCM), same GCM | Error, Clim | 
| Test2: Imperfect Cross-Validation | Historical (1981–2000) | Imperfect (from GCM), same GCM | Error, Clim | 


| Training Setup | Test Period | Test Experiment | Notes | Eval | 
|----------------|---------------|----------------|-------|------|
| ESD “pseudo-reality”<br>Period: 1961–1980 | Historical (1981–2000) | Perfect Cross-Validation | Same GCM, perfect predictors | Error, Clim | 
|  | 2041–2060 + 2081–2100 | Perfect Extrapolation | Same GCM, perfect predictors | change signal for mid/final term | 
|  | 2041–2060 + 2081–2100 | Imperfect Extrapolation| Same GCM, imperfect predictors | change signal for mid/final term| 
|  | 2041–2060 + 2081–2100 | Perfect Extrapolation (GCM Transferability) | Different GCM, perfect predictors  | change signal for mid/final term | 


- **Experiment 2: Emulator Hist+Future**: A 40-year (1961–1980 + 2081–2100) training period combining present and future climates, focused on testing interpolation and transferability of emulators.

| Test   | Test Period | Predictor type | Eval | 
|----------------|---------------|----------------|------|
| Test1: Perfect Cross-Validation | Historical (1981–2000) | Perfect (from RCM), same GCM | Error, Clim | 
| Test2: Imperfect Cross-Validation | Historical (1981–2000) | Imperfect (from GCM), same GCM | Error, Clim | 


| Training Setup | Test Period | Test Experiment | Notes | Eval | 
|----------------|---------------|----------------|-------|------|
|  | Historical (1981–2000) | Perfect Cross-Validation (GCM Transferability) | Different GCM, perfect predictors  | Error, Clim | 
|  | Historical (1981–2000) | Imperfect Cross-Validation (GCM Transferability) | Different GCM, imperfect predictors  | Error, Clim | 
|  | 2041–2060  | Perfect Interpolation | Same GCM, perfect predictors  | change signal | 
|  | 2041–2060  | Imperfect Interpolation | Same GCM, imperfect predictors  | change signal | 
|  | 2041–2060 | Perfect Interpolation (GCM Transferability) | Different GCM, perfect predictors  | change signal | 
|  | 2041–2060 | Imperfect Interpolation (GCM Transferability) | Different GCM, imperfect predictors  | change signal | 


## Model Training and Evaluation

A comprehensive set of evaluation metrics and accompanying code for assessing ML-based downscaling methods within the CORDEX ML-Bench framework is provided in this repository. Additional information and illustrative notebooks demonstrating their use are available in [./evaluation](/evaluation)


The [./format_predictions](./format_predictions) directory provides utilities and templates to help users structure their model outputs in the required NetCDF format for CORDEX ML-Bench evaluation. It includes ready-to-use NetCDF templates.

Please note that this directory is currently intended for internal use and may be subject to modification prior to public release.

## Contributing to the benchmark [PROVISIONAL]

## Submitting contributions

The registration and submission process is managed ****

The submission and automatic benchmarking process allows users to benchmark the performance of their own models against well-established reference approaches. The table below summarizes the contributing models and provides links to their respective implementation repositories when available (this table is automatically updated with new submissions).

| Model       | Description | Reference | Implementation |
|-------------|-------------|-----------|----------------|
| DeepESD     | Convolutional neural network  | [Baño-Medina et al., 2024](https://gmd.copernicus.org/articles/15/6747/2022/) | [GitHub repository]() |
| Model2        | | | |
| Model3        | | | |


## Scoreboard [PROVISIONAL]

The following scoreboard presents basic evaluation results for all contributing models (see *** for contribution instructions).

| Model              | RMSE (°C)  | MAE (°C)  | R²    | Training Time     | Inference Speed (samples/sec) |
|--------------------|------------|-----------|-------|-------------------|-------------------------------|
| DeepESD             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model2             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model3             | XXX        | XXX       | XXX   | XXX               | XXX                           |

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

## Citation

Link to the pre-print

## Contact

Include an issue in this repository
