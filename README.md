# CORDEX ML-Bench: A benchmark for Machine Learning-Based Regional Climate Downscaling

CORDEX ML-Bench is a benchmark designed to evaluate the performance of machine learning–based climate downscaling models across different regions covering both the standard (perfect prognosis ESD) and emulation climate downscaling approaches. It defines standardized training and test experiments assessing various downscaling challenges along with the corresponding datasets from Regional Climate Models (RCMs) driven by different Global Climate Models (GCMs). 

This repository contains all the materials, instructions, and datasets required to run the different experiments, as well as notebooks illustrating the process. It also includes instructions for registering and submitting contributions to participate in the online benchmarking, with results publicly reported in this page. 

**Development Status Notice**  
*This repository is currently under active development. As a result, the structure, documentation, datasets and experimental protocols may change in the near future. Users should be aware that updates may require adjustments to existing workflows. We recommend regularly checking for updates.*

## Dataset Overview

The benchmark covers three geographic regions: New Zealand (NZ), Europe (ALPS) and South Africa (SA) with ~10km target resolution. Training and test datasets (NetCDF files, approximately 5 GB per domain) are provided for each region, including common predictors (Z, U, V, T, Q at 850, 700, 500, and 300 hPa, as well as model orography) and predictands (daily temperature and precipitation). More information in [./data](/data)

<div align="center">
<img src="/images/CORDEX_ML-bench_domains.png" alt="NZ Domain" width="500"/>
</div>

## Experiments

The benchmark covers two experiments with different tests focusing on the standard (perfect prognosis ESD) and emulation climate downscaling approaches. For each region, data is obtained from a single RCM driven by two different GCMs, one used both for training and testing (denoted below as `same GCM`) and the other only used to test transferability (denoted as `different GCM`). 

For both experiments, training is based on perfect (upscaled) predictors from the RCM while test experiments explore both perfect and imperfect (from the driving GCM) predictors. Predictands (target for training) correspond to the RCM highres 10km temperature and precipitation output.

- **Experiment 1: _ESD Pseudo-Reality_**: A 20-year (1961–1980) training period in present climate conditions, designed to mimic the standard statistical climate downscaling approach and test extrapolation capabilities of the methods. 

| Test   | Test Period | Predictor type | Eval | 
|----------------|---------------|----------------|------|
| Test1: Perfect Cross-Validation | Historical (1981–2000) | Perfect (from RCM), same GCM | Error, Clim | 
| Test2: Imperfect Cross-Validation | Historical (1981–2000) | Imperfect (from GCM), same GCM | Error, Clim | 
| Test3: Perfect Extrapolation | 2041–2060 + 2080–2099 | Perfect (from RCM), same GCM | change signal for mid/final term | 
| Test4: Imperfect Extrapolation | 2041–2060 + 2080–2099 | Imperfect (from GCM), same GCM | change signal for mid/final term | 
| Test5: Perfect Extrapolation (GCM Transferability) | 2041–2060 + 2080–2099 | Perfect (from RCM), different GCM | change signal for mid/final term | 


- **Experiment 2: _Emulator Hist+Future_**: A 40-year (1961–1980 + 2080–2099) training period combining present and future climates, focused on testing interpolation and transferability of emulators.

| Test   | Test Period | Predictor type | Eval | 
|----------------|---------------|----------------|------|
| Test1: Perfect Cross-Validation | Historical (1981–2000) | Perfect (from RCM), same GCM | Error, Clim | 
| Test2: Imperfect Cross-Validation | Historical (1981–2000) | Imperfect (from GCM), same GCM | Error, Clim | 
| Test3: Perfect Interpolation  | 2041–2060  | Perfect (from RCM), same GCM | Change signal | 
| Test4: Imperfect Interpolation | 2041–2060  | Imperfect (from GCM), same GCM | Change signal | 
| Test5: Perfect Interpolation (GCM Transferability) | 2041–2060  | Perfect (from RCM), different GCM | Change signal | 
| Test6: Imperfect Interpolation (GCM Transferability) | 2041–2060  | Imperfect (from GCM), different GCM | Change signal | 


## Model Training and Evaluation

For each experiment, models must be trained for the two required target variables (temperature and precipition), jointly (multivariate) or individually. If possible, please train the models both with and without model orography as a covariate (this will correspond to two separate submissions; for more details, see “Contributing to the benchmark”). This will allow us to assess the importance of including such covariates in the models. Some examples of model training can be found in [./training](./training).

A comprehensive set of evaluation diagnostics and accompanying code for assessing ML-based downscaling methods within the CORDEX ML-Bench framework is provided in this repository ([./evaluation](/evaluation)). Similar diagnostics will be used for online benchmarking, so users can use therse as guidelines for developing their models.

## Contributing to the benchmark [PROVISIONAL]

Benchmarking allows users to benchmark the performance of their own models against well-established reference approaches. Contributing to the online benchmarking (with automatic evaluation and results publicly available in the benchmark table) requires model registration and uploading the test results for the different experiments. For more details on the registration and submission process see ([./submission](/submission)).

The table below summarizes the contributing models and provides links to their respective implementation repositories when available (this table is automatically updated with new submissions).

## Models contributing to the benchmark [PROVISIONAL]

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
