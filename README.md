# CORDEX ML-Bench: A benchmark for Machine Learning-Based Regional Climate Downscaling

CORDEX ML-Bench is a benchmark designed to evaluate the performance of machine learning–based climate downscaling models across different regions covering both the standard (perfect prognosis ESD) and emulation climate downscaling approaches. It defines standardized training and test experiments assessing various downscaling challenges along with the corresponding datasets from Regional Climate Models (RCMs) driven by different Global Climate Models (GCMs). 

This repository contains all the materials, instructions, and datasets required to run the bechmark. It also incldues instructions for registering and submitting contributions that will undergo automatic evaluation and benchmarking with results publicly reported in this page. Example scripts and notebooks for training, inference and evaluation are included to facilitate the process. 

**Development Status Notice**  
*This repository is currently under active development. As a result, the structure, documentation, datasets and experimental protocols may change in the near future. Users should be aware that updates may require adjustments to existing workflows. We recommend regularly checking for updates.*

## Dataset Overview

The benchmark covers three geographic domains: New Zealand (NZ), Europe (ALPS) and South Africa (SA) with ~10km target resolution. For the different experiments, training and test datasets (NetCDF files, approximately 5 GB per domain) are provided for each region, including common predictors (Z, U, V, T, Q at 850, 700, 500, and 300 hPa) and predictands (daily temperature and precipitation) derived from different RCMs. More information in [./data](/data)

## Experiments

The benchmark includes two training modes and the corresponding test experiments focusing on the standard (ESD) and emulation climate downscaling approaches:

- **ESD Pseudo-Reality**: A 20-year (1961–1980) training period in present climate conditions, designed to mimic the standard statistical climate downscaling approach and test extrapolation capabilities of the methods. 
- **Emulator Hist+Future**: A 40-year (1961–1980 + 2081–2100) training period combining present and future climates, focused on testing transferability of emulators.

*Test experiments for the ESD pseudo-reality training mode*

| Training Setup | Test Period | Test Experiment | Notes | Eval | 
|----------------|---------------|----------------|-------|------|
| ESD “pseudo-reality”<br>Period: 1961–1980 | Historical (1981–2000) | Perfect Cross-Validation | Same GCM, perfect predictors | Error, Clim | 
|  | Historical (1981–2000) | Imperfect Cross-Validation | Same GCM, imperfect predictors | Error, Clim | 
|  | 2041–2060 + 2081–2100 | Perfect Extrapolation | Same GCM, perfectly | change signal | 
|  | 2041–2060 + 2081–2100 | Imperfect Extrapolation| Same GCM, imperfectly | change signal | 
|  | 2081–2100 | Perfect Extrapolation (Hard Transferability) | Different GCM, perfectly | change signal | 


*Test experiments for the Emulator training mode*

| Training Setup | Test Period | Test Experiment | Notes | Eval | 
|----------------|---------------|----------------|-------|------|
| Emulator hist + future<br>Period: 1961–1980 + 2081–2100 | Historical (1981–2000) | Perfect Cross-Validation | Same GCM, perfectly | Error, Clim | 
|  | Historical (1981–2000) | Imperfect Cross-Validation | Same GCM, imperfectly | Error, Clim | 
|  | Historical (1981–2000) | Perfect Cross-Validation (Hard Transferability) | Different GCM, perfectly | Error, Clim | 
|  | Historical (1981–2000) | Imperfect Cross-Validation (Hard Transferability) | Different GCM, imperfectly | Error, Clim | 
|  | 2041–2060 + 2081–2100 | Perfect Interpolation | Same GCM, perfectly | change signal | 
|  | 2041–2060 + 2081–2100 | Imperfect Interpolation | Same GCM, imperfectly | change signal | 
|  | 2041–2060 + 2081–2100 | Perfect Interpolation (Hard Transferability) | Different GCM, perfectly | change signal | 
|  | 2041–2060 + 2081–2100 | Imperfect Interpolation (Hard Transferability) | Different GCM, imperfectly | change signal | 


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

| Model              | RMSE (°C)  | MAE (°C)  | R²    | Training Time     | Inference Speed (samples/sec) |
|--------------------|------------|-----------|-------|-------------------|-------------------------------|
| Model1             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model2             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model3             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model4             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model5             | XXX        | XXX       | XXX   | XXX               | XXX                           |

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

The [./format_predictions](./format_predictions) directory provides utilities and templates to help users structure their model outputs in the required NetCDF format for CORDEX ML-Bench evaluation. It includes ready-to-use NetCDF templates.

Please note that this directory is currently intended for internal use and may be subject to modification prior to public release.

## Citation

Link to the pre-print

## Contact

Include an issue in this repository
