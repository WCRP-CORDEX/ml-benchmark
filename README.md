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

## Contributing to the benchmark

Benchmarking allows users to benchmark the performance of their own models against well-established reference approaches. Contributing to the online benchmarking (with automatic evaluation and results publicly available in the benchmark table) requires model registration and uploading the test results for the different experiments. For more details on the registration and submission process see ([./submission](/submission)).

The table below summarizes the contributing models and provides links to their respective implementation repositories when available (this table is automatically updated with new submissions).

## Models contributing to the benchmark

| Model | Description | Reference | Implementation |
|-------|-------------|-----------|----------------|
| ViT-IFCAv1 | Vision Transformer (ViT) | [In preparation]() | [GitHub](https://github.com/jgonzalezab/ViT_CORDEX-ML-Bench) |
| Rossby-UNet | U-Net + CBAM (5 enc/dec layers) | [In preparation]() | [In preparation]() |
| DetUNet | U-Net + self-attention + FiLM (~3M params) | [Rampal et al., 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004668) | [GitHub](https://github.com/nram812/Generative-CORDEX-ML-Bench/tree/main/models) |
| CNRM-UNeT | Asymmetric U-Net + 1D bottleneck | [Doury et al., 2022](https://link.springer.com/article/10.1007/s00382-022-06343-9); [Doury et al., 2024](https://link.springer.com/article/10.1007/s00382-024-07350-8) | [GitHub](https://github.com/antoinedoury/RCM-Emulator) |
| Prithvi-UNet | Foundation model (Prithvi-WxC) + UNet decoder | [Schmude et al., 2024](https://arxiv.org/abs/2409.13598) | [GitHub](https://github.com/midatm1234/granite-wxc/tree/CORDEX_ML) |
| GNN4CD_Rall | Graph Neural Network (GATv2Conv, 10 layers) | [In Blasone et al., 2025](https://www.cambridge.org/core/journals/environmental-data-science/article/graph-neural-networks-for-hourly-precipitation-projections-at-the-convection-permitting-scale-with-a-novel-hybrid-imperfect-framework/97EEB267EA2AE5F9D87D50A9492264A7) | [GitHub](https://github.com/valebl/GNN4CD) |
| ParamUNET | U-Net outputting distribution parameters | [In preparation]() | [In preparation]() |
| DeepESD-IFCAv1 | Deep CNN (3 conv + 1 dense) | [González-Abad et al., 2025](https://journals.ametsoc.org/view/journals/aies/4/4/AIES-D-24-0121.1.xml) | [GitHub](https://github.com/jgonzalezab/deepESD_CORDEX-ML-Bench) |
| DeepESD-IDL | Deep CNN (3 conv + 1 dense) | [Soares et al., 2024](https://gmd.copernicus.org/articles/17/229/2024/) | [GitHub](https://github.com/rtomeidl/CordexML_DeepESD) |
| DeepSensor | Neural Process (parameterized GP) | [DeepSensor's Documentation](https://alan-turing-institute.github.io/deepsensor/) | [GitHub](https://github.com/alan-turing-institute/deepsensor) |
| ANN | Multi-layer perceptron (2 layers) | [Olmo and Bettolli, 2021](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7303) | [GitLab](https://gitlab.earth.bsc.es/molmo/ann_cordex_ml) |
| XGBoost | Gradient boosted decision trees | [Bushenkova et al., 2021](https://www.sciencedirect.com/science/article/pii/S2212095524001780) | [GitHub](https://github.com/rtomeidl/CordexML_XGBoost) |
| FlowMatching-v1 | Flow matching + ADM U-Net | [Wetherell 2026](https://arxiv.org/abs/2606.00281) | [GitHub](https://github.com/mo-tomaswetherell/mo-flow-cordexbench) |
| ResGAN | Residual WGAN (U-Net mean + GAN residual) | [Rampal et al., 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004668) | [GitHub](https://github.com/nram812/Generative-CORDEX-ML-Bench/tree/main/models) |
| SpaGAN | UNet2D GAN + convolutional discriminator | [Glawion et al., 2025](https://www.nature.com/articles/s41612-025-01103-y) | [GitHub](https://github.com/jpolz/ml-benchmark-spategan) |
| EnScale | Multi-step sparse localised downscaling | [Schillinger et al., 2025](https://arxiv.org/abs/2509.26258v2) | [GitHub](https://github.com/m-schillinger/cordexbench) |
| EnScale-linex | EnScale + linear residual model | [Schillinger et al., 2025](https://arxiv.org/abs/2509.26258v2) | [GitHub](https://github.com/m-schillinger/cordexbench) |
| UiBCorrDiff | Corrective diffusion (regression U-Net + EDM diffusion) | [Marani et al., 2025](https://www.nature.com/articles/s43247-025-02042-5) | [GitHub](https://github.com/joshdorrington/UiBcorrdiff) |
| CorrDiff-TW1 | Physics-inspired UNet regression + EDM diffusion | [Marani et al., 2025](https://www.nature.com/articles/s43247-025-02042-5) | [GitHub](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff) |
| ParamDiffusion | Diffusion on ParamUNET background | [In preparation]() | [In preparation]() |
| RCMGEM | NCSN++ diffusion (sub-VP SDE) | [Addison et al., 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005140) | [Zenodo](https://zenodo.org/records/18481925) |


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
