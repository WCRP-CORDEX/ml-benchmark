## Dataset Overview

The CORDEX ML-Bench Dataset is publicly available at [Zenodo](https://zenodo.org/records/15797226) as a `zip` file containing all the NetCDF files for the different experiments. The data is around 5 GB per domain. 

The notebook `./data_download.ipynb` explains how to download the data (for the different domains included so far) and notebook `./experiments.ipynb` provides a walkthrough of the dowloaded data, helping users understand data and how to configure train and test datasets for the different experiments forming the benchmark. We encourage users to carefully review this notebook to become familiar with the benchmark.

The dataset spans three geographic regions, each defined over domains of identical size (i.e., the same number of grid boxes in both predictor and predictand spaces, illustrated below by the wind and temperature fields, respectively) and with a common data structure. For each domain, the dataset comprises data derived from a single Regional Climate Model (RCM) driven by two different Global Climate Models (GCMs). The first GCM is used for both training and testing, while the other is used exclusively for testing transferability. 

-  **New Zealand (NZ) – 0.11° resolution**  
RCM model: **** (from CORDEX-CMIP6) <br>
Driving GCM model 1 (training and test): *** <br>
Driving GCM model 2 (test transferability): **** <br>
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_NZ.png" alt="NZ Domain" width="300"/>
</div>

- **Europe (ALPS) – 0.11° resolution** 
RCM model: Aladin63 (from CORDEX-CMIP5) <br>
Driving GCM model 1 (training and test): CNRM_r1i1p1 (historical and rcp85 scenarios). <br>
Driving GCM model 2 (test transferability): **** <br>
<div align="center">
  <img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_ALPS.png" alt="ALPS Domain" width="300"/>
</div>

- **South Africa (SA) – 0.10° resolution** 
RCM model: **** <br>
Driving GCM model 1 (training and test): *** <br>
Driving GCM model 2 (test transferability): **** <br>
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_SA.png" alt="SA Domain" width="300"/>
</div>

The `training` dataset includes common large-scale (~150km) `predictors` (Z, U, V, T, Q at 850, 700, 500, and 300 hPa) as well as highres model orogaphy (~10km) which can be used as co-variate in the models; preditands (`target` for training) correspond to the RCM highres ~10km temperature and precipitation output. This information is provided for the two benchmark training experiments focusing on the standard (ESD) and emulation downscaling approaches, denoted `ESD Pseudo-Reality` and `Emulator Hist+Future`. 

`Test` data includes both `perfect` (upscaled from the RCM) and `imperfect` (from the driving GCM) predictors both with ~150km resolution. 

## Data Structure

Each domain follows a consistent file structure, with the same subdirectories for training and testing data

```
Domain/
├── train/
│   ├── ESD_pseudo-reality/
│   │   ├── predictors/
│   │   └── target/
│   ├── Emulator_hist_future/
│   │   ├── predictors/
│   │   └── target/
└── test/
    ├── historical/
    │   ├── predictors/
    │   │   ├── perfect/
    │   │   └── imperfect/
    ├── mid_century/
    │   ├── predictors/
    │   │   ├── perfect/
    │   │   └── imperfect/
    └── end_century/
        └── predictors/
            ├── perfect/
            └── imperfect/
```

















