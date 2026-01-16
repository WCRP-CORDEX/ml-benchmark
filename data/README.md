## Dataset Overview

The CORDEX ML-Bench Dataset is publicly available at [Zenodo](https://zenodo.org/records/17957264) as a `zip` file containing all the NetCDF files for the different experiments. The data is around 5 GB per domain. 

The notebook `./data_download.ipynb` explains how to download the data for the different domains and notebook `./experiments.ipynb` provides a walkthrough of the dowloaded data, helping users understand data and how to configure train and test datasets for the different experiments forming the benchmark. We encourage users to carefully review this notebook to become familiar with the data.

The dataset spans three geographic regions, each defined over domains of identical size (i.e., the same number of grid boxes in both predictor and predictand spaces, illustrated below by the wind and temperature fields, respectively). For each domain, the dataset comprises data derived from a single Regional Climate Model (RCM) driven by two different Global Climate Models (GCMs). The first GCM is used for both training and testing, while the other is used exclusively for testing transferability. 

-  **New Zealand (NZ) – 0.11° resolution** <br>
RCM model: _CCAM_ (from CORDEX-CMIP6) <br>
Driving GCM model 1 (training and test): _ACCESS-CM2 (historical and ssp370 scenarios)_ <br>
Driving GCM model 2 (test transferability): _EC-Earth3 (historical and ssp370 scenarios)_ <br>
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_NZ.png" alt="NZ Domain" width="300"/>
</div>

- **Europe (ALPS) – 0.11° resolution** <br>
RCM model: _Aladin63_ (from CORDEX-CMIP5) <br>
Driving GCM model 1 (training and test): _CNRM-CM5 (historical and rcp85 scenarios)_ <br>
Driving GCM model 2 (test transferability): _MPI-ESM-LR (historical and rcp85 scenarios)_ <br> <br>
For this domain, the RCM simulations for the predictand are defined on a _Lambert Conformal Conic_ projection. When plotted on a regular lat–lon grid the domain appears curved (see the image below), but in their native projection the data lie on a square array. Therefore, the fields can be used directly for model training without reprojection (see [./data](/data) for more details). <br>
<div align="center">
  <img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_ALPS.png" alt="ALPS Domain" width="300"/>
</div>


- **South Africa (SA) – 0.10° resolution**  <br>
RCM model: **** <br>
Driving GCM model 1 (training and test): _ACCESS-CM2 (historical and *** scenarios)_ <br>
Driving GCM model 2 (test transferability): _NorESM2-MM (historical and *** scenarios)_  <br>
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_SA.png" alt="SA Domain" width="300"/>
</div>

<br>
<br>

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

















