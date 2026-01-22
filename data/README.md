## Dataset Overview

The CORDEX ML-Bench Dataset is publicly available at [Zenodo](https://zenodo.org/records/17957264) as a `zip` file per domain containing all the NetCDF files for the different experiments. The data is around 5 GB per domain. 

The notebook `./data_download.ipynb` explains how to download the data for the different domains and the notebook `./experiments.ipynb` provides a walkthrough of the dowloaded data, helping users understand data and how to configure train and test datasets for the different experiments forming the benchmark. We encourage users to carefully review this notebook to become familiar with the data.

The dataset spans three geographic regions, each defined over domains of identical size (i.e., the same number of grid boxes). For each domain, the dataset comprises data derived from a single Regional Climate Model (RCM) driven by two different Global Climate Models (GCMs). The first GCM is used for both training and testing, while the other is used exclusively for testing transferability. 

-  **New Zealand (NZ) – 0.11° resolution** <br>
RCM model: _CCAM-2203_ (from [CORDEX-CMIP6](https://wcrp-cordex.github.io/simulation-status/CORDEX_CMIP6_status.html#AUS-12)) <br>
Driving GCM model 1 (training and test): _ACCESS-CM2_r4i1p1f1 (historical and ssp370 scenarios)_ <br>
Driving GCM model 2 (test transferability): _EC-Earth3_r1i1p1f1 (historical and ssp370 scenarios)_ <br>
For this domain the predictors and predictand have been interpolated to regular lon/lat grids.
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_NZ.png" alt="NZ Domain" width="300"/>
</div>

- **South Africa (SA) – 0.10° resolution**  <br>
RCM model: **** <br>
Driving GCM model 1 (training and test): _ACCESS-CM2 (historical and *** scenarios)_ <br>
Driving GCM model 2 (test transferability): _NorESM2-MM (historical and *** scenarios)_  <br>
For this domain the predictors and predictand have been interpolated to regular lon/lat grids.
<div align="center">
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_SA.png" alt="SA Domain" width="300"/>
</div>
<br>

- **Europe (ALPS) – 0.11° resolution** <br>
RCM model: _Aladin63_ (from CORDEX-CMIP5) <br>
Driving GCM model 1 (training and test): _CNRM-CM5 (historical and rcp85 scenarios)_ <br>
Driving GCM model 2 (test transferability): _MPI-ESM-LR (historical and rcp85 scenarios)_ <br> 
For this domain, the predictors and predictand are provided on the _Lambert Conformal Conic_ projection as simulated by the Aladin63 RCM. This is transparent for the benchmark since the data is provided over the same 16x16 and 128x128 squared grids and can be used directly for model training without reprojection as in the other cases (see [./data](/data) for more details). When plotted on a regular lat–lon grid the domain appears curved (see the image below), but in their native projection the data lies on a square array (the `x` and `y` coordinates in the files represent the regular coordinates, whereas the `lat` and `lon` coordinates denotes the corresponding gepraphical latidue and longitue values, which are automatically used by standard ploting functions for plotting).  <br>
<div align="center">
  <img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_ALPS.png" alt="ALPS Domain" width="300"/>
</div>



The `training` dataset provides the information from the RCM needed to train the models, both the predictors and the targets (predictands, temperature and precipitation in this benchmark). Predictors included coarse large-scale (~200km) information for a set of common atmospheric variables (Z, U, V, T, Q,  at different height levels 850, 700, and 500 hPa) characterizing the 3D atmospheric state and are provided over a large squared domain (16x16) represented in the figures below displaying the wind fields (U,V). Predictands correspond to the RCM highres ~10km temperature and precipitation output over an inner square domain (128x128, represented in the figures below with the temperature fields). Highres model orogaphy (~10km) is also provided in the dataset and can be used as co-variate in the models. This information is provided for the two benchmark training experiments focusing on the standard (ESD) and emulation downscaling approaches, denoted `ESD Pseudo-Reality` and `Emulator Hist+Future`. The same models should be trained separately on both experiments. 

`Test` data includes both `perfect` (upscaled from the RCM, as in the training data) and `imperfect` (from the driving GCM) predictors over the same 16x16 domains. 

## Data Structure

Each domain follows a consistent file structure, with the same subdirectories for training and testing dat. Predictors and predictands are labelled with the name of the driving GCMs (note that the RCM model is the same for each domain), as illustrated in the scheme below for the New Zealand (NZ) domain.

```
Domain/
├── train/
│   ├── ESD_pseudo-reality/
│   │   ├── predictors/
│   │   │   ├── ACCESS-CM2_1961-1980.nc
│   │   │   └── static.nc
│   │   └── target/
│   │   │   └── pr_tasmax_ACCESS-CM2_1961-1980.nc
│   ├── Emulator_hist_future/
│   │   ├── predictors/
│   │   │   ├── ACCESS-CM2_1961-1980_2080-2099.nc
│   │   │   └── static.nc
│   │   └── target/
│   │   │   └── pr_tasmax_ACCESS-CM2_1961-1980_2080-2099.nc
└── test/
    ├── historical/
    │   ├── predictors/
    │   │   ├── perfect/
    │   │   │   ├── ACCESS-CM2_1981-2000.nc
    │   │   │   └── EC-Earth3_1981-2000.nc
    │   │   └── imperfect/
    │   │       ├── ACCESS-CM2_1981-2000.nc
    │   │       └── EC-Earth3_1981-2000.nc
    ├── mid_century/
    │   ├── predictors/
    │   │   ├── perfect/
    │   │   └── imperfect/
    └── end_century/
        └── predictors/
            ├── perfect/
            └── imperfect/
```

















