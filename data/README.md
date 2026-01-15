## Dataset Overview

The CORDEX ML-Bench dataset is publicly available at [Zenodo](https://zenodo.org/records/15797226) as a `zip` file containing all the NetCDF files for the different training and test experiments. The notebook `./data_download.ipynb` provides code for downloading this data (for any of the domains included so far). The data is around 5 GB per domain. After downloading, `./experiments.ipynb` provides a walkthrough of the data, helping users understand which data to use for training, which to use for evaluation, and what each dataset represents. We encourage users to carefully review this notebook to become familiar with the benchmark.

The CORDEX ML-Bench dataset spans three geographic domains:

<div align="center">

### New Zealand (NZ) – 0.11° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_NZ.png" alt="NZ Domain" width="300"/>
<br><br>

### Europe (ALPS) – 0.11° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_ALPS.png" alt="ALPS Domain" width="300"/>
<br><br>

### South Africa (SA) – 0.10° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_SA.png" alt="SA Domain" width="300"/>
<br><br>
<br><br>
</div>

Each region includes structured training and testing data derived from a particular Regional Climate Model (RCM) driven by different Global Climate Models (GCMs), allowing systematic evaluation across a range of experiments assessing different downscaling challenges.

The dataset provides two core training experiments:

- **ESD Pseudo-Reality**: A 20-year(1961–1980) historical training period using a single GCM, designed to mimic the training of standard Empirical Statistical Downscaling (ESD). 
- **Emulator Hist+Future**: A more comprehensive 40-year (1961–1980 + 2081–2100) training period combining historical and future climates, focused on training RCM downscaling emulators.
  
Predictors and target (predictand) for these experiments correspond to upscaled (large-scale) predictors and high-resolution predictands from the RCM, following the perfect training approach (Rampal et al. 2024).

The test dataset are common for both training setups and enable evaluation across multiple test experiments, based on different combinations of periods and perfect (from RCMs) and imperfect (from the driving GCM) predictors:

- **Historical (1981–2000)**: For cross-validation test experiments, using both perfect and imperfect predictors.
- **Mid-century (2041–2060) and End-century (2081–2100)**: For interpolation and extrapolation test experiments, using both perfect and imperfect predictors.

## Data Structure

Each domain follows a consistent file structure, with subdirectories for training and testing data, and further divisions by period, GCM, and evaluation type. Predictors include both dynamic variables (e.g., temperature, precipitation) and optional static fields (e.g., topography).

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

















