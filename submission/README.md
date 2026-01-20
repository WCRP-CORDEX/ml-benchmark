# Contributing to the Benchmark

Contributing to the benchmark involves two main steps:

1. **Preparing your submission** (generating and formatting predictions)  
2. **Registering your submission** (providing model details)

## 1. Preparing the Submission

Your submission must contain predictions for all predictors included in the `test` folder of the benchmark dataset (see [./data](/data) for details). The predictions must follow exactly the same directory structure as the `test` data. Specifically, your output directory must be organized as follows:

```
<submission_name>/
├── ALPS_Domain/
│   ├── Emulator_hist_future/
│   │   ├── historical/
│   │   │   ├── perfect/
│   │   │   │   ├── Predictions_pr_tasmax_CNRM-CM5_1981-2000.nc
│   │   │   │   └── Predictions_pr_tasmax_MPI-ESM-LR_1981-2000.nc
│   │   │   └── imperfect/
│   │   │       └── ...
│   │   ├── mid_century/
│   │   │   ├── perfect/
│   │   │   └── imperfect/
│   │   └── end_century/
│   │       ├── perfect/
│   │       └── imperfect/
│   ├── ESD_pseudo_reality/
│   │   └── ...
├── NZ_Domain/
│   └── ...
└── SA_Domain/
    └── ...
```

All prediction files must be named using the following convention:

```
Predictions_pr_tasmax_<GCM>_<period>.nc
```

The `<GCM>` and `<period>` values must match those in the corresponding predictor NetCDF file.

Each NetCDF file must include **both variables**: precipitation (`pr`) and maximum temperature (`tasmax`), stored as **separate variables within the same file**. We provide utilities, examples, and ready-to-use NetCDF templates in the `./format_predictions` directory to help you format your outputs correctly. In addition, a script is provided to help you generate a properly formatted submission (`submission.py`). It shows how to compute predictions for all available test predictor files and organize the outputs according to the required submission directory structure.  It uses the official templates in `format_predictions/templates/` to ensure that spatial coordinates, dimensions, and variable attributes are correctly defined in the resulting NetCDF files. We strongly recommend using or adapting this script to avoid formatting errors.

#### Special requirement for generative models

If you are submitting a generative or stochastic model, you must provide an ensemble of 10 members for each prediction. These ensemble members must be stored as an additional NetCDF dimension named `member`. The `member` dimension must have size 10. Both `pr` and `tasmax` variables must include this `member` dimension.

### Final submission package

Once all predictions are generated:

- Verify that all domains and all predictors are included.
- Compress the top-level directory (`<submission_name>/`) into a single `.zip` file.

This zip file is your official submission.

## 2. Registering the Submission

After your zip file is ready, you must register your submission so we can document your model and grant you access to upload your results for online evaluation.

### Registration procedure

1. Open an issue in the benchmark repository using the template [_Submit ML-Benchmark Prediction_](https://github.com/WCRP-CORDEX/ml-benchmark/issues/new?template=ml_benchmark_submission.yaml)
2. Complete the issue template carefully. This information is essential to properly identify, document, and validate your submission.
3. We will review your submission description to verify that it meets the benchmark requirements.
4. If accepted, we will send you a private form requesting the information needed to grant access to the evaluation cluster.
5. Once access is granted, you will be able to upload your zip file to our system for official evaluation. Detailed upload instructions will be provided at that stage.

The GitHub issue you open will serve as the main point of contact. Please monitor it in case we need clarification. Once everything is in order, we will reply with a link to the access form.

## 3. Important rules and consistency requirements

The registration process is essential to ensure that every submission is clearly documented. Please pay special attention to the following points.

### Required content

- Each submission must include predictions for both precipitation and maximum temperature.
- Each submission must include predictions for the full set of predictors contained in the `test` folder.
- If orography is used as a covariate, this must be clearly stated in the model description, and explicitly indicated in the name of the zip file, following the instructions provided in the submission issue template.

### Internal consistency of a submission

A single submission must be internally consistent across all domains and experiments, meaning it must use:

- The same overall architecture (e.g., U-Net, Transformer, diffusion model),
- The same training paradigm (e.g., standard training, adversarial training),

For univariate models, limited differences between variables are acceptable (for example, different loss functions or final activation layers for `pr` and `tasmax`). However, changing the architecture type or training paradigm between variables or domains within one submission is not allowed. If you wish to evaluate different approaches (e.g., GAN vs diffusion, U-Net vs Transformer), these must be submitted as separate submissions.
