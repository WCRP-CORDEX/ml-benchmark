# Contributing to the Benchmark

Contributing a model to the benchmark involves five steps:

1. **Prepare** your predictions and check them with the validation tool.
2. **Register** the submission by opening a GitHub issue.
3. **Approval**: the administrators review the issue and create a branch for your model.
4. **Evaluate** your predictions and send the results in a pull request.
5. **Publication**: once the pull request is merged, your model appears in the models table and the scoreboards in the website.

## 1. Preparing the Submission

Your submission must contain predictions for all predictors included in the `test` folder of the benchmark dataset (see [Data Structure](../data/README.md#data-structure) for details). The predictions must follow exactly the same directory structure as the `test` data. Specifically, your output directory must be organized as follows, where `<emulator_id>` is the identifier of your model (see [step 2](#2-registering-the-submission)):

```
<emulator_id>/
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

Each NetCDF file must include **both variables**: precipitation (`pr`) and maximum temperature (`tasmax`), stored as **separate variables within the same file**. We provide utilities, examples, and ready-to-use NetCDF templates in the `format_predictions` directory to help you format your outputs correctly. In addition, a script is provided to help you generate a properly formatted submission ([`submission.py`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/submission/submission.py)). This script shows how to compute predictions for all available test predictor files and organize the outputs according to the required submission directory structure. It uses the official templates in `format_predictions/templates/` to ensure that spatial coordinates, dimensions, and variable attributes are correctly defined in the resulting NetCDF files. We strongly recommend using or adapting this script to avoid formatting errors.

#### Special requirement for generative models

If you are submitting a generative or stochastic model, you must provide an ensemble of at least 5 members for each prediction. These ensemble members must be stored as an additional NetCDF dimension named `member`. Both `pr` and `tasmax` variables must include this `member` dimension.

### Validating the predictions

Before registering, check your predictions with the validation tool [`submission/validate.py`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/submission/validate.py). From the root of the repository, run:

```bash
python submission/validate.py <path>/<emulator_id>
```

The tool also accepts a `.zip` file of the directory. It lists the 72 required files and reports:

- **Errors**: a missing prediction file, a file that cannot be opened, a missing `pr` or `tasmax` variable, dimensions (names or order) or a spatial grid that differ from the templates, years of the period missing from the `time` axis, or fewer than 5 ensemble members.
- **Warnings**: files that do not match any test predictor (usually misnamed or misplaced), years outside the period, or missing `lat`/`lon` coordinates.

Your predictions are ready when the tool reports no errors. Run it with `--help` for further options, such as checking only some domains or experiments.

## 2. Registering the Submission

Open an issue in the benchmark repository using the template [_Submit ML-Benchmark Prediction_](https://github.com/WCRP-CORDEX/ml-benchmark/issues/new?template=ml_benchmark_submission.yaml) and complete it carefully: this information identifies and documents your model. Among other details, the form asks for:

- **Emulator identifier** (`emulator_id`): the name of your model, `{emulator name}-{version}`, with the suffix `-orog` if orography is used as a predictor. It becomes the name of your model's directory and branch, so use only letters, digits, `.`, `_` and `-`.
- **Dataset release**: whether the predictions were produced **before** (pre-release) or **after** (post-release) the public release of the test dataset. Pre-release submissions are evaluated by the administrators (see [step 4](#4-evaluating-the-submission)).

The issue is the main point of contact with the administrators: please monitor it in case we need clarification.

## 3. Approval

The administrators review the issue to verify that the submission meets the benchmark requirements (see [section 6](#6-important-rules-and-consistency-requirements)). Once approved, a branch named `submission/<emulator_id>` is created automatically in the benchmark repository. It contains two files for your model:

- `models/<emulator_id>/metadata.yaml`: the details from the issue, used to build the models table.
- `models/<emulator_id>/config.yaml`: the configuration file that you will use to evaluate your predictions.

A comment in the issue confirms the approval and repeats the instructions of the next steps.

## 4. Evaluating the Submission

**Pre-release submissions** are evaluated by the administrators, who will contact you in the issue to arrange the transfer of the predictions. You can skip the rest of this step.

**Post-release submissions** are evaluated by the contributors themselves, in a fork of the repository:

1. [Fork the repository](https://github.com/WCRP-CORDEX/ml-benchmark/fork) and get the branch of your submission:

   ```bash
   git clone https://github.com/<your-user>/ml-benchmark.git
   cd ml-benchmark
   git remote add upstream https://github.com/WCRP-CORDEX/ml-benchmark.git
   git fetch upstream submission/<emulator_id>
   git switch -c submission/<emulator_id> upstream/submission/<emulator_id>
   ```

2. Evaluate the model using the steps in [Evaluate your model](https://casadoj.github.io/ml-benchmark/contributing/evaluation/). The evaluation writes one `metrics_summary.json` file per training and test experiment in `models/<emulator_id>/evaluation/`.

3. Commit the evaluation and push it to your fork:

   ```bash
   git add models/<emulator_id>/evaluation/*/*/metrics_summary.json
   git commit -m "Add evaluation of <emulator_id>"
   git push origin submission/<emulator_id>
   ```

4. Open a pull request from the branch `submission/<emulator_id>` of your fork into the `main` branch of `WCRP-CORDEX/ml-benchmark`, and mention the issue (`#<issue number>`) in its description, so that both are linked.

## 5. Publication

The administrators review your pull request and may complete or correct the metadata. Once a maintainer has approved the pull request, both the models table and the scoreboards in the website are updated. The approval also closes your issue with the label _Processed submission_ and delets the branch `submission/<emulator_id>`.

## 6. Important rules and consistency requirements

The registration process is essential to ensure that every submission is clearly documented. Please pay special attention to the following points.

### Required content

- Each submission must include predictions for both precipitation and maximum temperature.
- Each submission must include predictions for the full set of predictors contained in the `test` folder.
- If orography is used as a covariate, this must be clearly stated in the model description, and explicitly indicated in the `emulator_id` with the suffix `-orog`, following the instructions provided in the submission issue template.

### Internal consistency of a submission

A single submission must be internally consistent across all domains and experiments, meaning it must use:

- The same overall architecture (e.g., U-Net, Transformer, diffusion model),
- The same training paradigm (e.g., standard training, adversarial training),

For univariate models, limited differences between variables are acceptable (for example, different loss functions or final activation layers for `pr` and `tasmax`). However, changing the architecture type or training paradigm between variables or domains within one submission is not allowed. If you wish to evaluate different approaches (e.g., GAN vs diffusion, U-Net vs Transformer), these must be submitted as separate submissions.
