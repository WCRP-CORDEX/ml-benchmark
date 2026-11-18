# Evaluating a model

CORDEX-ML-Bench utilises the framework [`ml-evaluation`](https://github.com/c3s-cordex-ml/ml-evaluation)
to evaluate the models. The script `evaluation/evaluate.py` applies the `ml-eval benchmark` 
tool on every domain, training/test experiment and variable, and
summarises the results in one JSON file per experiment.

## Before you start

- **Reference data**: the benchmark data must be in `data/Bench-data/` (see
  [`data_download.ipynb`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/feature/ml-evaluation/data/data_download.ipynb)).
- **Environment**: a Pixi or Conda environment with `ml-evaluation` installed;
  see [Requirements](https://wcrp-cordex.github.io/ml-benchmark/introduction/#requirements).

All commands below are run from the root of the repository.

## Steps

**1. Add the predictions.** Copy or link them to `models/<emulator_id>/predictions/`,
following the structure described in [`Submit your model`](../submission/README.md):

```
models/<emulator_id>/predictions/
├── ALPS_Domain/
├── NZ_Domain/
└── SA_Domain/
    └── <TRAINING_EXPERIMENT>/<horizon>/<case>/Predictions_pr_tasmax_<GCM>_<period>.nc
```

**2. Revise the configuration file**. If you followed the steps in [Submit your model](https://wcrp-cordex.github.io/ml-benchmark/contributing/submission/),
the branch `submission/<emulator_id>` already contains the configuration file (`models/<emulator_id>/config.yaml`).
Otherwise, copy the template into the model folder:

```bash
cp ./evaluation/config_template.yaml ./models/<emulator_id>/config.yaml
```

Either case, check that the configuration file is correct. By default only `model` needs 
to change; it must match the folder name `<emulator_id>`. The other fields select the domains, the
training/test experiments and the variables to evaluate, and, for ensemble
models, the maximum number of members to score (`max_members`, 5 by default).

**4. Run the evaluation**:

```bash
python ./evaluation/evaluate.py --config-file ./models/<emulator_id>/config.yaml
```

Useful options:

| Option | Effect |
|---|---|
| `-o`, `--overwrite` | Recompute experiments that already have a summary JSON (skipped by default). |
| `--keep-domain-files` | Keep the per-domain metric files (deleted by default once the summary is written). |
| `--aggregate-only` | Do not run `ml-eval` on each domain; only rebuild the summaries from the kept domain files. |
| `--use-dask` | Run `ml-eval benchmark` with Dask, which reads the data in chunks; use it if an evaluation runs out of memory (off by default). |

**5. Check the evaluation.** Make sure that every experiment in the configuration has a summary JSON covering all its variables and domains:

```bash
python ./evaluation/validate.py ./models/<emulator_id>/config.yaml
```

Missing summaries, variables or domains are reported as errors (exit code 1); null
metric values are reported as warnings.

## Output

For each combination of training and test experiment in the configuration, the evaluation writes one
summary file:

```
models/<emulator_id>/evaluation/<TRAINING_EXPERIMENT>/<TEST_EXPERIMENT>/metrics_summary.json
```

This JSON file contains some metadata and, for each variable, the metrics both aggregated
across domains and domain by domain:

```json
{
  "model": "<emulator_id>",
  "train_experiment": "ESD_pseudo_reality",
  "test_experiment": "Perfect_cross_validation",
  "ensemble": {"n_members": 10, "members": [0, 2, 5, 6, 8]},
  "variables": {
    "tasmax": {
      "rmse": {
        "aggregation": "mean",
        "aggregate": 1.23,
        "domains": {"ALPS": 1.10, "NZ": 0.98, "SA": 1.55}
      },
      ...
    },
    "pr": { ... }
  }
}
```

- **`ensemble`**: number of members in the predictions (`n_members`), and the
  positions of the members used to compute the metrics (`members`).
- **`aggregate`**: combination of the domain values using the aggregation function in 
  `aggregation`.

### Metrics

| Metric | Variable | Description | Units |
|---|---|---|---|
| `clim_mean` | both | Bias of the climatological mean | K, mm/day |
| `iav` | tasmax | Bias of the InterAnnual Variability (std. dev. of annual means) | K |
| `lhd` | pr | Logarithmic Histogram Distance between the daily distributions | – |
| `pss` | tasmax | Perkins skill score: overlap of the daily distributions (0–1) | – |
| `ralsd` | both | Radially Averaged Log Spectral Distance: error in the spatial structure | dB² |
| `rmse` | both | Root mean square error of the daily values | K, mm/day |
| `rx1day` | pr | Bias in mean Rx1day (annual max. 1-day precipitation) | mm/day |
| `sdii` | pr | Bias of the Simple Daily Intensity Index (mean precipitation on wet days) | mm/day |
| `txx` | tasmax | Bias in mean TXx (annual max. of daily max. temperature) | K |

All metrics are errors (lower is better), except PSS (higher is better). Biases
are spatial means of the absolute bias.

>**Note**. The metric definitions come from [`ml-evaluation`](https://github.com/c3s-cordex-ml/ml-evaluation/tree/main/README.md#evaluation-framework).

### Ensemble models

If the predictions have a member dimension, `ml-eval benchmark` randomly selects 
a subset of up to `max_members` as set in the configuration file (5 by default). The selection 
uses a fixed seed, so repeated runs score the same members.  All the metrics use this same subset, 
aggregated as follows:

| Metrics | Member aggregation |
|---|---|
| `rmse`, `clim_mean` | Metric computed over the ensemble mean |
| `rx1day`, `txx`, `sdii` | Index per member; bias averaged over members, then absolute value and spatial mean |
| `lhd`, `pss` | Metric computed per member, and averaged over members |
| `ralsd`, `iav` | One random member of the subset |

With `max_members: 1`, the model is scored on a single random member. The
setting is ignored in deterministic models.
