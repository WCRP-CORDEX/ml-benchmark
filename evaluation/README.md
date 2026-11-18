# Evaluating a model

This folder holds the script that computes the benchmark metrics of a model:
`evaluate.py`. It runs [`ml-evaluation`](https://github.com/c3s-cordex-ml/ml-evaluation)
(`ml-eval benchmark`) on every domain, training/test experiment and variable, and
summarises the results in one JSON file per experiment.

## Before you start

- **Reference data**: the benchmark data must be in `data/Bench-data/` (see
  [`data/data_download.ipynb`](../data/data_download.ipynb)). The targets are read from
  `data/Bench-data/<DOMAIN>_domain/test/<horizon>/target/`.
- **Environment**: a Pixi or Conda environment with `ml-evaluation` installed;
  see [Requirements](../README.md#requirements) in the main README.

All commands below are run from the root of the repository.

## Steps

**1. Add the predictions.** Copy or link them to `models/<MODEL>/predictions/`,
following the structure described in [`submission/README.md`](../submission/README.md):

```
models/<MODEL>/predictions/
├── ALPS_Domain/
├── NZ_Domain/
└── SA_Domain/
    └── <TRAINING_EXPERIMENT>/<horizon>/<case>/Predictions_pr_tasmax_<GCM>_<period>.nc
```

**2. Copy the configuration template** into the model folder:

```bash
cp ./evaluation/config_template.yaml ./models/<MODEL>/config.yaml
```

**3. Edit the configuration.** By default only `model` needs to change; it must
match the folder name `<MODEL>`. The other fields select the domains, the
training/test experiments and the variables to evaluate, and, for ensemble
models, the maximum number of members to score (`max_members`, 5 by default; see
[Ensemble models](#ensemble-models)). They are documented at the top of the file.

**4. Run the evaluation**, with either option:

- Pixi:
  ```bash
  pixi run evaluate --config-file ./models/<MODEL>/config.yaml
  ```
- Another environment (e.g. Conda): activate it, then
  ```bash
  python ./evaluation/evaluate.py --config-file ./models/<MODEL>/config.yaml
  ```

Useful options:

| Option | Effect |
|---|---|
| `-o`, `--overwrite` | Recompute experiments that already have a summary (skipped by default). |
| `--keep-domain-files` | Keep the per-domain metric files (deleted by default once the summary is written). |
| `--aggregate-only` | Do not run `ml-eval` on each domain; only rebuild the summaries from the kept domain files. |
| `--use-dask` | Run `ml-eval benchmark` with Dask, which reads the data in chunks; use it if an evaluation runs out of memory (off by default). |

**5. Check the evaluation.** Make sure that every experiment in the configuration has a summary covering all its variables and domains:

```bash
python ./evaluation/validate.py ./models/<MODEL>      # or ./models/<MODEL>/config.yaml
```

Missing summaries, variables or domains are reported as errors (exit code 1); null
metric values are reported as warnings. Domains are missing, for instance, when
their prediction files were not found (`evaluate.py` skips them with a warning).

## Output

For each training/test experiment in the configuration, the evaluation writes one
summary file:

```
models/<MODEL>/evaluation/<TRAINING_EXPERIMENT>/<TEST_EXPERIMENT>/metrics_summary.json
```

It contains some metadata and, for each variable, every metric both aggregated
across domains and domain by domain:

```json
{
  "model": "<MODEL>",
  "train_experiment": "ESD_pseudo_reality",
  "test_experiment": "Perfect_cross_validation",
  "ensemble": {"n_members": 10, "members": [0, 2, 5, 6, 8]},
  "variables": {
    "tasmax": {
      "rmse": {
        "aggregation": "quadratic_mean",
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
  positions of the members used to compute the metrics (`members`). For a
  deterministic model, `n_members` is 1 and `members` is `null`. Both are `null`
  if the domains disagree (e.g. a different number of members per domain).
- **`aggregate`**: combination of the domain values, with the rule given in
  `aggregation`: quadratic mean for RMSE and LHD, arithmetic mean otherwise. A
  missing value (`null`) in any domain makes the aggregate `null`; it is never
  dropped.

### Metrics

| Metric | Variable | Description | Units |
|---|---|---|---|
| `rmse` | both | Root mean square error of the daily values | K, mm/day |
| `clim_mean` | both | Bias of the climatological mean | K, mm/day |
| `ralsd` | both | Radially Averaged Log Spectral Distance: error in the spatial structure | dB² |
| `txx` | tasmax | Bias in mean TXx (annual max. of daily max. temperature) | K |
| `rx1day` | pr | Bias in mean Rx1day (annual max. 1-day precipitation) | mm/day |
| `pss` | tasmax | Perkins skill score: overlap of the daily distributions (0–1) | – |
| `lhd` | pr | Logarithmic Histogram Distance between the daily distributions | – |
| `iav` | tasmax | Bias of the InterAnnual Variability (std. dev. of annual means) | K |
| `sdii` | pr | Bias of the Simple Daily Intensity Index (mean precipitation on wet days) | mm/day |

All metrics are errors (lower is better), except PSS (higher is better). Biases
are spatial means of the absolute bias.

### Ensemble models

If the predictions have a member dimension, `ml-eval benchmark` handles the
members as in the CORDEX ML-Bench paper. It randomly selects a subset of up to
`max_members` members (5 by default, set in the configuration file). The
selection uses a fixed seed, so repeated runs score the same members. All the
metrics use this same subset, aggregated as follows:

| Metrics | Member aggregation |
|---|---|
| RMSE, bias of the climatological mean | Metric of the ensemble mean of the subset |
| Rx1day/TXx bias, SDII bias | Index per member; bias averaged over members, then absolute value and spatial mean |
| LHD, PSS | Score per member, averaged over members |
| RALSD, IAV bias | One random member of the subset |

With `max_members: 1`, the model is scored on a single random member. The
setting is ignored for deterministic models.
The metric definitions come from [`ml-evaluation`](https://github.com/c3s-cordex-ml/ml-evaluation/tree/main);
[`diagnostics.md`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/evaluation/README.md) describes the original indices and
diagnostics they were derived from.
