# CORDEX-ML-Bench plotting scripts

Full evaluation + publication-quality plot pipeline for CORDEX-ML-Bench.
One SLURM job produces **everything**: scorecards (per-domain + averaged),
rankings, daily-snapshot maps, historical climatologies, and climate-change
signal plots with model-mean distributions.

```
cordex_plots/
├── build_scorecard.py        # Score harvester + scorecard figures + rankings JSON
├── plot_daily_snapshot.py    # Single-day maps with PSD inset
├── plot_climatology.py       # Rx1day / TXx climatology with RMSE badges
├── plot_climate_change.py    # Future−historical signal with KDE of model-mean Δ
├── benchmark_utils.py        # Shared path / IO / colormap / PSD / RMSE helpers
├── model_rankings.json       # Written by build_scorecard.py (overwritten per run)
├── submit_plots.sh           # SLURM driver: scorecards → daily → clim → cc
└── README.md
```

## Pipeline overview

### Part 0 — `build_scorecard.py`

Walks `diagnostics/<MODEL>/<DOMAIN>_Domain/<experiment>/<variable>/<config>/metric_*.nc`,
extracts one absolute-mean scalar per file, and produces:

* **Eight scorecard figures** (PNG + PDF):
  - `scorecard_pr_average.png` — three-domain average for precipitation.
  - `scorecard_pr_ALPS.png`, `scorecard_pr_NZ.png`, `scorecard_pr_SA.png`.
  - Same four for `tasmax`.
* **`raw_scores.csv`** — every harvested `(model, domain, variable, experiment, config, metric, value)` scalar.
* **`model_rankings.json`** — overall ranks keyed by `(variable, experiment)`, consumed by the other three plot scripts.

Each scorecard panel mirrors the style in the reference figure you shared:
triangle-split cells (▲ CV, ▽ OOS), shaded by per-column **global rank**
(grey, darker = worse), and a blue rank column on the right (dark navy = rank 1).
Rows are grouped **Generative → Deterministic → Other**, with `+ orog`
variants rendered as tan indented siblings when they sit directly beneath
their non-orog base model.

Metric columns are per-variable:

| pr | tasmax |
|---|---|
| RMSE, Mean Bias, SDII, Rx1day, Lag-1, RALSD, PR-T | RMSE, Mean Bias, TXx, IAV, Lag-1, RALSD, PR-T |

### Part 1 — `plot_daily_snapshot.py`

6 small map panels (3 random top + 3 random bottom) plus a larger truth
panel, a shared colorbar, and a **radially-integrated power-spectrum
inset** on log-log axes (PSD ported from the WCRP-CORDEX reference
`evaluation/diagnostics.py`). Day selection:

- `--mode auto` : **wettest** day for pr, **hottest** for tasmax.
- `--mode wettest|hottest|random|date` for explicit control.

### Part 2 — `plot_climatology.py`

Same 6-panel layout, each prediction panel carries its **RMSE-vs-truth
badge** (black text on white pill). Rx1day and TXx = mean-over-years of
the annual maximum. pr uses BrBG; tasmax uses RdYlBu_r.

### Part 3 — `plot_climate_change.py`

Two-row figure:

1. **Maps** (6-panel + truth). Each prediction panel shows the Δ area-mean
   top-right and RMSE-vs-truth bottom-right.
2. **KDE subplot** of model-mean Δ signals — one value per ranked model
   (its area-mean Δ) smoothed with a Silverman-bandwidth Gaussian kernel.
   Ground-truth mean is marked as a thick black vertical line; each model
   is a dot along the baseline coloured by pool membership.

## Random sampling (consistent across all three plot scripts)

- **Top-3**: random 3 from the top-`--pool` (default 10) of the ranking.
- **Bottom-3**: random 3 from ranks `[N-bot_pool, N-bot_exclude)` — by
  default the bottom 15, skipping the 3 very-worst, so plots aren't
  dominated by catastrophic outliers.

Tunable: `--pool` `--bot-pool` `--bot-exclude` `--n-sample` `--seed`.

## Projections

- NZ, SA → regular lat/lon with `PlateCarree`.
- ALPS → rotated-pole curvilinear grid, drawn via
  `ax.pcolormesh(lon2d, lat2d, field, transform=PlateCarree())` using the
  2-D `lon(rlat, rlon)` / `lat(rlat, rlon)` coordinates stored in the file.

Coastlines (50 m Natural Earth) are drawn on every panel.

## Unit handling

- `tasmax` in Kelvin → automatically converted to °C.
- SA ground-truth `pr` is detected in kg m⁻² s⁻¹ and multiplied by 86 400.

## Running

### All-in-one

```bash
sbatch cordex_plots/submit_plots.sh
```

Environment-variable overrides:

```bash
DIAG_ROOT=/path/to/diagnostics \
TRUTH_ROOT=/path/to/target \
PRED_ROOT=/path/to/staging \
OUT_ROOT=/path/to/plots \
SEED=7 POOL=10 BOT_POOL=15 BOT_EXCLUDE=3 N_SAMPLE=3 DAILY_MODE=wettest \
sbatch submit_plots.sh
```

### Individual scripts

```bash
# Scorecards + rankings only
pixi run python build_scorecard.py \
    --diagnostics-root /path/to/diagnostics \
    --out-dir outputs/scorecard

# Daily snapshot
pixi run python plot_daily_snapshot.py \
    --domain NZ --gcm ACCESS-CM2 --variable pr \
    --experiment ESD_pseudo_reality --period historical \
    --truth-root /path/to/target --pred-root /path/to/staging \
    --out-dir outputs/daily --seed 42

# Climatology
pixi run python plot_climatology.py \
    --domain SA --gcm ACCESS-CM2 --variable tasmax \
    --experiment Emulator_hist_future --period historical \
    --truth-root ... --pred-root ... --out-dir outputs/climatology --seed 42

# Climate-change signal
pixi run python plot_climate_change.py \
    --domain ALPS --gcm CNRM-CM5 --variable tasmax \
    --experiment Emulator_hist_future --future-period end_century \
    --truth-root ... --pred-root ... --out-dir outputs/cc_signal --seed 42
```

## Paths expected on disk

```
<diagnostics-root>/               # from tree_diagnostics_updated.txt
└── <MODEL>/
    └── <DOMAIN>_Domain/          # capital _Domain
        └── <experiment>/
            └── <variable>/       # pr | tasmax
                └── <Config>/     # Perfect_cross_validation, Perfect_extrapolation,
                    │             # Perfect_interpolation, and imperfect variants
                    └── metric_*.nc

<truth-root>/
└── <DOMAIN>_domain/              # lowercase _domain
    └── test/<period>/target/pr_tasmax_<GCM>_<YEARS>.nc

<pred-root>/
└── <MODEL>/
    └── <DOMAIN>_Domain/
        └── <experiment>/<period>/<perfect|imperfect>/
            └── Predictions_pr_tasmax_<GCM>_<YEARS>.nc
```

`<period>` ∈ `historical` (1981-2000), `mid_century` (2041-2060),
`end_century` (2080-2099).
