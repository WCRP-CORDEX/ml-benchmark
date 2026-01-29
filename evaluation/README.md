# Evaluation Diagnostics

This module provides the set of evaluation diagnostics recommended by CORDEX-ML-Bench for assessing ML-based regional climate downscaling. It supports seasonal filtering and exposes both single-variable indices and diagnostic tools that compare predictions against reference datasets.

## Indices

Indices quantify particular characteristics or properties of a single simulation `x`, for example, the mean, specific quantiles, or extremes. These indices are always calculated for an individual simulation and often form the foundation for various diagnostics.

| Index | Description | Key parameters |
|-------|-------------|----------------|
| `mean` | Mean of a variable over the time dimension. | `x`, `var` |
| `quantile` | Specified time quantile of a variable. | `x`, `var`, `q` |
| `su` | Mean annual count of days above a temperature threshold (summer days). | `x`, `var`, `season`, `threshold` |
| `txx` | Mean annual maximum of daily maximum temperature. | `x`, `var`, `season` |
| `txn` | Mean annual minimum of daily maximum temperature. | `x`, `var`, `season` |
| `rx1day` | Mean annual maximum 1-day precipitation. | `x`, `var`, `season` |
| `sdii` | Mean precipitation on wet days above a threshold. | `x`, `var`, `season`, `wet_threshold` |
| `cdd` | Mean annual maximum dry spell length below a precipitation threshold. | `x`, `var`, `season`, `dry_threshold` |
| `cwd` | Mean annual maximum wet spell length above a precipitation threshold. | `x`, `var`, `season`, `wet_threshold` |
| `lag1_corr` | Lag-1 autocorrelation of a variable along time. | `x`, `var`, `season` |
| `interannual_var` | Interannual variability (standard deviation of annual means). | `x`, `var`, `season` |

## Diagnostics

Diagnostics compare two simulations, `x0` and `x1`, which correspond to the ground-truth (reference) data and the model predictions, respectively. These diagnostics are designed to evaluate different aspects of model performance. For example, to assess how well a model captures warm extremes, one might use the bias of the TXx index. Some diagnostics rely on specific indices, while others, such as RMSE, PSD, or the Wasserstein distance, do not depend on them.

To simplify comparisons between models in the benchmark, every diagnostic should be reducible to a single summary value. For example, the RMSE can be expressed as its mean across all relevant dimensions, and the bias of a specific index can be reported as the spatial mean of its absolute values.

| Diagnostic | Description | Key parameters |
|------------|-------------|----------------|
| `rmse` | Root mean square error between x0 and x1. | `x0`, `x1`, `var`, `season`, `dim` |
| `psd` | Power spectral density with radial averaging for 2D spatial fields. | `x0`, `x1`, `var`, `season` |
| `wasserstein_distance` | Wasserstein distance between the distributions of x0 and x1. With `spatial=False`, returns a single scalar (all values flattened); with `spatial=True`, returns a field of distances at each grid point (computed over time). | `x0`, `x1`, `var`, `season`, `spatial` |
| `bias_index` | Absolute or relative bias for any index function. | `x0`, `x1`, `index_fn`, `season`, index-specific kwargs |
| `ratio_index` | Ratio (x1 / x0) for any index function. | `x0`, `x1`, `index_fn`, `season`, index-specific kwargs |
| `bias_multivariable_correlation` | Bias in correlations between two variables. | `x0`, `x1`, `var_x`, `var_y`, `season` |

## Examples

### Tasmax diagnostics

```python
import xarray as xr
import indices, diagnostics

x0 = xr.open_dataset("target_tasmax.nc")
x1 = xr.open_dataset("prediction_tasmax.nc")

# RMSE over time dimension
rmse = diagnostics.rmse(x0, x1, var="tasmax", dim="time")

# Bias of annual maximum temperature
bias_txx = diagnostics.bias_index(x0, x1, index_fn=indices.txx, var="tasmax")

# Bias of mean temperature
bias_mean = diagnostics.bias_index(x0, x1, index_fn=indices._mean, var="tasmax")

# Bias of summer days count
bias_summer_days = diagnostics.bias_index(x0, x1, index_fn=indices.su, var="tasmax")

# Power Spectral Density
psd_x0, psd_x1 = diagnostics.psd(x0, x1, var="tasmax")

# Wasserstein distance (single scalar over all space-time)
wd_scalar = diagnostics.wasserstein_distance(x0, x1, var="tasmax")

# Wasserstein distance at each grid point (over time)
wd_field = diagnostics.wasserstein_distance(x0, x1, var="tasmax", spatial=True)
```

### Precipitation diagnostics

```python
import xarray as xr
import indices, diagnostics

x0 = xr.open_dataset("target_pr.nc")
x1 = xr.open_dataset("prediction_pr.nc")

# RMSE over time dimension
rmse = diagnostics.rmse(x0, x1, var="pr", dim="time")

# Bias of SDII
bias_sdii = diagnostics.bias_index(x0, x1, index_fn=indices.sdii, var="pr")

# Bias of annual maximum 1-day precipitation
bias_rx1day = diagnostics.bias_index(x0, x1, index_fn=indices.rx1day, var="pr")

# Bias of maximum wet spell length
bias_cwd = diagnostics.bias_index(x0, x1, index_fn=indices.cwd, var="pr")

# Power Spectral Density
psd_x0, psd_x1 = diagnostics.psd(x0, x1, var="pr")

# Wasserstein distance between precipitation distributions
wd_pr = diagnostics.wasserstein_distance(x0, x1, var="pr", season="summer")
```

### Multivariate correlation bias

```python
import xarray as xr
import diagnostics

# Load joint x0 dataset with both variables
x0 = xr.open_dataset("target_joint.nc")
x1 = xr.open_dataset("pred_joint.nc")

# Compute correlation bias between temperature and precipitation
correlation_bias = diagnostics.bias_multivariable_correlation(
    x0,
    x1,
    var_x="tasmax",
    var_y="pr"
)
