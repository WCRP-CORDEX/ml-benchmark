# Evaluation Diagnostics

This module provides the set of evaluation diagnostics recommended by CORDEX-ML-Bench for assessing ML-based regional climate downscaling. It supports seasonal filtering and exposes both single-variable climate indices and diagnostic tools that compare predictions against reference datasets.

## Indices

| Index | Description | Key parameters |
|-------|-------------|----------------|
| `su` | Mean annual count of days above a temperature threshold (summer days). | `var`, `season`, `threshold` |
| `txx` | Mean annual maximum of daily maximum temperature. | `var`, `season` |
| `txn` | Mean annual minimum of daily maximum temperature. | `var`, `season` |
| `rx1day` | Mean annual maximum 1-day precipitation. | `var`, `season` |
| `sdii` | Mean precipitation on wet days above a threshold. | `var`, `season`, `wet_threshold` |
| `cdd` | Mean annual maximum dry spell length below a precipitation threshold. | `var`, `season`, `dry_threshold` |
| `cwd` | Mean annual maximum wet spell length above a precipitation threshold. | `var`, `season`, `wet_threshold` |
| `lag1_corr` | Lag-1 autocorrelation of a variable along time. | `var`, `season` |
| `interannual_var` | Interannual variability (standard deviation of annual means). | `var`, `season` |

## Diagnostics

| Diagnostic | Description | Key parameters |
|------------|-------------|----------------|
| `rmse` | Root mean square error between prediction and target. | `var`, `season`, `dim` |
| `psd` | Power spectral density with radial averaging for 2D spatial fields. | `var`, `season` |
| `bias_index` | Absolute or relative bias for any index function. | `index_fn`, `season`, index-specific kwargs |
| `ratio_index` | Ratio (prediction / target) for any index function. | `index_fn`, `season`, index-specific kwargs |
| `bias_multivariable_correlation` | Bias in correlations between two variables. | `var_x`, `var_y`, `season` |

## Examples

### Tasmax diagnostics

```python
import xarray as xr
import indices, diagnostics

target = xr.open_dataset("target_tasmax.nc")
pred = xr.open_dataset("prediction_tasmax.nc")

# RMSE over time dimension
rmse = diagnostics.rmse(target, pred, var="tasmax", dim="time")

# Bias of annual maximum temperature
bias_txx = diagnostics.bias_index(target, pred, index_fn=indices.txx, var="tasmax")

# Bias of mean temperature
bias_mean = diagnostics.bias_index(target, pred, index_fn=indices._mean, var="tasmax")

# Bias of summer days count
bias_summer_days = diagnostics.bias_index(target, pred, index_fn=indices.su, var="tasmax")

# Power Spectral Density
psd_target, psd_pred = diagnostics.psd(target, pred, var="tasmax")
```

### Precipitation diagnostics

```python
import xarray as xr
import indices, diagnostics

target = xr.open_dataset("target_pr.nc")
pred = xr.open_dataset("prediction_pr.nc")

# RMSE over time dimension
rmse = diagnostics.rmse(target, pred, var="pr", dim="time")

# Bias of SDII
bias_sdii = diagnostics.bias_index(target, pred, index_fn=indices.sdii, var="pr")

# Bias of annual maximum 1-day precipitation
bias_rx1day = diagnostics.bias_index(target, pred, index_fn=indices.rx1day, var="pr")

# Bias of maximum wet spell length
bias_cwd = diagnostics.bias_index(target, pred, index_fn=indices.cwd, var="pr")

# Power Spectral Density
psd_target, psd_pred = diagnostics.psd(target, pred, var="pr")
```

### Multivariate correlation bias

```python
import xarray as xr
import diagnostics

# Load joint target dataset with both variables
target = xr.open_dataset("target_joint.nc")
pred = xr.open_dataset("pred_joint.nc")

# Compute correlation bias between temperature and precipitation
correlation_bias = diagnostics.bias_multivariable_correlation(
    target,
    pred,
    var_x="tasmax",
    var_y="pr"
)
