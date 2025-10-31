from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr

from numpy import fft

def _filter_by_season(x: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season label.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to filter.
    season : str | None
        Season name (winter, summer, spring, autumn) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """
    if season is None:
        return x
    if season == "winter":
        return x.where(x["time.season"] == "DJF", drop=True)
    if season == "summer":
        return x.where(x["time.season"] == "JJA", drop=True)
    if season == "spring":
        return x.where(x["time.season"] == "MAM", drop=True)
    if season == "autumn":
        return x.where(x["time.season"] == "SON", drop=True)
    return x

def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.
    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

def bias_index( 
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    relative: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the bias between x0 and x1 for a supplied index function.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    relative : bool, default False
        If True, return the relative bias.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Absolute or relative bias of the selected index.
    """
    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    x0_index = index_fn(x0, **call_kwargs)
    x1_index = index_fn(x1, **call_kwargs)
    bias = x1_index - x0_index

    if relative:
        return bias / x0_index
    else:
        return bias

def ratio_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the ratio between x1 and x0 for a supplied index function.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Ratio of the selected index (x1 / x0).
    """
    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    x0_index = index_fn(x0, **call_kwargs)
    x1_index = index_fn(x1, **call_kwargs)
    ratio = x1_index / x0_index
    return ratio


def bias_multivariable_correlation(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var_x: str,
    var_y: str,
    season: str | None = None,
) -> xr.Dataset:
    """
    Compute correlations between two variables for x0 and x1 datasets
    and return the correlation bias.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset containing ``var_x`` and ``var_y``.
    x1 : xr.Dataset
        Predicted dataset containing ``var_x`` and ``var_y``.
    var_x : str
        Name of the first variable used in the correlation computation.
    var_y : str
        Name of the second variable used in the correlation computation.
    season : str | None, optional
        Season to filter before computing the correlations.

    Returns
    -------
    xr.Dataset
        Dataset containing the x0 correlation, x1 correlation, signed
        correlation difference, and the requested correlation bias.
    """

    x0_filtered = _filter_by_season(x0, season)
    x1_filtered = _filter_by_season(x1, season)

    x0_corr = xr.corr(x0_filtered[var_x], x0_filtered[var_y], dim='time')
    x1_corr = xr.corr(x1_filtered[var_x], x1_filtered[var_y], dim='time')

    correlation_bias = x1_corr - x0_corr

    return xr.Dataset(
        {
            "correlation_x0": x0_corr,
            "correlation_x1": x1_corr,
            "correlation_bias": correlation_bias,
        }
    )


def rmse( 
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    dim: str | None = None,
) -> xr.Dataset:
    """
    Compute the root mean square error between x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the RMSE.
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE. If None, computes over all dimensions.

    Returns
    -------
    xr.Dataset
        Root mean square error between x0 and x1.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[[var]]
    x1_da = x1[[var]]

    squared_diff = (x1_da - x0_da) ** 2

    if dim is None:
        mse = squared_diff.mean()
    else:
        mse = squared_diff.mean(dim=dim)

    rmse_result = np.sqrt(mse)
    
    return rmse_result

def psd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
):
    """
    Compute the power spectral density for x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the PSD.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for x0 and x1.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    x0_np = np.nan_to_num(x0_da.values, nan=0.0)
    x1_np = np.nan_to_num(x1_da.values, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2

    psd_x0_list = [_radial_average(p) for p in power_x0]
    psd_x1_list = [_radial_average(p) for p in power_x1]

    avg_psd_x0 = np.mean(psd_x0_list, axis=0)
    avg_psd_x1 = np.mean(psd_x1_list, axis=0)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=["wavenumber"], name="PSD_x0")
    psd_x1_da = xr.DataArray(avg_psd_x1, dims=["wavenumber"], name="PSD_x1")

    return psd_x0_da, psd_x1_da

def frequency_index_bias(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    thresholds: list[float],
    season: str | None = None,
) -> xr.Dataset:
    """
    Compute the Frequency Index Bias (FBI) for precipitation events above thresholds.

    For each threshold, computes the ratio between the number of events in x1
    (predicted) and x0 (ground truth) above that threshold. Also computes a
    single aggregate metric measuring the deviation from the perfect score (1.0).

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    thresholds : list[float]
        List of precipitation thresholds to evaluate.
    season : str | None, optional
        Season to filter before computing the FBI.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - ``frequency_bias``: Frequency ratio for each threshold.
        - ``frequency_bias_mad``: Mean Absolute Deviation from perfect score (1.0),
          measuring how far the FBI is from the perfect value on average.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    fbi_ratios = {}
    for threshold in thresholds:
        x0_count = (x0_da > threshold).sum().values
        x1_count = (x1_da > threshold).sum().values
        
        ratio = float(x1_count) / float(x0_count) if x0_count > 0 else np.nan
        fbi_ratios[f"threshold_{threshold}"] = ratio

    fbi_da = xr.DataArray(
        list(fbi_ratios.values()),
        dims=["threshold"], 
        coords={"threshold": thresholds},
        name="FBI",
    )

    fbi_values = np.array(list(fbi_ratios.values()))
    valid_mask = ~np.isnan(fbi_values)
    if np.any(valid_mask):
        mean_absolute_deviation = np.mean(np.abs(fbi_values[valid_mask] - 1.0))
    else:
        mean_absolute_deviation = np.nan

    fbi_mad_da = xr.DataArray(
        mean_absolute_deviation,
        name="frequency_bias_mad",
        attrs={"long_name": "Mean Absolute Deviation from perfect FBI score (1.0)"},
    )

    return xr.Dataset(
        {
            "frequency_bias": fbi_da,
            "frequency_bias_mad": fbi_mad_da,
        }
    )