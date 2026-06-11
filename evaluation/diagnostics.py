from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
from tqdm import tqdm

from numpy import fft
from scipy.stats import wasserstein_distance_nd

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

def _histogram_counts(
    da: xr.DataArray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Compute histogram counts for a DataArray over the supplied bin edges.

    Parameters
    ----------
    da : xr.DataArray
        Input data array (any shape).
    bin_edges : np.ndarray
        Monotonically increasing array of bin edges.

    Returns
    -------
    np.ndarray
        One-dimensional array of counts with length ``len(bin_edges) - 1``.
    """
    arr = np.asarray(da.values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return np.histogram(arr, bins=bin_edges)[0].astype(np.float64)

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

def ralsd(psd_x0: xr.DataArray, psd_x1: xr.DataArray) -> float:
    """
    Compute the Radially Averaged Log Spectral Distance (RALSD).

    RALSD measures the distance between two radially averaged power spectra
    in log space (Harris et al., 2022). Lower values indicate better agreement
    between the spectra.

    Parameters
    ----------
    psd_x0 : xr.DataArray
        Power spectral density of the ground truth (as returned by psd function).
    psd_x1 : xr.DataArray
        Power spectral density of the prediction (as returned by psd function).

    Returns
    -------
    float
        Scalar RALSD value.
    """
    ratio = psd_x0.values / np.maximum(psd_x1.values, 1e-30)
    log_ratio = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    return float(np.mean(log_ratio ** 2))

def wasserstein_distance(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    spatial: bool = False,
) -> float | xr.Dataset:
    """
    Compute the Wasserstein distance between x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the Wasserstein distance.
    spatial : bool, default False
        If True, compute distance at each grid point across time dimension.
        If False, compute a single scalar distance by flattening both datasets.

    Returns
    -------
    float or xr.Dataset
        If spatial=False: single Wasserstein distance value.
        If spatial=True: Wasserstein distance field with same spatial dimensions as input.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    if not spatial:
        x0_flat = np.nan_to_num(x0_da.values, nan=0.0).flatten()
        x1_flat = np.nan_to_num(x1_da.values, nan=0.0).flatten()

        print('Computing Wasserstein distance...')
        distance = wasserstein_distance_nd(u_values=x0_flat, v_values=x1_flat)
        return distance

    else:
        x0_np = np.nan_to_num(x0_da.values, nan=0.0)
        x1_np = np.nan_to_num(x1_da.values, nan=0.0)

        time_dim = x0_np.shape[0]
        spatial_shape = x0_np.shape[1:]
        x0_reshaped = x0_np.reshape(time_dim, -1)
        x1_reshaped = x1_np.reshape(time_dim, -1)

        n_spatial = x0_reshaped.shape[1]
        distances = np.zeros(n_spatial)

        for i in tqdm(range(n_spatial), desc="Computing Wasserstein distances"):
            x0_temporal = x0_reshaped[:, i].flatten()
            x1_temporal = x1_reshaped[:, i].flatten()
            distances[i] = wasserstein_distance_nd(x0_temporal, x1_temporal)

        distance_field = distances.reshape(spatial_shape)

        wd_da = x0.isel(time=0).copy(deep=True)
        wd_da.attrs = {}
        wd_da[var].values = distance_field

        return wd_da

def perkins_skill_score(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
) -> float:
    """
    Compute the Perkins Skill Score between x0 and x1 datasets.

    The Perkins Skill Score (Perkins et al., 2007) measures the overlap
    between the probability distributions of two datasets. A score of 1
    indicates identical distributions, while 0 indicates no overlap.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the score.
    bins : int, default 100
        Number of histogram bins.
    value_range : tuple[float, float] | None, optional
        Explicit (min, max) range for the histogram. If None, the range
        is derived from the combined finite values of x0 and x1.

    Returns
    -------
    float
        Perkins Skill Score in the interval [0, 1].
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    if value_range is None:
        x0_vals = np.asarray(x0_da.values, dtype=np.float64).ravel()
        x1_vals = np.asarray(x1_da.values, dtype=np.float64).ravel()
        combined = np.concatenate([x0_vals[np.isfinite(x0_vals)],
                                   x1_vals[np.isfinite(x1_vals)]])
        lo, hi = float(combined.min()), float(combined.max())
    else:
        lo, hi = value_range

    bin_edges = np.linspace(lo, hi, bins + 1)

    x0_counts = _histogram_counts(x0_da, bin_edges)
    x1_counts = _histogram_counts(x1_da, bin_edges)

    x0_total = x0_counts.sum()
    x1_total = x1_counts.sum()
    x0_p = np.divide(x0_counts, x0_total,
                     out=np.zeros_like(x0_counts), where=x0_total > 0)
    x1_p = np.divide(x1_counts, x1_total,
                     out=np.zeros_like(x1_counts), where=x1_total > 0)

    return float(np.minimum(x0_p, x1_p).sum())

def lhd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
    min_count: int = 10,
) -> float:
    """
    Compute the Logarithmic Histogram Distance between x0 and x1 datasets.

    LHD quantifies the distance between two histograms in log-count space,
    considering only bins where both histograms have counts above
    ``min_count``. Lower values indicate better agreement.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the distance.
    bins : int, default 100
        Number of histogram bins.
    value_range : tuple[float, float] | None, optional
        Explicit (min, max) range for the histogram. If None, the range
        is derived from the combined finite values of x0 and x1.
    min_count : int, default 10
        Minimum count required in both histograms for a bin to be
        included in the computation.

    Returns
    -------
    float
        Scalar LHD value, or NaN if no bins satisfy the minimum count
        requirement.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    if value_range is None:
        x0_vals = np.asarray(x0_da.values, dtype=np.float64).ravel()
        x1_vals = np.asarray(x1_da.values, dtype=np.float64).ravel()
        combined = np.concatenate([x0_vals[np.isfinite(x0_vals)],
                                   x1_vals[np.isfinite(x1_vals)]])
        lo, hi = float(combined.min()), float(combined.max())
    else:
        lo, hi = value_range

    bin_edges = np.linspace(lo, hi, bins + 1)

    x0_counts = _histogram_counts(x0_da, bin_edges)
    x1_counts = _histogram_counts(x1_da, bin_edges)

    valid = (x0_counts > min_count) & (x1_counts > min_count)
    if not valid.any():
        return float("nan")

    return float(
        np.sqrt(np.mean(10.0 * np.log10(x0_counts[valid] / x1_counts[valid]) ** 2))
    )