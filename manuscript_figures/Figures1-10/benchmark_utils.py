"""
benchmark_utils.py  —  shared helpers for the CORDEX-ML-Bench plot suite.

What's new in this version
--------------------------
* Per-variable, per-experiment rankings (see model_rankings.json).
* ``sample_top_bottom`` — draws N models from the top-POOL and bottom-POOL
  at random, so every run shows a different mix of models.
* ``find_extreme_day`` — returns the date of the wettest / hottest day in
  the ground-truth series (used by the daily-snapshot script).
* Ground-truth precipitation for the SA domain is multiplied by 86 400 on
  load (it's stored in kg m⁻² s⁻¹); predictions are left alone.
* ``radial_psd`` — radially-integrated 2-D power spectrum, for the PSD
  inset on the daily plots.
* ``rmse`` — field-wise RMSE, for the RMSE annotations on the climatology
  and climate-change plots.
* ``get_crs`` — returns (map CRS, data CRS) tuples for cartopy; ALPS is
  plotted with a RotatedPole data transform, NZ/SA are PlateCarree.
* ``plot_map`` — geo-aware panel drawer with coastlines/borders.
* ``setup_nature_style`` — clean sans-serif rcParams for publication output.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------- #
# Domain / GCM / period registry
# --------------------------------------------------------------------------- #
DOMAIN_GCMS: dict[str, list[str]] = {
    "ALPS": ["CNRM-CM5", "MPI-ESM-LR"],
    "NZ":   ["ACCESS-CM2", "EC-Earth3"],
    "SA":   ["ACCESS-CM2", "NorESM2-MM"],
}
PERIOD_YEARS: dict[str, str] = {
    "historical":  "1981-2000",
    "mid_century": "2041-2060",
    "end_century": "2080-2099",
}
EXPERIMENTS = ("ESD_pseudo_reality", "Emulator_hist_future")


# --------------------------------------------------------------------------- #
# Colormaps (perceptual, print-friendly)
# --------------------------------------------------------------------------- #
_PRECIP_STOPS = [
    "#ffffff", "#e6f7f7", "#b7e2e2", "#7dc9c9",
    "#3aa6b9", "#2d7ca3", "#234b82", "#1b1b5a", "#3d1b66",
]
PRECIP_CMAP = LinearSegmentedColormap.from_list("precip_blues", _PRECIP_STOPS, N=256)

# A vivid, high-dynamic-range scale for daily precipitation snapshots.
# Goes white → very-pale teal → teal → blue → indigo → violet → magenta →
# fiery red → yellow-white, so the hottest pixels really pop. Intended for
# vmax clipped at the wettest-day 99.9th percentile so extreme convective
# cells stand out against a drizzle-dominated background.
_PRECIP_WET_STOPS = [
    "#ffffff",   # 0 mm — pure white
    "#edfbfb",   # drizzle
    "#c6ebeb",
    "#84d4d4",
    "#3cb6c4",
    "#2c87b4",   # steady rain
    "#2a57a0",
    "#2a2e87",
    "#3a1e72",
    "#711e68",
    "#a8195a",   # heavy rain
    "#d31f3a",
    "#ee5a1c",
    "#fbb03b",
    "#fff59e",   # extreme: near-white-yellow
]
PRECIP_WET_CMAP = LinearSegmentedColormap.from_list(
    "precip_wet", _PRECIP_WET_STOPS, N=512,
)

# BrBG-style diverging map for climatologies and climate-change signals in pr.
_BRBG_STOPS = [
    "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
    "#f5f5f5",
    "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
]
BRBG_CMAP = LinearSegmentedColormap.from_list("brbg_custom", _BRBG_STOPS, N=256)

# YlOrRd for temperature change signals (1-6 K feels like home on this scale).
_YLORRD_STOPS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
]
YLORRD_CMAP = LinearSegmentedColormap.from_list("ylorrd_custom", _YLORRD_STOPS, N=256)

_TEMP_STOPS = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]
TEMP_CMAP = LinearSegmentedColormap.from_list("temp_rdylbu", _TEMP_STOPS, N=256)

_PR_DIFF_STOPS = [
    "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
    "#ffffff",
    "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
]
PR_DIFF_CMAP = LinearSegmentedColormap.from_list("pr_brown_teal", _PR_DIFF_STOPS, N=256)

_T_DIFF_STOPS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#ffffff",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b",
]
T_DIFF_CMAP = LinearSegmentedColormap.from_list("t_rdbu", _T_DIFF_STOPS, N=256)


def cmap_for(variable: str, *, kind: str = "daily") -> "LinearSegmentedColormap":
    """Select a colormap that matches the context.

    kind ∈ {"daily", "climatology", "cc_signal"}.

    * daily        → precip: rich white-to-magenta PRECIP_WET_CMAP (wide
                     dynamic range); tasmax: RdYlBu_r-style TEMP_CMAP.
    * climatology  → precip: BrBG (brown-to-green); tasmax: TEMP_CMAP.
    * cc_signal    → precip: BrBG diverging (brown dry → green wet);
                     tasmax: YlOrRd sequential (0 → +6 K).
    """
    if variable == "pr":
        if kind == "cc_signal":
            return BRBG_CMAP
        if kind == "climatology":
            return BRBG_CMAP
        return PRECIP_WET_CMAP
    # tasmax
    if kind == "cc_signal":
        return YLORRD_CMAP
    return TEMP_CMAP


def pretty_var(variable: str) -> str:
    return {"pr": "Precipitation", "tasmax": "Max temperature"}[variable]


def units_for(variable: str, diverging: bool = False) -> str:
    if variable == "pr":
        return "mm day$^{-1}$"
    return "K" if diverging else "°C"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    truth_root: Path
    pred_root: Path

    def truth_file(self, domain: str, period_key: str, gcm: str) -> Path:
        years = PERIOD_YEARS[period_key]
        return (self.truth_root / f"{domain}_domain" / "test" / period_key
                / "target" / f"pr_tasmax_{gcm}_{years}.nc")

    def pred_file(self, model: str, domain: str, experiment: str,
                  period_key: str, gcm: str,
                  predictor_flavour: str = "perfect") -> Path:
        years = PERIOD_YEARS[period_key]
        return (self.pred_root / model / f"{domain}_Domain" / experiment
                / period_key / predictor_flavour
                / f"Predictions_pr_tasmax_{gcm}_{years}.nc")


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #
def load_rankings(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _strip_comments(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def get_ranking(rankings: dict, variable: str, experiment: str) -> list[str]:
    """Per-variable nested format preferred; falls back to the old flat
    ``{experiment: [...]}`` format for backward compatibility."""
    r = _strip_comments(rankings)
    if variable in r and isinstance(r[variable], dict):
        per_var = _strip_comments(r[variable])
        if experiment not in per_var:
            raise KeyError(f"Experiment {experiment!r} missing under rankings[{variable!r}]")
        return per_var[experiment]
    if experiment in r and isinstance(r[experiment], list):
        return r[experiment]
    raise KeyError(f"No ranking found for variable={variable!r}, experiment={experiment!r}")


def sample_top_bottom(
    ranking: list[str],
    available: Iterable[str],
    *,
    n_sample: int = 3,
    pool: int = 10,
    seed: int | None = None,
    bot_pool: int | None = None,
    bot_exclude: int = 0,
) -> tuple[list[str], list[str]]:
    """Random sample of ``n_sample`` models from the top slice and the
    bottom slice of the ranking, restricted to models whose files actually
    exist on disk.

    Parameters
    ----------
    pool : int
        Size of the **top** pool (first ``pool`` ranked models on disk).
    bot_pool : int, optional
        Size of the **bottom** pool. Defaults to ``pool``.
    bot_exclude : int, default 0
        Number of very-worst models to drop before sampling the bottom.
        With ``bot_pool=15, bot_exclude=3`` we sample from ranks
        ``[N-15, N-3)`` — the "bad but not catastrophic" band — so the
        resulting plots stay readable instead of being dominated by a
        single pathologically bad model.
    """
    if bot_pool is None:
        bot_pool = pool
    avail = set(available)
    present = [m for m in ranking if m in avail]
    if len(present) < 2 * n_sample + bot_exclude:
        raise RuntimeError(
            f"Only {len(present)} ranked models have files on disk; "
            f"need at least {2 * n_sample + bot_exclude}."
        )

    # Top pool: first `pool` ranked-and-present models.
    eff_top_pool = min(pool, len(present) - n_sample)
    eff_top_pool = max(eff_top_pool, n_sample)

    # Bottom pool: the slice [-bot_pool+bot_exclude, -bot_exclude) of present,
    # i.e. "worst bot_pool models, except drop the bot_exclude worst".
    eff_bot_pool = min(bot_pool, len(present) - n_sample)
    eff_bot_pool = max(eff_bot_pool, n_sample + bot_exclude)
    if bot_exclude > 0:
        bot_pool_slice = present[-eff_bot_pool:-bot_exclude]
    else:
        bot_pool_slice = present[-eff_bot_pool:]
    # Make sure nothing in bottom pool overlaps the top pool
    top_pool_slice = [m for m in present[:eff_top_pool]]
    bot_pool_slice = [m for m in bot_pool_slice if m not in top_pool_slice]
    if len(bot_pool_slice) < n_sample:
        # fallback: just drop the exclusion
        bot_pool_slice = [m for m in present[-eff_bot_pool:] if m not in top_pool_slice]

    rng = np.random.default_rng(seed)
    top_idx = rng.choice(len(top_pool_slice), size=n_sample, replace=False)
    bot_idx = rng.choice(len(bot_pool_slice), size=n_sample, replace=False)
    top = [top_pool_slice[i] for i in sorted(top_idx)]    # keep rank order within sample
    bot = [bot_pool_slice[i] for i in sorted(bot_idx)]
    return top, bot


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def discover_available(
    paths: Paths,
    ranking: list[str],
    *,
    domain: str,
    experiment: str,
    period_key,
    gcm: str,
    predictor_flavour: str = "perfect",
    verbose: bool = True,
) -> list[str]:
    periods = [period_key] if isinstance(period_key, str) else list(period_key)
    available, checked = [], []
    for m in ranking:
        missing = None
        for pk in periods:
            f = paths.pred_file(m, domain, experiment, pk, gcm, predictor_flavour)
            if not f.exists():
                missing = f
                break
        if missing is None:
            available.append(m)
        else:
            checked.append((m, missing))

    if verbose and not available:
        print("\n[discover] No prediction files found. Diagnostics:")
        print(f"[discover]   pred_root = {paths.pred_root}")
        print(f"[discover]   pred_root exists?  {paths.pred_root.exists()}")
        print(f"[discover]   domain={domain!r}  experiment={experiment!r}  "
              f"periods={periods}  gcm={gcm!r}  flavour={predictor_flavour!r}")
        print("[discover]   Sample paths that were checked and not found:")
        for m, p in checked[:3]:
            print(f"[discover]     {m:<25}  {p}")
            cur = p.parent
            while not cur.exists() and cur != cur.parent:
                cur = cur.parent
            if cur.exists():
                print(f"[discover]        ↳ first existing parent: {cur}")
                try:
                    siblings = sorted(x.name for x in cur.iterdir())[:8]
                    print(f"[discover]          contents (up to 8): {siblings}")
                except PermissionError:
                    pass
        print("[discover]")
    return available


# --------------------------------------------------------------------------- #
# Unit normalisation
# --------------------------------------------------------------------------- #
def _normalise(da: xr.DataArray, variable: str, *,
               domain: str | None = None, is_truth: bool = False) -> xr.DataArray:
    """Fix the two known quirks:
       * tasmax in Kelvin → Celsius
       * SA ground-truth pr in kg m-2 s-1 → mm day-1 (predictions are OK)
    """
    if variable == "tasmax":
        try:
            mx = float(da.max(skipna=True))
            if mx > 150:
                da = da - 273.15
                da.attrs["units"] = "degC"
        except ValueError:
            pass
    elif variable == "pr":
        if is_truth and domain == "SA":
            try:
                mx = float(da.max(skipna=True))
                if mx < 1.0:
                    da = da * 86400.0
                    da.attrs["units"] = "mm day-1"
            except ValueError:
                pass
    return da


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def open_truth(paths: Paths, domain: str, period_key: str, gcm: str,
               variable: str) -> tuple[xr.DataArray, xr.Dataset]:
    ds = xr.open_dataset(paths.truth_file(domain, period_key, gcm))
    da = _normalise(ds[variable], variable, domain=domain, is_truth=True)
    return da, ds


def open_prediction(paths: Paths, model: str, domain: str, experiment: str,
                    period_key: str, gcm: str, variable: str,
                    predictor_flavour: str = "perfect") -> xr.DataArray:
    f = paths.pred_file(model, domain, experiment, period_key, gcm, predictor_flavour)
    ds = xr.open_dataset(f)
    return _normalise(ds[variable], variable, domain=domain, is_truth=False)


def ensemble_mean(da: xr.DataArray) -> xr.DataArray:
    for name in ("member", "members", "realization", "ensemble"):
        if name in da.dims:
            return da.mean(name, keep_attrs=True)
    return da


# --------------------------------------------------------------------------- #
# Climate indices
# --------------------------------------------------------------------------- #
def rx1day(pr: xr.DataArray) -> xr.DataArray:
    
    annual_max = pr.groupby("time.year").max("time")
    clim = annual_max.mean("year", keep_attrs=True)
    clim = ensemble_mean(clim)
    clim.name = "rx1day"
    clim.attrs.update(long_name="Rx1day climatology", units="mm day-1")
    return clim


def txx(tasmax: xr.DataArray) -> xr.DataArray:
    annual_max = tasmax.groupby("time.year").max("time")
    clim = annual_max.mean("year", keep_attrs=True)
    clim = ensemble_mean(clim)
    clim.name = "TXx"
    clim.attrs.update(long_name="TXx climatology", units="degC")
    return clim


def climatology(da: xr.DataArray, variable: str) -> xr.DataArray:
    return rx1day(da) if variable == "pr" else txx(da)


# --------------------------------------------------------------------------- #
# Day-selection helpers
# --------------------------------------------------------------------------- #
def find_extreme_day(da: xr.DataArray, mode: str = "hottest") -> np.datetime64:
    """Return the time step at which the spatial maximum is largest
    (``hottest`` / ``wettest``) or smallest (``coldest``)."""
    da = ensemble_mean(da)
    other = [d for d in da.dims if d != "time"]
    spatial_max = da.mean(dim=other, skipna=True)
    if mode in ("hottest", "wettest"):
        idx = int(np.asarray(spatial_max.argmax().values).item())
    elif mode == "coldest":
        spatial_min = da.min(dim=other, skipna=True)
        idx = int(np.asarray(spatial_min.argmin().values).item())
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    return da["time"].values[idx]

def find_extreme_day_ranked(da: xr.DataArray, mode: str = "hottest",
                            rank: int = 1) -> np.datetime64:
    """Return the Nth most extreme day in the spatial-mean time series.

    Parameters
    ----------
    da : xr.DataArray
        Time series with a ``time`` dimension.
    mode : {"hottest", "wettest", "coldest"}
        Which extreme to rank by.
    rank : int, default 1
        1 = most extreme, 2 = second most extreme, etc.
        Clamped to [1, len(time)] so out-of-range values are safe.

    Notes
    -----
    Uses the area-mean (same as ``find_extreme_day``) so day selection is
    consistent with the combined-region plots.
    """
    da = ensemble_mean(da)
    other = [d for d in da.dims if d != "time"]
    series = da.mean(dim=other, skipna=True)
    values = np.asarray(series.values, dtype=float)
    rank = max(1, min(rank, len(values)))

    if mode in ("hottest", "wettest"):
        # argsort ascending → last elements are largest; reverse to get rank 1 = max
        sorted_idx = np.argsort(values)[::-1]
    elif mode == "coldest":
        sorted_idx = np.argsort(values)   # ascending: rank 1 = min
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")

    return da["time"].values[sorted_idx[rank - 1]]


def daily_max_series(da: xr.DataArray) -> np.ndarray:
    """Return the per-timestep spatial maximum as a 1-D numpy array.

    Used by the daily-snapshot plots to build a "distribution of daily
    maximum intensities" — for every day in the series we take the
    spatial max, and the KDE of those values shows whether each model
    reproduces the RCM's distribution of extreme events.
    """
    da = ensemble_mean(da)
    other = [d for d in da.dims if d != "time"]
    m = da.max(dim=other, skipna=True).values
    m = np.asarray(m, dtype=float).ravel()
    return m[np.isfinite(m)]


def gaussian_kde_1d(values, grid=None, n_grid: int = 256,
                    bandwidth: float | None = None):
    """Plain Gaussian KDE without requiring scipy.

    Returns ``(grid, density)`` where the density is normalised so that
    trapezoidal integration yields ~1. If ``grid`` is None, a 1% / 99%
    percentile range with ``n_grid`` points is used.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        if grid is None:
            grid = np.linspace(0, 1, n_grid)
        return grid, np.zeros_like(grid)
    if grid is None:
        lo = float(np.nanpercentile(v, 0.5))
        hi = float(np.nanpercentile(v, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        grid = np.linspace(lo, hi, n_grid)
    grid = np.asarray(grid, dtype=float)
    std = float(np.std(v, ddof=1)) if v.size > 1 else 1.0
    if std < 1e-9:
        std = 1e-6
    h = bandwidth if bandwidth is not None else 1.06 * std * v.size ** (-1 / 5)
    h = max(h, 1e-6)
    # Chunked sum to keep memory bounded
    dens = np.zeros_like(grid)
    chunk = 50000
    for i in range(0, v.size, chunk):
        z = (grid[:, None] - v[None, i:i + chunk]) / h
        dens += np.exp(-0.5 * z * z).sum(axis=1)
    dens /= v.size * h * np.sqrt(2 * np.pi)
    return grid, dens


def safe_random_date(times: xr.DataArray, seed: int | None) -> np.datetime64:
    rng = np.random.default_rng(seed)
    return times.values[int(rng.integers(0, times.size))]


# --------------------------------------------------------------------------- #
# Metrics: RMSE, radial PSD
# --------------------------------------------------------------------------- #
def rmse(pred: xr.DataArray, truth: xr.DataArray) -> float:
    pred = ensemble_mean(pred)
    truth = ensemble_mean(truth)
    p, t = xr.align(pred, truth, join="inner")
    diff = p.values - t.values
    return float(np.sqrt(np.nanmean(diff ** 2)))


def radial_psd(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2-D power spectral density.

    Ports the reference CORDEX-ML benchmark implementation:
    https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/evaluation/diagnostics.py

    Works on 2-D (lat, lon) or 3-D (time, lat, lon) fields. For 3-D inputs,
    one radial profile is computed per time step and the results are
    averaged. Any NaNs are replaced by 0 (to match the reference).

    Returns
    -------
    k : np.ndarray
        Integer wavenumber (pixels from the DC bin). Index 0 is the DC
        component; we exclude it when plotting on log scale.
    P : np.ndarray
        Radially-averaged power at each wavenumber.
    """
    data = np.asarray(field.values, dtype=float)
    # normalizing the power spectral density
    data = (data - data.mean())/data.std()#(data - data.min())/(data.max() - data.min())
    data = np.nan_to_num(data, nan=0.0)

    if data.ndim == 2:
        data = data[None, ...]
    elif data.ndim == 3:
        pass
    else:
        raise ValueError(f"radial_psd expects a 2D/3D field, got shape {data.shape}")

    # 2-D FFT per sample, shifted so DC sits at the centre
    Fk = np.fft.fftshift(np.fft.fft2(data, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(Fk) ** 2  # shape: (T, ny, nx)

    # Radial profile: integer-radius bincount centred on the image centre.
    ny, nx = power.shape[-2], power.shape[-1]
    y, x = np.indices((ny, nx))
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    r_flat = r.ravel()
    nr = np.bincount(r_flat)
    nr_safe = np.maximum(nr, 1)

    profiles = []
    for frame in power:
        tbin = np.bincount(r_flat, frame.ravel())
        profiles.append(tbin / nr_safe)
    P = np.mean(profiles, axis=0)

    k = np.arange(P.size, dtype=float)
    return k, P


# --------------------------------------------------------------------------- #
# 2-D lon/lat rendering (ALPS needs this because its grid is curvilinear)
# --------------------------------------------------------------------------- #
def has_2d_coords(ds: xr.Dataset, variable: str | None = None) -> bool:
    """Return True if the dataset carries 2-D ``lon(y,x)`` / ``lat(y,x)``
    coordinates (as happens on the rotated-pole ALPS grid)."""
    lon = ds.coords.get("lon", ds.coords.get("longitude"))
    lat = ds.coords.get("lat", ds.coords.get("latitude"))
    if lon is None or lat is None:
        return False
    return lon.ndim == 2 and lat.ndim == 2


def get_lonlat_2d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray] | None:
    """If the dataset has 2-D lon/lat coords, return them as numpy arrays."""
    lon = ds.coords.get("lon", ds.coords.get("longitude"))
    lat = ds.coords.get("lat", ds.coords.get("latitude"))
    if lon is None or lat is None or lon.ndim != 2 or lat.ndim != 2:
        return None
    return np.asarray(lon.values), np.asarray(lat.values)


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "grid.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Enlarged style used by the multi-region combined plots — labels in the
# combined figure need to stay readable when the whole page is scaled down.
LARGE_RC = {
    **NATURE_RC,
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
}


def setup_nature_style():
    import matplotlib as mpl
    mpl.rcParams.update(NATURE_RC)


def setup_large_style():
    """Larger fonts for multi-region combined plots."""
    import matplotlib as mpl
    mpl.rcParams.update(LARGE_RC)


# --------------------------------------------------------------------------- #
# Projections (cartopy)
# --------------------------------------------------------------------------- #
# Set this to False to skip coastlines/borders globally (e.g. offline).
_FEATURES_AVAILABLE: bool | None = None


def _probe_features() -> bool:
    """Check once whether Natural-Earth data is available; cache the answer."""
    global _FEATURES_AVAILABLE
    if _FEATURES_AVAILABLE is not None:
        return _FEATURES_AVAILABLE
    try:
        import cartopy.feature as cfeature  # noqa: F401
        from cartopy.io.shapereader import natural_earth
        # Try to resolve the 50m coastline file; this triggers a download
        # if not cached. If offline, we'll fall back to no features.
        natural_earth(resolution="10m", category="physical", name="coastline")
        _FEATURES_AVAILABLE = True
    except Exception as e:
        import warnings
        warnings.warn(f"Cartopy Natural-Earth data unavailable ({e!r}); "
                      "plots will render without coastlines/borders.",
                      stacklevel=2)
        _FEATURES_AVAILABLE = False
    return _FEATURES_AVAILABLE


def get_crs(domain: str, ds: xr.Dataset | None = None):
    """Return ``(map_crs, data_crs)`` for cartopy.

    NZ and SA use regular lat/lon so both are PlateCarree.
    ALPS is EURO-CORDEX rotated-pole; if the NetCDF carries a
    ``rotated_pole`` / ``rotated_latitude_longitude`` grid-mapping variable we
    read the pole location from it, otherwise we fall back to the
    EURO-CORDEX standard (-162°, 39.25°).
    """
    import cartopy.crs as ccrs
    plate = ccrs.PlateCarree()
    if domain != "ALPS":
        return plate, plate

    rp_lon, rp_lat = 20, 39.25
    if ds is not None:
        for name in ("rotated_pole", "rotated_latitude_longitude",
                     "rotated_latitude_longitude_crs"):
            if name in ds.variables:
                a = ds[name].attrs
                rp_lon = float(a.get("grid_north_pole_longitude", rp_lon))
                rp_lat = float(a.get("grid_north_pole_latitude", rp_lat))
                break
    return plate, ccrs.RotatedPole(pole_longitude=rp_lon, pole_latitude=rp_lat)


def _xy_coords(field: xr.DataArray):
    for xn, yn in (("lon", "lat"), ("longitude", "latitude"),
                   ("rlon", "rlat"), ("x", "y")):
        if xn in field.coords and yn in field.coords:
            return field[xn].values, field[yn].values
    return (field[field.dims[-1]].values, field[field.dims[-2]].values)


def plot_map(ax, field: xr.DataArray, *, cmap, vmin, vmax, data_crs,
             add_features: bool = True, lon2d=None, lat2d=None):
    """Draw a pcolormesh panel with coastlines and (light) borders.

    If ``lon2d`` / ``lat2d`` are provided (2-D curvilinear grids, e.g. ALPS
    rotated-pole), the field is drawn in PlateCarree and ``data_crs`` is
    ignored for the pcolormesh. This is what the user requested for ALPS:
    ``ax.pcolormesh(lon2d, lat2d, field, transform=PlateCarree())``.
    """
    if lon2d is not None and lat2d is not None:
        import cartopy.crs as ccrs
        im = ax.pcolormesh(
            lon2d, lat2d, field.values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto", transform=ccrs.PlateCarree(), rasterized=True,
        )
    else:
        x, y = _xy_coords(field)
        im = ax.pcolormesh(
            x, y, field.values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto", transform=data_crs, rasterized=True,
        )

    if add_features and _probe_features():
        try:
            ax.coastlines(resolution="10m", linewidth=0.6, color="#111", alpha=0.95)
        except Exception:
            pass
        try:
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                           linewidth=0.35, color="#333", alpha=0.75)
        except Exception:
            pass

    # Fit extent to the actual data
    try:
        import cartopy.crs as ccrs
        if lon2d is not None and lat2d is not None:
            ax.set_extent(
                [float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
                 float(np.nanmin(lat2d)), float(np.nanmax(lat2d))],
                crs=ccrs.PlateCarree(),
            )
        else:
            x, y = _xy_coords(field)
            ax.set_extent([float(np.min(x)), float(np.max(x)),
                           float(np.min(y)), float(np.max(y))],
                          crs=data_crs)
    except Exception:
        pass

    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#333")
    return im