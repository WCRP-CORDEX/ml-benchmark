#!/usr/bin/env python
"""
plot_combined_regions.py
========================
Produce a single page that stacks ALPS (top) + SA (middle) + NZ (bottom),
each region rendered as its own band with:

    * 3 top-sampled models + 3 bottom-sampled models
    * the ground-truth panel on the right
    * a colorbar below the small panels
    * a distribution inset under the ground-truth panel:
        - daily mode       → KDE of per-model spatial-max on the selected day
        - climatology mode → KDE of per-model RMSE vs truth
        - cc_signal mode   → KDE of per-model area-mean Δ, truth mean as vline

Large fonts (``bu.setup_large_style``) are used so labels stay readable when
the page is scaled down for presentations / papers.

Each region uses its own colour range so the high-precipitation SA days don't
wash out the ALPS/NZ panels.

Modes (``--mode``):

    climatology    Rx1day (pr) or TXx (tasmax), historical period.
    daily          Single-day snapshot (wettest for pr / hottest for tasmax)
                   chosen by area-mean on the truth. Use --day-rank 1 for
                   the single most extreme day, 2 for the second most
                   extreme, etc. (default: 1).
    cc_signal      Future minus historical climatology signal.
                   Use --future-period {mid_century|end_century} and
                   optionally --relative (pr only: express as %).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

import benchmark_utils as bu


INDEX_NAME = {"pr": "Rx1day", "tasmax": "TXx"}

# Default driving GCM per domain, matching submit_plots.sh
PRIMARY_GCM = {"ALPS": "CNRM-CM5", "NZ": "ACCESS-CM2", "SA": "ACCESS-CM2"}

# Per-region panel letters so the combined figure has unique labels a-u
REGION_LABELS = {
    "ALPS": list("abcdefg"),
    "SA":   list("hijklmn"),
    "NZ":   list("opqrstu"),
}


# --------------------------------------------------------------------------- #
def _stroke(lw, fg):
    import matplotlib.patheffects as pe
    return pe.withStroke(linewidth=lw, foreground=fg)


def _badge(ax, txt, *, loc="br"):
    x, y, va, ha = ((0.97, 0.05, "bottom", "right") if loc == "br"
                    else (0.97, 0.95, "top", "right"))
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va=va, ha=ha, color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#333", linewidth=0.5, alpha=0.92),
            zorder=10)


def _align_to_truth(pred, truth_like, domain):
    if domain == "ALPS":
        try:
            pred = pred.sel(y=truth_like.y.values, x=truth_like.x.values,
                            method="nearest")
        except Exception:
            pass
        return pred.transpose("y", "x")
    try:
        pred = pred.sel(lat=truth_like.lat.values, lon=truth_like.lon.values,
                        method="nearest")
    except Exception:
        pass
    return pred.transpose("lat", "lon")


def _area_mean(field) -> float:
    vals = np.asarray(field.values).ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _cc_signal(hist_clim, fut_clim, relative: bool) -> xr.DataArray:
    """Future minus historical climatology; optionally as % of historical."""
    if relative:
        denom = hist_clim.where(hist_clim > 0.1)
        return 100.0 * (fut_clim - hist_clim) / denom
    return fut_clim - hist_clim


# --------------------------------------------------------------------------- #
def _region_band(subfig, domain, gcm, *, args, paths, rankings, mode):
    """Draw a single region's top/bot + truth + distribution inset into a
    matplotlib SubFigure. Returns nothing; mutates the subfig in place.
    """
    ranking = bu.get_ranking(rankings, args.variable, args.experiment)

    # cc_signal needs files present for BOTH periods
    period_key = (["historical", args.future_period]
                  if mode == "cc_signal" else args.period)

    available = bu.discover_available(
        paths, ranking,
        domain=domain, experiment=args.experiment,
        period_key=period_key,
        gcm=gcm,
        predictor_flavour=args.predictor_flavour,
    )
    if len(available) < 2 * args.n_sample + args.bot_exclude:
        subfig.text(0.5, 0.5,
                    f"[{domain}] not enough models available on disk "
                    f"(n={len(available)})",
                    ha="center", va="center", fontsize=16, color="#a00")
        return

    top, bot = bu.sample_top_bottom(
        ranking, available,
        n_sample=args.n_sample, pool=args.pool,
        bot_pool=args.bot_pool, bot_exclude=args.bot_exclude,
        seed=args.seed,
    )
    print(f"[combined] {domain}: top={top}")
    print(f"[combined] {domain}: bot={bot}")

    rel = getattr(args, "relative", False) and args.variable == "pr"

    # ------------------------------------------------------------------ #
    # Build the truth field for this band
    # ------------------------------------------------------------------ #
    if mode == "cc_signal":
        truth_hist_da, _ = bu.open_truth(paths, domain, "historical",
                                         gcm, args.variable)
        truth_fut_da, truth_ds = bu.open_truth(paths, domain, args.future_period,
                                               gcm, args.variable)
        truth_hist_clim = bu.climatology(truth_hist_da, args.variable)
        truth_fut_clim  = bu.climatology(truth_fut_da,  args.variable)
        truth_field = _cc_signal(truth_hist_clim, truth_fut_clim, rel)
        unit_str    = "%" if rel else bu.units_for(args.variable, diverging=True)
        date_hint   = None

        if domain == "ALPS":
            truth_ds[args.variable] = truth_fut_da.transpose("time", "y", "x")
            truth_field = truth_field.transpose("y", "x")
        else:
            truth_ds[args.variable] = truth_fut_da.transpose("time", "lat", "lon")
            truth_field = truth_field.transpose("lat", "lon")

    elif mode == "climatology":
        truth, truth_ds = bu.open_truth(paths, domain, args.period, gcm, args.variable)
        if domain == "ALPS":
            truth_ds[args.variable] = truth_ds[args.variable].transpose("time", "y", "x")
        truth_field = bu.climatology(truth, args.variable)
        unit_str    = bu.units_for(args.variable)
        date_hint   = None
        if domain == "ALPS":
            truth_field = truth_field.transpose("y", "x")
        else:
            truth_field = truth_field.transpose("lat", "lon")

    else:  # daily
        truth, truth_ds = bu.open_truth(paths, domain, args.period, gcm, args.variable)
        if domain == "ALPS":
            truth_ds[args.variable] = truth_ds[args.variable].transpose("time", "y", "x")
        mode_k      = "hottest" if args.variable == "tasmax" else "wettest"
        day         = bu.find_extreme_day_ranked(truth, mode=mode_k,
                                                 rank=args.day_rank)
        date_hint   = str(np.datetime_as_string(day, unit="D"))
        truth_field = truth.sel(time=day, method="nearest")
        unit_str    = bu.units_for(args.variable)
        if domain == "ALPS":
            truth_field = truth_field.transpose("y", "x")
        else:
            truth_field = truth_field.transpose("lat", "lon")

    # ------------------------------------------------------------------ #
    # Build pred fields for the 6 sampled models
    # ------------------------------------------------------------------ #
    def _field_for(mname):
        if mode == "cc_signal":
            da_hist = bu.open_prediction(paths, mname, domain, args.experiment,
                                         "historical", gcm, args.variable,
                                         args.predictor_flavour)
            da_fut  = bu.open_prediction(paths, mname, domain, args.experiment,
                                         args.future_period, gcm, args.variable,
                                         args.predictor_flavour)
            out = _cc_signal(bu.climatology(da_hist, args.variable),
                             bu.climatology(da_fut,  args.variable), rel)
        elif mode == "climatology":
            da  = bu.open_prediction(paths, mname, domain, args.experiment,
                                     args.period, gcm, args.variable,
                                     args.predictor_flavour)
            out = bu.climatology(da, args.variable)
        else:  # daily
            da  = bu.open_prediction(paths, mname, domain, args.experiment,
                                     args.period, gcm, args.variable,
                                     args.predictor_flavour)
            out = bu.ensemble_mean(da).sel(time=day, method="nearest")
        return _align_to_truth(out, truth_field, domain)

    tops = {m: _field_for(m) for m in top}
    bots = {m: _field_for(m) for m in bot}

    # ------------------------------------------------------------------ #
    # Distribution data
    # ------------------------------------------------------------------ #
    all_vals: dict[str, float] = {}
    truth_scalar = None

    if mode == "cc_signal":
        for mname in available:
            try:
                all_vals[mname] = _area_mean(_field_for(mname))
            except Exception as e:
                warnings.warn(f"[{domain}] cc mean skip {mname}: {e}")
        truth_scalar = _area_mean(truth_field)

    elif mode == "climatology":
        for mname in available:
            try:
                all_vals[mname] = float(bu.rmse(_field_for(mname), truth_field))
            except Exception as e:
                warnings.warn(f"[{domain}] RMSE skip {mname}: {e}")
        truth_scalar = None

    else:  # daily
        for mname in available:
            try:
                da = bu.open_prediction(paths, mname, domain, args.experiment,
                                        args.period, gcm, args.variable,
                                        args.predictor_flavour)
                dslice = bu.ensemble_mean(da).sel(time=day, method="nearest")
                v = np.asarray(dslice.values, dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    all_vals[mname] = float(v.max())
            except Exception as e:
                warnings.warn(f"[{domain}] max skip {mname}: {e}")
        truth_scalar = float(np.nanmax(truth_field.values))

    # ------------------------------------------------------------------ #
    # Colour scale (per region)
    # ------------------------------------------------------------------ #
    stack = np.concatenate(
        [truth_field.values.ravel()]
        + [f.values.ravel() for f in tops.values()]
        + [f.values.ravel() for f in bots.values()]
    )
    stack = stack[np.isfinite(stack)]

    if mode == "cc_signal":
        cmap = bu.cmap_for(args.variable, kind="cc_signal")
        if args.variable == "pr":
            vlim = float(np.nanpercentile(np.abs(stack), 98))
            vlim = max(vlim, 1.0)
            vmin, vmax = -vlim, vlim
        else:
            vmax = float(np.nanpercentile(stack, 98))
            vmax = max(vmax, 1.0)
            vmin = 0.0
    elif args.variable == "pr":
        cmap = bu.cmap_for(args.variable, kind=mode)
        vmin = 0.0
        vmax = max(float(np.nanpercentile(stack, 99.5 if mode == "daily" else 99)), 5.0)
    else:
        cmap = bu.cmap_for(args.variable, kind=mode)
        vmin = float(np.nanpercentile(stack, 1))
        vmax = float(np.nanpercentile(stack, 99))

    map_crs, data_crs = bu.get_crs(domain, truth_ds)
    lonlat2d = bu.get_lonlat_2d(truth_ds) if domain == "ALPS" else None

    # ------------------------------------------------------------------ #
    # Draw into this subfigure
    # ------------------------------------------------------------------ #
    gs = subfig.add_gridspec(
        4, 7,
        width_ratios=[1, 1, 1, 0.08, 0.5, 1, 0.5],
        height_ratios=[1, 1, 0.03, 0.5],
        hspace=0.18, wspace=0.04,
        left=0.05, right=0.985, top=0.91, bottom=0.07,
    )

    # Region / TOP / BOTTOM labels on the left margin
    subfig.text(0.028, 0.6, domain, rotation=90,
                fontsize=22, fontweight="bold", style="italic",
                color="#222", va="center", ha="center")
    subfig.text(0.042, 0.72, "TOP", rotation=90, fontsize=14, color="#1a5e2a", va="center", ha="center")
    subfig.text(0.042, 0.42, "BOTTOM", rotation=90, fontsize=14, color="#8b1a1a", va="center", ha="center")

    labels = REGION_LABELS[domain]

    def small_panel(row, col, field, model, rank_label, letter):
        ax = subfig.add_subplot(gs[row, col], projection=map_crs)
        im = bu.plot_map(ax, field, cmap=cmap, vmin=vmin, vmax=vmax,
                         data_crs=data_crs,
                         lon2d=lonlat2d[0] if lonlat2d else None,
                         lat2d=lonlat2d[1] if lonlat2d else None)
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                color="white", path_effects=[_stroke(1.3, "#000")])
        ax.set_title(f"{model}", fontsize=15, pad=3, weight ='bold')

        if mode == "cc_signal":
            mean = _area_mean(field)
            sign = "+" if mean >= 0 else ""
            _badge(ax, f"Δ {sign}{mean:.2f} {unit_str}", loc="tr")
            _badge(ax, f"RMSE {bu.rmse(field, truth_field):.2f}", loc="br")
        elif mode == "climatology":
            _badge(ax, f"RMSE {bu.rmse(field, truth_field):.2f}", loc="br")
        return im

    for i, (name, f) in enumerate(tops.items()):
        small_panel(0, i, f, name, f"top {i+1}", labels[i])
    for i, (name, f) in enumerate(bots.items()):
        small_panel(1, i, f, name, f"bot {i+1}", labels[i + 3])

    # Ground truth panel
    ax_truth = subfig.add_subplot(gs[0:2, 4:], projection=map_crs)
    im = bu.plot_map(ax_truth, truth_field, cmap=cmap, vmin=vmin, vmax=vmax,
                     data_crs=data_crs,
                     lon2d=lonlat2d[0] if lonlat2d else None,
                     lat2d=lonlat2d[1] if lonlat2d else None)

    if mode == "cc_signal":
        hist_years = bu.PERIOD_YEARS["historical"]
        fut_years  = bu.PERIOD_YEARS[args.future_period]
        tmean = _area_mean(truth_field)
        sign  = "+" if tmean >= 0 else ""
        title = (f"Ground truth ({gcm})\n"
                 f"{fut_years} − {hist_years}   "
                 f"Δ {sign}{tmean:.2f} {unit_str}")
    else:
        title = f"Ground truth \n ({gcm})"
        if date_hint:
            title += f"   ({date_hint})"

    ax_truth.set_title(title, fontsize=18, fontweight="bold", pad=5)
    ax_truth.text(0.02, 0.98, labels[6], transform=ax_truth.transAxes,
                  fontsize=15, fontweight="bold", va="top", ha="left",
                  color="white", path_effects=[_stroke(1.6, "#000")])
    try:
        gl = ax_truth.gridlines(draw_labels=True, linewidth=0.4,
                                color="gray", alpha=0.55)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}
    except Exception:
        pass

    # Colorbar under the small panels
    cax  = subfig.add_subplot(gs[3, 0:3])
    bbox = cax.get_position()
    cax.set_position([bbox.x0 + 0.03, bbox.y0 + bbox.height * 0.35,
                      bbox.width - 0.06, bbox.height * 0.28])
    cb = subfig.colorbar(im, cax=cax, orientation="horizontal")
    if mode == "cc_signal":
        cb_label = f"Δ{INDEX_NAME[args.variable]}  [{unit_str}]"
    else:
        cb_label = f"{bu.pretty_var(args.variable)}  [{bu.units_for(args.variable)}]"
    cb.set_label(cb_label, fontsize=16, fontweight="bold")
    cb.ax.tick_params(labelsize=15)
    cb.outline.set_linewidth(0.6)

    # Distribution inset (bottom-right of band)
    ax_d = subfig.add_subplot(gs[3, 5])
    if mode == "cc_signal":
        _draw_cc_kde_inset(ax_d, all_vals, truth_scalar, tops, bots,
                           args.variable, unit_str)
    elif mode == "climatology":
        _draw_rmse_kde_inset(ax_d, all_vals, tops, bots, args.variable)
    else:
        _draw_max_kde_inset(ax_d, all_vals, truth_scalar, tops, bots, args.variable)


# --------------------------------------------------------------------------- #
# Distribution inset helpers
# --------------------------------------------------------------------------- #

def _draw_cc_kde_inset(ax, all_means, truth_mean, tops, bots, variable, unit_str):
    """KDE of per-model area-mean delta signal; truth mean as vertical line."""
    import matplotlib.lines as mlines

    if not all_means:
        ax.set_axis_off(); return

    names = list(all_means)
    vals  = np.asarray([all_means[n] for n in names], float)
    ok    = np.isfinite(vals)
    names = [n for n, k in zip(names, ok) if k]
    vals  = vals[ok]
    if vals.size == 0:
        ax.set_axis_off(); return

    lo = min(float(vals.min()), truth_mean if truth_mean is not None else float(vals.min()))
    hi = max(float(vals.max()), truth_mean if truth_mean is not None else float(vals.max()))
    pad = 0.08 * max(hi - lo, 1e-6)
    lo -= pad; hi += pad
    grid = np.linspace(lo, hi, 256)
    _, dens = bu.gaussian_kde_1d(vals, grid=grid)

    fill = "#80cdc1" if variable == "pr" else "#f4a582"
    line = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, dens, color=fill, alpha=0.45, zorder=2)
    ax.plot(grid, dens, color=line, linewidth=2.0, zorder=3)

    y0   = -0.03 * float(max(dens.max(), 1e-9))
    ts, bs = set(tops), set(bots)
    for n, v in zip(names, vals):
        c, z, s = (("#1a5e2a", 6, 70) if n in ts
                   else ("#8b1a1a", 6, 70) if n in bs
                   else ("#888",    4, 20))
        ax.scatter(v, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)

    if truth_mean is not None and np.isfinite(truth_mean):
        ax.axvline(truth_mean, color="black", linewidth=2.4, zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    ax.set_xlabel(f"Area-mean Δ  [{unit_str}]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_title(f"Δ signal across {vals.size} models", fontsize=11, pad=3,
                 fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        mlines.Line2D([], [], color=line,      lw=2.0,
                      label=f"KDE (n={vals.size})"),
        mlines.Line2D([], [], color="black",   lw=2.2, label="truth mean"),
        mlines.Line2D([], [], color="#1a5e2a", marker="o", lw=0,
                      markersize=7, label="top sampled"),
        mlines.Line2D([], [], color="#8b1a1a", marker="o", lw=0,
                      markersize=7, label="bot sampled"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              frameon=True, framealpha=0.9, handlelength=1.4, borderpad=0.3)


def _draw_rmse_kde_inset(ax, all_rmse, tops, bots, variable):
    if not all_rmse:
        ax.set_axis_off(); return
    names = list(all_rmse)
    vals  = np.asarray([all_rmse[n] for n in names], float)
    ok    = np.isfinite(vals)
    names = [n for n, k in zip(names, ok) if k]
    vals  = vals[ok]
    if vals.size == 0:
        ax.set_axis_off(); return
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.08 * max(hi - lo, 1e-6); lo = max(0.0, lo - pad); hi += pad
    grid = np.linspace(lo, hi, 256)
    _, dens = bu.gaussian_kde_1d(vals, grid=grid)
    fill = "#80cdc1" if variable == "pr" else "#f4a582"
    line = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, dens, color=fill, alpha=0.45, zorder=2)
    ax.plot(grid, dens, color=line, linewidth=2.0, zorder=3)
    y0   = -0.03 * float(max(dens.max(), 1e-9))
    ts, bs = set(tops), set(bots)
    for n, v in zip(names, vals):
        c, z, s = (("#1a5e2a", 6, 70) if n in ts
                   else ("#8b1a1a", 6, 70) if n in bs
                   else ("#888",    4, 20))
        ax.scatter(v, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    ax.set_xlabel(f"RMSE  [{bu.units_for(variable)}]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_title(f"RMSE across {vals.size} models", fontsize=11, pad=3)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5); ax.set_axisbelow(True)


def _draw_max_kde_inset(ax, all_max, truth_max, tops, bots, variable):
    if not all_max:
        ax.set_axis_off(); return
    names = list(all_max)
    vals  = np.asarray([all_max[n] for n in names], float)
    ok    = np.isfinite(vals)
    names = [n for n, k in zip(names, ok) if k]
    vals  = vals[ok]
    if vals.size == 0:
        ax.set_axis_off(); return
    lo = min(float(vals.min()), truth_max if truth_max is not None else float(vals.min()))
    hi = max(float(vals.max()), truth_max if truth_max is not None else float(vals.max()))
    pad = 0.08 * max(hi - lo, 1e-6); lo -= pad; hi += pad
    grid = np.linspace(lo, hi, 256)
    _, dens = bu.gaussian_kde_1d(vals, grid=grid)
    fill = "#80cdc1" if variable == "pr" else "#f4a582"
    line = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, dens, color=fill, alpha=0.45, zorder=2)
    ax.plot(grid, dens, color=line, linewidth=2.0, zorder=3)
    y0   = -0.03 * float(max(dens.max(), 1e-9))
    ts, bs = set(tops), set(bots)
    for n, v in zip(names, vals):
        c, z, s = (("#1a5e2a", 6, 70) if n in ts
                   else ("#8b1a1a", 6, 70) if n in bs
                   else ("#888",    4, 20))
        ax.scatter(v, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)
    if truth_max is not None and np.isfinite(truth_max):
        ax.axvline(truth_max, color="black", linewidth=2.4, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    ax.set_xlabel(f"Max intensity  [{bu.units_for(variable)}]",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_title(f"Max across {vals.size} models", fontsize=11, pad=3, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5); ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable",   required=True, choices=["pr", "tasmax"])
    p.add_argument("--experiment", required=True, choices=list(bu.EXPERIMENTS))
    p.add_argument("--period",     default="historical", choices=list(bu.PERIOD_YEARS))
    p.add_argument("--predictor-flavour", default="perfect",
                   choices=["perfect", "imperfect"])
    p.add_argument("--mode", required=True,
                   choices=["climatology", "daily", "cc_signal"])
    # cc_signal-specific
    p.add_argument("--day-rank", default=1, type=int,
                   help="daily mode: 1=most extreme day, 2=second most extreme, etc.")
    p.add_argument("--future-period", default="end_century",
                   choices=["mid_century", "end_century"],
                   help="Future period for cc_signal mode.")
    p.add_argument("--relative", action="store_true",
                   help="cc_signal + pr only: express delta as %% of historical.")
    p.add_argument("--truth-root", required=True,  type=Path)
    p.add_argument("--pred-root",  required=True,  type=Path)
    p.add_argument("--rankings",
                   default=Path(__file__).with_name("model_rankings.json"), type=Path)
    p.add_argument("--out-dir", default=Path("outputs/combined"), type=Path)
    p.add_argument("--n-sample",    default=3,  type=int)
    p.add_argument("--pool",        default=10, type=int)
    p.add_argument("--bot-pool",    default=15, type=int)
    p.add_argument("--bot-exclude", default=3,  type=int)
    p.add_argument("--seed",        default=42, type=int)
    args = p.parse_args()

    paths    = bu.Paths(truth_root=args.truth_root, pred_root=args.pred_root)
    rankings = bu.load_rankings(args.rankings)

    bu.setup_large_style()
    fig     = plt.figure(figsize=(19, 25), facecolor="white")
    subfigs = fig.subfigures(3, 1, hspace=0.033)

    for sf, domain in zip(subfigs, ("ALPS", "SA", "NZ")):
        gcm = PRIMARY_GCM[domain]
        try:
            _region_band(sf, domain, gcm,
                         args=args, paths=paths, rankings=rankings,
                         mode=args.mode)
        except Exception as e:
            sf.text(0.5, 0.5, f"[{domain}] failed: {e}",
                    ha="center", va="center", fontsize=16, color="#a00")

    # Suptitle
    if args.mode == "cc_signal":
        hist_years = bu.PERIOD_YEARS["historical"]
        fut_years  = bu.PERIOD_YEARS[args.future_period]
        rel_tag    = " (relative %)" if (args.relative and args.variable == "pr") else ""
        title_str  = (f"{args.variable.upper()}  ·  cc_signal{rel_tag}\n  "
                      f"{args.experiment.replace('_', ' ')}  ·  "
                      f"Δ{INDEX_NAME[args.variable]}  {fut_years} − {hist_years}")
    else:
        title_index = (INDEX_NAME[args.variable] if args.mode == "climatology"
                       else f"Case study — rank {args.day_rank} extreme day")
        title_str = (f"{args.variable.upper()}  ·  {args.mode}\n  "
                     f"{args.experiment.replace('_', ' ')}  ·  {title_index}")

    fig.suptitle(title_str, fontsize=29, fontweight="bold", y=1.05)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"combined_{args.variable}_{args.experiment}_{args.mode}_seed{args.seed}"
    if args.mode == "daily":
        stem += f"_rank{args.day_rank}"
    elif args.mode == "cc_signal":
        rel_tag = "_relpct" if (args.relative and args.variable == "pr") else ""
        stem   += f"_{args.future_period}{rel_tag}"

    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{stem}.pdf",           bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] wrote {args.out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()