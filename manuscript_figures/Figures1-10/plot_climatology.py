#!/usr/bin/env python
"""
plot_climatology.py
===================
Climatology panels for the CORDEX-ML-Bench. Each model's Rx1day (for pr)
or TXx (for tasmax) is computed across the selected 20-year window, then
compared against the ground-truth climatology. 3 models are drawn from
the top-``--pool``; 3 more from ranks ``[N-bot_pool, N-bot_exclude)``
(defaults: bottom-15 skipping the 3 very-worst).

Each panel carries an RMSE-vs-truth badge in its bottom-right corner,
printed in black against a white pill so it stays legible.

    pr      -> Rx1day  (annual-max 1-day precip, averaged over years)
    tasmax  -> TXx     (annual-max daily-max T, averaged over years)

Colour schemes:
* pr climatology   → BrBG (dry brown → wet green)
* tasmax climatology → RdYlBu_r-like TEMP_CMAP
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

import benchmark_utils as bu


INDEX_NAME = {"pr": "Rx1day", "tasmax": "TXx"}


def _badge(ax, txt):
    """Put a white-pill / black-text RMSE badge in the bottom-right."""
    ax.text(
        0.97, 0.05, txt,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        va="bottom", ha="right",
        color="black",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor="#333", linewidth=0.5,
                  alpha=0.92),
        zorder=10,
    )


def _letter(ax, letter):
    import matplotlib.patheffects as pe
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left",
            color="white",
            path_effects=[pe.withStroke(linewidth=1.4, foreground="#000")])


def _draw_rmse_kde(ax, all_rmse, tops, bots, variable):
    """KDE of climatology RMSE across all available models, with the
    sampled top-3 / bot-3 highlighted as coloured dots. Same visual
    grammar as the CC-signal KDE."""
    if not all_rmse:
        ax.text(0.5, 0.5, "no RMSE data", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="#888")
        ax.set_axis_off()
        return

    names = list(all_rmse.keys())
    vals = np.asarray([all_rmse[n] for n in names], dtype=float)
    finite = np.isfinite(vals)
    names = [n for n, k in zip(names, finite) if k]
    vals = vals[finite]

    lo = float(vals.min()); hi = float(vals.max())
    pad = 0.08 * max(hi - lo, 1e-6)
    lo = max(0.0, lo - pad); hi += pad
    grid = np.linspace(lo, hi, 256)
    _, dens = bu.gaussian_kde_1d(vals, grid=grid)

    fill_color = "#80cdc1" if variable == "pr" else "#f4a582"
    line_color = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, dens, color=fill_color, alpha=0.45, zorder=2)
    ax.plot(grid, dens, color=line_color, linewidth=2.0, zorder=3)

    y0 = -0.03 * float(max(dens.max(), 1e-9))
    top_set = set(tops); bot_set = set(bots)
    for n, v in zip(names, vals):
        if n in top_set:
            c, z, s = "#1a5e2a", 6, 80
        elif n in bot_set:
            c, z, s = "#8b1a1a", 6, 80
        else:
            c, z, s = "#888888", 4, 22
        ax.scatter(v, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)

    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    unit = bu.units_for(variable)
    ax.set_xlabel(f"RMSE  vs truth  [{unit}]",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_title("Distribution of climatology RMSE across models",
                 fontsize=13, pad=3, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=line_color, lw=2.0,
                      label=f"KDE of RMSE (n={vals.size})"),
        mlines.Line2D([], [], color="#1a5e2a", marker="o", lw=0,
                      markersize=8, label="top-3 sampled"),
        mlines.Line2D([], [], color="#8b1a1a", marker="o", lw=0,
                      markersize=8, label="bot-3 sampled"),
        mlines.Line2D([], [], color="#888888", marker="o", lw=0,
                      markersize=5, label="other models"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.92, handlelength=1.6, borderpad=0.3)


def make_figure(truth, tops, bots, *, variable, domain, gcm,
                experiment, period_key, truth_ds,
                all_rmse=None):
    bu.setup_nature_style()
    cmap = bu.cmap_for(variable, kind="climatology")

    stack = np.concatenate(
        [truth.values.ravel()]
        + [f.values.ravel() for f in tops.values()]
        + [f.values.ravel() for f in bots.values()]
    )
    stack = stack[np.isfinite(stack)]
    if variable == "pr":
        vmin = 0.0
        vmax = max(float(np.nanpercentile(stack, 99)), 5.0)
    else:
        vmin = float(np.nanpercentile(stack, 1))
        vmax = float(np.nanpercentile(stack, 99))

    map_crs, data_crs = bu.get_crs(domain, truth_ds)
    lonlat2d = bu.get_lonlat_2d(truth_ds) if domain == "ALPS" else None

    fig = plt.figure(figsize=(16.5, 9.0), facecolor="white")
    # 3-row GS: maps (rows 0-1), narrow colorbar row (row 2), KDE row (row 3 — sits under truth)
    gs = GridSpec(
        3, 5,
        width_ratios=[1, 1, 1, 0.08, 2.0],
        height_ratios=[1, 1, 0.32],
        hspace=0.22, wspace=0.08,
        left=0.045, right=0.97, top=0.90, bottom=0.07,
    )

    fig.text(0.015, 0.69, "TOP",    rotation=90, fontsize=16, fontweight="bold",
             color="#1a5e2a", va="center", ha="center")
    fig.text(0.015, 0.39, "BOTTOM", rotation=90, fontsize=16, fontweight="bold",
             color="#8b1a1a", va="center", ha="center")

    labels = list("abcdefg")

    def small_panel(row, col, field, model, rank_label, label_letter):
        ax = fig.add_subplot(gs[row, col], projection=map_crs)
        im = bu.plot_map(ax, field, cmap=cmap, vmin=vmin, vmax=vmax,
                         data_crs=data_crs,
                         lon2d=lonlat2d[0] if lonlat2d else None,
                         lat2d=lonlat2d[1] if lonlat2d else None)
        err = bu.rmse(field, truth)
        ax.set_title(f"{rank_label}   {model}", fontsize=14, pad=3)
        _letter(ax, label_letter)
        _badge(ax, f"RMSE = {err:.2f}")
        return im

    for i, (name, f) in enumerate(tops.items()):
        small_panel(0, i, f, name, f"top {i+1}", labels[i])
    for i, (name, f) in enumerate(bots.items()):
        small_panel(1, i, f, name, f"bot {i+1}", labels[i + 3])

    # Big truth panel
    ax_truth = fig.add_subplot(gs[0:2, 4], projection=map_crs)
    im = bu.plot_map(ax_truth, truth, cmap=cmap, vmin=vmin, vmax=vmax,
                     data_crs=data_crs,
                     lon2d=lonlat2d[0] if lonlat2d else None,
                     lat2d=lonlat2d[1] if lonlat2d else None)
    import matplotlib.patheffects as pe
    ax_truth.text(0.02, 0.98, labels[6], transform=ax_truth.transAxes,
                  fontsize=16, fontweight="bold", va="top", ha="left",
                  color="white",
                  path_effects=[pe.withStroke(linewidth=1.6, foreground="#000")])
    ax_truth.set_title(f"Ground truth — {gcm}", fontsize=16, pad=5,
                       fontweight="bold")
    try:
        gl = ax_truth.gridlines(draw_labels=True, linewidth=0.4,
                                color="gray", alpha=0.55)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}
    except Exception:
        pass

    # Colorbar (cols 0-2 only; KDE sits under the truth panel in col 4)
    cax = fig.add_subplot(gs[2, 0:3])
    bbox = cax.get_position()
    cax.set_position([bbox.x0 + 0.03, bbox.y0 + bbox.height * 0.30,
                      bbox.width - 0.06, bbox.height * 0.30])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{INDEX_NAME[variable]}  [{bu.units_for(variable)}]",
                 fontsize=15, fontweight="bold")
    cb.ax.tick_params(labelsize=13)
    cb.outline.set_linewidth(0.6)

    # RMSE distribution across all available models (sits below the truth panel)
    ax_rmse = fig.add_subplot(gs[2, 4])
    _draw_rmse_kde(ax_rmse, all_rmse, tops, bots, variable)

    years = bu.PERIOD_YEARS[period_key]
    fig.suptitle(
        f"{domain} · {experiment.replace('_', ' ')} · {period_key.replace('_', ' ')} ({years})"
        f"   |   {INDEX_NAME[variable]} climatology",
        fontsize=18, fontweight="bold", y=0.965,
    )
    return fig


# --------------------------------------------------------------------------- #
def save_netcdf(out_nc, truth, tops, bots, variable):
    arrays = {"truth": truth}
    rmse_vals = {}
    for r, (n, f) in enumerate(tops.items(), 1):
        arrays[f"top{r}__{n}"] = f
        rmse_vals[f"top{r}__{n}"] = bu.rmse(f, truth)
    for r, (n, f) in enumerate(bots.items(), 1):
        arrays[f"bot{r}__{n}"] = f
        rmse_vals[f"bot{r}__{n}"] = bu.rmse(f, truth)
    ds = xr.Dataset({k: v.reset_coords(drop=True) for k, v in arrays.items()})
    for k, v in rmse_vals.items():
        ds[k].attrs["rmse_vs_truth"] = float(v)
    ds.attrs.update(variable=variable, index=INDEX_NAME[variable])
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(bu.DOMAIN_GCMS))
    p.add_argument("--gcm", required=True)
    p.add_argument("--variable", required=True, choices=["pr", "tasmax"])
    p.add_argument("--experiment", required=True, choices=list(bu.EXPERIMENTS))
    p.add_argument("--period", default="historical", choices=list(bu.PERIOD_YEARS))
    p.add_argument("--predictor-flavour", default="perfect",
                   choices=["perfect", "imperfect"])
    p.add_argument("--truth-root", required=True, type=Path)
    p.add_argument("--pred-root", required=True, type=Path)
    p.add_argument("--rankings",
                   default=Path(__file__).with_name("model_rankings.json"), type=Path)
    p.add_argument("--out-dir", default=Path("outputs/climatology"), type=Path)
    p.add_argument("--n-sample", default=3, type=int)
    p.add_argument("--pool", default=10, type=int)
    p.add_argument("--bot-pool", default=15, type=int)
    p.add_argument("--bot-exclude", default=3, type=int)
    p.add_argument("--seed", default=None, type=int)
    args = p.parse_args()

    if args.gcm not in bu.DOMAIN_GCMS[args.domain]:
        raise SystemExit(f"GCM {args.gcm!r} not valid for domain {args.domain}.")

    paths = bu.Paths(truth_root=args.truth_root, pred_root=args.pred_root)
    rankings = bu.load_rankings(args.rankings)
    ranking = bu.get_ranking(rankings, args.variable, args.experiment)

    available = bu.discover_available(
        paths, ranking,
        domain=args.domain, experiment=args.experiment,
        period_key=args.period, gcm=args.gcm,
        predictor_flavour=args.predictor_flavour,
    )
    top, bot = bu.sample_top_bottom(
        ranking, available,
        n_sample=args.n_sample, pool=args.pool,
        bot_pool=args.bot_pool, bot_exclude=args.bot_exclude,
        seed=args.seed,
    )
    print(f"[clim] sampled top: {top}")
    print(f"[clim] sampled bot: {bot}  (from bottom-{args.bot_pool}, excluding worst-{args.bot_exclude})")

    truth_series, truth_ds = bu.open_truth(paths, args.domain, args.period,
                                           args.gcm, args.variable)
    truth_clim = bu.climatology(truth_series, args.variable)
    if args.domain == "ALPS":
        truth_clim = truth_clim#.reindex(y = sorted(truth_clim.y.values)).reindex(x= sorted(truth_clim.x.values))
        truth_clim = truth_clim.transpose("y","x")
        truth_ds = truth_ds#.reindex(y = sorted(truth_ds.y.values)).reindex(x= sorted(truth_ds.x.values))
        truth_ds[args.variable] = truth_ds[args.variable].transpose("time","y","x")
    else:
        truth_clim = truth_clim#.reindex(lat = sorted(truth_clim.lat.values)).reindex(lon= sorted(truth_clim.lon.values))
        truth_clim = truth_clim.transpose("lat","lon")
        truth_ds = truth_ds#.reindex(lat = sorted(truth_ds.lat.values)).reindex(lon= sorted(truth_ds.lon.values))
        truth_ds[args.variable] = truth_ds[args.variable].transpose("time","lat","lon")
    #truth_series = truth_series.reindex(

    tops, bots = {}, {}
    for m in top:
        da = bu.open_prediction(paths, m, args.domain, args.experiment,
                                args.period, args.gcm, args.variable,
                                args.predictor_flavour)
        climatology = bu.climatology(da, args.variable)
        #da = da.reindex(
        


        if args.domain == "ALPS":
            climatology = climatology#.sel(y = truth_clim.y.values, x = truth_clim.x.values, method ='nearest')
            climatology =climatology.transpose("y","x")
        else:
            climatology = climatology#.sel(lat = truth_clim.lat.values, x = truth_clim.lon.values, method ='nearest')
            climatology =climatology.transpose("lat","lon")

            
        tops[m] = climatology

        
    for m in bot:
        da = bu.open_prediction(paths, m, args.domain, args.experiment,
                                args.period, args.gcm, args.variable,
                                args.predictor_flavour)

        climatology = bu.climatology(da, args.variable)
        

        if args.domain == "ALPS":
            climatology = climatology#.sel(y = truth_clim.y.values, x = truth_clim.x.values, method ='nearest')
            climatology =climatology.transpose("y","x")
        else:
            climatology = climatology#.sel(lat = truth_clim.lat.values, x = truth_clim.lon.values, method ='nearest')
            climatology =climatology.transpose("lat","lon")
        bots[m] = climatology


    # RMSE vs truth for every available model (for the bottom-right KDE)
    import warnings
    all_rmse: dict[str, float] = {}
    for mname in available:
        try:
            da = bu.open_prediction(paths, mname, args.domain, args.experiment,
                                    args.period, args.gcm, args.variable,
                                    args.predictor_flavour)
            mclim = bu.climatology(da, args.variable)
            if args.domain == "ALPS":
                mclim = mclim.transpose("y", "x")
            else:
                mclim = mclim.transpose("lat", "lon")
            all_rmse[mname] = float(bu.rmse(mclim, truth_clim))
        except Exception as e:
            warnings.warn(f"RMSE scan skipping {mname!r}: {e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.domain}_{args.experiment}_{args.period}_{args.gcm}"
            f"_{args.variable}_{INDEX_NAME[args.variable]}")

    fig = make_figure(truth_clim, tops, bots,
                      variable=args.variable, domain=args.domain, gcm=args.gcm,
                      experiment=args.experiment, period_key=args.period,
                      truth_ds=truth_ds, all_rmse=all_rmse)
    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    save_netcdf(args.out_dir / f"{stem}.nc", truth_clim, tops, bots, args.variable)

    print(f"[clim] wrote {args.out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()