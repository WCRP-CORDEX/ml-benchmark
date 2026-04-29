#!/usr/bin/env python
"""
plot_daily_snapshot.py
======================
Single-day comparison of downscaled fields against the ground truth, for
3 randomly-chosen "top" models (from the top ``--pool``) and 3 randomly-
chosen "bottom" models (from ranks ``[N-bot_pool, N-bot_exclude)`` — we
deliberately skip the very worst ``--bot-exclude`` models, which are
typically so bad they swamp the plot).

Figure layout
-------------
    +------+------+------+  +----------+
    | top1 | top2 | top3 |  |          |
    +------+------+------+  |  TRUTH   |
    | bot1 | bot2 | bot3 |  |          |
    +------+------+------+  +----------+
    |    colour bar     |   |   PSD    |
    +-------------------+   +----------+

* Maps use cartopy with coastlines. For the ALPS domain we use the 2-D
  lon/lat coordinates as ``pcolormesh`` inputs with a PlateCarree
  transform, which handles the rotated-pole grid correctly.
* The PSD inset is the radially-integrated 2-D power spectrum, computed
  exactly as the WCRP-CORDEX reference (``evaluation/diagnostics.py``):
  shifted 2-D FFT → |·|² → integer-radius bincount.

Day selection
-------------
``--mode`` controls which day is plotted:
    * ``auto``     : wettest day for pr, hottest for tasmax (from truth).
    * ``wettest``  : day of largest spatial max precipitation.
    * ``hottest``  : day of largest spatial max temperature.
    * ``random``   : uniform random day (use ``--seed`` for reproducibility).
    * ``date``     : explicit ``YYYY-MM-DD`` via ``--date``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

import benchmark_utils as bu


# --------------------------------------------------------------------------- #
def select_day(truth_da, mode, date_str, seed, variable):
    if mode == "date":
        if not date_str:
            raise SystemExit("--mode date requires --date YYYY-MM-DD")
        return np.datetime64(date_str), "specified"
    if mode == "random":
        return bu.safe_random_date(truth_da["time"], seed), "random"
    if mode == "auto":
        mode = "hottest" if variable == "tasmax" else "wettest"
    return bu.find_extreme_day(truth_da, mode=mode), mode


def get_member(da, member):
    for name in ("member", "members", "realization", "ensemble"):
        if name in da.dims:
            if member is None:
                return da.mean(name, keep_attrs=True)
            return da.isel({name: member})
    return da


def _stroke(lw, fg):
    import matplotlib.patheffects as pe
    return pe.withStroke(linewidth=lw, foreground=fg)


# --------------------------------------------------------------------------- #
def make_figure(
    truth_day, top_fields, bot_fields, *,
    variable, domain, gcm, experiment, period_key, date_str, selection_mode,
    truth_ds, selection_hint,
    truth_max_scalar=None, all_model_maxes=None,
):
    bu.setup_nature_style()
    if domain == "ALPS":
        truth_day = truth_day.transpose("y","x")
        truth_ds[variable] = truth_ds[variable].transpose("time","y","x")
    else:
        truth_day = truth_day.transpose("lat","lon")
        truth_ds[variable] = truth_ds[variable].transpose("time","lat","lon")
    cmap = bu.cmap_for(variable, kind="daily")

    # Shared colour range across every panel.
    stack = np.concatenate([
        truth_day.values.ravel(),
        *[f.values.ravel() for f in top_fields.values()],
        *[f.values.ravel() for f in bot_fields.values()],
    ])
    stack = stack[np.isfinite(stack)]
    if variable == "pr":
        vmin = 0.0
        # For extreme (wettest) days, push vmax toward the 99.9th percentile
        # so the heavy-rain colours actually get used. For a typical day,
        # 99th is still fine.
        q = 99.9 if selection_hint == "wettest" else 99.0
        vmax = max(float(np.nanpercentile(stack, q)), 5.0)
    else:
        vmin = float(np.nanpercentile(stack, 1))
        vmax = float(np.nanpercentile(stack, 99))

    map_crs, data_crs = bu.get_crs(domain, truth_ds)
    lonlat2d = bu.get_lonlat_2d(truth_ds) if domain == "ALPS" else None

    fig = plt.figure(figsize=(16.5, 8.2), facecolor="white")
    gs = GridSpec(
        3, 5,
        width_ratios=[1, 1, 1, 0.08, 2.0],
        height_ratios=[1, 1, 0.75],
        hspace=0.22, wspace=0.08,
        left=0.045, right=0.97, top=0.90, bottom=0.06,
    )

    # Row labels
    fig.text(0.015, 0.705, "TOP",    rotation=90, fontsize=16, fontweight="bold",
             color="#1a5e2a", va="center", ha="center")
    fig.text(0.015, 0.425, "BOTTOM", rotation=90, fontsize=16, fontweight="bold",
             color="#8b1a1a", va="center", ha="center")

    labels = list("abcdefg")

    def small_panel(row, col, field, model, rank_label, label_letter):
        ax = fig.add_subplot(gs[row, col], projection=map_crs)
        im = bu.plot_map(ax, field, cmap=cmap, vmin=vmin, vmax=vmax,
                         data_crs=data_crs,
                         lon2d=lonlat2d[0] if lonlat2d else None,
                         lat2d=lonlat2d[1] if lonlat2d else None)
        ax.text(0.02, 0.97, f"{label_letter}", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                color="white", path_effects=[_stroke(1.4, "#000")])
        ax.set_title(f"{rank_label}   {model}", fontsize=14, pad=3)
        return im

    for i, (name, f) in enumerate(top_fields.items()):
        small_panel(0, i, f, name, f"top {i+1}", labels[i])
    for i, (name, f) in enumerate(bot_fields.items()):
        small_panel(1, i, f, name, f"bot {i+1}", labels[i + 3])

    # ---- big truth panel ----
    ax_truth = fig.add_subplot(gs[0:2, 4], projection=map_crs)
    im = bu.plot_map(ax_truth, truth_day, cmap=cmap, vmin=vmin, vmax=vmax,
                     data_crs=data_crs,
                     lon2d=lonlat2d[0] if lonlat2d else None,
                     lat2d=lonlat2d[1] if lonlat2d else None)
    ax_truth.text(0.02, 0.98, labels[6], transform=ax_truth.transAxes,
                  fontsize=16, fontweight="bold", va="top", ha="left",
                  color="white", path_effects=[_stroke(1.6, "#000")])
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

    # ---- colour bar (row 2, cols 0-2) ----
    cax = fig.add_subplot(gs[2, 0:3])
    bbox = cax.get_position()
    cax.set_position([bbox.x0 + 0.025, bbox.y0 + bbox.height * 0.55,
                      bbox.width - 0.05, bbox.height * 0.22])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{bu.pretty_var(variable)} [{bu.units_for(variable)}]",
                 fontsize=15, fontweight="bold")
    cb.ax.tick_params(labelsize=13)
    cb.outline.set_linewidth(0.6)

    # ---- Max-intensity distribution inset (row 2, col 4) ----
    ax_dist = fig.add_subplot(gs[2, 4])
    _draw_max_intensity_kde(
        ax_dist, variable,
        truth_max_scalar, all_model_maxes or {},
        top_fields, bot_fields,
    )

    # ---- headline ----
    fig.suptitle(
        f"{domain} · {experiment.replace('_', ' ')} · {period_key.replace('_', ' ')}   |   "
        f"{bu.pretty_var(variable)} — {date_str}  ({selection_mode})",
        fontsize=18, fontweight="bold", y=0.975,
    )
    return fig


def _draw_max_intensity_kde(ax, variable, truth_max, all_model_maxes,
                             top_fields, bot_fields):
    """Distribution of spatial-max values on this day across all available
    models. Truth shown as a thick black vertical line; sampled top-3/
    bot-3 highlighted as coloured dots; all other models as small grey
    dots. Analogous to the CC-signal KDE.
    """
    if not all_model_maxes:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="#888")
        ax.set_axis_off()
        return

    names = list(all_model_maxes.keys())
    vals = np.asarray([all_model_maxes[n] for n in names], dtype=float)
    finite = np.isfinite(vals)
    names = [n for n, k in zip(names, finite) if k]
    vals = vals[finite]

    lo = float(min(vals.min(), truth_max if truth_max is not None else vals.min()))
    hi = float(max(vals.max(), truth_max if truth_max is not None else vals.max()))
    pad = 0.08 * max(hi - lo, 1e-6)
    lo -= pad; hi += pad
    grid = np.linspace(lo, hi, 256)
    _, dens = bu.gaussian_kde_1d(vals, grid=grid)

    fill_color = "#80cdc1" if variable == "pr" else "#f4a582"
    line_color = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, dens, color=fill_color, alpha=0.45, zorder=2)
    ax.plot(grid, dens, color=line_color, linewidth=2.0, zorder=3)

    y0 = -0.03 * float(max(dens.max(), 1e-9))
    top_set = set(top_fields); bot_set = set(bot_fields)
    for n, v in zip(names, vals):
        if n in top_set:
            c, z, s = "#1a5e2a", 6, 80
        elif n in bot_set:
            c, z, s = "#8b1a1a", 6, 80
        else:
            c, z, s = "#888888", 4, 22
        ax.scatter(v, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)

    if truth_max is not None and np.isfinite(truth_max):
        ax.axvline(truth_max, color="black", linewidth=2.4, zorder=4)

    unit = bu.units_for(variable)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    ax.set_xlabel(
        f"{('Max precipitation' if variable == 'pr' else 'Max temperature')}  "
        f"on this day  [{unit}]",
        fontsize=15, fontweight="bold",
    )
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Distribution of max intensity across models",
                 fontsize=14, pad=3, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    import matplotlib.lines as mlines
    tmax_label = (f"truth max = {truth_max:.1f}"
                  if (truth_max is not None and np.isfinite(truth_max))
                  else "truth max (n/a)")
    handles = [
        mlines.Line2D([], [], color=line_color, lw=2.0,
                      label=f"KDE of model max (n={vals.size})"),
        mlines.Line2D([], [], color="black", lw=2.4, label=tmax_label),
        mlines.Line2D([], [], color="#1a5e2a", marker="o", lw=0,
                      markersize=8, label="top-3 sampled"),
        mlines.Line2D([], [], color="#8b1a1a", marker="o", lw=0,
                      markersize=8, label="bot-3 sampled"),
        mlines.Line2D([], [], color="#888888", marker="o", lw=0,
                      markersize=5, label="other models"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              frameon=True, framealpha=0.92, ncol=1,
              handlelength=1.6, borderpad=0.3)


def _draw_psd(ax, truth_day, top_fields, bot_fields):
    """Legacy radial PSD (kept for reference / optional use)."""
    top_colors = ["#1a5e2a", "#3fa050", "#7fbf7b"]
    bot_colors = ["#8b1a1a", "#d73027", "#f46d43"]

    def _plot(field, **kw):
        k, P = bu.radial_psd(field)
        mask = (k >= 1) & np.isfinite(P) & (P > 0)
        ax.plot(k[mask], P[mask], **kw)

    _plot(truth_day, color="black", linewidth=2.2, label="truth", zorder=5)
    for i, (name, f) in enumerate(top_fields.items()):
        _plot(f, color=top_colors[i % 3], linewidth=1.2, alpha=0.95,
              label=f"top {i+1}")
    for i, (name, f) in enumerate(bot_fields.items()):
        _plot(f, color=bot_colors[i % 3], linewidth=1.2, alpha=0.95,
              linestyle="--", label=f"bot {i+1}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber  k  [pixels]", fontsize=15)
    ax.set_ylabel("Radial power $P(k)$", fontsize=15)
    ax.tick_params(labelsize=13, which="both")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.55)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.3)
    ax.set_title("Radial power spectrum", fontsize=15, pad=3, fontweight="bold")
    ax.legend(fontsize=12, loc="lower left", frameon=False, ncol=2,
              handlelength=1.6, columnspacing=0.9, borderpad=0.2)


# --------------------------------------------------------------------------- #
def save_netcdf(out_nc, truth_day, top_fields, bot_fields, variable, date_str):
    arrays = {"truth": truth_day}
    for r, (n, f) in enumerate(top_fields.items(), 1): arrays[f"top{r}__{n}"] = f
    for r, (n, f) in enumerate(bot_fields.items(), 1): arrays[f"bot{r}__{n}"] = f
    ds = xr.Dataset({k: v.reset_coords(drop=True) for k, v in arrays.items()})
    ds.attrs.update(variable=variable, date=date_str,
                    description="Single-day snapshot: truth plus 3 top + 3 bottom models.")
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
    p.add_argument("--pred-root",  required=True, type=Path)
    p.add_argument("--rankings",
                   default=Path(__file__).with_name("model_rankings.json"), type=Path)
    p.add_argument("--out-dir", default=Path("outputs/daily"), type=Path)

    p.add_argument("--mode", default="auto",
                   choices=["auto", "hottest", "wettest", "random", "date"],
                   help="Day-selection mode. 'auto' = wettest for pr / hottest for tasmax.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (used with --mode date)")
    p.add_argument("--seed", default=None, type=int,
                   help="Used by --mode random AND by the model-sampling.")
    p.add_argument("--member", default=None, type=int,
                   help="Generative-model member to plot (default: ensemble mean).")

    p.add_argument("--n-sample", default=3, type=int,
                   help="How many models to draw from each pool (default 3).")
    p.add_argument("--pool", default=10, type=int,
                   help="Top pool size from the ranking (default 10).")
    p.add_argument("--bot-pool", default=15, type=int,
                   help="Bottom pool size; sample from ranks [N-bot-pool, N-bot-exclude).")
    p.add_argument("--bot-exclude", default=3, type=int,
                   help="Exclude the N very-worst models (default 3).")
    args = p.parse_args()

    if args.gcm not in bu.DOMAIN_GCMS[args.domain]:
        raise SystemExit(f"GCM {args.gcm!r} not valid for domain {args.domain}. "
                         f"Pick one of {bu.DOMAIN_GCMS[args.domain]}.")

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
    print(f"[daily] top pool (first {args.pool}): {ranking[:args.pool]}")
    print(f"[daily] sampled top: {top}")
    print(f"[daily] sampled bot: {bot}  (from bottom-{args.bot_pool}, excluding worst-{args.bot_exclude})")

    truth, truth_ds = bu.open_truth(paths, args.domain, args.period,
                                    args.gcm, args.variable)
    day, mode = select_day(truth, args.mode, args.date, args.seed, args.variable)
    date_str = str(np.datetime_as_string(day, unit="D"))
    truth_day = truth.sel(time=day, method="nearest")
    print(f"[daily] day selected: {date_str}  ({mode})")

    top_fields, bot_fields = {}, {}
    for m in top:
        da = bu.open_prediction(paths, m, args.domain, args.experiment,
                                args.period, args.gcm, args.variable,
                                args.predictor_flavour)
        da = get_member(da.sel(time=day, method="nearest"), args.member)
        if args.domain == "ALPS":
            da = da.transpose("y","x")
        else:
            da = da.transpose("lat","lon")
        top_fields[m] = da
    for m in bot:
        da = bu.open_prediction(paths, m, args.domain, args.experiment,
                                args.period, args.gcm, args.variable,
                                args.predictor_flavour)
        da = get_member(da.sel(time=day, method="nearest"), args.member)
        if args.domain == "ALPS":
            da = da.transpose("y","x")
        else:
            da = da.transpose("lat","lon")
        bot_fields[m] = da

    # Spatial max of every available model on the selected day — cheap scalar
    # loop for the intensity KDE.
    all_model_maxes: dict[str, float] = {}
    import warnings
    for mname in available:
        try:
            da = bu.open_prediction(paths, mname, args.domain, args.experiment,
                                    args.period, args.gcm, args.variable,
                                    args.predictor_flavour)
            day_slice = get_member(da.sel(time=day, method="nearest"), args.member)
            v = np.asarray(day_slice.values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                all_model_maxes[mname] = float(v.max())
        except Exception as e:
            warnings.warn(f"max-intensity scan skipping {mname!r}: {e}")
    truth_max_scalar = float(np.nanmax(truth_day.values))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.domain}_{args.experiment}_{args.period}_{args.gcm}"
            f"_{args.variable}_{date_str}_{mode}")

    fig = make_figure(
        truth_day, top_fields, bot_fields,
        variable=args.variable, domain=args.domain, gcm=args.gcm,
        experiment=args.experiment, period_key=args.period,
        date_str=date_str, selection_mode=mode, truth_ds=truth_ds,
        selection_hint=mode,
        truth_max_scalar=truth_max_scalar,
        all_model_maxes=all_model_maxes,
    )
    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    save_netcdf(args.out_dir / f"{stem}.nc",
                truth_day, top_fields, bot_fields, args.variable, date_str)

    print(f"[daily] wrote {args.out_dir / (stem + '.png')}")
    print(f"[daily] wrote {args.out_dir / (stem + '.pdf')}")
    print(f"[daily] wrote {args.out_dir / (stem + '.nc')}")


if __name__ == "__main__":
    main()