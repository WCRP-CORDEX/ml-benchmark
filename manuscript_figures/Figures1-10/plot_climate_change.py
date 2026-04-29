#!/usr/bin/env python
"""
plot_climate_change.py
======================
Climate-change signal  =  climatology(future)  -  climatology(historical).

Layout
------
    +------+------+------+  +----------+
    | top1 | top2 | top3 |  |          |
    +------+------+------+  |  TRUTH   |
    | bot1 | bot2 | bot3 |  |          |
    +------+------+------+  +----------+
    |         colour bar                |
    +-----------------------------------+
    | KDE of model-mean Δ signals       |
    |   + truth mean as vertical line   |
    +-----------------------------------+

* Top-3 / bottom-3 are sampled from the top-``--pool`` and from
  ``[N-bot_pool, N-bot_exclude)`` of the ranking, so the bottom row shows
  "bad but still reasonable" models rather than the absolute worst.
* ALPS is rendered via ``pcolormesh(lon2d, lat2d, field, transform=PlateCarree())``
  to handle the rotated-pole curvilinear grid.
* Each map panel shows the area-mean signal (e.g. "+2.5 K") in the top-right
  and RMSE-vs-truth in the bottom-right.
* The distribution subplot shows the **distribution of model-mean area
  averages** — one number per model — smoothed as a KDE. The ground-truth
  mean is marked as a vertical line. Top-3 and bottom-3 sampled models
  appear as ticks at the bottom.

Colour schemes
--------------
* pr signal      → BrBG diverging
* tasmax signal  → YlOrRd sequential (0 → P98, typically 0-6 K)
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


def signal(hist_clim, fut_clim, relative: bool) -> xr.DataArray:
    if relative:
        denom = hist_clim.where(hist_clim > 0.1)
        out = 100.0 * (fut_clim - hist_clim) / denom
    else:
        out = fut_clim - hist_clim
    return out


def _stroke(lw, fg):
    import matplotlib.patheffects as pe
    return pe.withStroke(linewidth=lw, foreground=fg)


def _area_mean(field) -> float:
    vals = np.asarray(field.values).ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _badge_black(ax, txt, *, loc):
    """Black text on a white pill. loc ∈ {'tr', 'br'}."""
    if loc == "tr":
        x, y, va, ha = 0.97, 0.95, "top", "right"
    else:
        x, y, va, ha = 0.97, 0.05, "bottom", "right"
    ax.text(
        x, y, txt, transform=ax.transAxes,
        fontsize=15, fontweight="bold", va=va, ha=ha,
        color="black",
        bbox=dict(boxstyle="round,pad=0.22",
                  facecolor="white", edgecolor="#333",
                  linewidth=0.5, alpha=0.92),
        zorder=10,
    )


# --------------------------------------------------------------------------- #
def make_figure(
    truth_sig, tops, bots, mean_signals, *,
    variable, domain, gcm, experiment, future_period, relative, truth_ds,
):
    bu.setup_nature_style()
    cmap = bu.cmap_for(variable, kind="cc_signal")
    unit_str = "%" if (relative and variable == "pr") else bu.units_for(variable, diverging=True)

    stack = np.concatenate(
        [truth_sig.values.ravel()]
        + [f.values.ravel() for f in tops.values()]
        + [f.values.ravel() for f in bots.values()]
    )
    stack = stack[np.isfinite(stack)]
    if variable == "pr":
        vlim = float(np.nanpercentile(np.abs(stack), 98))
        vlim = max(vlim, 1.0)
        vmin, vmax = -vlim, vlim
    else:
        vmax = float(np.nanpercentile(stack, 98))
        vmax = max(vmax, 1.0)
        vmin = 0.0

    map_crs, data_crs = bu.get_crs(domain, truth_ds)
    lonlat2d = bu.get_lonlat_2d(truth_ds) if domain == "ALPS" else None

    fig = plt.figure(figsize=(16.5, 10.8), facecolor="white")
    gs = GridSpec(
        3, 5,
        width_ratios=[1, 1, 1, 0.08, 2.0],
        height_ratios=[1, 1, 0.12],
        hspace=0.32, wspace=0.08,
        left=0.045, right=0.97, top=0.93, bottom=0.07,
    )

    fig.text(0.015, 0.78, "TOP",    rotation=90, fontsize=16,
             fontweight="bold", color="#1a5e2a", va="center", ha="center")
    fig.text(0.015, 0.555, "BOTTOM", rotation=90, fontsize=16,
             fontweight="bold", color="#8b1a1a", va="center", ha="center")

    labels = list("abcdefg")

    def small_panel(row, col, field, model, rank_label, label_letter):
        ax = fig.add_subplot(gs[row, col], projection=map_crs)
        im = bu.plot_map(ax, field, cmap=cmap, vmin=vmin, vmax=vmax,
                         data_crs=data_crs,
                         lon2d=lonlat2d[0] if lonlat2d else None,
                         lat2d=lonlat2d[1] if lonlat2d else None)
        mean = _area_mean(field)
        err = bu.rmse(field, truth_sig)
        corrcoef = np.corrcoef(field.values.ravel(), truth_sig.values.ravel()).T[0,1]
        sign = "+" if mean >= 0 else ""
        ax.set_title(f"{rank_label}   {model}", fontsize=14, pad=3)
        ax.text(0.02, 0.97, label_letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                color="white", path_effects=[_stroke(1.4, "#000")])
        _badge_black(ax, f"Δ {sign}{mean:.2f} {unit_str} \n $r_s$ = {corrcoef:.2f}", loc="tr")
        _badge_black(ax, f"RMSE = {err:.2f}", loc="br")
        return im

    for i, (name, f) in enumerate(tops.items()):
        small_panel(0, i, f, name, f"top {i+1}", labels[i])
    for i, (name, f) in enumerate(bots.items()):
        small_panel(1, i, f, name, f"bot {i+1}", labels[i + 3])

    # Big truth panel
    ax_truth = fig.add_subplot(gs[0:2, 4], projection=map_crs)
    im = bu.plot_map(ax_truth, truth_sig, cmap=cmap, vmin=vmin, vmax=vmax,
                     data_crs=data_crs,
                     lon2d=lonlat2d[0] if lonlat2d else None,
                     lat2d=lonlat2d[1] if lonlat2d else None)
    truth_mean = _area_mean(truth_sig)
    sign = "+" if truth_mean >= 0 else ""
    ax_truth.set_title(f"Ground truth — {gcm}   (Δ {sign}{truth_mean:.2f} {unit_str})",
                       fontsize=16, pad=5, fontweight="bold")
    ax_truth.text(0.02, 0.98, labels[6], transform=ax_truth.transAxes,
                  fontsize=13, fontweight="bold", va="top", ha="left",
                  color="white", path_effects=[_stroke(1.6, "#000")])
    try:
        gl = ax_truth.gridlines(draw_labels=True, linewidth=0.4,
                                color="gray", alpha=0.55)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}
    except Exception:
        pass

    # Colorbar in the slim middle row
    cax = fig.add_subplot(gs[2, 0:3])
    bbox = cax.get_position()
    cax.set_position([bbox.x0 + 0.03, bbox.y0 + bbox.height * 0.2,
                      bbox.width - 0.06, bbox.height * 0.55])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"Δ {INDEX_NAME[variable]}  [{unit_str}]",
                 fontsize=14, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.6)

    # ---- KDE of model-mean signals, full width ----
    ax_kde = fig.add_subplot(gs[2, 4:5])
    _draw_mean_kde(ax_kde, truth_mean, mean_signals, tops, bots,
                   variable, unit_str)

    # Headline
    hist_years = bu.PERIOD_YEARS["historical"]
    fut_years = bu.PERIOD_YEARS[future_period]
    fig.suptitle(
        f"{domain}  ·  {experiment.replace('_', ' ')}     "
        f"Δ{INDEX_NAME[variable]}  =  {fut_years}  −  {hist_years}",
        fontsize=18, fontweight="bold", y=0.98,
    )
    return fig


# --------------------------------------------------------------------------- #
def _gaussian_kde(values: np.ndarray, grid: np.ndarray, *, bandwidth=None):
    """Plain Gaussian KDE without requiring scipy."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(grid)
    n = v.size
    std = np.std(v, ddof=1) if n > 1 else 1.0
    if std < 1e-9:
        std = max(1e-6, float(abs(np.mean(v))) * 1e-3 + 1e-6)
    # Silverman's rule
    h = bandwidth if bandwidth is not None else 1.06 * std * n ** (-1/5)
    h = max(h, 1e-6)
    z = (grid[:, None] - v[None, :]) / h
    kern = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return kern.sum(axis=1) / (n * h)


def _draw_mean_kde(ax, truth_mean, mean_signals, tops, bots, variable, unit_str):
    """KDE of model-mean Δ signals. One value per model (its area mean)."""
    import matplotlib.lines as mlines

    names = list(mean_signals.keys())
    means = np.array([mean_signals[n] for n in names], dtype=float)
    finite = np.isfinite(means)
    means = means[finite]
    names = [n for n, keep in zip(names, finite) if keep]

    if means.size == 0:
        ax.text(0.5, 0.5, "no model means", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_axis_off()
        return

    # x-range: include truth mean plus padding
    x_lo = min(means.min(), truth_mean)*0.7
    x_hi = max(means.max(), truth_mean) * 1.25
    pad = 0.08 * max(x_hi - x_lo, 1e-6)
    x_lo -= pad; x_hi += pad
    grid = np.linspace(x_lo, x_hi, 400)
    pdf = _gaussian_kde(means, grid)

    # Fill + outline
    fill_color = "#80cdc1" if variable == "pr" else "#f4a582"
    line_color = "#01665e" if variable == "pr" else "#b2182b"
    ax.fill_between(grid, 0, pdf, color=fill_color, alpha=0.45, zorder=2)
    ax.plot(grid, pdf, color=line_color, linewidth=1.8, zorder=3,
            label=f"model-mean distribution (n={means.size})")

    # Individual model markers along the x axis, coloured by pool
    y0 = -0.03 * pdf.max()
    top_set = set(tops); bot_set = set(bots)
    for n, m in zip(list(mean_signals.keys()), list(mean_signals.values())):
        if not np.isfinite(m):
            continue
        if n in top_set:
            c, z, s = "#1a5e2a", 6, 70
        elif n in bot_set:
            c, z, s = "#8b1a1a", 6, 70
        else:
            c, z, s = "#888888", 4, 22
        ax.scatter(m, y0, s=s, color=c, edgecolors="black",
                   linewidths=0.4, zorder=z, clip_on=False)

    # Truth mean as a vertical reference
    ax.axvline(truth_mean, color="black", linewidth=2.2, zorder=4,
               label=f"truth mean = {truth_mean:.2f} {unit_str}")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(bottom=min(y0 * 1.5, -1e-6))
    ax.set_xlabel(f"Area-mean  Δ {INDEX_NAME[variable]}  [{unit_str}]",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("density", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(
        f"Distribution of model-mean Δ{INDEX_NAME[variable]} across all ranked models",
        fontsize=11.5, fontweight="bold", pad=4,
    )
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        mlines.Line2D([], [], color=line_color, lw=2.2,
                      label=f"KDE of model means (n={means.size})"),
        mlines.Line2D([], [], color="black", lw=2.2, label="truth mean"),
        mlines.Line2D([], [], color="#1a5e2a", marker="o", lw=0,
                      markersize=8, label="top-3 sampled"),
        mlines.Line2D([], [], color="#8b1a1a", marker="o", lw=0,
                      markersize=8, label="bot-3 sampled"),
        mlines.Line2D([], [], color="#888888", marker="o", lw=0,
                      markersize=5, label="other models"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.92, ncol=1,
              handlelength=1.8, borderpad=0.4)


# --------------------------------------------------------------------------- #
def save_netcdf(out_nc, truth_sig, tops, bots, mean_signals, variable,
                future_period, relative):
    arrays = {"truth_signal": truth_sig}
    rmse_vals = {}
    for r, (n, f) in enumerate(tops.items(), 1):
        arrays[f"top{r}__{n}"] = f
        rmse_vals[f"top{r}__{n}"] = bu.rmse(f, truth_sig)
    for r, (n, f) in enumerate(bots.items(), 1):
        arrays[f"bot{r}__{n}"] = f
        rmse_vals[f"bot{r}__{n}"] = bu.rmse(f, truth_sig)

    ds = xr.Dataset({k: v.reset_coords(drop=True) for k, v in arrays.items()})
    for k, v in rmse_vals.items():
        ds[k].attrs["rmse_vs_truth"] = float(v)
    ds.attrs.update(
        variable=variable,
        index=INDEX_NAME[variable],
        future_period=future_period,
        future_years=bu.PERIOD_YEARS[future_period],
        historical_years=bu.PERIOD_YEARS["historical"],
        relative=str(bool(relative)),
        truth_mean=float(_area_mean(truth_sig)),
        model_mean_signals=", ".join(f"{m}={v:.4f}" for m, v in mean_signals.items()),
    )
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)


# --------------------------------------------------------------------------- #
def load_and_index(paths, model, args, period_key):
    da = bu.open_prediction(paths, model, args.domain, args.experiment,
                            period_key, args.gcm, args.variable,
                            args.predictor_flavour)
    return bu.climatology(da, args.variable)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(bu.DOMAIN_GCMS))
    p.add_argument("--gcm", required=True)
    p.add_argument("--variable", required=True, choices=["pr", "tasmax"])
    p.add_argument("--experiment", required=True, choices=list(bu.EXPERIMENTS))
    p.add_argument("--future-period", default="end_century",
                   choices=["mid_century", "end_century"])
    p.add_argument("--predictor-flavour", default="perfect",
                   choices=["perfect", "imperfect"])
    p.add_argument("--relative", action="store_true",
                   help="pr only: express Δ as % of historical")
    p.add_argument("--truth-root", required=True, type=Path)
    p.add_argument("--pred-root", required=True, type=Path)
    p.add_argument("--rankings",
                   default=Path(__file__).with_name("model_rankings.json"), type=Path)
    p.add_argument("--out-dir", default=Path("outputs/cc_signal"), type=Path)
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
        period_key=["historical", args.future_period],
        gcm=args.gcm, predictor_flavour=args.predictor_flavour,
    )
    top, bot = bu.sample_top_bottom(
        ranking, available,
        n_sample=args.n_sample, pool=args.pool,
        bot_pool=args.bot_pool, bot_exclude=args.bot_exclude,
        seed=args.seed,
    )
    print(f"[ccs] n_available = {len(available)}")
    print(f"[ccs] sampled top: {top}")
    print(f"[ccs] sampled bot: {bot}  (from bottom-{args.bot_pool}, excluding worst-{args.bot_exclude})")

    rel = args.relative and args.variable == "pr"

    # Truth signal
    truth_hist, _ = bu.open_truth(paths, args.domain, "historical",
                                  args.gcm, args.variable)
    truth_fut,  truth_ds = bu.open_truth(paths, args.domain, args.future_period,
                                         args.gcm, args.variable)
    truth_sig = signal(bu.climatology(truth_hist, args.variable),
                       bu.climatology(truth_fut,  args.variable),
                       rel)
    if args.domain == "ALPS":
        truth_sig = truth_sig.transpose("y","x")
    else:
        truth_sig = truth_sig.transpose("lat","lon")

    # For every available model we only need the *area-mean* Δ signal,
    # but for the sampled top/bot we also keep the full field for the maps.
    sampled = set(top) | set(bot)
    sampled_sigs: dict[str, xr.DataArray] = {}
    mean_signals: dict[str, float] = {}

    for m in available:
        try:
            h = load_and_index(paths, m, args, "historical")
            f = load_and_index(paths, m, args, args.future_period)
            s = signal(h, f, rel)
            if args.domain == "ALPS":
                s = s.transpose("y","x")
            else:
                s = s.transpose("lat","lon")
            mean_signals[m] = _area_mean(s)
            if m in sampled:
                sampled_sigs[m] = s
        except Exception as e:
            warnings.warn(f"Skipping {m!r}: {e}")

    tops = {m: sampled_sigs[m] for m in top if m in sampled_sigs}
    bots = {m: sampled_sigs[m] for m in bot if m in sampled_sigs}

    if len(tops) < args.n_sample or len(bots) < args.n_sample:
        raise RuntimeError(
            f"Failed to compute signals for enough sampled models; "
            f"got {len(tops)} top, {len(bots)} bot."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rel_tag = "_relpct" if rel else ""
    stem = (f"{args.domain}_{args.experiment}_{args.gcm}_{args.variable}"
            f"_delta{INDEX_NAME[args.variable]}_{args.future_period}{rel_tag}")

    fig = make_figure(truth_sig, tops, bots, mean_signals,
                      variable=args.variable, domain=args.domain, gcm=args.gcm,
                      experiment=args.experiment, future_period=args.future_period,
                      relative=rel, truth_ds=truth_ds)
    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    save_netcdf(args.out_dir / f"{stem}.nc",
                truth_sig, tops, bots, mean_signals, args.variable,
                args.future_period, rel)

    print(f"[ccs] wrote {args.out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
