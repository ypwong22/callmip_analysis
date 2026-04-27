"""
Plot calibration/validation output from four netCDF files:
  *Cal_Prior.nc, *Cal_Posterior.nc, *Val_Prior.nc, *Val_Posterior.nc

Each variable gets one subplot. Prior (dashed) and Posterior (solid) are
overlaid. Observations (green, where available in obs_file) cover the Cal
period. Uncertainty intervals (_uc vars) are shown as ±1σ shading.
All series are smoothed with a 10-day rolling average.
A vertical line separates the Cal and Val periods.
"""

import glob
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

# ── configurations ────────────────────────────────────────────────────

cal_prior_path     = "../output/DK-Sor/E3SM.v3.0.2_Expt1_DK-Sor_Cal_Prior.nc"
cal_post_path      = "../output/DK-Sor/E3SM.v3.0.2_Expt1_DK-Sor_Cal_Posterior.nc"
val_prior_path     = "../output/DK-Sor/E3SM.v3.0.2_Expt1_DK-Sor_Val_Prior.nc"
val_post_path      = "../output/DK-Sor/E3SM.v3.0.2_Expt1_DK-Sor_Val_Posterior.nc"
out_path           = "../output/DK-Sor/postprocess_timeseries.png"
obs_file           = "../data/DK-Sor/DK-Sor_daily_aggregated_1997-2013_FLUXNET2015_Flux.nc"

KG_M2_S_TO_GC_M2_D = 1000.0 * 86400.0   # 8.64e7

# ── load & squeeze lat/lon ────────────────────────────────────────────────────

ds_cal_prior = xr.open_dataset(cal_prior_path, use_cftime=True).squeeze(["lat", "lon"], drop=True)
ds_cal_post  = xr.open_dataset(cal_post_path, use_cftime=True).squeeze(["lat", "lon"], drop=True)
ds_val_prior = xr.open_dataset(val_prior_path, use_cftime=True).squeeze(["lat", "lon"], drop=True)
ds_val_post  = xr.open_dataset(val_post_path, use_cftime=True).squeeze(["lat", "lon"], drop=True)
ds_obs       = xr.open_dataset(obs_file).squeeze(drop=True)


# ── variable lists ────────────────────────────────────────────────────────────

all_vars = [v for v in ds_cal_post.data_vars]
data_vars = [v for v in all_vars if not v.endswith("_uc")]
uc_vars   = {v.removesuffix("_uc") for v in all_vars if v.endswith("_uc")}


# ── time helpers ──────────────────────────────────────────────────────────────

def cftime_to_pandas(ct_array):
    """Convert cftime NoLeap array → pandas DatetimeIndex (ignores leap-day absence)."""
    return pd.DatetimeIndex(
        [pd.Timestamp(t.year, t.month, t.day) for t in ct_array]
    )


cal_time = cftime_to_pandas(ds_cal_prior.time.values)
val_time = cftime_to_pandas(ds_val_prior.time.values)
obs_time = pd.DatetimeIndex(ds_obs.time.values)

# boundary between Cal and Val periods
cal_end   = cal_time[-1]
val_start = val_time[0]
boundary  = cal_end + (val_start - cal_end) / 2   # midpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling10(time_idx, values, window=10):
    """Return (time_idx, smoothed_values) via a centred 10-day rolling mean."""
    s = pd.Series(values, index=time_idx)
    smoothed = s.rolling(window, center=True, min_periods=1).mean()
    return smoothed.index, smoothed.values


def get_series(ds_cal, ds_val, varname):
    cal_arr = ds_cal[varname].values
    val_arr = ds_val[varname].values
    combined = np.concatenate([cal_arr, val_arr])
    time_idx = cal_time.append(val_time)
    return time_idx, combined


def maybe_convert(values, units):
    """Convert kg/m2/s → gC/m2/d; leave everything else unchanged."""
    norm = units.replace(" ", "").lower()
    if norm in ("kg/m2/s", "kgm-2s-1"):
        return values * KG_M2_S_TO_GC_M2_D, "gC/m2/d"
    return values, units


# ── plot ──────────────────────────────────────────────────────────────────────

ncols = 2
nrows = math.ceil(len(data_vars) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 3.5 * nrows),
                         sharex=False, constrained_layout=True)
axes_flat = axes.flatten()

COLOR_PRIOR = "#4477AA"      # blue
COLOR_POST  = "#EE6677"      # red
COLOR_OBS   = "#228833"      # green

for idx, var in enumerate(data_vars):
    ax = axes_flat[idx]

    units = ds_cal_prior[var].attrs.get("units", "")
    long  = ds_cal_prior[var].attrs.get("long_name", var)

    # ── prior ──────────────────────────────────────────────────────────────
    t_prior, y_prior = get_series(ds_cal_prior, ds_val_prior, var)
    y_prior, plot_units = maybe_convert(y_prior, units)
    t_prior, y_prior = rolling10(t_prior, y_prior)
    ax.plot(t_prior, y_prior, color=COLOR_PRIOR, lw=0.8, ls="--",
            alpha=0.85, label="Prior")

    # ── posterior + optional uncertainty ──────────────────────────────────
    t_post, y_post = get_series(ds_cal_post, ds_val_post, var)
    y_post, _ = maybe_convert(y_post, units)
    t_post, y_post = rolling10(t_post, y_post)
    ax.plot(t_post, y_post, color=COLOR_POST, lw=1.0, ls="-",
            alpha=0.9, label="Posterior")

    if var in uc_vars:
        _, y_uc = get_series(ds_cal_post, ds_val_post, f"{var}_uc")
        y_uc, _ = maybe_convert(y_uc, units)
        _, y_uc = rolling10(t_post, y_uc)
        ax.fill_between(t_post, y_post - 2*y_uc, y_post + 2*y_uc,
                        color=COLOR_POST, alpha=0.20, linewidth=0)
        print(var, np.mean(y_post - 2*y_uc), np.mean(y_post + 2*y_uc))

    # ── observations ──────────────────────────────────────────────────────
    obs_varname = f"{var}_daily"
    if obs_varname in ds_obs:
        obs_vals = ds_obs[obs_varname].values.astype(float)
        if obs_varname == "NEE_daily":
            obs_vals *= -1
        valid = np.isfinite(obs_vals)
        t_obs, y_obs = rolling10(obs_time[valid], obs_vals[valid])
        ax.plot(t_obs, y_obs, color=COLOR_OBS, lw=1.0, ls="-",
                alpha=0.85, label="Obs")

        obs_uc_varname = f"{var}_uc_daily"
        if obs_uc_varname in ds_obs:
            obs_uc_vals = ds_obs[obs_uc_varname].values.astype(float)
            _, y_obs_uc = rolling10(obs_time[valid], obs_uc_vals[valid])
            ax.fill_between(t_obs, y_obs - y_obs_uc, y_obs + y_obs_uc,
                            color=COLOR_OBS, alpha=0.20, linewidth=0)

    # ── Cal / Val boundary ────────────────────────────────────────────────
    ax.axvline(boundary, color="black", lw=1.2, ls=":", alpha=0.7)

    # ── labels & formatting ───────────────────────────────────────────────
    ax.set_title(f"{var}\n{long}", fontsize=8, pad=3)
    ax.set_ylabel(plot_units, fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.text(0.15, 0.97, "Cal", transform=ax.transAxes,
            fontsize=7, va="top", ha="center", color="dimgray")
    ax.text(0.93, 0.97, "Val", transform=ax.transAxes,
            fontsize=7, va="top", ha="center", color="dimgray")

    if idx == 0:
        ax.legend(fontsize=7, loc="lower left", framealpha=0.6)


# ── hide unused panels ────────────────────────────────────────────────────────
for ax in axes_flat[len(data_vars):]:
    ax.set_visible(False)

# ── figure-level annotations ──────────────────────────────────────────────────
site_name = cal_prior_path.split("_Cal_")[0]
fig.suptitle(f"{site_name}  —  Prior vs Posterior time series\n"
             f"(dotted line = Cal/Val boundary, "
             f"±2σ shading where available)",
             fontsize=10, y=1.03)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)
