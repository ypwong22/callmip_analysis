"""Diagnose error distribution (Gaussian) and autocorrelation for CalLMIP MCMC outputs."""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

var_pairs = [('NEP', 'NEE'), ('EFLX_LH_TOT', 'Qle'), ('FSH', 'Qh')]
fig, axes = plt.subplots(len(var_pairs), 4, figsize=(20, 4 * len(var_pairs)))
fig.suptitle('Error Diagnostics: Gaussian Assumption & Autocorrelation', fontsize=14, fontweight='bold')

for row, (var_sim, var_obs) in enumerate(var_pairs):

    # ── Load sim (23936 MCMC samples × 657 10-day periods) ───────────────────
    sim_file = os.path.join(os.environ['SHARDIR'], '../zdr', 'CalLMIP',
                            f'chain_output_20260311_DK-Sor_ICB20TR_{var_sim}.csv')
    sim_data = np.loadtxt(sim_file, delimiter=',')   # (23936, 657)
    if var_sim == 'NEP':
        sim_data *= -1

    # ── Load obs, resample to 10D ─────────────────────────────────────────────
    obs_file = "../data/DK-Sor/DK-Sor_daily_aggregated_1997-2013_FLUXNET2015_Flux.nc"
    ds_obs = xr.open_dataset(obs_file).squeeze(drop=True)
    obs_vals = ds_obs[f'{var_obs}_daily'].resample(time='10D').mean().values.astype(float)
    ds_obs.close()

    # ── Align: obs is shorter; both start 1997-01-01 with same 10D grid ───────
    n = len(obs_vals)                                 # ~620 periods
    assert n <= sim_data.shape[1], "obs longer than sim — check time alignment"

    # Posterior summary at each obs time step
    sim_sub = sim_data[:, :n]                         # (23936, n)
    sim_one = sim_sub[0, :] # .mean(axis=0)          # (n,)  — posterior mean prediction

    # Errors and valid mask (obs NaN → skip)
    errors = sim_one - obs_vals
    valid = np.isfinite(errors)
    ev = errors[valid]                                # clean 1-D error series
    t  = np.where(valid)[0]                           # corresponding time indices

    # Posterior 5–95% interval at valid obs points (single percentile call)
    q05, q95 = np.percentile(sim_sub[:, valid], [5, 95], axis=0)

    # ── Panel 1: Residual time series with posterior uncertainty ──────────────
    ax = axes[row, 0]
    ax.fill_between(t, q05 - obs_vals[valid], q95 - obs_vals[valid],
                    alpha=0.25, color='steelblue', label='5–95% posterior')
    ax.plot(t, ev, color='steelblue', lw=0.8, label='Posterior mean error')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'{var_sim} / {var_obs}', fontweight='bold')
    ax.set_xlabel('10-day period index')
    ax.set_ylabel('Error (sim − obs)')
    ax.legend(fontsize=7)

    # ── Panel 2: Histogram + fitted normal density ────────────────────────────
    ax = axes[row, 1]
    mu, sigma = ev.mean(), ev.std(ddof=1)
    ax.hist(ev, bins=40, density=True, alpha=0.6, color='steelblue', label='Residuals')
    x = np.linspace(ev.min(), ev.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
            label=f'N(μ={mu:.3g}, σ={sigma:.3g})')
    stat_dp, p_dp = stats.normaltest(ev)   # D'Agostino–Pearson (valid for any n)
    ax.set_title(f"Histogram\nD'Agostino–Pearson: p = {p_dp:.3f}")
    ax.set_xlabel('Error')
    ax.legend(fontsize=7)

    # ── Panel 3: Normal QQ plot ───────────────────────────────────────────────
    ax = axes[row, 2]
    (osm, osr), (slope, intercept, r) = stats.probplot(ev)
    ax.scatter(osm, osr, s=6, alpha=0.5, color='steelblue')
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
    ax.set_title(f'QQ Plot  (r = {r:.4f})')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')

    # ── Panel 4: ACF + Ljung–Box test ─────────────────────────────────────────
    ax = axes[row, 3]
    plot_acf(ev, ax=ax, lags=40, alpha=0.05, title='')
    lb = acorr_ljungbox(ev, lags=[10, 20], return_df=True)
    p10, p20 = lb['lb_pvalue'].values
    ax.set_title(f'ACF (95% CI)\nLjung–Box: p(lag 10)={p10:.3f},  p(lag 20)={p20:.3f}')
    ax.set_xlabel('Lag (10-day periods)')

plt.tight_layout()
out_fig = '../output/DK-Sor/error_diagnostics_gaussian_acf.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved: {out_fig}")
