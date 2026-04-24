"""
Posterior uncertainty quantification via importance-sampling reweighting.

Given an ensemble of forward-model runs {theta_i, Y(theta_i)} drawn from a prior
p(theta), compute posterior moments (mean, variance, std) of target variables Y
at daily resolution by reweighting each ensemble member with normalized
importance weights W_i derived from the likelihood L(data | theta_i).

Reference: the attached document. Uniform-prior case => w_i proportional to
L(data|theta_i). Log-sum-exp stabilization is used throughout.

The likelihood is supplied as a callable so it can be swapped to match whatever
the MCMC procedure uses.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load the target variables to calculate posterior uncertainty for
# in one ensemble member
# ---------------------------------------------------------------------------

def _target_files(
    member_path: Path,
    file_prefix: str,
    year_bracket: tuple[int, int] | None = None,
) -> list[Path]:
    """Locate per-year daily NetCDFs named <file_prefix>.<year>-01-01-00000.nc.

    If ``year_bracket`` is provided as ``(start_year, end_year)``, only files whose
    year falls within the inclusive range are returned.
    """
    pattern = re.compile(rf"^{re.escape(file_prefix)}\.(\d{{4}})-01-01-00000\.nc")

    if year_bracket is not None:
        start, end = year_bracket

        matched: list[tuple[int, Path]] = []
        for p in member_path.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.name)
            if m is None:
                continue
            year = int(m.group(1))
            if year < start or year > end:
                continue
            matched.append((year, p))

        files = [p for _, p in sorted(matched)]

    else:
        files = sorted(
            p for p in member_path.iterdir()
            if p.is_file() and pattern.match(p.name)
        )

    if not files:
        bracket_msg = f" within years {year_bracket}" if year_bracket else ""
        raise FileNotFoundError(
            f"No files matching '{file_prefix}.YYYY-01-01-0000.nc'{bracket_msg} in {member_path}"
        )
    return files


def load_target_variable(
    member: EnsembleMember,
    varname: str,
    file_prefix: str,
    year_bracket: tuple[int, int] | None = None,
) -> xr.DataArray:
    """
    Concatenate all daily output files for one member into a
    DataArray with dims (time, grid) and coords lat(grid), lon(grid).
    """
    files = _target_files(member.path, file_prefix, year_bracket)

    per_file: list[xr.DataArray] = []
    for f in files:
        with xr.open_dataset(f, decode_times=True) as ds:
            if varname not in ds:
                raise KeyError(f"Variable '{varname}' missing in {f}")
            da = xr.decode_cf(ds)[varname]
            da = da.convert_calendar("standard", use_cftime=False)

            non_time_dims = [d for d in da.dims if d != "time"]
            if len(non_time_dims) != 1:
                raise ValueError(
                    f"{varname} in {f} has dims {da.dims}; expected "
                    "exactly one time dimension and one grid dimension."
                )

            per_file.append(da.isel(lndgrid=0).load())  # load into memory before closing the file

    return xr.concat(per_file, dim="time")


# ---------------------------------------------------------------------------
# Calculate the log-likelihood value of one ensemble member
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LikelihoodSpec:
    """
    Specification for one likelihood contribution.

    Each spec contributes an additive term to the total log-likelihood for an
    ensemble member. Multiple specs are summed, assuming independence across 
    the target variables.

    Attributes
    ----------
    sim_var : simulated variable name, read from <member>/<sim_file>.
    sim_prefix : basename of the simulated NetCDF (without year or .nc).
    obs_path : absolute path to the observed NetCDF.
    obs_var : observed variable name inside obs_path.
    sigma_var : name of the observational uncertainty variable inside obs_path.
    """
    sim_var: str
    sim_prefix: str
    obs_path: Path
    obs_var: str
    sigma_var: str

# A LikelihoodFn is a numpy-only kernel: it receives three 1-D float64 arrays
# of equal length (sim, obs, sigma), already broadcast / flattened / masked /
# dtype-coerced by the wrapper, and returns a scalar ln L(data | theta_i).
# Swap this out to match whatever likelihood the MCMC procedure uses.
LikelihoodFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def gaussian_log_likelihood(
    sim: np.ndarray,
    obs: np.ndarray,
    sigma: np.ndarray,
    include_normalization: bool = False,
) -> float:
    """
    Gaussian log-likelihood kernel.

        lnL = -0.5 * sum_j [ ln(2*pi*sigma_j^2) + ((d_j - m_j)/sigma_j)^2 ]

    Inputs
    ------
    sim, obs, sigma : 1-D numpy arrays of equal length, finite, sigma > 0.
        All preprocessing is the wrapper's job; this function does no
        validation beyond what the formula requires.

    Notes
    -----
    The normalization term ln(2*pi*sigma^2) cancels in the weight
    normalization when sigma does not depend on theta, so it is omitted by
    default. Set `include_normalization=True` if sigma is itself a
    calibrated parameter.
    """
    resid = (obs - sim) / sigma
    ll = -0.5 * np.sum(resid * resid)
    if include_normalization:
        ll += -0.5 * np.sum(np.log(2.0 * np.pi * sigma * sigma))
    return float(ll)


def load_likelihood_inputs(
    member: EnsembleMember,
    spec: LikelihoodSpec,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Align (sim, obs, sigma) DataArrays aligned along the time dimension,
    resample all three arrays to 10-day means, broadcast to a common 
    shape, flatten, coerce to float64, and drop entries where any
    of sim/obs/sigma is NaN or where sigma <= 0.

    Returns three 1-D float64 numpy arrays of equal length, ready to feed
    into a `LikelihoodFn`.
    """

    with xr.open_dataset(spec.obs_path) as ds_obs:
        for v in (spec.obs_var, spec.sigma_var):
            if v not in ds_obs:
                raise KeyError(f"{v} not in {spec.obs_path}")
        obs = ds_obs[spec.obs_var].load()
        sigma = ds_obs[spec.sigma_var].load()

    # load sim on subset of years
    year_bracket = None
    if "time" in obs.dims:
        years = obs["time"].dt.year
        year_bracket = (int(years.min()), int(years.max()))

    sim = load_target_variable(member, spec.sim_var, spec.sim_prefix, year_bracket)

    # unit match between obs and sim
    if spec.sim_var in ["NEP"]:
        # gC/m^2/s -> gC/m^2/d; flip sign: it seems this file is defined with the
        # normal NEE sign convention, instead of the ALMA sign convention
        sim *= -86400

    # Align on shared coords; drop points where any is missing.
    if all("time" in da.dims for da in (sim, obs, sigma)):
        sim, obs, sigma = xr.align(sim, obs, sigma, join="inner")
        sim = sim.resample(time="10D").mean()
        obs = obs.resample(time="10D").mean()
        sigma = sigma.resample(time="10D").mean()

    sim_v, obs_v, sig_v = xr.broadcast(sim, obs, sigma)

    sim_a = np.asarray(sim_v.values, dtype=np.float64).ravel()
    obs_a = np.asarray(obs_v.values, dtype=np.float64).ravel()
    sig_a = np.asarray(sig_v.values, dtype=np.float64).ravel()

    mask = (
        np.isfinite(sim_a)
        & np.isfinite(obs_a)
        & np.isfinite(sig_a)
        & (sig_a > 0)
    )
    if not mask.any():
        raise ValueError("No valid (sim, obs, sigma) triples for likelihood.")
    return sim_a[mask], obs_a[mask], sig_a[mask]


def member_log_likelihood(
    member: EnsembleMember,
    specs: Sequence[LikelihoodSpec],
    likelihood_fn: LikelihoodFn,
) -> float:
    """
    Sum log-likelihood contributions across all specs (assumes independence).
    """
    total = 0.0
    for spec in specs:
        sim_a, obs_a, sigma_a = load_likelihood_inputs(member, spec)
        ll = likelihood_fn(sim_a, obs_a, sigma_a)
        total += ll
        logger.debug("  %s / %s: lnL=%.4f (n=%d)",
                     member.member_id, spec.sim_var, ll, sim_a.size)
    return total


# ---------------------------------------------------------------------------
# Collect all the ensemble simulation directories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleMember:
    """Single ensemble member on disk."""
    member_id: str
    path: Path


def discover_members(
    ensemble_root: Path,
    pattern: re.Pattern,
) -> list[EnsembleMember]:
    """Return every subdirectory of `ensemble_root` whose name matches `pattern`."""
    if not ensemble_root.is_dir():
        raise FileNotFoundError(f"Ensemble root not found: {ensemble_root}")
    members = [
        EnsembleMember(member_id=d.name, path=d)
        for d in sorted(ensemble_root.iterdir())
        if d.is_dir() and pattern.match(d.name)
    ]
    if not members:
        raise RuntimeError(f"No ensemble members found under {ensemble_root}")
    logger.info("Found %d ensemble members under %s", len(members), ensemble_root)
    return members


# ---------------------------------------------------------------------------
# Calculate ensemble-member weights (log-sum-exp stabilized)
# from a list of log-likelihood values
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    """Normalized importance weights and diagnostics."""
    log_likelihood: np.ndarray  # shape (N,)
    weights: np.ndarray         # shape (N,), sums to 1
    n_eff: float                # Kish's effective sample size


def compute_weights(log_likelihood: np.ndarray) -> Weights:
    """
    Normalize log-likelihoods into weights via log-sum-exp.

    Assumes uniform prior (so w_i proportional to L_i). To support a non-uniform
    prior, pass in log_likelihood + log_prior(theta_i) - log_proposal(theta_i).
    """
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if not np.all(np.isfinite(ll)):
        n_bad = int(np.sum(~np.isfinite(ll)))
        logger.warning("%d of %d log-likelihoods are non-finite; assigning zero weight.", n_bad, ll.size)
        ll = np.where(np.isfinite(ll), ll, -np.inf)

    ll_max = np.max(ll)
    if not np.isfinite(ll_max):
        raise RuntimeError("All log-likelihoods are -inf; cannot form weights.")

    # Regularization:
    # exp(ll_i) overflows to inf or underflows to 0 for large/small ll_i
    # (log-likelihoods can be e.g. −10⁶).
    # Factor out the maximum: ll_i - ll_max ≤ 0 for all i, so exp(ll_i - ll_max) ∈ (0, 1]
    # — no overflow possible. The largest term becomes exp(0) = 1, anchoring the sum
    # away from zero — no underflow for the dominant members.
    w = np.exp(ll - ll_max)
    w_sum = w.sum()
    W = w / w_sum
    n_eff = 1.0 / np.sum(W ** 2)

    logger.info(
        "Weights: N=%d, N_eff=%.2f (%.1f%%), max W=%.3g",
        W.size, n_eff, 100.0 * n_eff / W.size, W.max(),
    )
    if n_eff < 50:
        logger.warning("N_eff=%.1f is below 50; posterior is dominated by few members.", n_eff)
    return Weights(log_likelihood=ll, weights=W, n_eff=float(n_eff))


# ---------------------------------------------------------------------------
# Weighted posterior moments over the ensemble
# ---------------------------------------------------------------------------

def weighted_moments(
    stacked: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted mean and variance along axis 0 (ensemble axis).

    stacked : (N, time, grid)
    weights : (N,), sums to 1

    Variance uses the plug-in estimator E[Y^2] - (E[Y])^2, which matches the
    formula in the document. NaNs in `stacked` are handled per-cell: weights
    are renormalized over the non-NaN members at each (time, grid) cell.
    """
    if stacked.shape[0] != weights.shape[0]:
        raise ValueError("Ensemble axis of `stacked` must match `weights`.")

    w = weights.reshape((-1,) + (1,) * (stacked.ndim - 1))  # (N,1,1)
    mask = np.isfinite(stacked)

    # Per-cell weight sums (handles NaNs).
    wsum = np.sum(np.where(mask, w, 0.0), axis=0)
    valid = wsum > 0

    # Weighted mean.
    num = np.sum(np.where(mask, w * stacked, 0.0), axis=0)
    mean = np.where(valid, num / np.where(valid, wsum, 1.0), np.nan)

    # Weighted second moment, then variance.
    num2 = np.sum(np.where(mask, w * stacked ** 2, 0.0), axis=0)
    ex2 = np.where(valid, num2 / np.where(valid, wsum, 1.0), np.nan)
    var = ex2 - mean ** 2
    var = np.where(var < 0, 0.0, var)  # clamp tiny negative numerical noise

    return mean, var


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class PosteriorUCConfig:
    output_path: Path
    ensemble_root: Path
    target_vars: Sequence[str]          # e.g. ["NEP", "EFLX_LH_TOT", "FSH"]
    target_years: tuple[int, int]       # e.g. (1997, 2014)
    target_file_prefix: str             # e.g. "daily_output"
    likelihood_specs: Sequence[LikelihoodSpec]
    likelihood_fn: LikelihoodFn = gaussian_log_likelihood
    include_mean: bool = True
    member_pattern: re.Pattern = re.compile(r"^g\d+$")


def compute_posterior_uncertainty(cfg: PosteriorUCConfig) -> xr.Dataset:
    members = discover_members(cfg.ensemble_root, cfg.member_pattern)

    # --- 1) Log-likelihood per member ---
    logger.info("Computing log-likelihood for %d members...", len(members))
    lls = np.empty(len(members), dtype=np.float64)
    for i, m in enumerate(members):
        lls[i] = member_log_likelihood(m, cfg.likelihood_specs, cfg.likelihood_fn)
        logger.info("  [%d/%d] %s: lnL=%.4f", i + 1, len(members), m.member_id, lls[i])

    # --- 2) Normalize to posterior weights ---
    weights = compute_weights(lls)

    # --- 3) Posterior moments per target variable ---
    out = xr.Dataset()
    out.attrs["description"] = (
        "Posterior uncertainty of target variables via importance-sampling "
        "reweighting. Weights derived from Gaussian-like likelihood; see "
        "`log_likelihood` and `weights` variables for per-member diagnostics."
    )
    out.attrs["n_members"] = len(members)
    out.attrs["n_effective"] = weights.n_eff

    # Per-member diagnostics (1-D along 'member').
    member_ids = np.array([m.member_id for m in members])
    out["member"] = ("member", member_ids)
    out["log_likelihood"] = ("member", weights.log_likelihood)
    out["weight"] = ("member", weights.weights)

    for varname in cfg.target_vars:
        logger.info("Loading target variable '%s' across ensemble...", varname)
        per_member = []
        ref_da = None
        for m in members:
            da = load_target_variable(m, varname, cfg.target_file_prefix, cfg.target_years)
            if ref_da is None:
                ref_da = da
            else:
                # Align to the reference so concatenation is safe.
                da = da.reindex_like(ref_da)
            per_member.append(da.values)

        stacked = np.stack(per_member, axis=0)  # (N, time, grid)
        mean, var = weighted_moments(stacked, weights.weights)

        # --- 4) (time, grid) ---
        da_uc = xr.DataArray(
            np.sqrt(var).reshape(-1,1),
            dims=("time", "lndgrid"),
            coords={
                "time": ref_da["time"],
                "lndgrid": [1],
            },
            name=f"{varname}_uc",
            attrs={"long_name": f"Posterior std. dev of {varname}",
                   "units": ref_da.attrs.get("units", "")}
        )
        out[da_uc.name] = da_uc

        if cfg.include_mean:
            da_mean = xr.DataArray(
                mean.reshape(-1,1),
                dims=("time", "lndgrid"),
                coords={
                    "time": ref_da["time"],
                    "lndgrid": [1],
                },
                name=f"{varname}_mean",
                attrs={"long_name": f"Posterior mean of {varname}",
                       "units": ref_da.attrs.get("units", "")},
            )

            out[da_mean.name] = da_mean

    # --- 5) Write ---
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(cfg.output_path, encoding=encoding)
    logger.info("Wrote %s", cfg.output_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Posterior uncertainty via importance-sampling reweighting.",
    )
    p.add_argument("--ensemble-root", type=Path, required=True,
                   help="Folder containing gXXXXX subdirectories.")
    p.add_argument("--target-vars", nargs="+", required=True,
                   help="Target variable names to propagate (e.g. GPP NEE).")
    p.add_argument("--target-years", nargs=2, type=int, metavar=("START", "END"), required=True,
                   help="Target time period defined by first and last year (e.g. 1997 2014)")
    p.add_argument("--target-file-prefix", required=True,
                   help="Prefix of per-year NetCDFs: <prefix>.<year>-01-01-00000.nc.")
    p.add_argument("--obs-vars", nargs="+", required=True,
                   help="Observed variable names corresponding to the target variable names.")    
    p.add_argument("--obs-file", required=True,
                   help="Full path to the observational data")
    p.add_argument("--output", type=Path, required=True,
                   help="Full path to the output file.")
    p.add_argument("--no-mean", action="store_true",
                   help="Skip writing <VAR>_mean.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    spec_list = [
        LikelihoodSpec(sim_var=sim, sim_prefix=args.target_file_prefix,
                       obs_path=Path(args.obs_file), obs_var=f"{obs}_daily", 
                       sigma_var=f'{obs}_uc_daily') \
        for (sim,obs) in zip(args.target_vars, args.obs_vars)
    ]

    cfg = PosteriorUCConfig(
        ensemble_root=args.ensemble_root,
        target_vars=args.target_vars,
        target_file_prefix=args.target_file_prefix,
        target_years=args.target_years,
        likelihood_specs=spec_list,
        likelihood_fn=gaussian_log_likelihood,
        output_path=args.output,
        include_mean=not args.no_mean,
    )
    compute_posterior_uncertainty(cfg)


if __name__ == "__main__":
    """
    Example usage

    python postprocess_uc.py --ensemble-root ${SHARDIR}/../zdr/CalLMIP/e3sm_run/UQ/ensembles/20260311_DK-Sor_ICB20TRCNPRDCTCBC_landuse \
        --target-vars NEP EFLX_LH_TOT FSH --target-years 1997 2014 --target-file-prefix 20260311_DK-Sor_ICB20TRCNPRDCTCBC_landuse.elm.h1 \
        --obs-vars NEE Qle Qh --obs-file ../data/DK-Sor/DK-Sor_daily_aggregated_1997-2013_FLUXNET2015_Flux.nc \
        --output ../output/DK-Sor/posterior_uncertainty.nc
    """
    main()