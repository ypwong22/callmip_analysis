#!/usr/bin/env python3
"""
postprocess_output.py
---------------------
Translate selected variables from a list of E3SM Land Model (ELM) history
files into the ALMA netCDF I/O standard and write a single output file with
all input files concatenated along the time dimension.

Variable correspondence
-----------------------
NEE = NEP
Qle = EFLX_LH_TOT
Qh = FSH
Ground heat flux = FGR
GPP = GPP
Reco = ER
Transpiration = QVEGT
Bare soil evaporation = QSOIL
Land surface temperature = derived from FIRE (Stefan-Boltzmann)
Total column soil moisture = H2OSOI or SOILLIQ
LAI = TLAI
Total aboveground biomass = TOTVEGC_ABG
Total soil carbon = TOTSOMC + TOTLITC

Source file assumptions
-----------------------
- Single grid cell (lndgrid == 1).
- Time dimension is unlimited.
- Vertical soil dimension is `levgrnd` (or `levsoi`).

Output
------
ALMA-compliant file with dimensions (lat=1, lon=1, time=N) for every
translated variable. Variable names, units, long_names, and standard_names
follow the ALMA table provided.

Usage (as a module)
-------------------
    from postprocess_output import translate_many
    translate_many(["file1.nc", "file2.nc"], "output.nc")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import cftime
import numpy as np
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_WATER = 1000.0  # kg/m^3, used to convert volumetric soil moisture to kg/m^2
STB = 5.67e-8       # W/m^2/K^4, Stefan-Boltzmann constant

# ---------------------------------------------------------------------------
# Variable mapping definition
# ---------------------------------------------------------------------------
@dataclass
class AlmaVar:
    """Specification for one ALMA output variable."""
    alma_name: str            # short_name_alma
    cmip_name: str            # short_name_cmip (kept as attribute)
    standard_name: str
    long_name: str
    units: str
    # Function: (src_dataset, h0_dataset_or_None) -> 1-D numpy array along time
    compute: Callable[[Dataset, Optional[Dataset]], np.ndarray]
    # Optional notes appended to the comment attribute
    comment: str = ""


def _read_1d(src: Dataset, name: str) -> np.ndarray:
    """Read a (time, lndgrid) variable and return it as a 1-D (time,) array."""
    if name not in src.variables:
        raise KeyError(f"Required source variable '{name}' not found.")
    v = src.variables[name][:]
    return np.asarray(v).reshape(-1)


def _read_soil_profile(src: Dataset, name: str) -> np.ndarray:
    """Read a (time, levgrnd, lndgrid) variable as (time, levgrnd)."""
    if name not in src.variables:
        raise KeyError(f"Required source variable '{name}' not found.")
    v = np.asarray(src.variables[name][:])
    # Expected shape: (time, levgrnd, lndgrid=1) -> squeeze last axis
    if v.ndim == 3:
        v = v[..., 0]
    return v


# --- Computation functions for each ALMA variable --------------------------
# All functions share the signature (src, h0=None) for consistency.

def _comp_nee(src, h0=None):
    # ELM NEP: positive = sink (atm -> land). ALMA: positive = downward (atm -> land).
    # Convert gC/m^2/s -> kg/m^2/s.
    return _read_1d(src, "NEP") * 1.0e-3


def _comp_qle(src, h0=None):
    return _read_1d(src, "EFLX_LH_TOT")


def _comp_qh(src, h0=None):
    return _read_1d(src, "FSH")


def _comp_qg(src, h0=None):
    return _read_1d(src, "FGR")


def _comp_gpp(src, h0=None):
    # gC/m^2/s -> kg/m^2/s; same kg-based convention as NEE.
    return _read_1d(src, "GPP") * 1.0e-3


def _comp_reco(src, h0=None):
    # gC/m^2/s -> kg/m^2/s.
    return _read_1d(src, "ER") * 1.0e-3


def _comp_transp(src, h0=None):
    # QVEGT is mm/s, equivalent to kg/m^2/s.
    return _read_1d(src, "QVEGT")


def _comp_bare_soil_evap(src, h0=None):
    return _read_1d(src, "QSOIL")


def _comp_lst(src, h0=None):
    return (_read_1d(src, "FIRE") / (0.97 * STB)) ** (1 / 4)


def _comp_soil_moisture_total(src, h0=None):
    """
    Total column soil moisture [kg/m^2].

    Preferred: H2OSOI (m^3/m^3) * DZSOI (m) * rho_water, summed over levgrnd.
    Backup: sum SOILLIQ over levgrnd (already kg/m^2 per layer).
    """
    if "H2OSOI" in src.variables:
        h2o = _read_soil_profile(src, "H2OSOI")   # (time, levgrnd) [m3/m3]
        dzsoi = np.diff(np.array([
            0, 0.0175, 0.0451, 0.0906, 0.1655, 0.2891, 0.4929, 0.8289,
            1.3828, 2.2961, 3.8019, 6.2845, 10.3775, 17.1259, 28.2520, 42.1032
        ]))
        n = min(h2o.shape[1], dzsoi.shape[0])
        h2o = np.where(h2o[:, :n] > 1e3, np.nan, h2o[:, :n])
        per_layer = h2o * dzsoi[:n][np.newaxis, :] * RHO_WATER
        return np.nansum(per_layer, axis=1)

    if "SOILLIQ" in src.variables:
        prof = _read_soil_profile(src, "SOILLIQ")
        return np.nansum(np.where(prof > 1e3, np.nan, prof), axis=1)

    raise KeyError("Neither SOILLIQ nor H2OSOI present for soil moisture.")


def _comp_lai(src, h0=None):
    return _read_1d(src, "TLAI")


def _comp_agb(src, h0=None):
    # gC/m2 => kgC/m2
    return _read_1d(src, "TOTVEGC_ABG") * 1e-3


def _comp_soil_c(src, h0=None):
    # gC/m2 => kgC/m2
    som = _read_1d(src, "TOTSOMC")
    lit = _read_1d(src, "TOTLITC") if "TOTLITC" in src.variables else 0.0
    return (som + lit) * 1e-3


# --- ALMA variable registry ------------------------------------------------
ALMA_VARS = [
    AlmaVar("NEE", "nep",
            "surface_net_downward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_all_land_processes_excluding_anthropogenic_land_use_change",
            "Net Ecosystem Exchange", "kg/m2/s", _comp_nee,
            comment="Sign flipped from ELM (source-positive) to ALMA (downward-positive)."),
    AlmaVar("Qle", "hfls",
            "surface_upward_latent_heat_flux",
            "Latent heat flux", "W/m2", _comp_qle),
    AlmaVar("Qh", "hfss",
            "surface_upward_sensible_heat_flux",
            "Sensible heat flux", "W/m2", _comp_qh),
    AlmaVar("Qg", "hfds",
            "surface_downward_heat_flux",
            "Ground heat flux", "W/m2", _comp_qg),
    AlmaVar("GPP", "gpp",
            "gross_primary_productivity_of_biomass_expressed_as_carbon",
            "Gross primary production", "kg/m2/s", _comp_gpp),
    AlmaVar("Reco", "reco",
            "surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_ecosystem_respiration",
            "Total ecosystem respiration", "kg/m2/s", _comp_reco),
    AlmaVar("TVeg", "tran",
            "Transpiration",
            "Vegetation transpiration", "kg/m2/s", _comp_transp),
    AlmaVar("ESoil", "es",
            "liquid_water_evaporation_flux_from_soil",
            "Bare soil evaporation", "kg/m2/s", _comp_bare_soil_evap),
    AlmaVar("RadT", "tr",
            "surface_radiative_temperature",
            "Surface Radiative Temperature", "K", _comp_lst,
            comment="Derived from FIRE upwelling longwave via Stefan-Boltzmann."),
    AlmaVar("SoilMoist", "mrso",
            "mass_content_of_water_in_soil",
            "Total column soil moisture", "kg/m2", _comp_soil_moisture_total,
            comment="Sum of H2OSOI over all soil layers; falls back to SOILLIQ."),
    AlmaVar("LAI", "lai",
            "leaf_area_index",
            "Leaf area index", "1", _comp_lai),
    AlmaVar("AGB", "cVegAbove",
            "aboveground_vegetation_carbon_content",
            "Total aboveground biomass carbon", "kg/m2", _comp_agb),
    AlmaVar("TotSoilCarb", "cSoil",
            "soil_carbon_content",
            "Carbon Mass in Soil Pool", "kg/m2", _comp_soil_c,
            comment="TOTSOMC + TOTLITC."),
]


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def _process_file(input_path: str) -> dict:
    """
    Open one ELM history file and compute all ALMA variables.

    Returns a dict with keys:
        time_vals, time_units, time_calendar, lat_vals, lon_vals,
        data (dict alma_name -> 1-D array), skipped (list of alma_names).
    """
    src = Dataset(input_path, "r")
    try:
        time_var = src.variables["time"]
        result = {
            "time_vals":    np.asarray(time_var[:]),
            "time_units":   getattr(time_var, "units", "days since 1850-01-01 00:00:00"),
            "time_calendar": getattr(time_var, "calendar", "noleap"),
            "lat_vals":     np.asarray(src.variables["lat"][:]).reshape(-1),
            "lon_vals":     np.asarray(src.variables["lon"][:]).reshape(-1),
            "data":   {},
            "skipped": [],
        }
        for spec in ALMA_VARS:
            try:
                result["data"][spec.alma_name] = spec.compute(src, None)
            except KeyError as e:
                print(f"[skip] {spec.alma_name} in {os.path.basename(input_path)}: {e}",
                      file=sys.stderr)
                result["skipped"].append(spec.alma_name)
        return result
    finally:
        src.close()


def _write_output(
    output_path: str,
    time_vals: np.ndarray,
    time_units: str,
    time_calendar: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    var_arrays: dict[str, np.ndarray],
    skipped: set[str],
    n_sources: int,
) -> None:
    """Write the merged ALMA-format output file."""
    dst = Dataset(output_path, "w", format="NETCDF4_CLASSIC")
    try:
        dst.createDimension("lat", 1)
        dst.createDimension("lon", 1)
        dst.createDimension("time", None)

        lat_v = dst.createVariable("lat", "f4", ("lat",))
        lat_v.units = "degrees_north"
        lat_v.long_name = "latitude"
        lat_v.standard_name = "latitude"
        lat_v[:] = lat_vals

        lon_v = dst.createVariable("lon", "f4", ("lon",))
        lon_v.units = "degrees_east"
        lon_v.long_name = "longitude"
        lon_v.standard_name = "longitude"
        lon_v[:] = lon_vals

        time_v = dst.createVariable("time", "f8", ("time",))
        time_v.units = time_units
        time_v.calendar = time_calendar
        time_v.long_name = "time"
        time_v.standard_name = "time"
        time_v[:] = time_vals

        written = []
        for spec in ALMA_VARS:
            if spec.alma_name not in var_arrays:
                continue
            arr = np.asarray(var_arrays[spec.alma_name], dtype=np.float32).reshape(-1, 1, 1)
            v = dst.createVariable(
                spec.alma_name, "f4", ("time", "lat", "lon"),
                fill_value=np.float32(1.0e36),
                zlib=True, complevel=4,
            )
            v.units = spec.units
            v.long_name = spec.long_name
            v.standard_name = spec.standard_name
            v.cmip_short_name = spec.cmip_name
            v.cell_methods = "time: mean"
            if spec.comment:
                v.comment = spec.comment
            v[:] = np.where(np.isnan(arr), 1.0e36, arr)
            written.append(spec.alma_name)

        dst.title = "ALMA-format translation of ELM history output"
        dst.Conventions = "CF-1.7"
        dst.source = f"Translated from {n_sources} ELM history file(s)"
        dst.source_format = "ALMA netCDF I/O standard"
        dst.history = f"Created by postprocess_output.py from {n_sources} file(s)"
        if skipped:
            dst.skipped_variables = ", ".join(sorted(skipped))

        print(f"Wrote {len(written)} variables: {', '.join(written)}")
        if skipped:
            print(f"Skipped {len(skipped)} variables: {', '.join(sorted(skipped))}",
                  file=sys.stderr)
    finally:
        dst.close()


def translate_many(input_paths: list[str], output_path: str) -> None:
    """
    Translate a list of ELM history files to ALMA format.

    All input files are concatenated along the time dimension into a single
    output file. Coordinate metadata (lat, lon, time units/calendar) is taken
    from the first file. Variables missing from any file are skipped globally.

    Parameters
    ----------
    input_paths : list of str
        Ordered list of ELM history (.nc) files to process.
    output_path : str
        Destination path for the merged ALMA-format output file.
    """
    if not input_paths:
        raise ValueError("input_paths must not be empty.")

    for path in input_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    file_results = [_process_file(p) for p in input_paths]

    # Coordinate metadata from the first file
    first = file_results[0]
    time_units    = first["time_units"]
    time_calendar = first["time_calendar"]
    lat_vals      = first["lat_vals"]
    lon_vals      = first["lon_vals"]

    time_vals = np.concatenate([r["time_vals"] for r in file_results])

    # Skip a variable if it was missing in any single file
    skipped_any = {name for r in file_results for name in r["skipped"]}

    var_arrays: dict[str, np.ndarray] = {}
    for spec in ALMA_VARS:
        if spec.alma_name in skipped_any:
            continue
        var_arrays[spec.alma_name] = np.concatenate(
            [r["data"][spec.alma_name] for r in file_results]
        )

    _write_output(
        output_path, time_vals, time_units, time_calendar,
        lat_vals, lon_vals, var_arrays, skipped_any, len(input_paths),
    )


if __name__ == '__main__':
    ################
    # configurations
    ################
    model_name = 'E3SM'
    model_version = 'v3.0.2'
    callmip_exp_no = '1'
    site_name = 'DK-Sor'

    prior_case = '20260422_DK-Sor_ICB20TRCNPRDCTCBC_default'
    post_case  = '20260422_DK-Sor_ICB20TRCNPRDCTCBC_optimized'

    cal_years = range(1997, 2014)
    val_years = range(2014, 2015)

    # generated by postprocess_uc.py
    ## uncertainty_file = ''
    uncertainty_file = '../output/DK-Sor/posterior_uncertainty.nc'

    ################
    # Conversion to calibration and validation
    ################
    prior_path = os.path.join(os.environ['SHARDIR'], '../zdr', 'CalLMIP', 'e3sm_run', prior_case, 'run')
    post_path = os.path.join(os.environ['SHARDIR'], '../zdr', 'CalLMIP', 'e3sm_run', post_case, 'run')

    # calibration period, prior
    translate_many(
        [os.path.join(prior_path, f'{prior_case}.elm.h1.{year}-01-01-00000.nc') for year in cal_years], 
        os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Cal_Prior.nc')
    )

    # calibration period, posterior
    translate_many(
        [os.path.join(post_path, f'{post_case}.elm.h1.{year}-01-01-00000.nc') for year in cal_years],
        os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Cal_Posterior.nc')
    )

    # validation period, prior
    translate_many(
        [os.path.join(prior_path, f'{prior_case}.elm.h1.{year}-01-01-00000.nc') for year in val_years], 
        os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Val_Prior.nc')
    )

    # validation period, posterior
    translate_many(
        [os.path.join(post_path, f'{post_case}.elm.h1.{year}-01-01-00000.nc') for year in val_years],
        os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Val_Posterior.nc')
    )


    # append the posterior uncertainty to the posterior files
    if len(uncertainty_file) > 0:
        src = Dataset(uncertainty_file, "r")

        # Decode uncertainty time to (year, month, day) tuples for calendar-agnostic matching.
        uc_tv = src.variables["time"]
        uc_dates = cftime.num2date(
            np.asarray(uc_tv[:], dtype=np.float64),
            getattr(uc_tv, "units", "days since 1997-01-01 00:00:00"),
            getattr(uc_tv, "calendar", "proleptic_gregorian"),
        )
        uc_ymd_to_idx = {(d.year, d.month, d.day): i for i, d in enumerate(uc_dates)}

        class _UcProxy:
            def __init__(self, ds, mapping):
                self._ds = ds
                self._mapping = mapping
            @property
            def variables(self):
                result = dict(self._ds.variables)
                for base, uc in self._mapping.items():
                    if uc in result:
                        result[base] = result[uc]
                return result

        proxy = _UcProxy(src, {
            "NEP": "NEP_uc",
            "EFLX_LH_TOT": "EFLX_LH_TOT_uc",
            "FSH": "FSH_uc",
        })
        uc_specs = [
            ("NEE_uc", ALMA_VARS[0]),
            ("Qle_uc", ALMA_VARS[1]),
            ("Qh_uc",  ALMA_VARS[2]),
        ]
        # Read full uncertainty arrays (covers a superset of the output time ranges).
        uc_arrays_full = {out_name: np.abs(spec.compute(proxy, None)) for out_name, spec in uc_specs}
        src.close()

        for out_path in [
            os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Cal_Posterior.nc'),
            os.path.join('..', 'output', 'DK-Sor', f'{model_name}.{model_version}_Expt1_{site_name}_Val_Posterior.nc'),
        ]:
            dst = Dataset(out_path, "a")

            # Decode destination times and find the corresponding indices in the uncertainty arrays.
            dst_tv = dst.variables["time"]
            dst_dates = cftime.num2date(
                np.asarray(dst_tv[:], dtype=np.float64),
                getattr(dst_tv, "units", "days since 1850-01-01 00:00:00"),
                getattr(dst_tv, "calendar", "noleap"),
            )
            uc_idx = np.array([uc_ymd_to_idx[(d.year, d.month, d.day)] for d in dst_dates])

            for out_name, spec in uc_specs:
                arr = np.asarray(uc_arrays_full[out_name][uc_idx], dtype=np.float32).reshape(-1, 1, 1)
                v = dst.createVariable(
                    out_name, "f4", ("time", "lat", "lon"),
                    fill_value=np.float32(1.0e36),
                    zlib=True, complevel=4,
                )
                v.units = spec.units
                v.long_name = spec.long_name
                v.standard_name = spec.standard_name
                v.cmip_short_name = spec.cmip_name
                v.cell_methods = "time: mean"
                v[:] = np.where(np.isnan(arr), 1.0e36, arr)
            dst.close()