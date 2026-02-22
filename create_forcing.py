""" 
Convert CalMIP flux site meteorological forcing to ELM forcings.
"""
import numpy as np
from netCDF4 import Dataset
import os, sys
import write_elm_met
import gapfill
import pandas as pd


# ------- user input -------------
site = "DK-Sor"
start_year = 1997
end_year = 2014
time_offset = +1  # Standard time offset from UTC (e.g. EST is -5)
npd = 48  # number of time steps per day (48 = half hourly)
mylon = 11.64464  # site longitude (0 to 360)
mylat = 55.48587  # site latitude
measurement_height = 57  # tower height (m)
filename = os.environ['WORLD'] + "/e3sm/inputdata/CalLMIP/DK-Sor_1997-2014_FLUXNET2015_Met.nc"
calc_flds = False  # use T and RH to comput FLDS (use if data missing or sparse)
leapdays = True  # input data has leap days (to be removed for ELM)
outdir = os.environ['WORLD'] + "/e3sm/inputdata/atm/datm7/CLM1PT_data/1x1pt_" + site + "/"  # Desired directory for ELM met inputs

# outvars   - met variables used as ELM inputs
# invars    - corresponding variables to be read from input file
# conv_add  - offset for converting units (e.g. C to K)
# conv_mult - multiplier for converting units (e.g. hPa to Pa, PAR to FSDS)
# valid_min - minimum acceptable value for this variable (set as NaN outside range)
# valid_max - maximum acceptable value for this variable (set as NaN outside range)

outvars =   ["TBOT",  "QBOT",   "WIND",  "PSRF",    "FSDS",    "FLDS", "PRECTmms"]
invars =    ["Tair",  "Qair",   "Wind", "Psurf",  "SWdown",  "LWdown",   "Precip"] # matching header of input file
conv_add =  [     0,       0,        0,       0,         0,         0,          0]
conv_mult = [     1,       1,        1,       1,         1,         1,          1]
valid_min = [180.00,    1e-4,        0,     8e4,         0,         0,          0]
valid_max = [350.00,    1e-1,       80,   1.5e5,      2500,       800,        0.2]

# ELM Variable names and units
# TBOT:     Air temperature at measurement (tower) height (K)
# QBOT:     Specific humidity (kg/kg)
# RH:       Relative humidity at measurment height (%)
# WIND:     Wind speeed at measurement height (m/s)
# PSRF:     air pressure at surface  (Pa)
# FSDS:     Incoming Shortwave radiation  (W/m2)
# FLDS:     Incoming Longwave radiation   (W/m2)
# PRECTmms: Precipitation       (kg/m2/s)


os.system("mkdir -p " + outdir)


# Load the data
with Dataset(filename) as nc:
    tvec = pd.to_datetime(
        nc['time'][:],
        unit="s",
        origin=pd.Timestamp("1997-01-01 00:00:00"),
    )

    metdata = {}
    for v in range(0, len(invars)):
        temp = nc[invars[v]][:,0,0].astype(float) * conv_mult[v] + conv_add[v]
        temp[(temp < valid_min[v]) | (temp > valid_max[v])] = np.nan

        # subset to range and drop leap days
        temp = temp[(tvec.year >= start_year) & (tvec.year <= end_year) & 
                    ~((tvec.month == 2) & (tvec.day == 29))]

        metdata[outvars[v]] = list(temp)

## Fill ZWT with linear cycle first
#metdata['ZWT'] = gapfill.linear(metdata['ZWT'])

# Fill missing values with diurnal mean
for key in metdata:
    gapfill.diurnal_mean(metdata[key], npd=npd)

out_fname = outdir + "/all_hourly.nc"
#write_elm_output_zwt.write_elm_metvars(
write_elm_met.bypass_format(
    out_fname,
    metdata,
    mylat,
    mylon,
    start_year,
    end_year,
    edge=0.1,
    time_offset=time_offset,
    calc_lw=calc_flds,
    zbot=measurement_height,
)
