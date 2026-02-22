import os
import subprocess
import re
from netCDF4 import Dataset
import numpy as np

ccsm_input = os.path.join(os.environ['WORLD'], 'e3sm/inputdata')

srcfile = os.path.join(ccsm_input, 'atm/datm7/CO2/fco2_datm_rcp4.5_1765-2500_c130312.nc')
tgtfile = os.path.join(ccsm_input, 'CalLMIP/fco2_datm_rcp4.5_1765-2500_cTRENDY.nc')

subprocess.run(['cp', srcfile, tgtfile], check=True)

with open(os.path.join(ccsm_input, 'CalLMIP', 'CO2_1700_2024_TRENDYv2025.txt')) as f:
    alllines = f.readlines()
    co2_data = [(int(ln.split('=')[0][-4:]), float(ln.split('=')[1])) for ln in alllines]
    co2_data = np.array(co2_data)

nc = Dataset(tgtfile, 'r+')
nc['CO2'][:260,0,0] = co2_data[(co2_data[:,0] >= 1765), 1]
nc.sync()
nc.close()