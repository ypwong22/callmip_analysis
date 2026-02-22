from netCDF4 import Dataset
import numpy as np
import subprocess


def calc_q(e_in, pres):
    """ Function for cacluating saturation specific humidity """
    myq = 0.622 * e_in / (pres - 0.378*e_in)
    return myq

def esat(t):
    """ Coefficients for calculating saturation vapor pressure

    Lowe, P. R., and J. M. Ficke (1974)
    The Computation of Saturation Vapor Pressure
    NOAA Technical Report NWS 4"""

    a = [6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04, \
            3.031240396e-06, 2.034080948e-08, 6.136820929e-11]
    b = [6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04, \
            5.824720280e-06, 4.838803174e-08, 1.838826904e-10]

    myesat = np.where(
        t >= 0, # reference = water,
        a[0]+t*(a[1]+t*(a[2]+t*(a[3]+t*(a[4]+t*(a[5]+t*a[6]))))),
        # reference = ice
        b[0]+t*(b[1]+t*(b[2]+t*(b[3]+t*(b[4]+t*(b[5]+t*b[6])))))
    )

    return myesat


def bypass_format(filename, met_data, lat, lon, startyear, endyear, edge=0.1, time_offset=0, calc_lw = False, zbot=10):
    """Write an ELM single-point atmospheric forcing file in netCDF format.

    Writes all variables in met_data to a netCDF file with the dimensions and
    attributes expected by ELM's bypass (point-mode) forcing reader. Coordinate
    variables (DTIME, LONGXY, LATIXY) and scalar metadata (start_year, end_year)
    are added after dimension-packing with ncpdq.

    Parameters
    ----------
    filename : str
        Path of the output netCDF file. Overwritten if it already exists.
    met_data : dict
        Dictionary mapping ELM variable names (e.g. 'TBOT', 'RH', 'FSDS') to
        1-D arrays of length nt (total number of timesteps across all years).
    lat : float
        Latitude of the site in degrees N.
    lon : float
        Longitude of the site in degrees E.
    startyear : int
        First calendar year of the forcing data.
    endyear : int
        Last calendar year of the forcing data (inclusive).
    edge : float, optional
        Intended half-width of the gridcell in degrees (currently unused).
    time_offset : float, optional
        UTC offset of the source data, in hours. Set this to the UTC offset of
        the input timezone to align the output with UTC. For example, use -6
        for data originally in UTC-6 (data is shifted to later positions, with
        the start padded by the first value) and +2 for UTC+2 (data shifted to
        earlier positions, end padded by the last value). Default is 0.
    calc_lw : bool, optional
        If True, overwrite FLDS with downwelling longwave estimated from TBOT
        and the derived actual vapor pressure using the Brutsaert (1975)
        emissivity parameterization. FLDS and at least one of QBOT, RH, or VPD
        must be present in met_data. Default is False.

    Notes
    -----
    Any subset of {QBOT, RH, VPD} may be supplied; the missing variables are
    always derived automatically from TBOT and PSRF using the priority order
    QBOT > RH > VPD. The inverse formulas used are:
      e - actual vapor pressure (hPa)

      from QBOT:  e = QBOT * (PSRF / 100) / (0.622 + 0.378 * QBOT) (PSRF in Pa)
      from RH:    e = esat(T) * RH / 100
      from VPD:   e = esat(T) - VPD / 100    (VPD in Pa, esat in hPa)
    zbot : float, optional
        Reference height (m) for wind and temperature observations. Written as
        the ZBOT variable if it is not already present in met_data. Default
        is 10.
    """
    metvars = list(met_data.keys())

    units={}
    units['TBOT'] = 'K'
    units['TSOIL'] = 'K'
    units['RH'] = '%'
    units['WIND'] = 'm/s'
    units['FSDS'] = 'W/m2'
    units['PAR'] = 'umol/m2/s'
    units['FLDS'] = 'W/m2'
    units['PSRF'] = 'Pa'
    units['PRECTmms'] = 'kg/m2/s'
    units['QBOT'] = 'kg/kg'
    units['ZBOT'] = 'm'
    units['VPD'] = 'Pa'
    long_names = {}
    long_names['TBOT'] = 'temperature at the lowest atm level (TBOT)'
    long_names['RH'] = 'relative humidity at the lowest atm level (RH)'
    long_names['WIND'] = 'wind at the lowest atm level (WIND)'
    long_names['FSDS'] = 'incident solar (FSDS)'
    long_names['FLDS'] = 'incident longwave (FLDS)'
    long_names['PSRF'] = 'pressure at the lowest atm level (PSRF)'
    long_names['PRECTmms'] = 'precipitation (PRECTmms)'
    long_names['QBOT'] = 'specific humidity at the lowest atm level (QBOT)'
    long_names['ZBOT'] = 'observational height (ZBOT)'
    long_names['VPD'] = 'vapor pressure deficit at the lowest atm level (VPD)'

    nt = len(met_data[metvars[0]])
    npd = int(np.round(nt/(endyear-startyear+1))/365)

    all_hourly = Dataset(filename,'w')
    all_hourly.createDimension('DTIME', nt)
    all_hourly.createDimension('gridcell',1)
    all_hourly.createDimension('scalar',1)
    for v in metvars:
        all_hourly.createVariable(v, 'f', ('gridcell','DTIME',))
        nshift = int(abs(time_offset*int(npd/24)))
        if time_offset < 0:
            all_hourly[v][0,nshift:] = met_data[v][:-1*nshift]
            all_hourly[v][0,0:nshift] = met_data[v][0]
        elif time_offset > 0:
            all_hourly[v][0,:-1*nshift] = met_data[v][nshift:]
            all_hourly[v][0,-1*nshift:] = met_data[v][nt-1]
        else:
            all_hourly[v][0,:] = met_data[v][:]
        all_hourly[v].units = units[v]
        all_hourly[v].long_name = long_names[v]
        all_hourly[v].mode = 'time-dependent'

    # Derive actual vapor pressure from the highest-priority humidity variable
    esat_vals = esat(all_hourly['TBOT'][0,:]-273.15)
    P_hPa = all_hourly['PSRF'][0,:]/100.
    if 'QBOT' in metvars:
        mye = all_hourly['QBOT'][0,:] * P_hPa / (0.622 + 0.378*all_hourly['QBOT'][0,:])
    elif 'RH' in metvars:
        mye = esat_vals * all_hourly['RH'][0,:]/100.
    elif 'VPD' in metvars:
        mye = esat_vals - all_hourly['VPD'][0,:]/100.
    else:
        mye = None

    if mye is not None:
        if 'QBOT' not in metvars:
            all_hourly.createVariable('QBOT','f',('gridcell','DTIME',))
            all_hourly['QBOT'][0,:] = calc_q(mye, P_hPa)
            all_hourly['QBOT'].units = units['QBOT']
            all_hourly['QBOT'].long_name = long_names['QBOT']
            all_hourly['QBOT'].mode = 'time-dependent'
        if 'RH' not in metvars:
            all_hourly.createVariable('RH','f',('gridcell','DTIME',))
            all_hourly['RH'][0,:] = mye/esat_vals*100.
            all_hourly['RH'].units = units['RH']
            all_hourly['RH'].long_name = long_names['RH']
            all_hourly['RH'].mode = 'time-dependent'
        if 'VPD' not in metvars:
            all_hourly.createVariable('VPD','f',('gridcell','DTIME',))
            all_hourly['VPD'][0,:] = (esat_vals - mye)*100.
            all_hourly['VPD'].units = units['VPD']
            all_hourly['VPD'].long_name = long_names['VPD']
            all_hourly['VPD'].mode = 'time-dependent'

    if calc_lw:
        stebol = 5.67e-8
        ea = 0.70 + 5.95e-5*mye*np.exp(1500.0/all_hourly['TBOT'][0,:])
        all_hourly['FLDS'][0,:] = ea * stebol * (all_hourly['TBOT'][0,:]) ** 4

    if 'ZBOT' not in metvars:
        all_hourly.createVariable('ZBOT','f',('gridcell','DTIME',))
        all_hourly['ZBOT'][:,:] = zbot
        all_hourly['ZBOT'].units = units['ZBOT']
        all_hourly['ZBOT'].long_name = long_names['ZBOT']
        all_hourly['ZBOT'].mode = 'time-dependent'

    all_hourly.close()

    subprocess.run(['ncpdq', filename, filename+'_pk'], check=True)
    subprocess.run(['mv', filename+'_pk', filename], check=True)
    output_data = Dataset(filename,'a')
    output_data.createDimension('scalar', 1)
    output_data.createVariable('DTIME', 'f8', 'DTIME')
    output_data.variables['DTIME'].long_name='observation time'
    output_data.variables['DTIME'].units='days since '+str(startyear)+'-01-01 00:00:00'
    output_data.variables['DTIME'].calendar='noleap'
    n_years = endyear-startyear+1
    output_data.variables['DTIME'][:] = np.cumsum(np.ones([n_years*365*npd], float)/npd)-0.5/npd
    output_data.createVariable('LONGXY', 'f8', 'gridcell')
    output_data.variables['LONGXY'].long_name = "longitude"
    output_data.variables['LONGXY'].units = 'degrees E'
    output_data.variables['LONGXY'][:] = lon 
    output_data.createVariable('LATIXY', 'f8', 'gridcell')
    output_data.variables['LATIXY'].long_name = "latitude"
    output_data.variables['LATIXY'].units = 'degrees N'
    output_data.variables['LATIXY'][:] = lat 
    output_data.createVariable('start_year', 'i4', 'scalar')
    output_data.variables['start_year'][:] = startyear
    output_data.createVariable('end_year', 'i4', 'scalar')
    output_data.variables['end_year'][:] = endyear
    output_data.close()


    #for y in range(startyear,endyear):
    #  for m in range(0,12):
    #    mst = str(101+m)[1:]
    #    monthly = Dataset('1x1pt_'+site+'/'+str(y)+'-'+mst+'.nc','w')

