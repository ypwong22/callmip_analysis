import numpy as np


def diurnal_mean(var, window=10, npd=24):

  diurnal_mean = np.zeros([365,npd], float)
  for d in range(0,365):
    for h in range(0,npd):
      diurnal_mean[d,h] = np.nanmean(var[max(d-window,0)*npd+h:min(d+window,364)*npd+h:npd])
  
  for i in range(0,len(var)):
    if (np.isnan(var[i])):
      d = int((i % 365*npd) / npd)
      h = i % npd
      var[i] = diurnal_mean[d,h]



def linear(var):
  #debug
  #data = np.array([np.nan, np.nan, 1, 2, 3, np.nan, np.nan])
  #print(linear(data))
  #data = np.array([1, np.nan, np.nan, 1, 2, 3, np.nan, 3])
  #print(linear(data))
  #data = np.array([1, 2, 3, np.nan, np.nan, np.nan, 1, 2, 3, np.nan, np.nan, -1])
  #print(linear(data))

  mask = np.isnan(var).astype(int)
  start_of_nan = np.where(np.diff(np.insert(mask, 0, 0)) == 1)[0]
  end_of_nan = np.where(np.diff(np.append(mask, 0)) == -1)[0]

  if np.sum(mask) == len(var):
    raise Exception('Cannot interpolate because everything is NaN.')

  for start, end in zip(start_of_nan, end_of_nan):
    if start == 0:
      var[:(end+1)] = np.array([var[(end+1)]] * (end + 1))
    elif end == (len(var)-1):
      var[start:] = np.array([var[(start-1)]] * (end + 1))
    else:
      var[start:(end+1)] = var[(start-1)] + np.cumsum(np.ones(end-start+1)) / (end-start+2) * (var[(end+1)] - var[(start-1)])

  return var