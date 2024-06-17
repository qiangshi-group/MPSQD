# Parameters and date types, Qiang Shi, Nan Sheng, Weizhong Guan

import sys
import numpy as np
def init_tds(fname,input_type,multi_mole=False):
  params = {}
#get data from model
  if(input_type=='d' or input_type=='D'):
    from . import model_define as md
  elif(input_type=='m' or input_type=='M'):
    from . import model_mctdh as md
  else:
    print("Unknow input flie type")
    sys.exit()
  estimator = md.GetParameters(fname)

#collect parameters from estimator and check the value of these parameters
  for key in ['init_state']:
    if (estimator.inp[key] < 0):
      print("Wrong value: "+key)
      sys.exit()
    else:
      params[key] = estimator.inp[key]
  for key in ['nrtt']:
    if (estimator.inp[key] <= 0):
      print("Wrong value: "+key)
      sys.exit()
    else:
      params[key] = estimator.inp[key]

#MPO parameters
  for key in ['small','nrmax','custom_nb','need_trun']:
    params[key] = estimator.inp[key]
  params['e0'] = estimator.e0
  params['omega'] = estimator.omega
  params['coef1'] = estimator.coef1
  params['coef2'] = estimator.coef2
  params['nlevel'] = estimator.inp['nmode']
  params['ndvr'] = estimator.inp['nstate']
  if(params['custom_nb']):
    params['nb1'] = estimator.nb1
  else:
    params['nbv'] = estimator.inp['nbv']

#check the value of these parameters
  for key in ['nstate','nmode','small','nrmax']:
    if (estimator.inp[key] <= 0):
      print("Wrong value: "+key)
      sys.exit()

  if multi_mole:
    params['nmole'] = estimator.inp['nmole']
    if (params['nmole'] != None):
      if(isinstance(params['nmole'], int)):
        if(params['nmole'] < 1):
          print("Wrong value: nmole="+params['nmole'])
          sys.exit()
        else:
          for imole in range(params['nmole']):
            params['e0'][imole*params['ndvr']:(imole+1)*params['ndvr'],imole*params['ndvr']:(imole+1)*params['ndvr']] = params['e0'][:params['ndvr'],:params['ndvr']]
            params['coef1'][imole*params['ndvr']:(imole+1)*params['ndvr'],imole*params['ndvr']:(imole+1)*params['ndvr'],imole*params['nlevel']:(imole+1)*params['nlevel']] = params['coef1'][:params['ndvr'],:params['ndvr'],:params['nlevel']]
            params['coef2'][imole*params['ndvr']:(imole+1)*params['ndvr'],imole*params['ndvr']:(imole+1)*params['ndvr'],imole*params['nlevel']:(imole+1)*params['nlevel'],imole*params['nlevel']:(imole+1)*params['nlevel']] = params['coef2'][:params['ndvr'],:params['ndvr'],:params['nlevel'],:params['nlevel']]
          params['omega'] = np.tile(params['omega'],(params['nmole']))
          params['ndvr'] = params['ndvr']*params['nmole']
          params['nlevel'] = params['nlevel']*params['nmole']
      else:
        print("Wrong value: nmole="+params['nmole'])
        sys.exit()
    else:
      print("Parameter missing: nmole")
      sys.exit()
  else:
    params['e0'] = params['e0'][:params['ndvr'],:params['ndvr']]
    params['coef1'] = params['coef1'][:params['ndvr'],:params['ndvr'],:params['nlevel']]
    params['coef2'] = params['coef2'][:params['ndvr'],:params['ndvr'],:params['nlevel'],:params['nlevel']]
  return params
