import sys
import re
import numpy as np
from ..models.construct_tds import write_file_define
AU2EV = 27.2113845
def get_all(fr):
  data = fr.readlines()
  fr.close()
  data_re = []
  for item in data:
    data_re.append(item.strip().replace(' ', '').lower())
#everything after "#" will be ignored
    if(re.search('#', item)):
      begin = re.search('#', item).span()[0]
      item = item[:begin]
  return data_re

def load_input(input_name):
  data_re = get_all(open(input_name))
  parameters = {}
  e0_0 = {}
  omega_0 = {}
  coef1_0 = {}
  coef2_0 = {}
  try:
    for k in data_re:
      if '=' in k:
        key, valunit = k.split("=")[0:2]
        value, unit = valunit.split(",")[0:2]
        param = {}
        if unit == 'ev':
          param = float(value) / AU2EV
        else:
          param = float(value)
        parameters[key] = param
    for k in data_re:
      if re.search(u'\|\ds\d+&\d+\|\d+q\^2', k):
        result = re.search(u'(\w+).*?s(\d+).*?(\d+)\|(\d+).*', k)
        key, istate1, istate2, imode = result.groups()
        coef2_0[(int(istate1)-1, int(istate2)-1, int(imode)-2, int(imode)-2)] = parameters[key]
      elif re.search(u'\|\ds\d+&\d+\|\d+q\|\d+q', k):
        result = re.search(u'.*\*(\w+).*?s(\d+).*?(\d+)\|(\d+).*\|(\d+).*', k)
        key, istate1, istate2, imode1, imode2 = result.groups()
        coef2_0[(int(istate1)-1, int(istate2)-1, int(imode1)-2, int(imode2)-2)] = parameters[key]
      elif re.search(u'\|\ds\d+&\d+\|\d+q', k):
        result = re.search(u'(\w+).*?s(\d+).*?(\d+)\|(\d+).*', k)
        key, istate1, istate2, imode = result.groups()
        coef1_0[(int(istate1)-1, int(istate2)-1, int(imode)-2)] = parameters[key]
      elif re.search(u'\|\d+ke', k):
        result = re.search(u'.*\*(\w+)\|(\d+)ke', k)
        key, imode = result.groups()
        omega_0[int(imode)-2] = parameters[key]
      elif re.search(u'\d+&\d+', k):
        result = re.search(u'(\w+).*\|\d+s(\d+)&(\d+)', k)
        key, istate1, istate2 = result.groups()
        if '-' in k:
          e0_0[(int(istate1)-1, int(istate2)-1)] = -1*parameters[key]
        else:
          e0_0[(int(istate1)-1, int(istate2)-1)] = parameters[key]
  except(ValueError,KeyError):
    print("Wrong value: "+k)
    sys.exit()
  statelist = [x[0] for x in e0_0.keys()] + [x[1] for x in e0_0.keys()] + [x[0] for x in coef1_0.keys()] + [x[1] for x in coef1_0.keys()] + [x[0] for x in coef2_0.keys()] + [x[1] for x in coef2_0.keys()]
  modelist = [x for x in omega_0.keys()]
  nstate = max(statelist)+1
  nmode = max(modelist)+1
  params = {}
  params["ndvr"] = nstate
  params["nlevel"] = nmode
  e0 = np.zeros((nstate, nstate), dtype=np.float64)
  omega = np.zeros(nmode, dtype=np.float64)
  coef1 = np.zeros((nstate, nstate, nmode), dtype=np.float64)
  coef2 = np.zeros((nstate, nstate, nmode, nmode), dtype=np.float64)
  for key in e0_0.keys():
    istate1,istate2 = key
    e0[istate1, istate2] = e0_0[key]
    e0[istate2, istate1] = e0_0[key]
  for key in omega_0.keys():
    omega[key] = omega_0[key]
  for key in coef1_0.keys():
    istate1,istate2,imode = key
    coef1[istate1, istate2, imode] = coef1_0[key]
    coef1[istate2, istate1, imode] = coef1_0[key]
  for key in coef2_0.keys():
    istate1,istate2,imode1,imode2 = key
    coef2[istate1, istate2, imode1, imode2] = coef2_0[key]
    coef2[istate2, istate1, imode1, imode2] = coef2_0[key]
    coef2[istate1, istate2, imode2, imode1] = coef2_0[key]
    coef2[istate2, istate1, imode2, imode1] = coef2_0[key]
  params['e0'] = e0
  params['omega'] = omega
  params['coef1'] = coef1
  params['coef2'] = coef2
  return params

def mctdh2inp(input_name,output_name):
  params = load_input(input_name)
  write_file_define(params,output_name)
  return

def mctdh2python(input_name):
  params = load_input(input_name)
  fp = open("model.py",'w')
  output = "# Energy units are in au...\n"
  fp.write(output)
  output = "import numpy as np\n\n"
  fp.write(output)
  output = "nmode = "+str(params['nlevel'])+"\n"
  fp.write(output)
  output = "nstate = "+str(params['ndvr'])+"\n"
  fp.write(output)
  output = "e0 = np.zeros((nstate,nstate),dtype=np.float64)\n"
  fp.write(output)
  e0_off = params['e0'].copy() - np.diag(np.diagonal(params['e0']))
  for i in range(params['ndvr']):
    for j in range(i+1):
      if(np.abs(params['e0'][i,j])>1e-12):
        output = "e0["+str(i)+","+str(j)+"] = "+str(params['e0'][i,j])+"\n"
        fp.write(output)
  output = "for i in range(nstate):\n"
  fp.write(output)
  output = "  for j in range(i):\n"
  fp.write(output)
  output = "    e0[j,i] = e0[i,j]\n"
  fp.write(output)
  fp.write("\n")
  output = "omega = np.zeros(nmode,dtype=np.float64)\n"
  fp.write(output)
  for i in range(params['nlevel']):
    output = "omega["+str(i)+"] = "+str(params['omega'][i])+"\n"
    fp.write(output)
  fp.write("\n")
  output = "coef1 = np.zeros((nstate,nstate,nmode),dtype=np.float64)\n"
  fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i+1):
      for k in range(params['nlevel']):
        if(np.abs(params['coef1'][i,j,k])>1e-12):
          output = "coef1["+str(i)+","+str(j)+","+str(k)+"] = "+str(params['coef1'][i,j,k])+"\n"
          fp.write(output)
  output = "for i in range(nstate):\n"
  fp.write(output)
  output = "  for j in range(i):\n"
  fp.write(output)
  output = "    for k in range(nmode):\n"
  fp.write(output)
  output = "      coef1[j,i,k] = coef1[i,j,k]\n"
  fp.write(output)
  fp.write("\n")
  output = "coef2 = np.zeros((nstate,nstate,nmode,nmode),dtype=np.float64)\n"
  fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i+1):
      for k in range(params['nlevel']):
        for l in range(k+1):
          if(np.abs(params['coef2'][i,j,k,l])>1e-12):
            output = "coef2["+str(i)+","+str(j)+","+str(k)+","+str(l)+"] = "+str(params['coef2'][i,j,k,l])+"\n"
            fp.write(output)
  output = "for i in range(nstate):\n"
  fp.write(output)
  output = "  for j in range(i):\n"
  fp.write(output)
  output = "    for k in range(nmode):\n"
  fp.write(output)
  output = "      for l in range(k):\n"
  fp.write(output)
  output = "        coef2[j,i,k,l] = coef2[i,j,k,l]\n"
  fp.write(output)
  output = "        coef2[j,i,l,k] = coef2[i,j,k,l]\n"
  fp.write(output)
  output = "        coef2[i,j,l,k] = coef2[i,j,k,l]\n"
  fp.write(output)
  fp.write("\n")
  fp.flush()
  return
