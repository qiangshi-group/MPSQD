#!/usr/bin/env python3
import re
import glob
import sys
import numpy as np
import warnings

class GetParameters(object):
  AU2EV = 27.2113845
  AU2FS = 2.418884254e-2

  def __init__(self,fname):
#get the name of input file
    if(fname == None):
      warnings.warn("No input file name is given, try to find input file automaticly", UserWarning)
      self.inp_file = self.check_input('*.inp')
    else:
      self.inp_file = fname

#get data from the input file
    try:
      self.inp, self.modes = self.get_inp_para(self.inp_file)
    except (FileNotFoundError):
      warnings.warn("Given file name \"" + fname + "\" is wrong, try to find input file automaticly", UserWarning)
      self.inp_file = self.check_input('*.inp')
      self.inp, self.modes = self.get_inp_para(self.inp_file)
    print("Input file: " + self.inp_file)

    default_value = {}
    default_value['custom_nb'] = False
    default_value['need_trun'] = True
    default_value['nrmax'] = 0
    default_value['small'] = 1e-13


#parameter collecting
    self.inp['nmode'] = len(self.modes.keys())
    try:
#necessary parameters with no default value
      for key in ['energy_unit']:
        if (not key in self.inp):
          print("Parameter missing: " + key)
          sys.exit()
      for key in ['nstate','init_state','nrtt']:
        self.inp[key] = int(self.inp[key])
    except (KeyError):
      print("Parameter missing: "+key)
      sys.exit()
    except (ValueError):
      print("Wrong value: "+key+" = "+self.inp[key])
      sys.exit()

#optional parameters and have default value
    for key in ['custom_nb','need_trun']:
      try:
        if self.inp[key] == 'true':
          self.inp[key] = True
        elif self.inp[key] == 'false':
          self.inp[key] = False
        else:
          self.inp[key] = default_value[key]
      except (KeyError):
        self.inp[key] = default_value[key]
      except (ValueError):
        massage = "Wrong value: "+key+"="+str(self.inp[key])+", use default value: "+str(default_value[key])
        warnings.warn(massage, UserWarning)
        self.inp[key] = default_value[key]
    for key in ['nrmax']:
      try:
        self.inp[key] = int(self.inp[key])
      except (KeyError):
        self.inp[key] = default_value[key]
      except (ValueError):
        massage = "Wrong value: "+key+"="+str(self.inp[key])+", use default value: "+str(default_value[key])
        warnings.warn(massage, UserWarning)
        self.inp[key] = default_value[key]
    for key in ['small']:
      try:
        self.inp[key] = float(self.inp[key])
      except (KeyError):
        self.inp[key] = default_value[key]
      except (ValueError):
        massage = "Wrong value: "+key+"="+str(self.inp[key])+", use default value: "+str(default_value[key])
        warnings.warn(massage, UserWarning)
        self.inp[key] = default_value[key]

#nbv will be read even when custom_nb is True, however it will be optional in this case 
    try:
      self.inp['nbv'] = int(self.inp['nbv'])
    except (KeyError):
      if (not self.inp['custom_nb']):
        print("Parameter missing: nbv")
        sys.exit()
    except (ValueError):
      if (not self.inp['custom_nb']):
        print("Wrong value: nbv = "+self.inp['nbv'])
        sys.exit()

#nmole will be used only when multi_mole is True, the mistakes about it will not matter otherwise
    self.with_nmole = False
    try:
      self.inp['nmole'] = int(self.inp['nmole'])
      self.with_nmole = True
    except (KeyError):
      self.inp['nmole'] = None
    except (ValueError,KeyError):
      self.inp['nmole'] = self.inp['nmole']

#get number of basises of bath
    if self.inp['custom_nb']:
      self.nb1 = self.get_primary_basis()

#get Hamiltonian parameters
    self.e0, self.omega, self.coef1, self.coef2 = self.get_parameters()
    if self.inp['energy_unit'] == 'ev':
      self.e0 = self.e0 / GetParameters.AU2EV
      self.omega = self.omega / GetParameters.AU2EV
      self.coef1 = self.coef1 / GetParameters.AU2EV
      self.coef2 = self.coef2 / GetParameters.AU2EV
    return

#search for the input file
  def check_input(self, extension):
    inpfile = glob.glob(pathname=extension)
    if len(inpfile) == 0:
      print("There is no input files, please add one!")
      sys.exit()
    if len(inpfile) > 1:
      print("There are at least two input files, please keep one!")
      sys.exit()
    return inpfile[0]

#get data from input file
  def get_inp_para(self, filename):
    fr = open(filename)
    data = fr.readlines()
    fr.close()
    data_re = []
    for item in data:
      data_re.append(item.strip().replace(' ', '').lower())
    order = 0
    inp = {}
    modes = {}
    for item in data_re:
#everything after "#" will be ignored
      if(re.search('#', item)):
        begin = re.search('#', item).span()[0]
        item = item[:begin]
#lines with "=" will be collected as a parameter
      if '=' in item:
        key, value = item.split("=")[0:2]  #only the first "=" will be read
        inp[key] = value
#lines begin with "modes\|" will be collected as modes
      elif re.match('modes\\|', item):
        for imode in item.replace('modes|', '').split('|'):
          if(len(imode) == 0):
            warnings.warn("Empty mode name, ignore")
            continue
          if(imode in modes):
            warnings.warn("Repeated mode name, ignore")
            continue
          modes[imode] = order
          order += 1
    return inp, modes

#get the nb of modes when custom_nb is true
  def get_primary_basis(self):
    nb1 = np.zeros(self.inp['nmode'], dtype=int)
    for mode in self.modes.keys():
      try:
        nb1[self.modes[mode]] = int(self.inp[mode])
        if (nb1[self.modes[mode]] <= 0):
          print("Wrong value: nb of "+mode)
          sys.exit()
      except KeyError:
        if('nbv' in self.inp):  #nbv is default value for nb1
          nb1[self.modes[mode]] = self.inp['nbv']
          warnings.warn("Parameter missing: nb of "+mode, UserWarning)
        else:
          print("Parameter missing: nb of "+mode)
          sys.exit()
      except ValueError:
        print("Wrong value: nb of "+mode)
        sys.exit()
    return nb1

#get Hamiltonian parameters
  def get_parameters(self):
    if(self.with_nmole):
      nmole = self.inp['nmole']
    else:
      nmole = 1
    e0 = np.zeros((self.inp['nstate']*nmole, self.inp['nstate']*nmole), dtype=np.float64)
    omega = np.zeros(self.inp['nmode'], dtype=np.float64)
    coef1 = np.zeros((self.inp['nstate']*nmole, self.inp['nstate']*nmole, self.inp['nmode']*nmole), dtype=np.float64)
    coef2 = np.zeros((self.inp['nstate']*nmole, self.inp['nstate']*nmole, self.inp['nmode']*nmole, self.inp['nmode']*nmole), dtype=np.float64)
    parameters = {}
    for item in self.inp.keys():
      try:
        if re.match('v_s(\\d+)_s(\\d+)', item):  #off-diagonal constant term. example:V_S0_S1 = 0.00
          result = re.match(u'v_s(\\d+)_s(\\d+)', item)
          istate, jstate = result.groups()
          e0[int(istate), int(jstate)] = self.inp[item]
          e0[int(jstate), int(istate)] = self.inp[item]
        elif re.match('v_s(\\d+)', item):  #on-diagonal constant term. example:V_S0 = -0.42300
          result = re.match(u'v_s(\\d+)', item)
          istate = result.groups()[0]
          e0[int(istate), int(istate)] = self.inp[item]
        elif re.match('omega', item):  #harmonic term. example:omega_v10a = 0.11390
          result = re.match(u'omega_(\\w+)', item)
          imode = result.groups()[0]
          omega[self.modes[imode]] = self.inp[item]
        elif re.match('kl_s(\\d+)_s(\\d+)_(\\w+)', item):  #off-diagonal linar term. example:kl_S0_S1_v10a = 0.20804
          result = re.match(u'kl_s(\\d+)_s(\\d+)_(\\w+)', item)
          istate, jstate, imode = result.groups()
          coef1[int(istate), int(jstate), self.modes[imode]] = self.inp[item]
          coef1[int(jstate), int(istate), self.modes[imode]] = self.inp[item]
        elif re.match('kl_s(\\d+)_(\\w+)', item):  #on-diagonal linar term. example:kl_S0_v6a = 0.09806
          result = re.match(u'kl_s(\\d+)_(\\w+)', item)
          istate, imode = result.groups()
          coef1[int(istate), int(istate), self.modes[imode]] = self.inp[item]
        elif re.match('kb_s(\\d+)_s(\\d+)_(\\w+)_(\\w+)', item):  #off-diagonal bilinear term. example:kb_S0_S1_v10a_v1 = 0.00553
          result = re.match(u'kb_s(\\d+)_s(\\d+)_(\\w+)_(\\w+)', item)
          istate, jstate, imode, jmode = result.groups()
          coef2[int(istate), int(jstate), self.modes[imode], self.modes[jmode]] = self.inp[item]
          coef2[int(jstate), int(istate), self.modes[imode], self.modes[jmode]] = self.inp[item]
          coef2[int(istate), int(jstate), self.modes[jmode], self.modes[imode]] = self.inp[item]
          coef2[int(jstate), int(istate), self.modes[jmode], self.modes[imode]] = self.inp[item]
        elif re.match('kb_s(\\d+)_(\\w+)_(\\w+)', item):  #on-diagonal bilinear term. example:kb_S1_v6a_v1 = -0.00298
          result = re.match(u'kb_s(\\d+)_(\\w+)_(\\w+)', item)
          istate, imode, jmode = result.groups()
          coef2[int(istate), int(istate), self.modes[imode], self.modes[jmode]] = self.inp[item]
          coef2[int(istate), int(istate), self.modes[jmode], self.modes[imode]] = self.inp[item]
        elif re.match('kq_s(\\d+)_s(\\d+)_(\\w+)', item):  #off-diagonal quadratic term. example:kq_S0_S1_v10a = 0.00553
          result = re.match(u'kq_s(\\d+)_s(\\d+)_(\\w+)', item)
          istate, jstate, imode = result.groups()
          coef2[int(istate), int(jstate), self.modes[imode], self.modes[imode]] = self.inp[item]
        elif re.match('kq_s(\\d+)_(\\w+)', item):  #on-diagonal quadratic term. example:kq_S1_v10a = -0.01159
          result = re.match(u'kq_s(\\d+)_(\\w+)', item)
          istate, imode = result.groups()
          coef2[int(istate), int(istate), self.modes[imode], self.modes[imode]] = self.inp[item]
      except(IndexError,KeyError):
        print("Wrong state or mode: \""+item+" = "+self.inp[item]+"\"")
        sys.exit()
      except(ValueError):
        print("Wrong value: \""+item+" = "+self.inp[item]+"\"")
        sys.exit()
    return e0, omega, coef1, coef2
