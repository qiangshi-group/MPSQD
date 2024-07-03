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

#open the input file
    try:
      inpfr = open(self.inp_file)
    except (FileNotFoundError):
      warnings.warn("Given file name \"" + fname + "\" is wrong, try to find input file automaticly", UserWarning)
      self.inp_file = self.check_input('*.inp')
      inpfr = open(self.inp_file)

    self.inp = self.get_inp_para(inpfr)
    print("inp file: " + self.inp_file)

    default_value = {}
    default_value['custom_nb'] = False
    default_value['need_trun'] = True
    default_value['nrmax'] = 0
    default_value['small'] = 1e-13

#parameter collecting
    try:
#necessary parameters with no default value
      for key in ['nstate','init_state','nmode','nrtt']:
        self.inp[key] = int(self.inp[key])
    except (KeyError):
      print("Parameter missing: "+key)
      sys.exit()
    except (ValueError):
      print("Wrong value: "+key)
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

    if (not self.inp['custom_nb']):
      try:
        for key in ['nbv']:
          self.inp[key] = int(self.inp[key])
      except (KeyError):
        print("Parameter missing: "+key)
        sys.exit()
      except (ValueError):
        print("Wrong value: "+key)
        sys.exit()

#get or search for the operator file
    try:
      self.op_file = self.inp['opname']+'.op'
      opfr = open(self.op_file)
    except(KeyError,FileNotFoundError):
      warnings.warn("Open given operator file failed, try to find operator file automaticly", UserWarning)
      self.op_file = self.check_input('*.op')
      opfr = open(self.op_file)
    print("op file: " + self.op_file)

#get number of basises of bath
    if self.inp['custom_nb']:
      self.nb1 = self.get_primary_basis(opfr)

#get Hamiltonian parameters
    opfr = open(self.op_file)
    self.e0, self.omega, self.coef1, self.coef2 = self.get_op_para(opfr)
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

#get line from input file
  def get_all(self, fr):
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

#get data from input file
  def get_inp_para(self, fr):
    data_re = self.get_all(fr)
    inp = {}
    for item in data_re:
#lines with "=" will be collected as a parameter
      if '=' in item:
        key, value = item.split("=")[0:2]  #only the first "=" matters
        inp[key] = value
    return inp

#get the nb of every mode when custom_nb is true
  def get_primary_basis(self, fr):
    order = 0
    data_re = self.get_all(fr)
    nb1 = np.zeros(self.inp['nmode'], dtype=int)
    for item in data_re:
      if re.search('modes\\|', item):
        for mode in item.replace('modes|', '').split('|'):
          if mode != 'el':
            try:
              nb1[order] = self.inp[mode]
              if (nb1[order] <= 0):
                print("Wrong value: nb of "+mode)
                sys.exit()
            except KeyError:
              print("Wrong mode: nb of "+mode)
              sys.exit()
            except ValueError:
              print("Wrong value: nb of "+mode)
              sys.exit()
            order += 1
    return nb1

#get Hamiltonian parameters
  def get_op_para(self, fr):
    e0 = np.zeros((self.inp['nstate'], self.inp['nstate']), dtype=np.float64)
    omega = np.zeros(self.inp['nmode'], dtype=np.float64)
    coef1 = np.zeros((self.inp['nstate'], self.inp['nstate'], self.inp['nmode']), dtype=np.float64)
    coef2 = np.zeros((self.inp['nstate'], self.inp['nstate'], self.inp['nmode'], self.inp['nmode']), dtype=np.float64)
    data_re = self.get_all(fr)
    parameters = {}
    try:
      for k in data_re:
        if '=' in k:
          key, valunit = k.split("=")[0:2]
          value, unit = valunit.split(",")[0:2]
          param = {}
          if unit == 'ev':
            param = float(value) / GetParameters.AU2EV
          else:
            param = float(value)
          parameters[key] = param
      for k in data_re:
        if re.search(u'\\|\\ds\\d+&\\d+\\|\\d+q\\^2', k):
          result = re.search(u'(\\w+).*?s(\\d+).*?(\\d+)\\|(\\d+).*', k)
          key, istate1, istate2, imode = result.groups()
          coef2[int(istate1)-1, int(istate2)-1, int(imode)-2, int(imode)-2] = parameters[key]
          coef2[int(istate2)-1, int(istate1)-1, int(imode)-2, int(imode)-2] = parameters[key]
        elif re.search(u'\\|\\ds\\d+&\\d+\\|\\d+q\\|\\d+q', k):
          result = re.search(u'.*\\*(\\w+).*?s(\\d+).*?(\\d+)\\|(\\d+).*\\|(\\d+).*', k)
          key, istate1, istate2, imode1, imode2 = result.groups()
          coef2[int(istate1)-1, int(istate2)-1, int(imode1)-2, int(imode2)-2] = parameters[key]
          coef2[int(istate1)-1, int(istate2)-1, int(imode2)-2, int(imode1)-2] = parameters[key]
          coef2[int(istate2)-1, int(istate1)-1, int(imode1)-2, int(imode2)-2] = parameters[key]
          coef2[int(istate2)-1, int(istate1)-1, int(imode2)-2, int(imode1)-2] = parameters[key]
        elif re.search(u'\\|\\ds\\d+&\\d+\\|\\d+q', k):
          result = re.search(u'(\\w+).*?s(\\d+).*?(\\d+)\\|(\\d+).*', k)
          key, istate1, istate2, imode = result.groups()
          coef1[int(istate1)-1, int(istate2)-1, int(imode)-2] = parameters[key]
          coef1[int(istate2)-1, int(istate1)-1, int(imode)-2] = parameters[key]
        elif re.search(u'\\|\\d+ke', k):
          result = re.search(u'.*\\*(\\w+)\\|(\\d+)ke', k)
          key, imode = result.groups()
          omega[int(imode)-2] = parameters[key]
        elif re.search(u'\\d+&\\d+', k):
          result = re.search(u'(\\w+).*\\|\\d+s(\\d+)&(\\d+)', k)
          key, istate1, istate2 = result.groups()
          if '-' in k:
            e0[int(istate1)-1, int(istate2)-1] = -1*parameters[key]
            e0[int(istate2)-1, int(istate1)-1] = -1*parameters[key]
          else:
            e0[int(istate1)-1, int(istate2)-1] = parameters[key]
            e0[int(istate2)-1, int(istate1)-1] = parameters[key]
    except(IndexError):
      print("Wrong state or mode: "+k)
      sys.exit()
    except(ValueError,KeyError):
      print("Wrong value: "+k)
      sys.exit()
    return e0, omega, coef1, coef2
