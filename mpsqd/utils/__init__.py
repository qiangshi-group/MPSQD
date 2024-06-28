from .mpsdef import MPS, MPO, MPO2MPS, MPS2MPO, calc_overlap, calc_element, calc_rho, print_mps, print_mpo
from .prod_tensor import prod_tensor_mat as _prod_tensor_mat
from .add_tensor import add_tensor as _add_tensor
from .read_write_mpo import read_mpo_file, write_mpo_file, read_mps_file, write_mps_file
import sys
au2fs = 2.418884254e-2

def multiply_tensor(pall,rin,small=1e-14,nrmax=50,need_trun=True):
  argdict = {'small':small,'nrmax':nrmax,'need_trun':need_trun}
  try:
    for key in ['nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['small']:
      argdict[key] = float(argdict[key])
    for key in ['small']:
      if (argdict[key] <= 0):
        print("Wrong value in prod_tensor_mat: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax']:
      if (argdict[key] < 0):
        print("Wrong value in prod_tensor_mat: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in prod_tensor_mat: need_trun  = ",argdict['need_trun'])
      sys.exit()
    if(rin.__class__.__name__ != "MPS"):
      print("Wrong value in prod_tensor_mat: improper MPS")
      sys.exit()
    if(pall.__class__.__name__ != "MPO"):
      print("Wrong value in prod_tensor_mat: improper MPS")
      sys.exit()
  except (ValueError):
    print("Wrong value in prod_tensor_mat: ",key," = ",argdict[key])
    sys.exit()
  return _prod_tensor_mat(pall,rin,argdict['small'],argdict['nrmax'],argdict['need_trun'])

def add_tensor(r1,r2,coeff=1,small=1e-14,nrmax=50,need_trun=True):
  argdict = {"coeff":coeff,'small':small,'nrmax':nrmax,'need_trun':need_trun}
  try:
    for key in ['nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['small']:
      argdict[key] = float(argdict[key])
    for key in ['coeff']:
      argdict[key] = complex(argdict[key])
    for key in ['small']:
      if (argdict[key] <= 0):
        print("Wrong value in add_tensor: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax']:
      if (argdict[key] < 0):
        print("Wrong value in add_tensor: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in add_tensor: need_trun  = ",argdict['need_trun'])
      sys.exit()
    if(not (r1.__class__.__name__ in ["MPS","MPO"])):
      print("Wrong value in add_tensor: improper MPS or MPO r1")
      sys.exit()
    if(not (r2.__class__.__name__ in ["MPS","MPO"])):
      print("Wrong value in add_tensor: improper MPS or MPO r2")
      sys.exit()
  except (ValueError):
    print("Wrong value in add_tensor: ",key," = ",argdict[key])
    sys.exit()
  if r1.__class__.__name__!=r2.__class__.__name__:
    print("Wrong value in add_tensor: add MPO to MPS")
    sys.exit()

  if r1.__class__.__name__=='MPS' and r2.__class__.__name__=='MPS':
    return _add_tensor(r1,r2,coeff,argdict['small'],argdict['nrmax'],argdict['need_trun'])
  if r1.__class__.__name__=='MPO' and r2.__class__.__name__=='MPO':
    return MPS2MPO(_add_tensor(MPO2MPS(r1),MPO2MPS(r2),argdict['coeff'],argdict['small'],argdict['nrmax'],argdict['need_trun']))

def truncate_tensor(rin,small=1e-14,nrmax=50):
  argdict = {'small':small,'nrmax':nrmax}
  try:
    for key in ['nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['small']:
      argdict[key] = float(argdict[key])
    for key in ['small']:
      if (argdict[key] <= 0):
        print("Wrong value in trun_tensor: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax']:
      if (argdict[key] < 0):
        print("Wrong value in trun_tensor: ",key," = ",argdict[key])
        sys.exit()
  except (ValueError):
    print("Wrong value in trun_tensor: ",key," = ",argdict[key])
    sys.exit()
  if(rin.__class__.__name__ == "MPS" or rin.__class__.__name__ == "MPO"):
    return rin.truncation(small,nrmax)
  else:
    print("Wrong value in trun_tensor: improper tensor")
    sys.exit()
