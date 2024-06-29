from .tdvp import tdvp1 as _tdvp1
from .tdvp import tdvp2 as _tdvp2
import sys
def tdvp1site(rin,pall,dt,update_type='krylov',mmax=30,nsteps=1):
  argdict = {'dt':dt,'mmax':mmax,'nsteps':nsteps,'update_type':update_type}
  try:
    for key in ['nsteps','mmax']:
      argdict[key] = int(argdict[key])
    for key in ['dt']:
      argdict[key] = float(argdict[key])
    for key in ['dt','nsteps','mmax']:
      if (argdict[key] <= 0):
        print("Wrong value in tdvp: ",key," = ",argdict[key])
        sys.exit()
    key = 'update_type'
    argdict[key] = argdict[key].lower()
    if (not argdict[key] in ['krylov','rk4']):
      print("Wrong value in tdvp: update_type = ",update_type)
      sys.exit()
    if(rin.__class__.__name__ != "MPS"):
      print("Wrong value in tdvp: improper MPS")
      sys.exit()
    if(pall.__class__.__name__ != "MPO"):
      print("Wrong value in tdvp: improper MPO")
      sys.exit()
  except (ValueError):
    print("Wrong value in tdvp: ",key," = ",argdict[key])
    sys.exit()
  for istep in range(nsteps):
    rin = _tdvp1(rin,pall,argdict['dt'],argdict['update_type'],argdict['mmax'])
  return rin
def tdvp2site(rin,pall,dt,mmax=30,small=1e-13,nrmax=50,nsteps=1):
  argdict = {'dt':dt,'mmax':mmax,'small':small,'nrmax':nrmax,'nsteps':nsteps}
  try:
    for key in ['nsteps','mmax']:
      argdict[key] = int(argdict[key])
    for key in ['dt','small','nrmax']:
      argdict[key] = float(argdict[key])
    for key in ['dt','nsteps','mmax','small']:
      if (argdict[key] <= 0):
        print("Wrong value in tdvp: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax']:
      if (argdict[key] < 0):
        print("Wrong value in tdvp: ",key," = ",argdict[key])
        sys.exit()
    if(rin.__class__.__name__ != "MPS"):
      print("Wrong value in tdvp: improper MPS")
      sys.exit()
    if(pall.__class__.__name__ != "MPO"):
      print("Wrong value in tdvp: improper MPO")
      sys.exit()
  except (ValueError):
    print("Wrong value in tdvp: ",key," = ",argdict[key])
    sys.exit()
  for istep in range(nsteps):
    rin = _tdvp2(rin,pall,argdict['dt'],argdict['mmax'],argdict['small'],argdict['nrmax'])
  return rin
