from .ksltt import ksltt as _ksltt
import sys
def ksltt(rin,pall,dt,mmax=30,nsteps=1):
  argdict = {'dt':dt,'mmax':mmax,'nsteps':nsteps}
  try:
    for key in ['nsteps','mmax']:
      argdict[key] = int(argdict[key])
    for key in ['dt']:
      argdict[key] = float(argdict[key])
    for key in ['dt','nsteps','mmax']:
      if (argdict[key] <= 0):
        print("Wrong value in ksltt: ",key," = ",argdict[key])
        sys.exit()
    if(rin.__class__.__name__ != "MPS"):
      print("Wrong value in ksltt: improper MPS")
      sys.exit()
    if(pall.__class__.__name__ != "MPO"):
      print("Wrong value in ksltt: improper MPO")
      sys.exit()
  except (ValueError):
    print("Wrong value in ksltt: ",key," = ",argdict[key])
    sys.exit()
  for istep in range(nsteps):
    rin = _ksltt(rin,pall,argdict['dt'],argdict['mmax'])
  return rin
__all__=["ksltt"]
