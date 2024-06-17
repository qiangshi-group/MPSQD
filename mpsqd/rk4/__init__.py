from .rk4 import rk4 as _rk4
def rk4(rin,pall,dt,small=1e-14,nrmax=50,nsteps=1,need_trun=True):
  argdict = {'dt':dt,'small':small,'nrmax':nrmax,'nsteps':nsteps,'need_trun':need_trun}
  try:
    for key in ['nsteps','nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['dt','small']:
      argdict[key] = float(argdict[key])
    for key in ['dt','small','nsteps']:
      if (argdict[key] <= 0):
        print("Wrong value in rk4: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax']:
      if (argdict[key] < 0):
        print("Wrong value in rk4: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in rk4: need_trun  = ",argdict['need_trun'])
      sys.exit()
    if(rin.__class__.__name__ != "MPS"):
      print("Wrong value in rk4: improper MPS")
      sys.exit()
    if(pall.__class__.__name__ != "MPO"):
      print("Wrong value in rk4: improper MPO")
      sys.exit()
  except (ValueError):
    print("Wrong value in prod_tensor_mat: ",key," = ",argdict[key])
    sys.exit()
  for istep in range(nsteps):
    rin = _rk4(rin,pall,argdict['dt'],argdict['small'],argdict['nrmax'],argdict['need_trun'])
  return rin
__all__=["rk4"]
