import numpy as np
import sys
from ..rk4 import rk4
from ..tdvp import tdvp1site, tdvp2site
from ..utils import MPS, MPO, write_mps_file, write_mpo_file
from .construct_sb import construct_sb as _construct_sb
from .construct_holstein import construct_holstein as _construct_holstein
from .init_rho import init_mps as _init_mps
from .construct_tds import construct_tds, init_tds, write_file_define
au2fs = 2.418884254e-2

def construct_sb(nlevel,ndvr,nbv,eta,omega,beta,eps,vcoupling,small=1e-14,nrmax=50,need_trun=True):
  argdict = {'nlevel':nlevel,'ndvr':ndvr,'nbv':nbv,'eta':eta,'omega':omega,'beta':beta,'eps':eps,'vcoupling':vcoupling,'small':small,'nrmax':nrmax,'need_trun':need_trun}
  try:
    for key in ['nlevel','ndvr','nbv','nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['eta','omega','beta','eps','vcoupling','small']:
      argdict[key] = float(argdict[key])
    for key in ['nlevel','ndvr','nbv','omega','beta','small']:
      if (argdict[key] <= 0):
        print("Wrong value in construct_sb: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax','eta']:
      if (argdict[key] < 0):
        print("Wrong value in construct_sb: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in construct_sb: need_trun  = ",argdict['need_trun'])
      sys.exit()
  except (ValueError):
    print("Wrong value in construct_sb: ",key," = ",argdict[key])
    sys.exit()
  nb = np.zeros(argdict['nlevel']+2,dtype=int)
  nb[0] = argdict['ndvr']
  nb[1:argdict['nlevel']+1] = argdict['nbv']
  nb[argdict['nlevel']+1] = argdict['ndvr']
  return argdict, _construct_sb(argdict['nlevel'],argdict['ndvr'],nb,argdict['eta'],argdict['omega'],argdict['beta'],argdict['eps'],argdict['vcoupling'],argdict['small'],argdict['nrmax'],argdict['need_trun'])

def construct_holstein(nbv,nsite,nl1,eta,omega,beta,delta,small=1e-14,nrmax=50,need_trun=True):
  argdict = {'nbv':nbv,'nsite':nsite,'nl1':nl1,'eta':eta,'omega':omega,'beta':beta,'delta':delta,'small':small,'nrmax':nrmax,'need_trun':need_trun}
  try:
    for key in ['nbv','nsite','nl1','nrmax']:
      argdict[key] = int(argdict[key])
    for key in ['eta','omega','beta','delta','small']:
      argdict[key] = float(argdict[key])
    for key in ['nbv','nsite','nl1','omega','beta','small']:
      if (argdict[key] <= 0):
        print("Wrong value in construct_holstein: ",key," = ",argdict[key])
        sys.exit()
    for key in ['nrmax','eta']:
      if (argdict[key] < 0):
        print("Wrong value in construct_holstein: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in construct_holstein: need_trun  = ",argdict['need_trun'])
      sys.exit()
  except (ValueError):
    print("Wrong value in construct_holstein: ",key," = ",argdict[key])
    sys.exit()
  nlevel = argdict['nsite']*argdict['nl1']
  nb = np.zeros(nlevel+2,dtype=int)
  nb[0] = argdict['nsite']
  nb[1:nlevel+1] = argdict['nbv']
  nb[nlevel+1] = argdict['nsite']
  return argdict, _construct_holstein(nb,argdict['nsite'],argdict['nl1'],argdict['eta'],argdict['omega'],argdict['beta'],argdict['delta'],argdict['small'],argdict['nrmax'],argdict['need_trun'])

def init_mps(rho0,nlevel,nbv,nrtt=1,small=1e-14,need_trun=True):
  argdict = {'nlevel':nlevel,'nbv':nbv,'nrtt':nrtt,'small':small,'need_trun':need_trun}
  try:
    for key in ['nlevel','nbv','nrtt']:
      argdict[key] = int(argdict[key])
    for key in ['small']:
      argdict[key] = float(argdict[key])
    for key in ['nlevel','nbv','nrtt','small']:
      if (argdict[key] <= 0):
        print("Wrong value in init_mps: ",key," = ",argdict[key])
        sys.exit()
    if (not(argdict['need_trun'] in [True,False])):
      print("Wrong value in init_mps: need_trun  = ",argdict['need_trun'])
      sys.exit()
    if (len(np.shape(rho0)) != 2 or np.shape(rho0)[0]!=np.shape(rho0)[1]):
      print("Wrong value in init_mps: improper rho0")
      sys.exit()
  except (ValueError):
    print("Wrong value in init_mps: ",key," = ",argdict[key])
    sys.exit()
  except (IndexError):
    print("Wrong value in init_mps: improper rho0")
    sys.exit()

  ndvr = np.shape(rho0)[0]
  nb = np.zeros(argdict['nlevel']+2,dtype=int)
  nb[0] = ndvr
  nb[1:argdict['nlevel']+1] = argdict['nbv']
  nb[argdict['nlevel']+1] = ndvr
  return argdict, _init_mps(rho0,argdict['nlevel'],nb,argdict['nrtt'],argdict['small'],argdict['need_trun'])

class HEOM():
  def prop(self,dt,nsteps,out_file_name="output.dat",prop_type="1tdvp",update_type="krylov",mmax=30,small=1e-13,nrmax=50,need_trun=True):
    argdict = {'prop_type':prop_type}
    argdict['prop_type'] = argdict['prop_type'].lower()
    if (not(argdict['prop_type'] in ['1tdvp','2tdvp','rk4'])):
      print("Wrong value in HEOM.prop: prop_type=",prop_type)
      sys.exit()
    ndvr = self.rin.ndim()[1,0]
    rout = np.zeros((int(ndvr*(ndvr+1)/2+1),nsteps),dtype=np.complex128)
    s1 = "  "
    rin = self.rin
    rho1 = rin.calc_rho()
    rout[0,0] = 0.0
    for i in range(ndvr):
      for j in range(i+1):
        rout[int(i*(i+1)/2+j+1),0] = rho1[i,j]
    output = str(0.0)+ s1
    for i in range(ndvr):
      for j in range(i+1):
        output = output + str(rho1[i,j]) + s1
    output = output + '\n'
    fp = open(out_file_name,'w')
    fp.write(output)
    fp.flush()
    for istep in range(1,nsteps):
      print("istep =", istep)
      if(argdict['prop_type']=="1tdvp"):
        rin = tdvp1site(rin, self.pall, dt, update_type, mmax=mmax)
      elif(argdict['prop_type']=="rk4"):
        rin = rk4(rin, self.pall, dt, small=small,nrmax=nrmax,need_trun=need_trun)
      else:
        rin = tdvp2site(rin, self.pall, dt, mmax=mmax, small=small, nrmax=nrmax)
      rho1 = rin.calc_rho()
      output = str(istep*dt) + s1
      for i in range(ndvr):
        for j in range(i+1):
          output = output + str(rho1[i,j]) + s1
      output = output + '\n'
      fp.writelines(output)
      fp.flush()
      rout[0,istep] = istep*dt
      for i in range(ndvr):
        for j in range(i+1):
          rout[int(i*(i+1)/2+j+1),istep] = rho1[i,j]
    self.rout = rin
    return rout

  def write_mpo_file(self,pall_file):
    write_mpo_file(self.pall,pall_file)
    return

  def write_mps_file(self,pall_file):
    write_mps_file(self.rin,pall_file)
    return

class Spin_boson(HEOM):
  def __init__(self,nlevel,nbv,eta,omega,beta,eps,vcoupling,rho0,small=1e-14,nrmax=50,nrtt=1,need_trun=True):
    self.params, self.pall = construct_sb(nlevel,np.shape(rho0)[0],nbv,eta,omega,beta,eps,vcoupling,small,nrmax,need_trun)
    mpsparams, self.rin = init_mps(rho0,nlevel,nbv,nrtt,small,need_trun)
    self.params.update(mpsparams)
    return

class Holstein(HEOM):
  def __init__(self,nbv,nl1,eta,omega,beta,delta,rho0,small=1e-14,nrmax=50,nrtt=1,need_trun=True):
    self.params, self.pall = construct_holstein(nbv,np.shape(rho0)[0],nl1,eta,omega,beta,delta,small,nrmax,need_trun)
    mpsparams, self.rin = init_mps(rho0,np.shape(rho0)[0]*nl1,nbv,nrtt,small,need_trun)
    self.params.update(mpsparams)
    return

class Schrodinger():
  def construct(self,construct_type=None):
    try:
      if(construct_type == None):
        construct_type1 = None
      else:
        construct_type1 = str(construct_type.lower())
        if (not construct_type1 in ['simple','term']):
          print("Unrecognized construct_type in Schrodinger.construct: ",construct_type0," . Default construct_type will be used.")
          construct_type1 = None
    except (AttributeError):
      print("Unrecognized construct_type in Schrodinger.construct: ",construct_type0," . Default construct_type will be used.")
      construct_type1 = None
    self.pall,self.rin = construct_tds(self.params,construct_type1)
    return

  def prop(self,dt,nsteps,out_file_name="output.dat",prop_type="1tdvp",update_type="krylov",mmax=30,small=1e-14,nrmax=50,need_trun=True):
    prop_type = prop_type.lower()
    argdict = {'prop_type':prop_type}
    argdict['prop_type'] = argdict['prop_type'].lower()
    if (not(argdict['prop_type'] in ['1tdvp','2tdvp','rk4'])):
      print("Wrong value in Schrodinger.prop: prop_type=",prop_type)
      sys.exit()
    s1 = "  "
    rin = self.rin
    ndvr = rin.ndim()[1,0]
    rout = np.zeros((ndvr+1,nsteps),dtype=np.float64)
    pop = rin.calc_pop()
    rout[0,0] = 0.0
    for i in range(ndvr):
      rout[i+1,0] = pop[i]
    output = (str(0.0)+ s1)
    for i in range(ndvr):
      output = output + str(pop[i].real) + s1
    output = output + '\n'
    fp = open(out_file_name,'w')
    fp.write(output)
    fp.flush()
    for istep in range(1,nsteps):
      print("istep =", istep)
      if(argdict['prop_type']=="1tdvp"):
        rin = tdvp1site(rin, self.pall, dt, update_type, mmax=mmax)
      elif(argdict['prop_type']=="rk4"):
        rin = rk4(rin, self.pall, dt, small=small,nrmax=nrmax,need_trun=need_trun)
      else:
        rin = tdvp2site(rin, self.pall, dt, mmax=mmax, small=small, nrmax=nrmax)
      pop = rin.calc_pop()
      rout[0,istep] = istep*dt
      for i in range(ndvr):
        rout[i+1,istep] = pop[i]
      output = str(istep*dt*au2fs)+ s1
      for i in range(ndvr):
        output = output + str(pop[i].real) + s1
      output = output + '\n'
      fp.writelines(output)
      fp.flush()
    self.rout = rin
    return rout

  def write_mpo_file(self,pall_file):
    write_mpo_file(self.pall,pall_file)
    return

  def write_mps_file(self,pall_file):
    write_mps_file(self.rin,pall_file)
    return

  def write_inputfile(self,filename):
    write_file_define(self.params,filename)
    return

class Vibronic(Schrodinger):
  def __init__(self,fname=None,input_type="d"):
    self.params = init_tds(fname,input_type)
    return

class Frenkel(Schrodinger):
  def __init__(self,fname=None,input_type="d",multi_mole=True):
    self.params = init_tds(fname,input_type,multi_mole)
    return
