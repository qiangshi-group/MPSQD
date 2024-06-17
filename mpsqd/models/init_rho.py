import sys
import numpy as np
from ..utils import MPS

def init_mps(rho0,nlevel,nb,nrtt=1,small=1e-14,need_trun=True):
  ndvr = np.shape(rho0)[0]

  u1,s1,vt1 = np.linalg.svd(rho0,full_matrices=False)
  if need_trun:
    ii = 0
    for i in range(ndvr):
      if s1[i]>small:
        ii += 1
  else:
    ii = len(s1)
  print("rank of initial rho =", ii)
  if ii==0:
    sys.exit("initial density operator has rank 0, stop")
  elif (ii > nrtt):
    sys.exit("rank of initial density larger than nrtt, stop")

# initial rhoall
  rhoall = MPS(nlevel+2,nb**2)

# vl and vr
  vl = np.zeros((1, ndvr, nrtt),dtype=np.complex128)
  vr = np.zeros((nrtt, ndvr, 1),dtype=np.complex128)
  for i in range(ndvr):
    for j in range(ii):
      vl[0,i,j] = u1[i,j]*np.sqrt(s1[j])
      vr[j,i,0] = vt1[j,i]*np.sqrt(s1[j])

# vmid
  vmid = []
  for i in range(1,nlevel+1):
    vtmp = np.zeros((nrtt, nb[i], nrtt),dtype=np.complex128)
    for j in range(ii):
      vtmp[j,0,j] = 1.0
    vmid.append(vtmp)

# store them in nodes
  rhoall.nodes.append((vl))
  rhoall.nodes = rhoall.nodes + vmid
  rhoall.nodes.append((vr))
  print("length of rhoall: {}".format(len(rhoall.nodes)))
  print("rhoall initialization done!")
  return rhoall
