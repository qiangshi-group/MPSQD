import sys
import numpy as np
import params as pa
from mpsqd.utils import MPS

def init_rho():

  rho0 = np.zeros((pa.ndvr,pa.ndvr),dtype=np.complex128)
  rho0[0,0] = 1.0


  u1,s1,vt1 = np.linalg.svd(rho0,full_matrices=False)
  ii = 0
  for i in range(1,pa.ndvr+1,1):
    if s1[i-1]>pa.small:
      ii += 1
  print("rank of initial rho =", ii)
  if ii==0:
    sys.exit("initial density operator has rank 0, stop")
  elif (ii > pa.nrtt):
    sys.exit("rank of initial density larger than nrtt, stop")

# initial rhoall
  rhoall = MPS(pa.nmps)

# vl and vr
  vl = np.zeros((1, pa.ndvr, pa.nrtt),dtype=np.complex128)
  vr = np.zeros((pa.nrtt, pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(ii):
      vl[0,i,j] = u1[i,j]
      vr[j,i,0] = vt1[j,i]

# vmid
  vmid = []
  for i in range(1,pa.nlevel+1):
    #print("vmid:({})".format(i))
    vtmp = np.zeros((pa.nrtt, pa.nb[i], pa.nrtt),dtype=np.complex128)
    for j in range(ii):
      vtmp[j,0,j] = 1.0
      vmid.append(vtmp)

# store them in nodes
  rhoall.nodes.append((vl))
  for i in range(pa.nlevel):
    rhoall.nodes.append((vmid[i]))
  rhoall.nodes.append((vr))
  rhoall.nb = pa.nb
  print("length of rhoall: {}".format(len(rhoall.nodes)))
  print("rhoall initialization done!")
  return rhoall
