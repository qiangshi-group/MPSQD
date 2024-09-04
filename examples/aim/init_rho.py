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
    if s1[i-1]>pa.small_mps:
      ii += 1
  if ii==0:
    sys.exit("initial density operator has rank 0, stop")
  print('rank',ii)
# initial rhoall
  rhoall = MPS(pa.nlevel+2)

# vrhol and vrhor
  vrhol = np.zeros((1, pa.ndvr, ii),dtype=np.complex128)
  vrhor = np.zeros((ii, pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(ii):
      vrhol[0,i,j] = u1[i,j]*np.sqrt(s1[j])
      vrhor[j,i,0] = vt1[j,i]*np.sqrt(s1[j])

# vmid
  vmid = []
  for i in range(pa.nlevel-1):
    #print("vmid:({})".format(i))
    vtmp = np.zeros((1, pa.nb[i+2], 1),dtype=np.complex128)
    vtmp[0,0,0] = 1.0
    vmid.append(vtmp)
  vtmp = np.zeros((1, pa.nb[pa.nlevel+1], 1),dtype=np.complex128)
  vtmp[0,0,0] = 1.0
  vmid.append(vtmp)

# store them in nodes
  rhoall.nodes.append(vrhol.copy())
  rhoall.nodes.append(vrhor.copy())

  for i in range(pa.nlevel):
    rhoall.nodes.append(vmid[i].copy())

  rhoall.nb = pa.nb

  print("length of rhoall: {}".format(len(rhoall.nodes)))
  print("rhoall initialization done!")

  return rhoall
