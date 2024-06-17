import numpy as np
import functools
from ...utils import add_tensor as _add_tensor
from ...utils import MPS

def construct(nlevel,nb,e0,coef1,coef2,hsys,xtmp1,xtmp2,small,nrmax,need_trun=True):
  add_tensor = functools.partial(_add_tensor,small=small,nrmax=nrmax,need_trun=need_trun)
  pall = inimpo(nlevel,nb,e0,hsys,add_tensor)
  for i in range(nb[0]):
    for j in range(i,nb[0]):
      temppall = mpo(nlevel,nb,i, j, coef1[i,j,:], coef2[i,j,:,:], xtmp1 ,xtmp2,add_tensor)
      pall = add_tensor(pall,temppall, 1.0)
      print(str([i,j])+"done")

  return pall
#============================================================
def inimpo(nlevel,nb,e0,hsys,add_tensor):
  rtmp0 = MPS(nlevel+1,nb)
  for k in range(nlevel+1):
    vtmp = np.reshape(np.eye(nb[k],dtype=np.complex128),(1,nb[k]**2,1),order='F') 
    rtmp0.nodes.append(vtmp)

#e0 term
  mpo = rtmp0.copy()
  vtmp = np.reshape(e0,(1,nb[0]**2,1),order='F').astype(np.complex128)
  mpo.nodes[0] = vtmp.copy()*(-1.0j)
  print("e0 term done")

# harmonic term
  for k in range(nlevel):
    rtmp = rtmp0.copy()
    rtmp.nodes[k+1] = hsys[k].copy().astype(np.complex128)
    mpo = add_tensor(mpo,rtmp, -1.0j)
  print("harmonic term done")
  return mpo
#============================================================
# coef1 and coef2 are the first and second order coefficients
def mpo(nlevel,nb,i1, j1, coef1, coef2, xtmp1, xtmp2, add_tensor):

#------------------------------------------------------
  rtmp0 = MPS(nlevel+1,nb)
#------------------------------------------------------
# construct vl
  vl = np.zeros((1, nb[0]**2, 1),dtype=np.complex128)
  vl[0,i1*nb[0]+j1,0] = 1.0
  vl[0,j1*nb[0]+i1,0] = 1.0
  rtmp0.nodes.append(vl)
#------------------------------------------------------
# other terms for the vibrational modes
  for k in range(nlevel):
    vtmp = np.reshape(np.eye(nb[k+1],dtype=np.complex128),(1,nb[k+1]**2,1),order='F')
    rtmp0.nodes.append(vtmp)

#======================================================
  mpo = rtmp0.copy()
  mpo.nodes[0] *= 0
#======================================================
# the linear terms, note that the h\omega part goes to split_bath
  for k in range(nlevel):
    if (np.abs(coef1[k]) < 1e-12 and np.abs(coef2[k,k]) < 1e-12):
      continue
# copy from the template
    rtmp1 = rtmp0.copy()

# need to midify the kth node
    rtmp1.nodes[k+1] = coef1[k]*xtmp1[k].copy() + coef2[k,k]*xtmp2[k].copy()
    mpo = add_tensor(mpo,rtmp1, -1.0j)
  print("linear + quadratic terms done")
#======================================================
# now the second order cross terms
  for k1 in range(nlevel):
    for k2 in range(k1):
# copy from the template
      rtmp1 = rtmp0.copy()
# need to midify the k1 and k2 nodes
# important, there is a 2.0 facotor for the cross terms
      if (np.abs(coef2[k2,k1]) > 1e-12):
        print("k1, k2 =", k1, k2, coef2[k2,k1])
# split coef2 to each node
        rtmp1.nodes[k1+1] = np.sqrt(abs(coef2[k2,k1]))*xtmp1[k1].copy()
        rtmp1.nodes[k2+1] = coef2[k2,k1]/np.sqrt(abs(coef2[k2,k1]))*xtmp1[k2].copy()
        mpo = add_tensor(mpo,rtmp1, -2.0j)

  return mpo
