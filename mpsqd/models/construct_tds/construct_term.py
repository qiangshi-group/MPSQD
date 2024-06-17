import numpy as np
import functools
from ...utils import add_tensor as _add_tensor
from ...utils import MPS

def construct(nlevel,nb,e0,coef1,coef2,hsys,xtmp1,xtmp2,small,nrmax,need_trun=True):

  add_tensor = functools.partial(_add_tensor,small=small,nrmax=nrmax,need_trun=need_trun)
  rtmp0 = MPS(nlevel+1,nb)
  for k in range(nlevel+1):
    if (k==0):
      vtmp = np.zeros((1, nb[k]**2, 1),dtype=np.complex128)
    else:
      vtmp = np.reshape(np.eye(nb[k],dtype=np.complex128),(1,nb[k]**2,1),order='F')
    rtmp0.nodes.append(vtmp)

#e0 term
  mpo = rtmp0.copy()
  vtmp = np.reshape(e0,(1,nb[0]**2,1),order='F').astype(np.complex128)
  mpo.nodes[0] = vtmp.copy()*(-1.0j)
  print("e0 term done")

# harmonic term
  mpo1 = rtmp0.copy()
  for k in range(nlevel):
    rtmp = rtmp0.copy()
    rtmp.nodes[0] = np.reshape(np.eye(nb[0],dtype=np.complex128),(1,nb[0]**2,1),order='F')
    rtmp.nodes[k+1] = hsys[k].copy().astype(np.complex128)
    mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -1.0j)
  print("harmonic term done")

# diagonal liner term
  mpo1 = rtmp0.copy()
  for k in range(nlevel):
    if(np.abs([coef1[i,i,k] for i in range(nb[0])]).max() < 1e-12):
      continue
    rtmp = rtmp0.copy()
    for i1 in range(nb[0]):
      ii = i1*nb[0]+i1
      rtmp.nodes[0][0,ii,0] = coef1[i1,i1,k]
    rtmp.nodes[k+1] = xtmp1[k].copy().astype(np.complex128)
    mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -1.0j)
  print("diagonal liner term done")

# diagonal quadratic term
  mpo1 = rtmp0.copy()
  for k in range(nlevel):
    if(np.abs([coef2[i,i,k,k] for i in range(nb[0])]).max() < 1e-12):
      continue
    rtmp = rtmp0.copy()
    for i1 in range(nb[0]):
      ii = i1*nb[0]+i1
      rtmp.nodes[0][0,ii,0] = coef2[i1,i1,k,k]
    rtmp.nodes[k+1] = xtmp2[k].copy().astype(np.complex128)
    mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -1.0j)
  print("diagonal quadratic term done")

# diagonal bilinear term
  mpo1 = rtmp0.copy()
  for k1 in range(nlevel):
    for k2 in range(k1):
      if(np.abs([coef2[i,i,k2,k1] for i in range(nb[0])]).max() < 1e-12):
        continue
      rtmp = rtmp0.copy()
      for i1 in range(nb[0]):
        ii = i1*nb[0]+i1
        rtmp.nodes[0][0,ii,0] = coef2[i1,i1,k2,k1]
      rtmp.nodes[k1+1] = xtmp1[k1].copy().astype(np.complex128)
      rtmp.nodes[k2+1] = xtmp1[k2].copy().astype(np.complex128)
      mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -2.0j)
  print("diagonal bilinear term done")

  al = {}
  for i in range(nb[0]-1):
    for j in range(i+1,nb[0]):
      al[(i,j)] = 1

# off-diagonal liner term
  mpo1 = rtmp0.copy()
  for k in range(nlevel):
    for key in al.keys():
      if (np.abs(coef1[key[0],key[1],k]) > 1e-12):
        al[key] = 0
    while True:
      re,al = getlist(nb[0],al)
      if (len(re) == 0):
        break
      for key in re:
        rtmp = rtmp0.copy()
        rtmp.nodes[0][0,key[0]*nb[0]+key[1],0] = coef1[key[0],key[1],k]
        rtmp.nodes[0][0,key[1]*nb[0]+key[0],0] = coef1[key[0],key[1],k]
        rtmp.nodes[k+1] = xtmp1[k].copy().astype(np.complex128)
        mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -1.0j)
  print("off-diagonal liner term done")

# off-diagonal quadratic term
  mpo1 = rtmp0.copy()
  for k in range(nlevel):
    for key in al.keys():
      if (np.abs(coef2[key[0],key[1],k,k]) > 1e-12):
        al[key] = 0
    while True:
      re,al = getlist(nb[0],al)
      if (len(re) == 0):
        break
      for key in re:
        rtmp = rtmp0.copy()
        rtmp.nodes[0][0,key[0]*nb[0]+key[1],0] = coef2[key[0],key[1],k,k]
        rtmp.nodes[0][0,key[1]*nb[0]+key[0],0] = coef2[key[0],key[1],k,k]
        rtmp.nodes[k+1] = xtmp2[k].copy().astype(np.complex128)
        mpo1 = add_tensor(mpo1,rtmp,1.0)
  mpo = add_tensor(mpo,mpo1, -1.0j)
  print("off-diagonal quadratic term done")

# off-diagonal bilinear term
  mpo1 = rtmp0.copy()
  for k1 in range(nlevel):
    for k2 in range(k1):
      for key in al.keys():
        if (np.abs(coef2[key[0],key[1],k2,k1]) > 1e-12):
          al[key] = 0
      while True:
        re,al = getlist(nb[0],al)
        if (len(re) == 0):
          break
        for key in re:
          ii1 = key[0]*nb[0]+key[1]
          ii2 = key[1]*nb[0]+key[0]
          rtmp = rtmp0.copy()
          rtmp.nodes[0][0,ii1,0] = coef2[key[0],key[1],k2,k1]
          rtmp.nodes[0][0,ii2,0] = coef2[key[0],key[1],k2,k1]
          rtmp.nodes[k1+1] = xtmp1[k1].copy().astype(np.complex128)
          rtmp.nodes[k2+1] = xtmp1[k2].copy().astype(np.complex128)
          mpo1 = add_tensor(mpo1,rtmp, 1.0)
  mpo = add_tensor(mpo,mpo1, -2.0j)
  print("off-diagonal bilinear term done")

  return mpo

def getlist(ndvr,al):
  co = {}
  re = []
  for i in range(ndvr):
    co[i] = 0
  for i in range(ndvr-1):
    if (co[i] == 1):
      continue
    for j in range(i+1,ndvr):
      if (co[j] == 1):
        continue
      if (al[(i,j)] == 0):
        re.append((i,j))
        co[i] = 1
        co[j] = 1
        al[(i,j)] = 1
        break
  return(re,al)
