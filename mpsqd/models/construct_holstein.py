import numpy as np
import functools
from ..utils import MPS,MPS2MPO,add_tensor
from .setupfreqs import setupfreqs_debye as setupfreqs

def construct_holstein(nb,nsite,nl1,eta,omega,beta,delta,small,nrmax,need_trun=True):
  nlevel = nsite*nl1
  wval,gval,gval_i,k0,normfac = setupfreqs(nlevel,eta,omega,beta)
  _add_tensor = functools.partial(add_tensor,small=small,nrmax=nrmax,need_trun=need_trun)

  hsys = np.zeros((nsite,nsite),dtype=np.float64)
  for i in range(nsite):
    i1 = i-1
    if i1 >= 0:
      hsys[i1,i] = delta
      hsys[i,i1] = delta
    i2 = i+1
    if i2 < nsite:
      hsys[i2,i] = delta
      hsys[i,i2] = delta
  hsys[0,nsite-1] = delta
  hsys[nsite-1,0] = delta

  p0 = MPS(nlevel+2,nb**2)

#---------------------------------------------------------
# vl and vr
  vl = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
  vr = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
  for i in range(nsite):
    ii = i*nsite+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0


#---------------------------------------------------------
# vmid
  vmid = []
  for i in range(nlevel):
    vtmp = np.zeros((1, nb[i+1]**2, 1),dtype=np.complex128)
    for j in range(nb[i+1]):
      ii = j*nb[i+1]+j
      vtmp[0,ii,0] = 1.0
    vmid.append(vtmp)

#---------------------------------------------------------
# add to p0.nodes
  p0.nodes.append(vl)

  for i in range(nlevel):
    p0.nodes.append(vmid[i])

  p0.nodes.append(vr)

#  rout = delta_all5(p0)

  drho = p0.copy()
  rtmp1 = p0.copy()
  nlen = len(p0.nodes)

#-----------------------------------------------------------
# the Hamiltonian term
  vl = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
  vr = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
  for i in range(nsite):
    for j in range(nsite):
      ii = j*nsite + i
      vl[0,ii,0] = -1.0j*hsys[i,j]
      vr[0,ii,0] =  1.0j*hsys[j,i]


  drho.nodes[0] = vl
  rtmp1.nodes[nlen-1] = vr
  drho = _add_tensor(drho,rtmp1,1.0)
  print("after hsys, shape of rout",drho.ndim())

#-----------------------------------------------------------
# the k0 term
  print("k0=",k0)
  rtmp1 = p0.copy()
  rtmp2 = p0.copy()
  vl = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
  vr = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)


  for i in range(nsite):
    ii = i*nsite+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0


  rtmp1.nodes[0] = vl
  rtmp2.nodes[nlen-1] = vr
  rtmp1 = _add_tensor(rtmp1,rtmp2,1.0)


  for i in range(nsite):
    rtmp2 = p0.copy()
    vl = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
    vr = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)

    ii = i*nsite+i
    vl[0,ii,0] = -2.0
    vr[0,ii,0] = 1.0


    rtmp2.nodes[0] = vl
    rtmp2.nodes[nlen-1] = vr


    rtmp1 = _add_tensor(rtmp1,rtmp2,1.0)

  drho = _add_tensor(drho,rtmp1,-1.0*k0)


  print("after k0, shape of rout",drho.ndim())

#-----------------------------------------------------------
# the diagonal terms
  for i in range(nsite):
    for j in range(nl1):
      ii = j + i*nl1
      rtmp1 = p0.copy()
      vtmp = np.zeros((1, nb[ii+1]*nb[ii+1], 1),dtype=np.complex128)

      for k in range(nb[ii+1]):
        jj = k*nb[ii+1]+k
        vtmp[0,jj,0] = -1.0*k*wval[j]


      rtmp1.nodes[ii+1] = vtmp.copy()
      drho = _add_tensor(drho,rtmp1,1.0)


  print("after diagonal, shape of rout",drho.ndim())

#  return drho
#========================================================
# the hierarchical terms
  for i in range(nsite):

    ii = i*nl1

    for k in range(nl1):


      rtmp1 = p0.copy()
      rtmp2 = p0.copy()


      vl = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)
      vr = np.zeros((1, nsite*nsite, 1),dtype=np.complex128)


      jj = i + i*nsite

      vl[0,jj,0] = 1.0
      vr[0,jj,0] = 1.0


      vtmp1 = np.zeros((1, nb[ii+k+1]*nb[ii+k+1], 1),dtype=np.complex128)
      vtmp2 = np.zeros((1, nb[ii+k+1]*nb[ii+k+1], 1),dtype=np.complex128)


      for j in range(nb[ii+k+1]):

#-------------------------------------------------------------------------
        j1 = j-1
        if j1>=0:
          jj = j1*nb[ii+k+1]+j
          vtmp1[0,jj,0] += gval_i[k]/normfac[k]*np.sqrt(1.0+j1)
          vtmp1[0,jj,0] -= 1.0j*gval[k]/normfac[k]*np.sqrt(1.0+j1)

          vtmp2[0,jj,0] += gval_i[k]/normfac[k]*np.sqrt(1.0+j1)
          vtmp2[0,jj,0] += 1.0j*gval[k]/normfac[k]*np.sqrt(1.0+j1)

#-------------------------------------------------------------------------
        j1 = j+1
        if j1<nb[ii+k+1]:
          jj = j1*nb[ii+k+1]+j
          vtmp1[0,jj,0] -= 1.0j*normfac[k]*np.sqrt(j1*1.0)
          vtmp2[0,jj,0] += 1.0j*normfac[k]*np.sqrt(j1*1.0)


      rtmp1.nodes[0] = vl
      rtmp1.nodes[ii+k+1] = vtmp1

      rtmp2.nodes[nlen-1] = vr
      rtmp2.nodes[ii+k+1] = vtmp2

      drho = _add_tensor(drho,rtmp1,1.0)
      drho = _add_tensor(drho,rtmp2,1.0)

  print("after hierachial, shape of rout",drho.ndim())
  rout = MPS2MPO(drho)
  return rout
