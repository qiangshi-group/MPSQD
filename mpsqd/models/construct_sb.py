import numpy as np
import functools
from ..utils import MPS,MPS2MPO,add_tensor
from .setupfreqs import setupfreqs_debye as setupfreqs

#==========================================================
def construct_sb(nlevel,ndvr,nb,eta,omega,beta,eps,vcoupling,small,nrmax,need_trun=True):
  wval,gval,gval_i,k0,normfac = setupfreqs(nlevel,eta,omega,beta)
  hsys = np.array(([eps,vcoupling],[vcoupling,-eps]),dtype=np.float64)
  sdvr = np.array(([-1.0,1.0]),dtype=np.float64)

  _add_tensor = functools.partial(add_tensor,small=small,nrmax=nrmax,need_trun=need_trun)

  print("ndvr =", ndvr)
  print("wval =", wval)
  print("gval =", gval)
  print("gval_i =", gval_i)
  print("normfac =", normfac)
  print("k0 =", k0)

  print("sdvr =", sdvr)
  print("hsys =", hsys)

  p0 = MPS(nlevel+2,nb**2)

# vl and vr
  vl = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)
  for i in range(ndvr):
    ii = i*ndvr+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0

# vmid
  vmid = []
  for i in range(nlevel):
    vtmp = np.zeros((1, nb[i+1]**2, 1),dtype=np.complex128)
    for j in range(nb[i+1]):
      ii = j*nb[i+1]+j
      vtmp[0,ii,0] = 1.0
    vmid.append(vtmp)

# add to p0.nodes
  p0.nodes.append(vl)
  for i in range(nlevel):
    p0.nodes.append(vmid[i])
  p0.nodes.append(vr)

  print("p0 calculation done!")

  drho = p0.copy()
  rtmp1 = p0.copy()
  nlen = len(p0.nodes)

#========================================================
# the Hamiltonian term
  vl = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)
  for i in range(ndvr):
    for j in range(ndvr):
      ii = j*ndvr + i
      vl[0,ii,0] = -1.0j*hsys[i,j]
      vr[0,ii,0] =  1.0j*hsys[j,i]

  drho.nodes[0] = vl
  rtmp1.nodes[nlen-1] = vr
  drho = _add_tensor(drho,rtmp1,1.0)

#========================================================
# the k0 term
  rtmp1 = p0.copy()
  rtmp2 = p0.copy()
  vl = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, ndvr*ndvr, 1),dtype=np.complex128)

  for i in range(ndvr):
    ii = i*ndvr+i
    vl[0,ii,0] = sdvr[i]**2
    vr[0,ii,0] = sdvr[i]**2

  rtmp1.nodes[0] = vl
  rtmp2.nodes[nlen-1] = vr
  rtmp1 = _add_tensor(rtmp1,rtmp2,1.0)


  rtmp2 = p0.copy()
  for i in range(ndvr):
    for j in range(ndvr):
      ii = i*ndvr+i
      jj = j*ndvr+j
      vl[0,ii,0] = -2.0*sdvr[i]
      vr[0,jj,0] = sdvr[j]


  rtmp2.nodes[0] = vl
  rtmp2.nodes[nlen-1] = vr

  rtmp1 = _add_tensor(rtmp1,rtmp2,1.0)
  drho = _add_tensor(drho,rtmp1,-1.0*k0)

#========================================================
# the diagonal terms
  for i in range(1,nlen-1):
    rtmp1 = p0.copy()
    vtmp = np.zeros((1, nb[i]*nb[i], 1),dtype=np.complex128)
    for j in range(nb[i]):
      jj = j*nb[i]+j
      vtmp[0,jj,0] = -1.0*j*wval[i-1]

    rtmp1.nodes[i] = vtmp
    drho = _add_tensor(drho,rtmp1,1.0)

#========================================================
# the hierarchical terms
  for i in range(1,nlen-1):

    rtmp1 = p0.copy()
    rtmp2 = p0.copy()
    vl = rtmp1.nodes[0].copy()
    vr = rtmp2.nodes[nlen-1].copy()

    for j in range(ndvr):
      jj = j*ndvr+j
      vl[0,jj,0] = sdvr[j]
      vr[0,jj,0] = sdvr[j]

    vtmp1 = np.zeros((1, nb[i]*nb[i], 1),dtype=np.complex128)
    vtmp2 = np.zeros((1, nb[i]*nb[i], 1),dtype=np.complex128)

    for j in range(nb[i]):

#-------------------------------------------------------------------------
      j1 = j-1
      if j1>=0:
        jj = j1*nb[i]+j
        vtmp1[0,jj,0] += gval_i[i-1]/normfac[i-1]*np.sqrt(1.0+j1)
        vtmp1[0,jj,0] -= 1.0j*gval[i-1]/normfac[i-1]*np.sqrt(1.0+j1)

        vtmp2[0,jj,0] += gval_i[i-1]/normfac[i-1]*np.sqrt(1.0+j1)
        vtmp2[0,jj,0] += 1.0j*gval[i-1]/normfac[i-1]*np.sqrt(1.0+j1)

#-------------------------------------------------------------------------
      j1 = j+1
      if j1<nb[i]:
        jj = j1*nb[i]+j
        vtmp1[0,jj,0] -= 1.0j*normfac[i-1]*np.sqrt(j1*1.0)
        vtmp2[0,jj,0] += 1.0j*normfac[i-1]*np.sqrt(j1*1.0)


    rtmp1.nodes[0] = vl
    rtmp1.nodes[i] = vtmp1

    rtmp2.nodes[nlen-1] = vr
    rtmp2.nodes[i] = vtmp2

    drho = _add_tensor(drho,rtmp1,1.0)
    drho = _add_tensor(drho,rtmp2,1.0)

  rout = MPS2MPO(drho)
  return rout
