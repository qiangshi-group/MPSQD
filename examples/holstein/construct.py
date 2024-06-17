import sys
import numpy as np
import params as pa
from mpsqd.utils import MPS, add_tensor

def construct():

  p0 = MPS(pa.nlevel+2)
  p0.nb = pa.nbmat

#---------------------------------------------------------
# vl and vr
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    ii = i*pa.ndvr+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0


#---------------------------------------------------------
# vmid
  vmid = []
  for i in range(pa.nlevel):
    vtmp = np.zeros((1, pa.nbmat[i+1], 1),dtype=np.complex128)
    for j in range(pa.nb[i+1]):
      ii = j*pa.nb[i+1]+j
      vtmp[0,ii,0] = 1.0
    vmid.append(vtmp)

#---------------------------------------------------------
# add to p0.nodes
  p0.nodes.append(vl)

  for i in range(pa.nlevel):
    p0.nodes.append(vmid[i])

  p0.nodes.append(vr)

  rout = delta_all5(p0)

  return rout


#===========================================================
def delta_all5(rho):

  drho = rho.copy()
  rtmp1 = rho.copy()
  nlen = len(rho.nodes)

#-----------------------------------------------------------
# the Hamiltonian term
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(pa.ndvr):
      ii = j*pa.ndvr + i
      vl[0,ii,0] = -1.0j*pa.hsys[i,j]
      vr[0,ii,0] =  1.0j*pa.hsys[j,i]


  drho.nodes[0] = vl
  rtmp1.nodes[nlen-1] = vr
  drho = add_tensor(drho,rtmp1,1.0,pa.small,pa.nrmax)
  print("after hsys, shape of rout",drho.ndim())

#-----------------------------------------------------------
# the k0 term
  print("k0=",pa.k0)
  rtmp1 = rho.copy()
  rtmp2 = rho.copy()
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)


  for i in range(pa.ndvr):
    ii = i*pa.ndvr+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0


  rtmp1.nodes[0] = vl
  rtmp2.nodes[nlen-1] = vr
  rtmp1 = add_tensor(rtmp1,rtmp2,1.0,pa.small,pa.nrmax)


  for i in range(pa.ndvr):
    rtmp2 = rho.copy()
    vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
    vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)

    ii = i*pa.ndvr+i
    vl[0,ii,0] = -2.0
    vr[0,ii,0] = 1.0

    rtmp2.nodes[0] = vl
    rtmp2.nodes[nlen-1] = vr

    rtmp1 = add_tensor(rtmp1,rtmp2,1.0,pa.small,pa.nrmax)

  drho = add_tensor(drho,rtmp1,-1.0*pa.k0,pa.small,pa.nrmax)


  print("after k0, shape of rout",drho.ndim())

#-----------------------------------------------------------
# the diagonal terms
  for i in range(pa.nsite):
    for j in range(pa.nl1):
      ii = j + i*pa.nl1
      rtmp1 = rho.copy()
      vtmp = np.zeros((1, pa.nb[ii+1]*pa.nb[ii+1], 1),dtype=np.complex128)

      for k in range(pa.nb[ii+1]):
        jj = k*pa.nb[ii+1]+k
        vtmp[0,jj,0] = -1.0*k*pa.wval[j]


      rtmp1.nodes[ii+1] = vtmp.copy()
      drho = add_tensor(drho,rtmp1,1.0,pa.small,pa.nrmax)


  print("after diagonal, shape of rout",drho.ndim())

#========================================================
# the hierarchical terms
  for i in range(pa.nsite):

    ii = i*pa.nl1

    for k in range(pa.nl1):

      rtmp1 = rho.copy()
      rtmp2 = rho.copy()

      vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
      vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)

      jj = i + i*pa.ndvr

      vl[0,jj,0] = 1.0
      vr[0,jj,0] = 1.0

      vtmp1 = np.zeros((1, pa.nb[ii+k+1]*pa.nb[ii+k+1], 1),dtype=np.complex128)
      vtmp2 = np.zeros((1, pa.nb[ii+k+1]*pa.nb[ii+k+1], 1),dtype=np.complex128)

      for j in range(pa.nb[ii+k+1]):

#-------------------------------------------------------------------------
        j1 = j-1
        if j1>=0:
          jj = j1*pa.nb[ii+k+1]+j
          vtmp1[0,jj,0] += pa.gval_i[k]/pa.normfac[k]*np.sqrt(1.0+j1)
          vtmp1[0,jj,0] -= 1.0j*pa.gval[k]/pa.normfac[k]*np.sqrt(1.0+j1)

          vtmp2[0,jj,0] += pa.gval_i[k]/pa.normfac[k]*np.sqrt(1.0+j1)
          vtmp2[0,jj,0] += 1.0j*pa.gval[k]/pa.normfac[k]*np.sqrt(1.0+j1)

#-------------------------------------------------------------------------
        j1 = j+1
        if j1<pa.nb[ii+k+1]:
          jj = j1*pa.nb[ii+k+1]+j
          vtmp1[0,jj,0] -= 1.0j*pa.normfac[k]*np.sqrt(j1*1.0)
          vtmp2[0,jj,0] += 1.0j*pa.normfac[k]*np.sqrt(j1*1.0)


      rtmp1.nodes[0] = vl
      rtmp1.nodes[ii+k+1] = vtmp1

      rtmp2.nodes[nlen-1] = vr
      rtmp2.nodes[ii+k+1] = vtmp2

      drho = add_tensor(drho,rtmp1,1.0,pa.small,pa.nrmax)
      drho = add_tensor(drho,rtmp2,1.0,pa.small,pa.nrmax)

  print("after hierachial, shape of rout",drho.ndim())
  return drho
