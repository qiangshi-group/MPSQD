import sys
import numpy as np
import params as pa
import model as md
from mpsqd.utils import add_tensor, MPS

def construct():
  for i in range(pa.ndvr):
    for j in range(i,pa.ndvr):
      temppall = delta_all5(i, j, md.e0[i,j], md.coef1[i,j,:], md.coef2[i,j,:,:])
      if (i == 0 and j == 0):
        pall = temppall.copy()
      else:
        pall = add_tensor(pall,temppall, 1.0)
      print(str([i,j])+"done")

  print("pall constructed...")
  print("shape of pall =", pall.ndim())

  return pall


#============================================================
# coef1 and coef2 are the first and second order coefficients
def delta_all5(i1, j1, delta, coef1, coef2):

# vmid is a template with identity matrix
  vmid,idiag = gen_identy_mpo(i1,j1)

#======================================================
  mpo = MPS(pa.nmps)

#------------------------------------------------------
# this is the constant term...
  kk0 = 0  # no delta term
  if (abs(delta) > 1e-12):
    rtmp1 = MPS(pa.nmps)
    rtmp1.nb = pa.nbmat
    for i in range(pa.nlevel+1):
      rtmp1.nodes.append((vmid[i]))

    #rtmp1.nodes[0].tensor *= delta
    rtmp1.nodes[0] = -1.0j * delta*rtmp1.nodes[0]
    mpo = rtmp1.copy()
    kk0 = 1


  print("delta term done")
#======================================================
# the linear terms, note that the h\omega part goes to split_bath
  for k in range(pa.nlevel):
# copy from the template
    rtmp1 = MPS(pa.nmps)
    rtmp1.nb = pa.nbmat

    for i in range(pa.nlevel+1):
      rtmp1.nodes.append((vmid[i]))

# need to midify the kth node
    vtmp = np.zeros((1, pa.nb[k+1], 1),dtype=np.complex128)
    for i in range(pa.nb[k+1]):
      vtmp[0,i,0] = coef1[k]*pa.sdvr_all[k][i] + coef2[k,k]*pa.sdvr_all[k][i]**2

    rtmp1.nodes[k+1] = (vtmp)

    if (k == 0 and kk0 == 0):
      mpo = rtmp1.copy()
      mpo.nodes[0] *= -1.0j
    else:
      mpo = add_tensor(mpo,rtmp1, -1.0j,pa.small,pa.nrmax)

  print("linear + qaudratic terms done")
#======================================================
# now the second order cross terms
  for k1 in range(pa.nlevel):
    for k2 in range(k1):
# copy from the template
      rtmp1 = MPS(pa.nmps)
      rtmp1.nb = pa.nbmat
      for i in range(pa.nlevel+1):
        rtmp1.nodes.append(vmid[i])
# need to midify the k1 and k2 nodes
# important, there is a 2.0 facotor for the cross terms
      if (np.abs(coef2[k2,k1]) > 1e-12):
        print("k1, k2 =", k1, k2, coef2[k2,k1])
        vtmp1 = np.zeros((1, pa.nb[k1+1], 1),dtype=np.complex128)
        vtmp2 = np.zeros((1, pa.nb[k2+1], 1),dtype=np.complex128)
# split coef2 to each node
        for i in range(pa.nb[k1+1]):
          vtmp1[0,i,0] = np.sqrt(abs(coef2[k2,k1]))*pa.sdvr_all[k1][i]

        for i in range(pa.nb[k2+1]):
          vtmp2[0,i,0] = coef2[k2,k1]/np.sqrt(abs(coef2[k2,k1]))*pa.sdvr_all[k2][i]

        rtmp1.nodes[k1+1] = vtmp1
        rtmp1.nodes[k2+1] = vtmp2
        mpo = add_tensor(mpo,rtmp1, -2.0j,pa.small,pa.nrmax)

# this is the diagonal part....
  if (idiag == 1):
    for k in range(pa.nlevel):
# copy from the template
      rtmp1 = MPS(pa.nmps)
      rtmp1.nb = pa.nbmat
      for i in range(pa.nlevel+1):
        rtmp1.nodes.append(vmid[i])

# need to midify the kth node
      vtmp = np.zeros((1, pa.nb[k+1], 1),dtype=np.complex128)
      for i in range(pa.nb[k+1]):
        vtmp[0,i,0] = md.omega[k] * pa.pot0_all[k][i]

      rtmp1.nodes[k+1] = vtmp
      mpo = add_tensor(mpo,rtmp1, -1.0j,pa.small,pa.nrmax)

  print("bilinear terms done")

  return mpo


def gen_identy_mpo(i1,j1):
#------------------------------------------------------
  vmid = []
#------------------------------------------------------
# construct vl
  vl = np.zeros((1, pa.nb[0]**2, 1),dtype=np.complex128)
  # diagonal term
  if (i1 == j1):
    idiag = 1
    ii = i1*pa.nb[0]+i1
    vl[0,ii,0] = 1.0
  # off-diagonal terms  
  else:
    idiag = 0
    ii = i1*pa.nb[0]+j1
    vl[0,ii,0] = 1.0

    ii = j1*pa.nb[0]+i1
    vl[0,ii,0] = 1.0

  vmid.append(vl)

#------------------------------------------------------
# other terms for the vibrational modes
  for i in range(pa.nlevel):
    vtmp = np.zeros((1, pa.nb[i+1], 1),dtype=np.complex128)
    # this is a indentity matrix
    vtmp[0,:,0] = 1.0
    vmid.append(vtmp)
  return vmid, idiag
