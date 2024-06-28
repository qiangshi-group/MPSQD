from mpsqd.utils import MPS, add_tensor
import numpy as np
import params as pa
import model as md

#------------------------------------------------------------
# xtmp is the matrix for the x operator...
xtmp = np.zeros((pa.nbv, pa.nbv),dtype=np.complex128)
for i in range(pa.nbv):
  for j in range(pa.nbv):
    if (i == j-1):
      xtmp[i,j] = np.sqrt(0.5 * j)
    if (i == j+1):
      xtmp[i,j] = np.sqrt(0.5 * i)
print("xtmp done")

# xtmp2 is the matrix for the x2 operator...
xtmp2 = np.zeros((pa.nbv, pa.nbv),dtype=np.complex128)
for i in range(pa.nbv):
  for j in range(pa.nbv):
    if (i == j):
      xtmp2[i,j] = (0.5+j)
    if (i == j+2):
      xtmp2[i,j] = 0.5*np.sqrt(i*(i-1))
    if (i == j-2):
      xtmp2[i,j] = 0.5*np.sqrt(j*(j-1))

print("xtmp2 done")

#def construct():
#
#  pall00 = mpo(0, 0, -md.delta, md.acoef, md.a2coef)
#
#  pall11 = mpo(1, 1, md.delta, md.bcoef, md.b2coef)
#  pall = add_tensor(pall00, pall11, 1.0)
#
#  pall01 = mpo(0, 1, 0.0, md.ccoef, md.c2coef)
#  pall = add_tensor(pall, pall01, 1.0)
#
#  print("pall constructed...")
#  print("shape of pall =", pall.ndim())
#
#  return pall
def construct():
  for i in range(pa.ndvr):
    for j in range(i,pa.ndvr):
      temppall = mpo(i, j, md.e0[i,j], md.coef1[i,j,:], md.coef2[i,j,:,:])
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
def mpo(i1, j1, delta, coef1, coef2):

#------------------------------------------------------
# vmid is a template with identity matrix
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
    vtmp = np.zeros((1, pa.nb[i+1]**2, 1),dtype=np.complex128)
    # this is a indentity matrix
    for j in range(pa.nb[i+1]):
      ii = j*pa.nb[i+1]+j
      vtmp[0,ii,0] = 1.0
    vmid.append(vtmp)

#======================================================
  mpo = MPS(pa.nlevel+1)

#------------------------------------------------------
# this is the constant term...
  kk0 = 0  # no delta term
  if (abs(delta) > 1e-12):
    rtmp1 = MPS(pa.nlevel+1)
    rtmp1.nb = pa.nbmat
    for i in range(pa.nlevel+1):
      rtmp1.nodes.append(vmid[i])

    rtmp1.nodes[0] = -1.0j * delta*rtmp1.nodes[0]
    mpo = rtmp1.copy()
    kk0 = 1


  print("delta term done")
#======================================================
# the linear terms, note that the h\omega part goes to split_bath
  for k in range(pa.nlevel):
# copy from the template
    rtmp1 = MPS(pa.nlevel+1)
    rtmp1.nb = pa.nbmat
    for i in range(pa.nlevel+1):
      rtmp1.nodes.append(vmid[i])


# need to midify the kth node
    vtmp = np.zeros((1, pa.nb[k+1]*pa.nb[k+1], 1),dtype=np.complex128)
    for i in range(pa.nb[k+1]):
      for j in range(pa.nb[k+1]):
        jj = j*pa.nb[k+1]+i
        vtmp[0,jj,0] = coef1[k]*xtmp[i,j] + coef2[k,k]*xtmp2[i,j]


    rtmp1.nodes[k+1] = vtmp
    if (k == 0 and kk0 == 0):
      mpo = rtmp1.copy()
      mpo.nodes[0] *= -1.0j
      #print("dimension of mpo",mpo.ndim())
    else:
      mpo = add_tensor(mpo,rtmp1, -1.0j, pa.small,pa.nrmax)
      #print("dimension of mpo",mpo.ndim())


  print("linear + qaudratic terms done")
#======================================================
# now the second order cross terms
  for k1 in range(pa.nlevel):
    for k2 in range(k1):
# copy from the template
      rtmp1 = MPS(pa.nlevel+1)
      rtmp1.nb = pa.nbmat
      for i in range(pa.nlevel+1):
        rtmp1.nodes.append(vmid[i])
# need to midify the k1 and k2 nodes
# important, there is a 2.0 facotor for the cross terms
      if (np.abs(coef2[k2,k1]) > 1e-12):
        print("k1, k2 =", k1, k2, coef2[k2,k1])
        vtmp1 = np.zeros((1, pa.nb[k1+1]*pa.nb[k1+1], 1),dtype=np.complex128)
        vtmp2 = np.zeros((1, pa.nb[k2+1]*pa.nb[k2+1], 1),dtype=np.complex128)
# split coef2 to each node
        for i in range(pa.nb[k1+1]):
          for j in range(pa.nb[k1+1]):
            jj = j*pa.nb[k1+1]+i
            vtmp1[0,jj,0] = np.sqrt(abs(coef2[k2,k1]))*xtmp[i,j]

        for i in range(pa.nb[k2+1]):
          for j in range(pa.nb[k2+1]):
            jj = j*pa.nb[k2+1]+i
            vtmp2[0,jj,0] = coef2[k2,k1]/np.sqrt(abs(coef2[k2,k1]))*xtmp[i,j]


        rtmp1.nodes[k1+1] = vtmp1
        rtmp1.nodes[k2+1] = vtmp2
        mpo = add_tensor(mpo,rtmp1, -2.0j,pa.small,pa.nrmax)

# this is the diagonal part....
  if (idiag == 1):
    for k in range(pa.nlevel):
# copy from the template
      rtmp1 = MPS(pa.nlevel+1)
      rtmp1.nb = pa.nbmat
      for i in range(pa.nlevel+1):
        rtmp1.nodes.append(vmid[i])


# need to midify the kth node
      vtmp = np.zeros((1, pa.nb[k+1]*pa.nb[k+1], 1),dtype=np.complex128)
      for i in range(pa.nb[k+1]):
        jj = i*pa.nb[k+1]+i
        vtmp[0,jj,0] = md.omega[k] * (i+0.5)


      rtmp1.nodes[k+1] = vtmp
      mpo = add_tensor(mpo,rtmp1, -1.0j,pa.small,pa.nrmax)

  print("bilinear terms done")

  return mpo


