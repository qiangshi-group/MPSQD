import sys
import numpy as np
import params as pa
from mpsqd.utils import MPS, add_tensor

#==========================================================
#Generate the identity MPO
def identy_mpo():

  p0 = MPS(pa.nlevel+2)
  p0.nb = pa.nb

# vl and vr
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    ii = i*pa.ndvr+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0

# vmid
  vmid = []

  for i in range(pa.nlevel):

    vtmp = np.zeros((1, pa.nbmat[i+2], 1),dtype=np.complex128)

    for j in range(pa.nb[i+2]):
      ii = j*pa.nb[i+2]+j
      vtmp[0,ii,0] = 1.0

    vmid.append(vtmp)

# add to p0.nodes
  p0.nodes.append(vl)

  p0.nodes.append(vr)

  for i in range(pa.nlevel):
    p0.nodes.append(vmid[i])

  print("p0 calculation done!")
  return p0

#===========================================================
def construct():

  rho = identy_mpo()
  drho = rho.copy()
  rtmp1 = rho.copy()
  nlen = len(rho.nodes)

  print("nlen =", nlen)
#-----------------------------------------------------------
# the Hamiltonian term
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(pa.ndvr):
      ii = j*pa.ndvr + i
      vl[0,ii,0] = -1.0j*pa.hsys[i,j]
      vr[0,ii,0] =  1.0j*pa.hsys[j,i]


  drho.nodes[0] = vl.copy()
  rtmp1.nodes[1] = vr.copy()
  drho = add_tensor(drho,rtmp1,1.0,pa.small_mpo,pa.nrmax)
  print("after hsys, shape of rout",drho.ndim())
#-----------------------------------------------------------
# currently no k0 term
#-----------------------------------------------------------
# the diagonal terms
  for j in range(pa.nlevel):
    rtmp1 = rho.copy()
    vtmp = np.zeros((1, pa.nb[j+2]*pa.nb[j+2], 1),dtype=np.complex128)

    for k in range(pa.nb[j+2]):
      jj = k*pa.nb[j+2]+k
      vtmp[0,jj,0] = -1.0*k*pa.wval_ct[j]


    rtmp1.nodes[j+2] = vtmp.copy()
    drho = add_tensor(drho,rtmp1,1.0,pa.small_mpo,pa.nrmax)


  print("after diagonal term, shape of rout",drho.ndim())

#========================================================
# the hierarchical terms
  for i in range(pa.nlevel):

    rtmp1 = rho.copy()
    rtmp2 = rho.copy()

#-------------------------------------------------------------
# the first part from j-1
    # vl for rtmp1, vr for rtmp2
    vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
    vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
 
    for i1 in range(pa.ndvr):
      for j1 in range(pa.ndvr):
        jj = j1*pa.ndvr + i1
        vl[0,jj,0] = pa.aop[i1,j1,i]
        vr[0,jj,0] = pa.aop[j1,i1,i]  


    # important, this gives the signs needed
    for j in range(pa.nlevel):
      rtmp1.nodes[j+2][:,2,:] = -1.0*rtmp1.nodes[j+2][:,2,:]  
      rtmp1.nodes[j+2][:,3,:] = -1.0*rtmp1.nodes[j+2][:,3,:]  

    for j in range(i):
      rtmp1.nodes[j+2][:,2,:] = -1.0*rtmp1.nodes[j+2][:,2,:]
      rtmp1.nodes[j+2][:,3,:] = -1.0*rtmp1.nodes[j+2][:,3,:]

      rtmp2.nodes[j+2][:,2,:] = -1.0*rtmp2.nodes[j+2][:,2,:] 
      rtmp2.nodes[j+2][:,3,:] = -1.0*rtmp2.nodes[j+2][:,3,:] 

    # these are used for mode i
    vtmp1 = np.zeros((1, pa.nb[i+2]*pa.nb[i+2], 1),dtype=np.complex128)
    vtmp2 = np.zeros((1, pa.nb[i+2]*pa.nb[i+2], 1),dtype=np.complex128)


    for j in range(pa.nb[i+2]):
      j1 = j-1
      if (j1 >= 0):
#+++++++++++ rtmp1 ++++++++++++++++++++++++++++++++++++++
        jj = j1*pa.nb[i+2]+j
        vtmp1[0,jj,0] -= 1.0j/pa.normfac[i]*pa.gval_ct[i]
#+++++++++++ rtmp2 ++++++++++++++++++++++++++++++++++++++
        vtmp2[0,jj,0] += 1.0j/pa.normfac[i]*pa.gval_ct_bar[i]

    
    #beware that rtmp1/rtmp2's i+1 th node and head node have been override!
    rtmp1.nodes[0] = vl.copy()
    rtmp1.nodes[i+2] = vtmp1.copy()

    rtmp2.nodes[1] = vr.copy()
    rtmp2.nodes[i+2] = vtmp2.copy()

    drho = add_tensor(drho,rtmp1,1.0,pa.small_mpo,pa.nrmax)
    drho = add_tensor(drho,rtmp2,1.0,pa.small_mpo,pa.nrmax)


#-------------------------------------------------------------------------
# the second part from j+1
  for i in range(pa.nlevel):
    rtmp1 = rho.copy()
    rtmp2 = rho.copy()

    # vl for rtmp1, vr for rtmp2
    vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
    vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
 
    for i1 in range(pa.ndvr):
      for j1 in range(pa.ndvr):
        jj = j1*pa.ndvr + i1
        vl[0,jj,0] = pa.aop_bar[i1,j1,i]
        vr[0,jj,0] = pa.aop_bar[j1,i1,i]  


    # important, this gives the signs needed
    for j in range(pa.nlevel):
      rtmp2.nodes[j+2][:,2,:] = -1.0*rtmp2.nodes[j+2][:,2,:]
      rtmp2.nodes[j+2][:,3,:] = -1.0*rtmp2.nodes[j+2][:,3,:]

    for j in range(i+1, pa.nlevel):
      rtmp1.nodes[j+2][:,2,:] = -1.0*rtmp1.nodes[j+2][:,2,:]  
      rtmp1.nodes[j+2][:,3,:] = -1.0*rtmp1.nodes[j+2][:,3,:]  

      rtmp2.nodes[j+2][:,2,:] = -1.0*rtmp2.nodes[j+2][:,2,:]  
      rtmp2.nodes[j+2][:,3,:] = -1.0*rtmp2.nodes[j+2][:,3,:]  

    # these are used for mode i
    vtmp1 = np.zeros((1, pa.nb[i+2]*pa.nb[i+2], 1),dtype=np.complex128)
    vtmp2 = np.zeros((1, pa.nb[i+2]*pa.nb[i+2], 1),dtype=np.complex128)


    for j in range(pa.nb[i+2]):
      j1 = j+1
      if (j1<pa.nb[i+2]):
#+++++++++++ rtmp1 ++++++++++++++++++++++++++++++++++++++
        jj = j1*pa.nb[i+2]+j
        vtmp1[0,jj,0] -= 1.0j*pa.normfac[i]
#+++++++++++ rtmp2 ++++++++++++++++++++++++++++++++++++++
        vtmp2[0,jj,0] += 1.0j*pa.normfac[i]


    #beware that rtmp1/rtmp2's i+1 th node and head node have been override!
    rtmp1.nodes[0] = vl.copy()
    rtmp1.nodes[i+2] = vtmp1.copy()

    rtmp2.nodes[1] = vr.copy()
    rtmp2.nodes[i+2] = vtmp2.copy()

    drho = add_tensor(drho,rtmp1,1.0,pa.small_mpo,pa.nrmax)
    drho = add_tensor(drho,rtmp2,1.0,pa.small_mpo,pa.nrmax)


  print("after hierachial, shape of rout",drho.ndim())
  for i in range(nlen):
    print("maxval for node", i, " = ", np.max(np.abs(drho.nodes[i])))

  return drho
