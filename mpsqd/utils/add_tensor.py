import numpy as np
import sys
from .mpsdef import MPS
from .trun import trun_tensor

# note, coeff only applies to the first node
def add_tensor(r1,r2,coeff,small,nrmax,need_trun=True):

  rtmp = MPS(r1.length,r1.nb)
  nlen = r1.length

  type_r1 = r1.nodes[0].dtype
  type_r2 = r2.nodes[0].dtype
  if (type_r1 != type_r2):
    print("r1 and r2 has different type, not supported yet")
    sys.exit(1)
  ndim1 = r1.ndim()
  ndim2 = r2.ndim()
#=============================================
# the left matrix, we should have m1=m2=1 here
  jj = ndim1[2,0]+ndim2[2,0]
  n1 = ndim1[2,0]
  # combine data
  vtmp = np.zeros((1,ndim1[1,0],jj),dtype=type_r1)
  vtmp[0,:,0:n1] = r1.nodes[0][0,:,:]
  vtmp[0,:,n1:jj] = coeff*r2.nodes[0][0,:,:]
  # add to the nodes
  rtmp.nodes.append(vtmp)

#================================================
# add all the intermediate matrices
  for i in range(1,nlen-1):
    m1 = ndim1[0,i]
    n1 = ndim1[2,i]
    ii = ndim1[0,i] + ndim2[0,i]
    jj = ndim1[2,i] + ndim2[2,i]

    # combine data
    vtmp = np.zeros((ii,ndim1[1,i],jj),dtype=type_r1)
    vtmp[0:m1,:,0:n1] = r1.nodes[i]
    vtmp[m1:ii,:,n1:jj] = r2.nodes[i]
    # add to the nodes
    rtmp.nodes.append(vtmp)


#=============================================
# the right matrix, we should have n1=n2=1 here

  m1 = ndim1[0,nlen-1]
  ii = ndim1[0,nlen-1] + ndim2[0,nlen-1]
  # combine data
  vtmp = np.zeros((ii,ndim1[1,nlen-1],1),dtype=type_r1)
  vtmp[0:m1,:,0] = r1.nodes[nlen-1][:,:,0]
  vtmp[m1:ii,:,0] = r2.nodes[nlen-1][:,:,0]

  # add to the nodes
  rtmp.nodes.append(vtmp)

#===========================================
# truncate the new tensor
  if need_trun:
    rtmp = trun_tensor(rtmp,small,nrmax)
  return rtmp
