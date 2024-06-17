import numpy as np
import sys
from .mpsdef import MPS
from .trun import trun_tensor
#===========================================
# this calculates the product of two tensors, with one one contraction
# also truncate the final result
def prod_tensor_mat(mat1,r1,small,nrmax,need_trun=True):

  rtmp = MPS(len(r1.nodes),r1.nb)

#----------------------------------------------------------
  for i in range(r1.length):
    vtmp1 = mat1.nodes[i]

    vtmp = np.tensordot(vtmp1, r1.nodes[i], (2,1))

    rtmp1 = vtmp.transpose((0,3,1,2,4))
    n1 = rtmp1.shape
    rtmp2 = np.reshape(rtmp1,(n1[0]*n1[1], n1[2], n1[3]*n1[4]),order='F')

    # add to the nodes
    rtmp.nodes.append(rtmp2)

#----------------------------------------------------------
# truncate the new tensor
  if need_trun:
    rtmp = trun_tensor(rtmp,small,nrmax)
  return rtmp
