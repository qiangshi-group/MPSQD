import numpy as np
import scipy.linalg as lg
from .split import split_qr, split_svd_rq

#==========================================
# truncation based on the 2-sweep method
# Note, added normaliation similar to the orginal fortran code....
# also use a custom svd truncation, the one with tn may cause problems somehow

def trun_tensor(rin,small,nrmax):

  rout = rin.__class__(rin.length,rin.nb)
  rout.nodes = []
  nlen = rin.length
  nrm = []
#----------------------------------------------------
  # QR for the first matrix
  q, r = split_qr(rin.nodes[0])
  rout.nodes.append(q)

  # normalize
  nrml = np.sqrt(np.sum(np.abs(r)**2))
  if (nrml < 1.e-3): nrml = 1.0

  nrm.append(nrml)
  r*= 1.0/nrml

#----------------------------------------------------
# middle ones
  for i in range(1, nlen-1,1):
    rtmp = np.tensordot(r,rin.nodes[i],axes=([1],[0]))
    q, r = split_qr(rtmp)
    rout.nodes.append(q)

    nrml=np.sqrt(np.sum(np.abs(r)**2))
    if (nrml < 1.e-3): nrml = 1.0
    nrm.append(nrml)
    r *= 1.0/nrml
#----------------------------------------------------
# the last one
  rout.nodes.append(np.tensordot(r,rin.nodes[nlen-1],axes=([1],[0])))
#----------------------------------------------------
  # the real truncation from the right
  rin = trun_tensor_right(rout,small,nrmax)
#----------------------------------------------------
  # get the renormalization factors back 
  nrml = np.sum(np.log(nrm))
  nrml = np.exp(nrml/nlen)
  for i in range(nlen):
    rin.nodes[i] *= nrml

  return rin

#==============================================
def trun_tensor_right(rin,small,nrmax):
  rout = rin.copy()
  nlen = rin.length

#----------------------------------------------------
  # split useing svd, the right matrix
  u1, vt = split_svd_rq(rin.nodes[nlen-1],small,nrmax)

  #can not use u and vt directly, some issues with dangling edge
  rout.nodes[nlen-1] = vt

#----------------------------------------------------
  #intermediate terms
  for i in range(nlen-2,0,-1):
    rtmp = np.tensordot(rin.nodes[i],u1,axes=([2],[0]))
    u1, vt = split_svd_rq(rtmp,small,nrmax)
    rout.nodes[i] = vt

#----------------------------------------------------
# the left matrix
  rout.nodes[0] = np.tensordot(rin.nodes[0],u1,axes=([2],[0]))
  return rout
