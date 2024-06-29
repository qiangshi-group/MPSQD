import numpy as np
def calc_pop(rin):
  ndvr = rin.ndim()[1,0]
  pop = np.zeros((ndvr),dtype=np.float64)
  for i in range(ndvr):
    psi = rin.copy()
    tmp = rin.nodes[0][:,i,:].copy()
    psi.nodes[0][:,:,:] = 0.0
    psi.nodes[0][:,i,:] = tmp
    pop[i] = calc_overlap(psi, psi).real
  return pop

def calc_overlap(r1, r2):
# the 1st node
  xx1 = np.conj(r1.nodes[0])
  yy1 = r2.nodes[0]
  rtmp = np.tensordot(xx1,yy1, axes=([0,1],[0,1]))
  for i in range(1,r1.length):
#------------------------------------------------------
    xx1 = np.conj((r1.nodes[i]))
    yy1 = r2.nodes[i]
    rtmp = np.tensordot(rtmp,yy1,axes=([1],[0]))
    rtmp = np.tensordot(xx1,rtmp,axes=([0,1],[0,1]))
  return rtmp[0,0]
