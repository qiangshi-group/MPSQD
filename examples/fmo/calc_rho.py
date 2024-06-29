import numpy as np
def calc_rho(rin):
# the first two nodes
  rho = np.dot(rin.nodes[0][0,:,:],rin.nodes[1][:,0,:])
#------------------------------------------------------
# intermediate ones
  for i in range(2,rin.length-1):
    rho = np.dot(rho,rin.nodes[i][:,0,:])
#------------------------------------------------------
# the right node
  rho = np.dot(rho, rin.nodes[rin.length-1][:,:,0])
  return rho
