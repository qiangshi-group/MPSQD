import numpy as np
import params as pa

def calc_rho(rin):

  nlen = len(rin.nodes)
  if(nlen>2): 
  #------------------------------------------------------
  # the last two nodes
    rho = np.dot(rin.nodes[nlen-2][:,0,:],rin.nodes[nlen-1][:,0,0])
  #------------------------------------------------------
  # intermediate ones
    for i in range(nlen-3,1,-1):
      rho = np.dot(rin.nodes[i][:,0,:],rho)
  #------------------------------------------------------
  # the left node
    rho = np.tensordot(rin.nodes[1],rho,axes=((2),(0)))
    rho = np.tensordot(rin.nodes[0][0,:,:],rho,axes=((1),(0)))
  elif(nlen==2):
    rho = np.dot(rin.nodes[0][0,:,:],rin.nodes[1][:,:,0])
  else:
    print('err in calc_rho: nlen=',nlen)
  return rho

def calc_current(rin):
  
  nlen = len(rin.nodes)
  rhotmp1=np.zeros((pa.ndvr,pa.ndvr),dtype=np.complex128)
  rhotmp2=np.zeros((pa.ndvr,pa.ndvr),dtype=np.complex128)

  #left lead
  for j in range(4*pa.nlevel1):

    jnode = j+2
    #the last two nodes 
    if(jnode==nlen-1):
      rho = np.dot(rin.nodes[nlen-2][:,0,:],rin.nodes[nlen-1][:,1,0])
    elif(jnode==nlen-2):
      rho = np.dot(rin.nodes[nlen-2][:,1,:],rin.nodes[nlen-1][:,0,0])
    else:
      rho = np.dot(rin.nodes[nlen-2][:,0,:],rin.nodes[nlen-1][:,0,0])

    #intermediate nodes
    for inode in range(nlen-3,1,-1):
      if(inode==jnode):
        rho = np.dot(rin.nodes[inode][:,1,:],rho)
      else:
        rho = np.dot(rin.nodes[inode][:,0,:],rho)

    # the left node
    rho = np.tensordot(rin.nodes[1],rho,axes=((2),(0)))
    rho = pa.normfac[j]*np.tensordot(rin.nodes[0][0,:,:],rho,axes=((1),(0)))

    if(j<pa.nlevel1 or (j>=2*pa.nlevel1 and j<3*pa.nlevel1)):
      rhotmp1 = rhotmp1 + np.dot(rho.copy(),pa.aop_bar[:,:,j])
    else:
      rhotmp1 = rhotmp1 - np.dot(pa.aop_bar[:,:,j],rho.copy())

  #right lead
  for j in range(4*pa.nlevel1,pa.nlevel):

    jnode = j+2
    #the last two nodes 
    if(jnode==nlen-1):
      rho = np.dot(rin.nodes[nlen-2][:,0,:],rin.nodes[nlen-1][:,1,0])
    elif(jnode==nlen-2):
      rho = np.dot(rin.nodes[nlen-2][:,1,:],rin.nodes[nlen-1][:,0,0])
    else:
      rho = np.dot(rin.nodes[nlen-2][:,0,:],rin.nodes[nlen-1][:,0,0])

    #intermediate nodes
    for inode in range(nlen-3,1,-1):
      if(inode==jnode):
        rho = np.dot(rin.nodes[inode][:,1,:],rho)
      else:
        rho = np.dot(rin.nodes[inode][:,0,:],rho)

    # the left node
    rho = np.tensordot(rin.nodes[1],rho,axes=((2),(0)))
    rho = pa.normfac[j]*np.tensordot(rin.nodes[0][0,:,:],rho,axes=((1),(0)))

    if(j<4*pa.nlevel1+pa.nlevel2 or (j>=4*pa.nlevel1+2*pa.nlevel2 and j<4*pa.nlevel1+3*pa.nlevel2)):
      rhotmp2 = rhotmp2 + np.dot(rho.copy(),pa.aop_bar[:,:,j])
    else:
      rhotmp2 = rhotmp2 - np.dot(pa.aop_bar[:,:,j],rho.copy())

  #tot_current from left,right and average 
  z1=1j*np.trace(rhotmp1)
  z2=1j*np.trace(rhotmp2)
  ave_z=(z1-z2)/2.0
  return z1,z2,ave_z
