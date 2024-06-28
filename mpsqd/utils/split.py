import numpy as np
import scipy.linalg as lg
#=================================================================
# this do the rq split for a 3D tensor, based on (a,b,c) => (a,b*c)
# need to reshape q to 3D
def split_rq(xx):

#----------------------
  y1,m,y2 = np.shape(xx)
  yy = np.reshape(xx,(y1,y2*m),order='F')
  r, q = lg.rq(yy,mode='economic')
  q1 = np.reshape(q,(min(y1,y2*m),m,y2),order='F')

  return r, q1

#=================================================================
# this do the rq split for a 3D tensor, based on (a,b,c) => (a*b,c)
# need to reshape q to 3D
def split_qr(xx):

#----------------------
  y1,m,y2 = np.shape(xx)
  yy = np.reshape(xx,(y1*m,y2),order='F')
  q, r = lg.qr(yy,mode='economic')
  q1 = np.reshape(q, (y1, m, min(y1*m, y2)),order='F')

  return q1, r

def split_svd(yy,small,nrmax=0):
  u1, s1, vt1 = lg.svd(yy, full_matrices=False, lapack_driver='gesvd')
#----------------------
  # find jmax
  for i in range(len(s1)):
    if (s1[i] < s1[0]*small):
      break

  jmax = i+1
  if (jmax > nrmax and nrmax > 0):
    jmax = nrmax
#-----------------------------------
# do the truncation
  u2  = u1[:,0:jmax]
  vt2 = vt1[0:jmax,:]
  return u2, vt2, s1, jmax

#====================================================
# this is our version of the split_w_truncation function
# using svd
def split_svd_rq(xx,small,nrmax):
#----------------------
  # do the svd
  y1,m,y2 = np.shape(xx)
  yy = np.reshape(xx,(y1,y2*m),order='F')
  u2, vt2, s1, jmax = split_svd(yy,small,nrmax)
  for i in range(jmax):
    u2[:,i]*=s1[i]
  q1 = np.reshape(vt2,(jmax,m,y2),order='F')
  return u2, q1

# reshape to U and V ,U is left orthogonal
def split_svd_qr_2tdvp(xx,kdims,small,nrmax):
  #----------------------
  # do the svd
  yy = np.reshape(xx,(kdims[0]*kdims[1], kdims[2]*kdims[3]))
  u2, vt2, s1, jmax = split_svd(yy,small,nrmax)
  for i in range(jmax):
    vt2[i,:]  *= s1[i]

  #------------------------------------
  # do the reshape
  q1 = np.reshape(u2,(kdims[0],kdims[1],jmax))
  v3 = np.reshape(vt2,(jmax,kdims[2],kdims[3]))

  return q1, v3

# reshape to U and V ,V is right orthogonal
def split_svd_rq_2tdvp(xx,kdims,small,nrmax):
  #----------------------
  # do the svd
  yy = np.reshape(xx,(kdims[0]*kdims[1], kdims[2]*kdims[3]))
  u2, vt2, s1, jmax = split_svd(yy,small,nrmax)
  for i in range(jmax):
    u2[:,i]  *= s1[i]

  #------------------------------------
  # do the reshape
  q1 = np.reshape(u2,(kdims[0],kdims[1],jmax))
  v3 = np.reshape(vt2,(jmax,kdims[2],kdims[3]))

  return q1, v3
