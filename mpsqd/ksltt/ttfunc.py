import sys
import numpy as np
from scipy.linalg import expm

def phia_next(phi0,mat1,x1,y1,iright):

  dim1 = x1.shape
  dim2 = y1.shape
  dim3 = phi0.shape
  dima = mat1.shape

  rx1,m,rx2 = dim1
  ry1,n,ry2 = dim2
  ra1 = dima[0]
  ra2 = dima[3]

  if iright == 1: # from right to left
    if (dim3[0]!=rx2 or dim3[1]!=ra2 or dim3[2]!=ry2):
        sys.exit("unmatched phi0 size in _next_phi")

    xx1 = np.conj(x1)
    yy1 = y1
    rtmp2 = np.tensordot(xx1,phi0,axes=((2),(0)))
    rtmp3 = np.tensordot(rtmp2,mat1,axes=((1,2),(1,3)))
    phi = np.tensordot(rtmp3,yy1,axes=((1,3),(2,1)))


  else: # from left to right
    if (dim3[0]!=rx1 or dim3[1]!=ra1 or dim3[2]!=ry1):
          sys.exit("unmatched phi0 size in _next_phi")
    xx1 = np.conj(x1)
    yy1 = y1
    rtmp2 = np.tensordot(xx1,phi0,axes=((0),(0)))
    rtmp3 = np.tensordot(rtmp2,mat1,axes=((0,2),(1,0)))
    phi = np.tensordot(rtmp3,yy1,axes=((1,2),(0,1)))
  return phi

#======================================================
def delta_ssol(phi1,phi2,y1):
  rtmp1 = np.tensordot(phi1,y1,axes=((2),(0)))
  dy = np.tensordot(rtmp1,phi2,axes=((1,2),(1,2)))
  dy *= -1.0
  return dy

#==========================================
def delta_ksol(mat1,phi1,phi2,y1):
  dim1 = phi1.shape
  dim2 = phi2.shape
  dima = mat1.shape
  dimy = y1.shape
  rx1,ra1,ry1 = dim1
  rx2,ra2,ry2 = dim2
  m = n = dimy[1]

  if (dima[0] != ra1 or dima[3] != ra2):
      sys.exit("array size does not match in delta_ksol for mat1")

  if ((dimy[0] != ry1 or dimy[2] != ry2)):
      sys.exit("array size does not match in delta_ksol for x1")
  rtmp2 = np.tensordot(phi1,y1,axes=((2),(0)))
  rtmp3 = np.tensordot(rtmp2,mat1,axes=((1,2),(0,2)))
  dy = np.tensordot(rtmp3,phi2,axes=((1,3),(2,1)))

  return dy

#=====================================
def ddot2_ssol(vec1, vec2):

  xx1 = np.conj(vec1)
  yy1 = vec2

  ddot2 = np.tensordot(xx1,yy1,axes=((0,1), (0,1)))
  return ddot2

#=====================================
def dnorm2_ssol(vec1):

  dnorm2 = ddot2_ssol(vec1,vec1)

  return np.sqrt(dnorm2)

#=====================================
def ddot2_ksol(vec1, vec2):

  xx1 = np.conj(vec1)
  yy1 = vec2
  ddot2 = np.tensordot(xx1,yy1,axes=((0,1,2), (0,1,2)))

  return ddot2

#=====================================
def dnorm2_ksol(vec1):

  dnorm2 = ddot2_ksol(vec1,vec1)

  return np.sqrt(dnorm2)

def expmv(mmax, dt, yy, delta_v, ddot2_v, dnorm2_v, phi1, phi2, mat1=None):
#----------------------
  vm = []
  hmat = np.zeros((mmax,mmax),dtype=np.complex128)

#--------------------------
# the first vector
  rtmp0 = np.sqrt(ddot2_v(yy,yy))
  y0 = yy/rtmp0
  vm.append(y0)
#--------------------------------------------
  for j in range(mmax):
    # dy1
    if (mat1 is None):
      dy1 = delta_v(phi1,phi2,vm[j])
    else:
      dy1 = delta_v(mat1,phi1,phi2,vm[j])
    dy1 *= dt

    # calculate the overlap
    for i in range(j+1):
      hmat[i,j] = ddot2_v(vm[i], dy1)

    # orthogonaization 
    for i in range(j+1):
      dy1 -=  hmat[i,j]*vm[i]

    # new basis
    rtmp = np.sqrt(ddot2_v(dy1,dy1))
# important, if rtmp = 0, the break, and set array dimension to j
    if (rtmp > 1.e-13):
      if (j < mmax-1):
        hmat[j+1,j] = rtmp
    # need to get a new y0, otherwise not good....
        y0 = dy1/rtmp
        vm.append(y0)
      else:
        print("Warning: The dimension of Krylov space reached mmax")
    else:
      break

  jmax = j + 1
#--------------------------------------------
# calculate exp(hmat)
  exph = expm(hmat[0:jmax,0:jmax])
#--------------------------------------------
# the new yy
  vtmp = np.zeros(yy.shape,dtype=np.complex128)

  for i in range(jmax):
    vtmp += rtmp0*exph[i,0]*vm[i]

  yy = vtmp

  return yy
