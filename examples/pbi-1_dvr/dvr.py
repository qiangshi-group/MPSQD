# 1d-dvr code by Shuocang Zhang and Qiang Shi
# based on the Colbert and Miller paper in JCP, 1992

import numpy as np
import sys
from scipy.linalg import expm

#============================================================
# harmonic potential with coefficient k1
def har1(r,k1):
  return 0.5*k1*r*r

#==============================================================
# first construct kintetic enery matrix ek(nr,nr)
# then the potential energy on a grid
#===============================================================
# ndvr is the number of basis
# the wavefunctions from 1 to N-1
# ndvr = N-1, N = ndvr+1
# distance is from a to b, b-a should be given
# notice the grid is different from in V, no edge
# the real rangle is from a-x to b+x
# so the distance is b-a+2x
# nr is the dvr space dimension
# ndvr is the podvr space dimension
# rmin and rmax is the grid edge
#==============================================================
def dvrtemplate(nr,ndvr,rmin,rmax):

#---------------------------------------------------------------
# the template harmonic oscillator has mass=1 and omega=1
  mass1 = 1.0
  k1 = 1.0

#---------------------------------------------------------------------
# calculate the grid and distance
  dr = (rmax-rmin)/(nr-1)
  dis = rmax-rmin+2*dr

# set the grids
  xgrid = np.zeros(nr,dtype=np.float64)
  for i in range(nr):
    xgrid[i] = rmin + i*dr

#------------------------------------------------------------------------
# construct the Ek matrix
  N = nr+1
  ek = np.zeros((nr,nr),dtype=np.float64)
  for i in range(nr):
    for j in range(nr):
      ii = i+1
      jj = j+1
      if i == j:
        ek[i,j] = (2*N**2+1)/3-1/(np.sin(np.pi*ii/(N)))**2
      else:
        ek[i,j] = (-1)**(ii-jj)*(1/(np.sin(np.pi*(ii-jj) \
                  /(2*N)))**2-1/(np.sin(np.pi*(ii+jj)/(2*N)))**2)

  ek *= np.pi**2/(4*mass1*dis**2)

# diagonalize to test the kinetic part, should be e_k \propto k^2
  ek_ev,ek_vr = np.linalg.eigh(ek)

#-------------------------------------------------------------------------
# construct the V matrix
  pot0 = np.zeros((nr),dtype=np.float64)

  for i in range(nr):
    pot0[i] = har1(xgrid[i],k1)

# plot pot0 and pot1
  ev = np.diag(pot0)

# add them together 
  hamil = ek + ev

# diagonalize the hamil matrix to obtaion energy and eigenfunctions
  h_ev,h_vr = np.linalg.eigh(hamil)

#-----------------------------------------------------------------------------
# potential optimized dvr
# contract to form smat
# smat = phi(ndvr,nr)*x(nr,nr)*phi(nr,ndvr)

# construct the diagonal x matrix
  xmat = np.diag(xgrid)

# the lowest ndvr eigen functions phi(nr,ndvr)  
  phi = h_vr[:,0:ndvr]

# get the smat
  smat = np.linalg.multi_dot([phi.transpose(1,0),xmat,phi])

# diagonalize to sdvr 
  sdvr, sphi = np.linalg.eigh(smat)

#-------------------------------------------------------------------------------
# construct the kinetic operator in the po-dvr basis
# the new space is (ndvr*ndvr)
  eng = h_ev[:ndvr]
  energy = np.diag(eng)
  
# the new hamiltonian operator is sphi^T*energy*sphi
  hnew = np.linalg.multi_dot([sphi.transpose(1,0),energy,sphi])

# get the potential surface in sdvr representation
  pot0 = np.zeros((ndvr),dtype=np.float64)
  for i in range(ndvr):
    pot0[i] = har1(sdvr[i],k1)

  v0 = np.diag(pot0)

# the new kinetic energy operator in the po-dvr basis
  eknew = hnew-v0

#--------------------------------------------------------------------------------
# we now can get the initial state from the eigenstate of h0
# default is the ground state
  vinit = 0
#  h0_ev, h0_vr = np.linalg.eigh(hnew)
#  init_phi = h0_vr[:,vinit]
  init_phi = sphi[vinit,:]

# return values: po-dvr points; initial wavefunction; 
# transfer matrix between po-dvr and eigenstat; new kinetic operator
# diagonal potential operator
  return sdvr, init_phi, sphi, eknew, pot0

#=====================================================
# setup the dvr points for each mode
def dvrsetup(nr, nbv, omega, dt):

  ndvr = nbv
  print("nlen = ", len(omega))
  print("omega = ", omega)
#-------------------------------------------
# estimate xmax based on nbv
  e1 = nbv-0.5
  if (nbv < 10):
    e1 = e1*10
  else:
    e1 = e1*4

  xmax = np.sqrt(2.0*e1)
  print("xmax =", xmax)

# the template data
  sdvr, init_phi, sphi, eknew, pot0 = dvrtemplate(nr, ndvr, -xmax, xmax)

  sdvr_all = []
  sphi_all = []
  pot0_all = []
  init_wave_all = []
  vkprop_all = []

  nlen = len(omega)

  for i in range(nlen):

    sdvr_all.append(sdvr.copy())
  
    init_wave_all.append(init_phi.copy())

    sphi_all.append(sphi.copy())

    pot0_all.append(pot0.copy())

    vkprop = expm(-0.5j*dt*omega[i]*eknew)
    vkprop_all.append(vkprop)

  return sdvr_all, sphi_all, pot0_all, init_wave_all, vkprop_all
