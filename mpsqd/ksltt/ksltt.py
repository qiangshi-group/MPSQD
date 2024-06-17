import numpy as np
import scipy.linalg as lg
from ..utils import MPS
from . import ttfunc as ttf
from ..utils.split import split_rq,split_qr

def ksltt(rin,pall,dt,mmax=30):

  dt2 = 0.5*dt

  nlen = len(rin.nodes)
  r1 = MPS(nlen,rin.nb)
  r2 = MPS(nlen,rin.nb)
  r3 = MPS(nlen,rin.nb)

  phia = []

  # phia[nlevel+2]
  vtmp = np.ones((1,1,1),dtype=np.complex128)
  phia.append(vtmp)

#======================================================
# ortho from right
  r, q = split_rq(rin.nodes[nlen-1])
  
  r1.nodes.append(q)
  u1 = r
# phia[nlevel+1]
  phia.append(ttf.phia_next(phia[0],pall.nodes[nlen-1],q,q,1))
#-------------------------------------------------------
# intermediate terms
  for i in range(nlen-2,0,-1):

    rtmp = np.tensordot(rin.nodes[i], u1, axes=((2),(0)))

    r, q = split_rq(rtmp)
    r1.nodes.append(q)
    u1 = r

    phia.append(ttf.phia_next(phia[nlen-1-i],pall.nodes[i],q,q,1))

# the left matrix

  rtmp = np.tensordot(rin.nodes[0], u1, axes=((2),(0)))
  r1.nodes.append(rtmp)

  r1.nodes.reverse()

# add phia[0] and reverse to normal
  vtmp = np.ones((1,1,1),dtype=np.complex128)
  phia.append(vtmp)
  phia.reverse()

#===============================================
###### the first part of KSL, from left to right
  for i in range(nlen-1):
    phi1 = phia[i]
    phi2 = phia[i+1]

    if (i == 0):
      ksol = r1.nodes[i].copy()
    else:
      ksol = r2.nodes[i].copy()


    #for islice in range(pa.rk4slice):
    ksol = ttf.expmv(mmax, dt2, ksol, ttf.delta_ksol, ttf.ddot2_ksol, ttf.dnorm2_ksol, phi1, phi2, pall.nodes[i])
    q, r = split_qr(ksol)


    if (i == 0):
      r2.nodes.append(q)
    else:
      r2.nodes[i] = q


    phi1 = ttf.phia_next(phi1,pall.nodes[i],q,q,0)
    phia[i+1] = phi1

# ?? need to copy
    ssol = r
    ssol = ttf.expmv(mmax, dt2, ssol, ttf.delta_ssol, ttf.ddot2_ssol, ttf.dnorm2_ssol, phi1, phi2)

    rtmp1 = np.tensordot(ssol,r1.nodes[i+1],axes=((1),(0)))
    r2.nodes.append(rtmp1)

#--------------------------------------------------------
### right most part

  phi1 = phia[nlen-1]
  phi2 = phia[nlen]
  ksol = r2.nodes[nlen-1].copy()
  ksol = ttf.expmv(mmax, dt2, ksol, ttf.delta_ksol, ttf.ddot2_ksol, ttf.dnorm2_ksol, phi1, phi2, pall.nodes[nlen-1])

  r2.nodes[nlen-1] = ksol
#===================================================================
###### the second part of KSL, from right to left
  for i in range(nlen-1,0,-1):

    phi1 = phia[i]
    phi2 = phia[i+1]

    if (i==nlen-1):
      ksol = r2.nodes[i].copy()
    else:
      ksol = r3.nodes[nlen-1-i].copy()

    #for islice in range(pa.rk4slice):
    ksol = ttf.expmv(mmax, dt2, ksol, ttf.delta_ksol, ttf.ddot2_ksol, ttf.dnorm2_ksol, phi1, phi2, pall.nodes[i])
    r, q = split_rq(ksol)

    if (i==nlen-1):
      r3.nodes.append(q)
    else:
      r3.nodes[nlen-1-i] = q

    phi2 = ttf.phia_next(phi2,pall.nodes[i],q,q,1)
    phia[i] = phi2

    ssol = r

    #for islice in range(pa.rk4slice):
    ssol = ttf.expmv(mmax, dt2, ssol, ttf.delta_ssol, ttf.ddot2_ssol, ttf.dnorm2_ssol, phi1, phi2)

    rtmp1 = np.tensordot(r2.nodes[i-1],ssol,axes=((2),(0)))
    r3.nodes.append(rtmp1)

  r3.nodes.reverse()
  ### the left most matrix
  phi1 = phia[0]
  phi2 = phia[1]
  ksol = r3.nodes[0].copy()
  ksol = ttf.expmv(mmax, dt2, ksol, ttf.delta_ksol, ttf.ddot2_ksol, ttf.dnorm2_ksol, phi1, phi2, pall.nodes[0])

  r3.nodes[0] = ksol

  return r3
