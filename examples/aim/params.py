import sys
import numpy as np
from mpsqd.utils import MPS

# constants
pie = 3.1415926535897932
#The accuracy in constructing the MPO
small_mpo = 1e-14
#The accuracy in doing MPS trucation
small_mps = 1e-8

hbar = 1.0

#simulation parameters
#the Kylov space in update the nodes
mmax = 20
nrtt = 100  
#max rank
nrmax = 100 
nsteps = 600 
dt = 0.005

# read/write pall
write_pall = False
read_pall = False

nlevel1 = 8
nlevel2 = 8
nlevel = 4*(nlevel1 + nlevel2)

ndvr = 4
nb1 = 2

nb = np.zeros(nlevel+2,dtype=int)
nb[0] = ndvr
nb[1] = ndvr
nb[2:nlevel+2] = nb1
nbmat = nb**2

phi = 5.0/hbar
beta = 1.0

mu_l = 0.5*phi
eta_l = 1.0/hbar
omegac_l = 10.0/hbar
omega0_l = 0.0

mu_r = -mu_l
eta_r = 1.0/hbar
omegac_r = 10.0/hbar
omega0_r = 0.0

eps_1 = -4.0/hbar
eps_2 = -4.0/hbar
ucoup = 8.0/hbar

hsys = np.zeros((ndvr,ndvr),dtype=np.float64)
hsys[1,1] = eps_1
hsys[2,2] = eps_2
hsys[3,3] = eps_1 + eps_2 + ucoup
aplus1 = np.zeros((ndvr,ndvr),dtype=np.float64)
aplus1[1,0] = 1.0
aplus1[3,2] = 1.0
aminus1 = np.zeros((ndvr,ndvr),dtype=np.float64)
aminus1[0,1] = 1.0
aminus1[2,3] = 1.0

aplus2 = np.zeros((ndvr,ndvr),dtype=np.float64)
aplus2[2,0] = 1.0
aplus2[3,1] = -1.0
aminus2 = np.zeros((ndvr,ndvr),dtype=np.float64)
aminus2[0,2] = 1.0
aminus2[1,3] = -1.0
print("use pade to SOP expansion")

# this sets up the correlation function decompositon
# for the electrode, based on the Pade approximation
def setup_freqs_pade_ct2_fk0(nlevel,omegac,omega0,mu,eta):
  nlength = nlevel1
  npade = nlevel1 - 1
  wval_ctp = np.zeros((nlength),dtype=np.complex128)
  gval_ctp = np.zeros((nlength),dtype=np.complex128)
  wval_ctm = np.zeros((nlength),dtype=np.complex128)
  gval_ctm = np.zeros((nlength),dtype=np.complex128)
  pole = np.zeros((npade),dtype=np.float64)
  residue = np.zeros((npade),dtype=np.float64)
# OK, we work on the Lorentizen part only
# Notice that Lorentz spectrum is :eta*gamma^2/((omega-omega0)^2+gamma^2)/(2*pi)
# or no (2*pi)
# it seems our original mode corresponds to the minus operator
  wval_ctp[0] = omegac - 1.0j*omega0
  wval_ctm[0] = omegac + 1.0j*omega0
  gval_ctp[0] = np.pi*eta*omegac/(1.0+np.exp(beta*(omega0-mu+1.0j*omegac)))/(2.0*np.pi)
  gval_ctm[0] = np.pi*eta*omegac/(1.0+np.exp(-beta*(omega0-mu-1.0j*omegac)))/(2.0*np.pi)
  print("gval[0] =", gval_ctp[0], gval_ctm[0])

# Pade decoposition of the Fermi function
# calculate the poles
# this is for the Fermi function
  temp = 1.0
  dummy = np.zeros((2*npade,2*npade),dtype=np.float64)
  for i in range(1,2*npade):
    dummy[i-1,i] = 1.0/np.sqrt(temp*(temp+2.0))
    dummy[i,i-1] = 1.0/np.sqrt(temp*(temp+2.0))
    temp = temp+2.0
  diag,_ = np.linalg.eig(dummy)
  diag.sort()

  for i in range(npade):
    pole[i] = 2.0/diag[2*npade-i-1]

# calculate the residues
# first, poles for P_N
  diag[:] = 0.0
  dummy[:,:] = 0.0
# this is for ther Fermi function
  temp = 3.0
  for i in range(1, 2*npade-1):
    dummy[i-1,i] = 1.0/np.sqrt(temp*(temp+2.0))
    dummy[i,i-1] = 1.0/np.sqrt(temp*(temp+2.0))
    temp=temp+2.0
  diag,_ = np.linalg.eig(dummy[:2*npade-1,:2*npade-1])
  diag.sort()

  dummy = pole**2

  for i in range(npade-1):
    diag[i] = (2.0/diag[2*npade-i-2])**2

  for i in range(npade):
    if(i == npade-1):
      temp = 0.5*npade*(2*npade+1)
    else:
      temp = 0.5*npade*(2*npade+1)*(diag[i]-dummy[i])/(dummy[npade-1]-dummy[i])
    for j in range(i):
      temp = temp*(diag[j]-dummy[i])/(dummy[j]-dummy[i])
    for j in range(i+1, npade-1):
      temp = temp*(diag[j]-dummy[i])/(dummy[j]-dummy[i])
    residue[i]=temp

  print("npade =", npade)
  for i in range(npade):
    print(pole[i], residue[i])

  for i in range(npade):
# the aplus parts
    omegtmp = (pole[i]/beta - 1.0j*mu)
    jwtmp = eta*omegac**2/((1.0j*omegtmp-omega0)**2+omegac**2)/(2.0*np.pi)
    gval_ctp[i+1] = - 2.0*np.pi*1.0j*jwtmp/beta*residue[i]
    wval_ctp[i+1] = omegtmp
# the aminus parts
    omegtmp = (pole[i]/beta + 1.0j*mu)
    jwtmp = eta*omegac**2/((-1.0j*omegtmp-omega0)**2+omegac**2)/(2.0*np.pi)
    gval_ctm[i+1] = - 2.0*np.pi*1.0j*jwtmp/beta*residue[i]
    wval_ctm[i+1] = omegtmp

# now we need to recalculate gval[0]
# plus part
  ctemp = 0.5*np.pi*eta
  omegtmp = omega0 + 1.0j*omegac - mu
  for i in range(npade):
    ctemp = ctemp - 2.0*np.pi*eta*residue[i]*beta*omegtmp/(beta**2*omegtmp**2+dummy[i])
# we should notice the (2.0*pi) is the same coefficient of the spectrum density
  gval_ctp[0] = omegac*ctemp/(2.0*np.pi)
# minus part
  ctemp = 0.5*np.pi*eta
  omegtmp = omega0 - 1.0j*omegac - mu
  for i in range(npade):
    ctemp = ctemp + 2.0*np.pi*eta*residue[i]*beta*omegtmp/(beta**2*omegtmp**2+dummy[i])
# we should notice (2.d0*pi) comes from the spectrum density function.
  gval_ctm[0] =omegac*ctemp/(2.0*np.pi)
  print("modified gval[0] =", gval_ctp[0], gval_ctm[0])
  return wval_ctp,wval_ctm,gval_ctp,gval_ctm

wval_ctp_l,wval_ctm_l,gval_ctp_l,gval_ctm_l = setup_freqs_pade_ct2_fk0(nlevel1,omegac_l,omega0_l,mu_l,eta_l)
wval_ctp_r,wval_ctm_r,gval_ctp_r,gval_ctm_r = setup_freqs_pade_ct2_fk0(nlevel2,omegac_r,omega0_r,mu_r,eta_r)

#setup the array
wval_ct = np.zeros((nlevel),dtype=np.complex128)
gval_ct = np.zeros((nlevel),dtype=np.complex128)
gval_ct_bar = np.zeros((nlevel),dtype=np.complex128)
aop = np.zeros((ndvr,ndvr,nlevel),dtype=np.float64)
aop_bar = np.zeros((ndvr,ndvr,nlevel),dtype=np.float64)

for i in range(nlevel1):
  wval_ct[i] = wval_ctp_l[i]
  wval_ct[nlevel1+i] = wval_ctm_l[i]

  gval_ct[i] = gval_ctp_l[i]
  gval_ct[nlevel1+i] = gval_ctm_l[i]

  gval_ct_bar[i] = np.conj(gval_ctm_l[i])
  gval_ct_bar[nlevel1+i] = np.conj(gval_ctp_l[i])

  aop[:,:,i] = aplus1
  aop[:,:,nlevel1+i] = aminus1

  aop_bar[:,:,i] = aminus1
  aop_bar[:,:,nlevel1+i] = aplus1

wval_ct[2*nlevel1:4*nlevel1] = wval_ct[:2*nlevel1]
gval_ct[2*nlevel1:4*nlevel1] = gval_ct[:2*nlevel1]
gval_ct_bar[2*nlevel1:4*nlevel1] = gval_ct_bar[:2*nlevel1]

for i in range(nlevel1):
  aop[:,:,2*nlevel1+i] = aplus2
  aop[:,:,3*nlevel1+i] = aminus2
  aop_bar[:,:,2*nlevel1+i] = aminus2
  aop_bar[:,:,3*nlevel1+i] = aplus2

for i in range(nlevel2):
  wval_ct[4*nlevel1+i] = wval_ctp_r[i]
  wval_ct[4*nlevel1+nlevel2+i] = wval_ctm_r[i]

  gval_ct[4*nlevel1+i] = gval_ctp_r[i]
  gval_ct[4*nlevel1+nlevel2+i] = gval_ctm_r[i]

  gval_ct_bar[4*nlevel1+i] = np.conj(gval_ctm_r[i])
  gval_ct_bar[4*nlevel1+nlevel2+i] = np.conj(gval_ctp_r[i])

  aop[:,:,4*nlevel1+i] = aplus1
  aop[:,:,4*nlevel1+nlevel2+i] = aminus1

  aop_bar[:,:,4*nlevel1+i] = aminus1
  aop_bar[:,:,4*nlevel1+nlevel2+i] = aplus1

wval_ct[4*nlevel1+2*nlevel2:nlevel] = wval_ct[4*nlevel1:4*nlevel1+2*nlevel2]
gval_ct[4*nlevel1+2*nlevel2:nlevel] = gval_ct[4*nlevel1:4*nlevel1+2*nlevel2]
gval_ct_bar[4*nlevel1+2*nlevel2:nlevel] = gval_ct_bar[4*nlevel1:4*nlevel1+2*nlevel2]

for i in range(nlevel2):
  aop[:,:,4*nlevel1+2*nlevel2+i] = aplus2
  aop[:,:,4*nlevel1+3*nlevel2+i] = aminus2
  aop_bar[:,:,4*nlevel1+2*nlevel2+i] = aminus2
  aop_bar[:,:,4*nlevel1+3*nlevel2+i] = aplus2

wval_ct = wval_ct/hbar
gval_ct = gval_ct/(hbar**2)
gval_ct_bar = gval_ct_bar/(hbar**2)

normfac = np.zeros((nlevel),dtype=np.float64)
for i in range(nlevel):
  if(i<4*nlevel1):
    normfac[i] = (wval_ct[i]*np.sqrt(eta_l)).real
  else:
    normfac[i] = (wval_ct[i]*np.sqrt(eta_r)).real

print('normfac',normfac)

s1 = "  "
#write out the freqs
fp_freq = open('opa-wgval.dat','w')
for i in range(2*nlevel1):
  output = (str(i+1) + s1 + str((wval_ct[i]*hbar).real) + s1 + str((wval_ct[i]*hbar).imag) \
            + s1 + str(gval_ct[i].real) + s1 + str(gval_ct[i].imag)+'\n')
  fp_freq.write(output)
for i in range(4*nlevel1,4*nlevel1+2*nlevel2):
  output = (str(i+1) + s1 + str((wval_ct[i]*hbar).real) + s1 + str((wval_ct[i]*hbar).imag) \
            + s1 + str(gval_ct[i].real) + s1 + str(gval_ct[i].imag)+'\n')
  fp_freq.write(output)
fp_freq.close()

#draw the correlation function
nstepsp=2**15
dtp = 0.05
bc_mode={'lp':['Lp',0,nlevel1],'lm':['Lm',nlevel1,2*nlevel1],\
           'rp':['Rp',4*nlevel1,4*nlevel1+nlevel2],'rm':['Rm',4*nlevel1+nlevel2,4*nlevel1+2*nlevel2]}
for i in bc_mode:
  f_bc = open('opa-bcorr'+bc_mode[i][0]+str(nlevel1)+'.dat','w')
  for istep in range(nstepsp):
    t1 = dtp*istep
    ctmp = 0.0
    for jlevel in range(bc_mode[i][1],bc_mode[i][2]):
      ctmp += gval_ct[jlevel]*hbar**2*np.exp(-t1*wval_ct[jlevel]*hbar)
    f_bc.write(str(t1)+s1+str(ctmp.real)+s1+str(ctmp.imag)+'\n')
  f_bc.close()

########### setup all parameters ################
print("============================================================")
print("python code for the AIM model")
print("nlevel =", nlevel)
print("nb =", nb)
print("nbmat =", nbmat)

print("ndvr =", ndvr)
print("wval_ct =", wval_ct)
print("gval_ct =", gval_ct)
print("gval_ct_bar =", gval_ct_bar)
print("normfac =", normfac)

print("hsys =", hsys)

fp1 = open('opa-hsys.dat','w')
for i in range(ndvr):
  for j in range(ndvr):
    output = (str(i+1)+ s1 + str(j+1)
              + s1 + str(hsys[i,j])+'\n')
    fp1.write(output)
fp1.close()
