import sys
import numpy as np

# constants
au2fs = 0.02418884254
small = 1e-16
mmax = 15
nrtt = 90
nrmax = 40
nsteps = 20000

#model parameters
beta = 1.0
vcoupling = 0.1
eps = 0.0

nmbo = 1
pk = []
omega_b =[]
gamma_b = [0.8]
for i in range(nmbo):
  omega_b.append(np.sqrt(1.0**2 - gamma_b[i]**2))
  pk.append(omega_b[i]**2*gamma_b[i] + gamma_b[i]**3)
delta = vcoupling
dt = 0.005

nl1    = 3
nsite  = 21
nsiteeq = 1
ndvr   = nsite
ndvreq = nsiteeq
nlevel = nsite*nl1
nleveleq = nsiteeq*nl1
nb1    = 20
nb2    = nb1
nmps = nlevel+2

nb = np.zeros(nmps,dtype=int)
nb[0] = ndvr
nb[1] = nb1
nb[2:nlevel+1] = nb2
nb[nlevel+1] = ndvr

# read/write pall
write_pall = False

nbmat = nb**2

# setup the key variables
def setupfreqs():
  wval = np.empty(nl1,dtype=np.complex128)
  gval = np.empty(nl1,dtype=np.complex128)
  gval_i = np.empty(nl1,dtype=np.complex128)

  gval_i[:] = 0.0
  for i in range(nmbo):
    pk1 = pk[i]/8.0/gamma_b[i]/omega_b[i]
    wval[2*i]   = gamma_b[i] + 1.0j*omega_b[i]
    wval[2*i+1] = gamma_b[i] - 1.0j*omega_b[i]
    wtmp1 = 0.5*beta*(-1.0j*gamma_b[i] + omega_b[i])
    wtmp2 = 0.5*beta*(+1.0j*gamma_b[i] + omega_b[i])
    gval[2*i]   = pk1*(np.exp(wtmp1) + np.exp(-wtmp1))/(np.exp(wtmp1) - np.exp(-wtmp1))
    gval[2*i+1] = pk1*(np.exp(wtmp2) + np.exp(-wtmp2))/(np.exp(wtmp2) - np.exp(-wtmp2))
    gval_i[2*i] = -1.0j*pk1
    gval_i[2*i+1] = 1.0j*pk1
  for i in range(1,nl1-2*nmbo+1,1):
    omegatmp = 2.0*np.pi*i/beta
    jwtmp = 0.0
    for j in range(nmbo):
      jwtmp += pk[j]/((-omegatmp**2+omega_b[j]**2+gamma_b[j]**2)**2+4*omega_b[j]**2*omegatmp**2)
    gval[i+2*nmbo-1] = -2.0*jwtmp/beta*omegatmp
    wval[i+2*nmbo-1] = omegatmp
  k0 = 0.0
  for i in range(nl1-2*nmbo+1,10000,1):
    omegatmp = 2.0*np.pi*i/beta
    jwtmp = 0.0
    for j in range(nmbo):
      jwtmp += pk[j]/((-omegatmp**2+omega_b[j]**2+gamma_b[j]**2)**2+4*omega_b[j]**2*omegatmp**2)
    k0 -= 2.0*jwtmp/beta
  a0coef = 0.0
  normfac = np.empty(nl1,dtype=np.float128)
  for i in range(1,nl1+1,1):
     normfac[i-1] = np.sqrt(abs(gval[i-1]))
     a0coef += np.real(gval[i-1])

  normfac = np.sqrt(a0coef)*np.ones(nl1)

  return wval, gval, gval_i, k0, normfac
wval,gval,gval_i,k0,normfac = setupfreqs()

print("ndvr =", ndvr)
print("wval =", wval)
print("gval =", gval)
print("gval_i =", gval_i)
print("normfac =", normfac)
print("k0 =", k0)

#setup hsys
hsys = np.zeros((nsite,nsite),dtype=np.float64)
for i in range(nsite):
  hsys[i,i] = eps
  i1 = i-1
  if i1 >= 0:
    hsys[i1,i] = delta
    hsys[i,i1] = delta
  i2 = i+1
  if i2 < nsite:
    hsys[i2,i] = delta
    hsys[i,i2] = delta
hsys[0,nsite-1] = delta
hsys[nsite-1,0] = delta

print("hsys =", hsys)

fp1 = open('hsys.txt','w')
s1 = "  "

for i in range(ndvr):
  for j in range(ndvr):
    output = (str(i+1)+ s1 + str(j+1)
              + s1 + str(hsys[i,j])+'\n')
    fp1.write(output)

fp1.close()

print("============================================================")
print("python code for pyrazine population and correlation function")
print("nlevel =", nlevel)
print("nb =", nb)
print("nbmat =", nbmat)
