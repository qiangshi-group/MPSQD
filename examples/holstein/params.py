import sys
import numpy as np

# constants
au2fs = 0.02418884254
small = 1e-16
mmax = 8
nrtt = 20
nrmax = 20
nsteps = 50000

#model parameters
eta = 0.5
gamma = 1.0
beta = 1.0
vcoupling = 1.0
eps = 0.0

omega = gamma
delta = vcoupling
dt = 0.001

nl1    = 5
nsite  = 7
ndvr   = nsite
nlevel = nsite*nl1
nb1    = 10
nb2    = nb1

# read/write pall
write_pall = False

# paramaters

nmps = nlevel+2
nb = np.zeros(nlevel+2,dtype=int)
nb[0] = ndvr
nb[1] = nb1
nb[2:nlevel+1] = nb2
nb[nlevel+1] = ndvr

nbmat = nb**2

# setup the key variables
def setupfreqs():
  wval = np.empty(nl1,dtype=np.complex128)
  gval = np.empty(nl1,dtype=np.complex128)
  gval_i = np.empty(nl1,dtype=np.complex128)

  gval_i[:] = 0.0
  gval[0] = eta*omega/2.0/np.tan(0.5*beta*omega)
  gval_i[0] = -eta*omega/2.0
  wval[0] = omega

  for i in range(1,nl1,1):
      omegatmp = 2.0*np.pi*i/beta
      gval[i] = -2.0*eta*omega/beta*omegatmp/(omega**2-omegatmp**2)
      wval[i] = omegatmp

  k0 = (eta/beta-gval[0])/omega

  for i in range(2,nl1+1,1):
      k0 = k0-gval[i-1]/wval[i-1]

  a0coef = 0.0
  normfac = np.empty(nl1,dtype=np.float128)
  for i in range(1,nl1+1,1):
      normfac[i-1] = np.sqrt(abs(gval[i-1]))
      a0coef += np.real(gval[i-1])

  normfac = np.sqrt(a0coef)*np.ones(nl1)
  return wval,gval,gval_i,k0,normfac

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
