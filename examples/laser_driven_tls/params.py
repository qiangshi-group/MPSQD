import sys
import numpy as np

# physics constants
pie = 3.1415926535897932
au2fs = 0.02418884254
au2cm = 219474.63068
bolz = 1.3806503e-23
au2joule = 4.35974394e-18

# constants
small = 1e-15
mmax = 10 
nb1 = 10
nrtt = 10
nrmax = 50
nsteps = 4000
nlevel = 4
ndvr = 2

# model parameters
omega = 300.0/au2cm
eta = 4.0*300.0/au2cm
vcoupling = 0.0
temperature = 300.0
k_temp = bolz * temperature / au2joule
beta = 1.0/k_temp
eps = 20000.0/au2cm
# this is 1 Debye
mua = 0.3934303
# laser pulse parameters
fwhm1  = 15.0/au2fs
gamma1 = (2.0*np.sqrt(np.log(2.0))/fwhm1)**2
omega1 = eps
e10 = 5.e-4
dt = 1.0

# read/write pall
write_pall = False

# paramaters
nmps = nlevel+2
nb = np.zeros(nmps,dtype=int)
nb[0] = ndvr
nb[1:nlevel+1] = nb1
nb[nlevel+1] = ndvr

nbmat = nb**2

# setup the key variables
def setupfreqs():
  wval = np.empty(nlevel,dtype=np.complex128)
  gval = np.empty(nlevel,dtype=np.complex128)
  gval_i = np.empty(nlevel,dtype=np.complex128)

  gval_i[:] = 0.0
  gval[0] = eta*omega/2.0/np.tan(0.5*beta*omega)
  gval_i[0] = -eta*omega/2.0
  wval[0] = omega

  for i in range(1,nlevel,1):
      omegatmp = 2.0*np.pi*i/beta
      gval[i] = -2.0*eta*omega/beta*omegatmp/(omega**2-omegatmp**2)
      wval[i] = omegatmp

  k0 = (eta/beta-gval[0])/omega

  for i in range(2,nlevel+1,1):
      k0 = k0-gval[i-1]/wval[i-1]

  a0coef = 0.0
  normfac = np.empty(nlevel,dtype=np.float64)
  for i in range(1,nlevel+1,1):
      normfac[i-1] = np.sqrt(abs(gval[i-1]))
      a0coef += np.real(gval[i-1])

  normfac = np.sqrt(a0coef)*np.ones(nlevel)
  return wval,gval,gval_i,k0,normfac

wval,gval,gval_i,k0,normfac = setupfreqs()

print("ndvr =", ndvr)
print("wval =", wval)
print("gval =", gval)
print("gval_i =", gval_i)
print("normfac =", normfac)
print("k0 =", k0)

hsys = np.array(([0.0,vcoupling],[vcoupling,eps]),dtype=np.float64)
sigmax = np.array(([0.0,1.0],[1.0,0.0]),dtype=np.float64)
sdvr = np.array(([0.0,1.0]),dtype=np.float64)

print("sdvr =", sdvr)
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
