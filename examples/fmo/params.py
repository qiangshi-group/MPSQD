import sys
import numpy as np

# constants
pie = 3.1415926535897932
small = 1e-14
mmax = 7
nb1 = 5
nrtt = 80
nrmax = 60
nsteps = 400
dt = 150.0
au2fs = 2.418884254e-2

# paramaters
nsite = 7
nbath = 74
nl1 = 2*nbath
ndvr = nsite
nlevel = nl1*nsite
read_pall = True


nb = np.zeros(nlevel+2,dtype=int)
nb[0] = ndvr
nb[1:nlevel+1] = nb1
nb[nlevel+1] = ndvr


# setup the key variables
def setup_sys():
  au2cm = 219474.63068
# setup the system Hamiltonian
  hsys = np.zeros((nsite,nsite),dtype=np.float64)
  infile=(open("hsys.dat", 'r'))
  for i in range(nsite):
    strs = infile.readline().split()
    for j in range(nsite):
      hsys[i,j] = strs[j]
  hsys = hsys / au2cm

# setup the \nu and \gamma arrays
  omega = np.zeros((nbath),dtype=np.float64)
  cbath = np.zeros((nbath),dtype=np.float64)
  infile=(open("fmo-cbath.dat", 'r'))
  for ibath in range(nbath):
    strs = infile.readline()
    omega[ibath] = strs.split()[0]
    cbath[ibath] = strs.split()[1]

  print("omega:")
  print(omega)
  print("cbath:")
  print(cbath)

  temperature = 77.0
  bolz = 1.3806503e-23
  au2joule = 4.35974394e-18
  k_temp = bolz * temperature / au2joule
  beta = 1.0/k_temp
  k0 = 0.0
  print("beta =", beta)

#  call setup_freqs3
  wval = np.zeros((nlevel),dtype=np.complex128)
  gval = np.zeros((nlevel),dtype=np.complex128)
  gval_i = np.zeros((nlevel),dtype=np.complex128)
  normfac = np.zeros((nlevel),dtype=np.complex128)
  for ibath in range(nbath):
# real part
    wval[2*ibath] = -1.0j*omega[ibath]
    wval[2*ibath+1] = 1.0j*omega[ibath]
    gval[2*ibath] = cbath[ibath]**2/np.tanh(0.5*omega[ibath]*beta)/omega[ibath]
    gval[2*ibath+1] = gval[2*ibath]
# the imaginary ones 
    gval_i[2*ibath] = 1.0j*cbath[ibath]**2/omega[ibath]
    gval_i[2*ibath+1]   = - gval_i[2*ibath]


  gval   = 0.25*gval
  gval_i = 0.25*gval_i

# the renormalization factor
  a0coef = 0.0
  for i in range(nl1):
    a0coef = a0coef + np.real(gval[i])
  normfac[:nl1] = np.sqrt(a0coef)


# gval etc for each site
  for i in range(2, nsite):
    gval[(i-1)*nl1:i*nl1] = gval[:nl1]
    gval_i[(i-1)*nl1:i*nl1] = gval_i[:nl1]
    wval[(i-1)*nl1:i*nl1] = wval[:nl1]
    normfac[(i-1)*nl1:i*nl1] = normfac[:nl1]

  return hsys,gval,gval_i,wval,k0,normfac

hsys,gval,gval_i,wval,k0,normfac = setup_sys()

# copy from the fortran data
print("ndvr =", ndvr)
print("wval =", wval)
print("gval =", gval)
print("gval_i =", gval_i)
print("normfac =", normfac)
print("k0 =", k0)
print("hsys =", hsys)

fp1 = open('hsys.txt','w')
s1 = "  "

for i in range(ndvr):
  for j in range(ndvr):
    output = (str(i+1)+ s1 + str(j+1)
              + s1 + str(hsys[i,j])+'\n')
    fp1.write(output)

fp1.close()

########### setup all parameters ################

nbmat = nb**2

print("============================================================")
print("python code for pyrazine population and correlation function")
print("nlevel =", nlevel)
print("nb =", nb)
print("nbmat =", nbmat)
