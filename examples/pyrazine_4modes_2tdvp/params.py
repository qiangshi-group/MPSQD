import sys
import numpy as np

# constants
small = 1e-14
mmax = 40
nbv = 40
nrtt = 50
nrmax = 50
nsteps = 1000
nlevel = 4
ndvr = 2

init_state = 1
dt = 2

# read/write pall
read_pall = False
write_pall = False

# paramaters

nmps = nlevel+1
nb = np.zeros(nlevel+1,dtype=int)
nb[0] = ndvr
nb[1:] = nbv
nbmat = nb**2

print("============================================================")
print("python code for pyrazine population and correlation function")
print("nlevel =", nlevel)
print("nb =", nb)
