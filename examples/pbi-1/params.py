import sys
import numpy as np

# constants
small = 1e-14
mmax = 20
nbv = 10
nrtt = 70
nrmax = 50
nsteps = 600
nlevel = 60
ndvr = 6

init_state = 0
dt = 20

# read/write pall
read_pall = False
write_pall = True#False

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
