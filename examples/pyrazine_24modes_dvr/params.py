import sys
import numpy as np
import model as md
import dvr

# constants
small = 1e-13
ndvr = 2
mmax = 40
nbv = 110
nlevel = md.nmode
nmps = nlevel + 1
nrtt = 120
nrmax = 50
nsteps = 10
dt = 10.6707

read_pall = False
write_pall = False
pall_file = "pall4mod"

########### setup all parameters ################
#nb and nbmat
nb = nbv*np.ones(nlevel+1,dtype=int)
nb[0] = 2

nbmat = nb.copy()
nbmat[0] = nb[0]**2

#-----------------------------
# setup the dvr basis realted stuff
nr = 256
sdvr_all, sphi_all, pot0_all, init_wave_all, vkprop_all = dvr.dvrsetup(nr,nbv,md.omega,dt)
