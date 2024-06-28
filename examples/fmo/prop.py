from time import time
import numpy as np
import params as pa
from mpsqd.tdvp import tdvp1

au2fs = 2.418884254e-2
def prop(rin, pall):

  s1 = "  "

#==============================================
  # time step 0
  rho1 = rin.calc_rho()
#--------------------------------------------
  fp1 = []
  for i in range(pa.ndvr):
    fname = "pop"+str(i+1)+".dat"
    fp1.append(open(fname, 'w'))
  
#--------------------------------------------
  for i in range(pa.ndvr):
    output = (str(0.0)+ s1 + str(rho1[i,i].real) +'\n')
    fp1[i].write(output)
    fp1[i].flush()


  print("dimension of rin =", rin.ndim())
#==============================================
  for istep in range(1,pa.nsteps+1):

    print("istep =", istep)

    # the propagation step for hsys
    rin = tdvp1(rin, pall, pa.dt)
    rho1 = rin.calc_rho()

#-------------------------------------------
    for i in range(pa.ndvr):
      output = str(istep*pa.dt*pa.au2fs)+ s1 + str(rho1[i,i].real) +'\n'
      fp1[i].write(output)
      fp1[i].flush()


#-------------------------------------------
  for i in range(pa.ndvr):
    fp1[i].close()
