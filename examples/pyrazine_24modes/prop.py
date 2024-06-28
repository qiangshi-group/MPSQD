from time import time
import numpy as np
from mpsqd.tdvp import tdvp1
import sys
from mpsqd.utils import calc_overlap
import params as pa

au2fs = 2.418884254e-2
def prop(rin, pall):
  s1 = "  "
  rin0 = rin.copy()
#==============================================
  # time step 0
  output = (str(0.0)+ s1)
  pop = rin.calc_pop()
  acf = calc_overlap(rin,rin0)
  ndvr = np.shape(pall.nodes[0])[1]
  for i in range(ndvr):
    output = output + str(pop[i].real) + s1
  output = output + '\n'
  fp = open('pop.dat','w')
  fp.write(output)
  fp.flush()
  output = (str(0.0)+ s1 + str(acf) +'\n')
  fp1 = open('acf.dat','w')
  fp1.writelines(output)
  fp1.flush()
  print("dimension of rin =", rin.ndim())

#==============================================
  for istep in range(1,pa.nsteps+1):
    print("istep =", istep)
    # the propagation step for hsys
    rin = tdvp1(rin, pall, pa.dt, mmax=pa.mmax)
    pop = rin.calc_pop()
    acf = calc_overlap(rin,rin0)
    output = (str(istep*pa.dt*au2fs)+ s1)
    for i in range(ndvr):
      output = output + str(pop[i].real) + s1
    output = output + '\n'
    fp.writelines(output)
    fp.flush()
    output = (str(istep*pa.dt*au2fs)+ s1 + str(acf) +'\n')
    fp1.writelines(output)
    fp1.flush()
  return rin
