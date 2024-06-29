from time import time
import numpy as np
import params as pa
from mpsqd.tdvp import tdvp1site
from calc_wave import calc_pop

au2fs = 2.418884254e-2

def prop(rin, pall):

  s1 = "  "

#==============================================
  # time step 0
  pop = calc_pop(rin)
  output = (str(0.0)+ s1)
  ndvr = np.shape(pall.nodes[0])[1]
  for i in range(ndvr):
    output = output + str(pop[i].real) + s1
  output = output + '\n'
  fp = open('output.dat','w')
  fp.write(output)
  fp.flush()


  print("dimension of rin =", rin.ndim())
#==============================================
  for istep in range(1,pa.nsteps+1):

    print("istep =", istep)

#---------------------------------------------------
    # the half step for the kinetic part
    rin = pro_rho_kinet(rin)
    # the propagation step for hsys
    rin = tdvp1site(rin, pall, pa.dt, mmax=pa.mmax)

    # another half step for the kinetic part
    rin = pro_rho_kinet(rin)

#---------------------------------------------------
    pop = calc_pop(rin)
    output = (str(istep*pa.dt*au2fs)+ s1)
    for i in range(ndvr):
      output = output + str(pop[i].real) + s1
    output = output + '\n'
    # write to files
    fp.writelines(output)
    fp.flush()

#================================================
def pro_rho_kinet(r1):

  for i in range(pa.nlevel):
    rtmp1 = r1.nodes[i+1]
    r1.nodes[i+1] = np.tensordot(pa.vkprop_all[i],rtmp1,axes=((1),(1))).transpose(1,0,2)

  return r1
