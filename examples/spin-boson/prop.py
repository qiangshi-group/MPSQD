import numpy as np
import params as pa
from mpsqd.tdvp import tdvp1site
from calc_rho import calc_rho

def prop(rin, pall):
  s1 = "  "
#==============================================
  # time step 0
  rho1 = calc_rho(rin)

  output = (str(0)+ s1 + str(rho1[0,0].real)
                + s1 + str(rho1[1,1].real)
                + s1 + str(rho1[0,1].real)+'\n')


  fp = open('output.dat','w')
  fp.write(output)
  fp.flush()

  print("dimension of rin =", rin.ndim())
#==============================================
  for istep in range(1,pa.nsteps+1):

    print("istep =", istep)

    # the propagation step for hsys
    rin = tdvp1site(rin, pall, pa.dt)

    rho1 = calc_rho(rin)

    output = (str(istep*pa.dt) + s1 + str(rho1[0,0].real)
                + s1 + str(rho1[1,1].real)
                + s1 + str(rho1[0,1].real)+'\n')


    # write to files
    fp.writelines(output)
    fp.flush()
  return
