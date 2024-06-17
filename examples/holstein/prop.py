import numpy as np
import params as pa
import mpsqd.ksltt as ksl

def prop(rin, pall):
  s1 = "  "
#==============================================
  # time step 0
  rho1 = rin.calc_rho()

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
    rin = ksl.ksltt(rin, pall, pa.dt)

    rho1 = rin.calc_rho()

    output = (str(istep*pa.dt) + s1 + str(rho1[0,0].real)
                + s1 + str(rho1[1,1].real)
                + s1 + str(rho1[0,1].real)+'\n')


    # write to files
    fp.writelines(output)
    fp.flush()
  return
