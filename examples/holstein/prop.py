import numpy as np
import params as pa
from calc_rho import calc_rho
from mpsqd.tdvp import tdvp1site


def prop(rin, pall):
  s1 = "  "
#==============================================
  # time step 0
  rho1 = calc_rho(rin)
  fp1 = open('msd.dat','w')
  fp = []
  for i in range(pa.nsite):
    output = (str(0)+ s1 + str(rho1[i,i].real) + '\n')
    fp.append(open('pop'+str(i)+'.dat','w'))
    fp[i].write(output)
    fp[i].flush()

  output1 = (str(0) + s1 + str(0) + '\n')
  fp1.write(output1)
  fp1.flush

  print("dimension of rin =", rin.ndim())
#==============================================
  for istep in range(1,pa.nsteps+1):

    print("istep =", istep)

    # the propagation step for hsys
    rin = tdvp1site(rin, pall, pa.dt)

    rho1 = calc_rho(rin)
    for i in range(pa.nsite):
      output = (str(istep*pa.dt) + s1 + str(rho1[i,i].real)+'\n')
    # write to files
      fp[i].writelines(output)
      fp[i].flush()
    # calculate the MSD
    msd = 0.0
    jj = 0
    for i in range(pa.nsite):
      jj = i
      if jj > pa.nsite/2:
        jj = pa.nsite - jj
      msd += jj**2*rho1[jj,jj].real
    output1 = (str(istep*pa.dt) + s1 + str(msd)+'\n')
    fp1.writelines(output1)
    fp1.flush()
  return
