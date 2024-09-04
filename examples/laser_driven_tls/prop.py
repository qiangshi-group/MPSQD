import numpy as np
import params as pa
from scipy.linalg import expm
from mpsqd.tdvp import tdvp1site
from calc_rho import calc_rho

def prop(rin, pall):
  s1 = "  "
#==============================================
  # time step 0
  rho1 = calc_rho(rin)

  fp = open('pop.dat','w')
  fp2=open('mu.dat','w')
  fp3=open('light_field.dat','w')

  output = (str(0)+ s1 + str(rho1[0,0].real)
                + s1 + str(rho1[1,1].real)
                + s1 + str(rho1[0,1].real)+'\n')



  fp.write(output)
  fp.flush()

  print("dimension of rin =", rin.ndim())
#==============================================
  for istep in range(1,pa.nsteps+1):

    if (istep%10 == 0):
      print("istep =", istep)

    # calculate the electric field
    et = pa.e10*np.exp(-pa.gamma1*(istep*pa.dt-2*pa.fwhm1)**2)* \
                np.cos(pa.omega1*(istep*pa.dt-2*pa.fwhm1))

    # the propgator based on electric field
    psplit1  = expm(0.5j*pa.dt*et*pa.sigmax)
    psplit1t = expm(-0.5j*pa.dt*et*pa.sigmax)

    # the split operator stuff, first half step
    rin.nodes[0]  = np.tensordot(psplit1, rin.nodes[0],axes=((1),(1))).transpose((1,0,2))
    rin.nodes[-1]  = np.tensordot(rin.nodes[-1], psplit1t, axes=((1),(0))).transpose((0,2,1))

    # the propagation step for all the rest
    rin = tdvp1site(rin, pall, pa.dt)

    # the split operator stuff, second half step
    rin.nodes[0]  = np.tensordot(psplit1, rin.nodes[0],axes=((1),(1))).transpose((1,0,2))
    rin.nodes[-1]  = np.tensordot(rin.nodes[-1], psplit1t, axes=((1),(0))).transpose((0,2,1))

    rho1 = calc_rho(rin)

    output = (str(istep*pa.dt*pa.au2fs) + s1 + str(rho1[0,0].real)
                + s1 + str(rho1[1,1].real)
                + s1 + str(rho1[0,1].real)+'\n')


    # write to files
    fp.writelines(output)
    fp.flush()

    # write the dipole moment
    mu = np.trace(np.dot(pa.sigmax,rho1))
    output2 = (str((istep+1)*pa.dt*pa.au2fs) + s1 + str(mu.real) + s1 + str(mu.imag)+'\n')
    fp2.writelines(output2)
    fp2.flush()

    # write the electronic field
    output3 = (str((istep+1)*pa.dt*pa.au2fs) + s1 + str(et.real) + s1 + str(et.imag)+'\n')
    fp3.writelines(output3)
    fp3.flush()

  return
