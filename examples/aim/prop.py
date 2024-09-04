import time
import numpy as np
import params as pa
import calc_rho as cr
from mpsqd.utils import MPS
from mpsqd.tdvp import tdvp1site
from mpsqd.rk4 import rk4

def prop(rin, pall):

  s1 = "  "

#==============================================
  # time step 0
  istep = 0
  fp_diag = open('opr-pop.dat','w')

  fpc1 = open('opr-lcurrent.dat','w')
  fpc2 = open('opr-rcurrent.dat','w')
  fpc3 = open('opr-avecurrent.dat','w')

  fbp1 = open('opr-bridge_pop.dat','w')

  fptr = open('opr-trace.dat','w')
  fpds = open('opr-dspin12.dat','w')

  print("dimension of rin =", rin.ndim())

  fptime = open('opr-time.dat','w')

  fpall={1:fp_diag,3:fpc1,4:fpc2,5:fpc3,6:fbp1,7:fptr,8:fpds,11:fptime}
#===============================================
# initial step with rk4
  wr_PhyQuant(0,rin,fp_diag,fbp1,fpds,fptr,fpc1,fpc2,fpc3) 

  istep += 1
  print('initial rk4')
  rin = rk4(rin, pall, 0.1*pa.dt, small=pa.small_mps, nrmax=pa.nrmax)
  rin = expand_nrtt(rin,pa.nrtt)

  print('initial ksl')
  rin = tdvp1site(rin, pall, 0.9*pa.dt, mmax=pa.mmax)
  print("dimension of rin =", rin.ndim())

  wr_PhyQuant(1,rin,fp_diag,fbp1,fpds,fptr,fpc1,fpc2,fpc3) 

  while(istep<pa.nsteps):
    istep += 1
    itime_st=time.time()
    print("istep =", istep)
  
    rin = tdvp1site(rin, pall, pa.dt, mmax=pa.mmax)
  
    wr_PhyQuant(istep,rin,fp_diag,fbp1,fpds,fptr,fpc1,fpc2,fpc3) 
  
    print( "Time for istep =",istep, time.time()-itime_st)
    fptime.write(str(istep)+s1+str(time.time()-itime_st)+'\n')
    fptime.flush()
  
  return rin

def expand_nrtt(rin,nbond_dim):
  nlen = len(rin.nodes)
  rout = MPS(nlen)
  rout.nb = rin.nb

  type_of_rho = rin.nodes[0].dtype

#---------------------------------------------
# vl node
  n0, n1, n2 = rin.nodes[0].shape
  nn = nbond_dim
  # need to zero and then add, otherwise strange problems
  vtmp = np.zeros((n0,n1,nn),dtype=type_of_rho)
  vtmp[:,:,:n2] = rin.nodes[0][:,:,:n2]
  rout.nodes.append(vtmp)

#---------------------------------------------
# vmid nodes
  for i in range(1,nlen-1):
    n0, n1, n2 = rin.nodes[i].shape
    nn = nbond_dim
  # need to zero and then add, otherwise strange problems
    vtmp = np.zeros((nn,n1,nn),dtype=type_of_rho)
    vtmp[:n0,:,:n2] = rin.nodes[i]
    rout.nodes.append(vtmp)

#----------------------------------------------
# the vr node
  n0, n1, n2 = rin.nodes[nlen-1].shape
  nn = nbond_dim
 # need to zero and then add, otherwise strange problems
  vtmp = np.zeros((nn,n1,n2),dtype=type_of_rho)
  vtmp[:n0,:,:] = rin.nodes[nlen-1][:n0,:,:]
  rout.nodes.append(vtmp)

  return rout

#write out physical quantity
#fp_diag   : rho population
#fbp  : bridge population
#fpds : dspin
#fptr : trace
#fpc : current
def wr_PhyQuant(istep,rin,fp_diag,fbp,fpds,fptr,fpc1,fpc2,fpc3):
  s1 = '  '
  rho1 = cr.calc_rho(rin)
  output = (str(istep*pa.dt) + s1 + str(rho1[0,0].real) \
              + s1 + str(rho1[1,1].real) \
              + s1 + str(rho1[2,2].real) \
              + s1 + str(rho1[3,3].real)+'\n')

  # write to files
  fp_diag.writelines(output);fp_diag.flush()

  #bridge population of AIM
  brid_pop = rho1[1,1] + rho1[2,2] + 2.0 * rho1[3,3]
  outbdp = (str(istep*pa.dt)+ s1 + str(brid_pop.real) + s1 + str(brid_pop.imag)+'\n')
  fbp.write(outbdp);fbp.flush()

  #spin difference of AIM
  dspin = rho1[2,2]-rho1[1,1]
  outds = (str(istep*pa.dt) +s1+ str(dspin.real) +s1+ str(dspin.imag) +'\n')
  fpds.write(outds);fpds.flush()

  #write out trace
  trace0= np.trace(rho1)
  output = (str(istep*pa.dt) + s1  + str(trace0.real) +s1+ str(trace0.imag) +'\n')
  fptr.writelines(output);fptr.flush()

  #the current
  z1,z2,z3 = cr.calc_current(rin)
  output1 = (str(istep*pa.dt)+s1+str(z1.real)+s1+str(z1.imag)+'\n')
  output2 = (str(istep*pa.dt)+s1+str(z2.real)+s1+str(z2.imag)+'\n')
  output3 = (str(istep*pa.dt)+s1+str(z3.real)+s1+str(z3.imag)+'\n')
  fpc1.writelines(output1);fpc1.flush()
  fpc2.writelines(output2);fpc2.flush()
  fpc3.writelines(output3);fpc3.flush()

  return
