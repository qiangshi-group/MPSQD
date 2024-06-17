# This is the spin-boson model with MPS and python
# This is based on the orignal python code by Nan Sheng
# Qiang Shi, Sept. 2020

import params as pa
import prop as pp
import init_rho as ir
import construct as cst
from mpsqd.utils import MPS2MPO, write_mpo_file

#===============================
# the main function
def main():

  # calculate the initial wavefunction
  rhoall = ir.init_rho()

#----------------------------------------
# get pall
# construct the MPO
  pall = cst.construct()
  if (pa.write_pall):
    # write the MPO to file
    write_mpo_file(pall,pa.pall_file)

  print("++pall done++")

  # copy pall
  pallnew = MPS2MPO(pall)
  print("++reshape of pall done++")

  # propagation and output
  pp.prop(rhoall, pallnew)

  print("all done!")

#==============================================
# the main heom TT code

main()
