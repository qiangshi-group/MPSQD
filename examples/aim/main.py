######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# 23.09: Calculate the impurity spectral function 
# Reference: Xiaohan Dan, Meng Xu, J. T. Stockburger, J. Ankerhold, and Qiang Shi
# "Efficient low-temperature simulations for 
# fermionic reservoirs with the hierarchical equations
# of motion method: Application to the Anderson impurity model"
# PHYSICAL REVIEW B 107, 195429 (2023).  
# main.py: the main program

from time import time
import sys
import params as pa
import numpy as np
import prop as pp
import init_rho as ir
import construct as cst
from mpsqd.utils import MPS2MPO, write_mpo_file, read_mpo_file

def TTHEOM():

  time0 = time()
  # calculate the initial wavefunction
  rhoall = ir.init_rho()

#----------------------------------------
# get pall when MPO is stored
  if (pa.read_pall):
    # read MPO from file
    pall = read_mpo_file('aim_tensor')
  else:
    # construct the MPO
    pall = cst.construct()
    pall = MPS2MPO(pall)
    if (pa.write_pall):
      # write the MPO to file
      write_mpo_file(pall,'aim_tensor')

  print("++pall done++")

  req = pp.prop(rhoall, pall)
  print("all done!")
  print("running time = "+ str(time()-time0) + " second")
  return

TTHEOM()
