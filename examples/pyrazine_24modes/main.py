# This is the vibronic model with MPS and python
# This is based on the orignal python code by Nan Sheng
# Qiang Shi, Sept. 2020

import prop as pp
import params as pa
import init_wave as iw
import construct as cst
from mpsqd.utils import read_mpo_file, write_mpo_file, MPS2MPO
import sys

#===============================
# the main function
def main():

  # calculate the initial wavefunction
  waveall = iw.init_wave()

#----------------------------------------
# get pall
  if (pa.read_pall):
    # read MPO from file
    pall = read_mpo_file('24py_tensor')
  else:
    # construct the MPO
    pall = cst.construct()
    pall = MPS2MPO(pall)
    if (pa.write_pall):
      # write the MPO to file
      write_mpo_file(pall,'24py_tensor')
##----------------------------------------
#  # read MPO from file
#  pall = read_mpo_file('py_tensor')

  print("++pall done++")


  # propagation and output
  pp.prop(waveall, pall)


  print("all done!")
  return
#==============================================
# the main heom TT code

main()
