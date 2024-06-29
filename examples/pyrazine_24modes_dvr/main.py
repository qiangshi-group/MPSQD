# This is the vibronic model with MPS and python
# This is based on the orignal python code by Nan Sheng
# Qiang Shi, Sept. 2020

import prop as pp
import params as pa
import init_wave as iw
import construct as cst
from mpsqd.utils import read_mpo_file, write_mpo_file, MPO
import sys
import numpy as np

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
    pall = copy2tensorMat0(pall)
    if (pa.write_pall):
      # write the MPO to file
      write_mpo_file(pall,'24py_tensor')

  print("++pall done++")


  # propagation and output
  pp.prop(waveall, pall)


  print("all done!")
  return

def copy2tensorMat0(node1):
  rout = MPO(pa.nmps)
  rout.nb = node1.nb
  nlen = len(node1.nodes)
  for i in range(nlen):
    ra1, rmid, ra2 = node1.nodes[i].shape
    print(ra1, rmid, ra2)
    if ( rmid == 4):
      vtmp = np.reshape(node1.nodes[i],(ra1,pa.nb[i],pa.nb[i],ra2),order='F')
    else:
      vtmp = node1.nodes[i]
    rout.nodes.append(vtmp)
  return rout

#==============================================
# the main code
main()
