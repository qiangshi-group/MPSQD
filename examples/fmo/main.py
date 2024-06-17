# This is based on the orignal python code by Nan Sheng
# Qiang Shi, Feb. 2020

from time import time
import sys
import numpy as np
import prop as pp
import params as pa
import construct as cst
import init_rho as ir
from mpsqd.utils import MPS2MPO, read_mpo_file, read_mps_file

def TTHEOM():

  # get the initial wavefunction and pall
  if(pa.read_pall):
    rhoall = read_mps_file('fmo')
    pallnew = read_mpo_file('fmo')
  else:
    pall = cst.construct()
    pallnew = MPS2MPO(pall)
    rhoall = ir.init_rho()

  print("++pall done++")

  # propagation and output
  pp.prop(rhoall, pallnew)

  print("all done!")
  return

TTHEOM()
