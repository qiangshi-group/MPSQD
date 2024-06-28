import sys
import numpy as np
from ..utils import add_tensor, multiply_tensor

def rk4(wave,mat1,delta_t,small=1e-14,nrmax=50,need_trun=True):

  dt2 = delta_t/2.0
  dt6 = delta_t/6.0

  dwave1 = multiply_tensor(mat1,wave,small,nrmax,need_trun)
  wave0 = add_tensor(wave,dwave1,dt2,small,nrmax,need_trun)

  dwave2 = multiply_tensor(mat1,wave0,small,nrmax,need_trun)
  wave0 = add_tensor(wave,dwave2,dt2,small,nrmax,need_trun)

  dwave3 = multiply_tensor(mat1,wave0,small,nrmax,need_trun)
  wave0 = add_tensor(wave,dwave3,delta_t,small,nrmax,need_trun)

  dwave3 = add_tensor(dwave3,dwave2,1.0,small,nrmax,need_trun)
  dwave2 = multiply_tensor(mat1,wave0,small,nrmax,need_trun)

  wave = add_tensor(wave,dwave1,dt6,small,nrmax,need_trun)
  wave = add_tensor(wave,dwave3,2.0*dt6,small,nrmax,need_trun)
  wave = add_tensor(wave,dwave2,dt6,small,nrmax,need_trun)

  return wave
