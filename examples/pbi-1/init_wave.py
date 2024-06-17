import sys
import numpy as np
import params as pa
from mpsqd.utils import MPS

def init_wave():
  waveall = MPS(pa.nlevel+1)
  waveall.nb = pa.nb.copy()
  vl = np.zeros((1, pa.nb[0], pa.nrtt),dtype=np.complex128)
  vl[0,pa.init_state,0] = 1.0
  waveall.nodes.append(vl)
  for i in range(pa.nlevel-1):
    vtmp = np.zeros((pa.nrtt, pa.nb[i+1], pa.nrtt),dtype=np.complex128)
    vtmp[0,0,0] = 1.0
    waveall.nodes.append(vtmp.copy())
  vr = np.zeros((pa.nrtt, pa.nb[pa.nlevel], 1),dtype=np.complex128)
  vr[0,0,0] =1.0 
  waveall.nodes.append(vr)

  print("waveall initialization done!")
  return waveall
