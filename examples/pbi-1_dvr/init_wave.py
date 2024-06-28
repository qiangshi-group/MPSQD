import sys
import numpy as np
import params as pa
from mpsqd.utils import MPS

def init_wave():

  waveall = MPS(pa.nmps)

#----------------------------------------------------------------
# vl and vr
  vl = np.zeros((1, pa.ndvr, pa.nrtt),dtype=np.complex128)
  vr = np.zeros((pa.nrtt, pa.nb[pa.nlevel], 1),dtype=np.complex128)

  vl[0,0,0] = 1.0
  vr[0,:,0] = pa.init_wave_all[pa.nlevel-1].copy()


#----------------------------------------------------------------
# vmid
  vmid = []
  for i in range(pa.nlevel-1):
    vtmp = np.zeros((pa.nrtt, pa.nb[i+1], pa.nrtt),dtype=np.complex128)
    
    vtmp[0,:,0] = pa.init_wave_all[i].copy()
    vmid.append(vtmp)


#----------------------------------------------------------------
# store them in nodes
  waveall.nodes.append(vl)

  for i in range(pa.nlevel-1):
    waveall.nodes.append(vmid[i])

  waveall.nodes.append(vr)


  waveall.nb = pa.nb
  print("length of waveall: {}".format(len(waveall.nodes)))
  print("waveall initialization done!")


  return waveall
