# This is based on the orignal python code by Nan Sheng
# Qiang Shi, Feb. 2020

import sys, os
import numpy as np
from .mpsdef import MPS, MPO

#==============================================
def write_mpo_file(pall,pall_file):
  os.makedirs(pall_file, exist_ok=True) 
  mpodata = {}
  for k in range(pall.length):
    mpodata[str(k)] = pall.nodes[k]
  filename = pall_file + "/" + pall_file+"_mpo"
  np.savez(filename, **mpodata)
  print("pall written to file")
  return

def write_mps_file(waveall,pall_file):
  os.makedirs(pall_file, exist_ok=True)
  mpsdata = {}
  filename = pall_file + "/" + pall_file+"_mps"
  for k in range(waveall.length):
    mpsdata[str(k)] = waveall.nodes[k]
  np.savez(filename, **mpsdata)
  return

#==============================================
def read_mpo_file(pall_file):
  filename = pall_file + "/" + pall_file+"_mpo.npz"
  try:
    mpofile = np.load(filename)
  except FileNotFoundError:
    print("Did not find MPO file: " + filename)
    sys.exit()
  except ValueError:
    print("Unexpected tensor: " + filename)
    sys.exit()
  nlen = len(mpofile)
  pall = MPO(nlen)
  indexr=1
  try:
    for k in range(nlen):
      tmptensor = mpofile[str(k)]
      pall.nodes.append(tmptensor)
      if (not len(np.shape(tmptensor)) in [3,4]):
        print("Unexpected tensor: " + filename)
        sys.exit()
      if (np.shape(tmptensor)[0] != indexr):
        print("Mismatching ndim: " + filename)
        sys.exit()
      if (len(np.shape(tmptensor)) == 4):
        if (np.shape(tmptensor)[1] != np.shape(tmptensor)[2]):
          print("Mismatching ndim: " + filename)
          sys.exit()
      indexr = np.shape(tmptensor)[-1]
      pall.nb[k] = np.shape(tmptensor)[1]
  except (KeyError, IndexError):
    print("Unexpected tensor: " + filename)
    sys.exit()
  if (indexr != 1):
    print("Mismatching ndim: " + filename)
    sys.exit()
  return pall

def read_mps_file(pall_file):
  filename = pall_file + "/" + pall_file+"_mps.npz"
  try:
    mpsfile = np.load(filename)
  except FileNotFoundError:
    print("Did not find MPS file: " + filename)
    sys.exit()
  except ValueError:
    print("Unexpected tensor: " + filename)
    sys.exit()
  nlen = len(mpsfile)
  rout = MPS(nlen)
  indexr = 1
  try:
    for k in range(nlen):
      tmptensor = mpsfile[str(k)]
      rout.nodes.append(tmptensor)
      if (len(np.shape(tmptensor)) != 3):
        print("Unexpected tensor: " + filename)
        sys.exit()
      if (np.shape(tmptensor)[0] != indexr):
        print("Mismatching ndim: " + filename)
        sys.exit()
      indexr = np.shape(tmptensor)[2]
      rout.nb[k] = np.shape(tmptensor)[1]
  except (KeyError, IndexError):
    print("Unexpected tensor: " + filename)
    sys.exit()
  if (indexr != 1):
    print("Mismatching ndim: " + filename)
    sys.exit()

  print("pall read from file")
  return rout
