import sys
import numpy as np
from .trun import trun_tensor

#==========================================
# definition of the MPS/MPO
class Tensortrain():
  def __init__(self,dnlevel,nb=None):
    self.nodes = []
    self.length = dnlevel
    if(nb is None):
      self.nb = np.empty(dnlevel,dtype=int)
    else:
      self.nb = nb.copy()
    return

  def ndim(self):
    nindex = len(np.shape(self.nodes[0]))
    ndim = np.empty((nindex,self.length),dtype=int)
    for i in range(self.length):
      ndim[:,i] = np.shape(self.nodes[i])
    return ndim

#------------------------------------------
# copy the TT to a new one...
  def copy(self):
    rout = self.__class__(self.length,self.nb)
    for node in self.nodes:
      rout.nodes.append(node.copy())
    return rout

#==============================================
# mpo class
class MPO(Tensortrain):
  def truncation(self,small=1e-14,nrmax=50):
    return MPS2MPO(trun_tensor(MPO2MPS(self),small,nrmax))

  def print_tensor(self,iflag="rank"):
    print_mpo(self,iflag)
    return

#=============================================
# mps class
class MPS(Tensortrain):
  def truncation(self,small=1e-14,nrmax=50):
    return trun_tensor(self,small,nrmax)

  def calc_pop(self):
    ndvr = self.ndim()[1,0]
    pop = np.zeros((ndvr),dtype=np.float64)
    for i in range(ndvr):
      psi = self.copy()
      tmp = psi.nodes[0][:,i,:].copy()
      psi.nodes[0][:,:,:] = 0.0
      psi.nodes[0][:,i,:] = tmp
      pop[i] = calc_overlap(psi, psi).real
    return pop

  def calc_rho(self):
    return calc_rho(self)

  def print_tensor(self,iflag="rank"):
    print_mps(self,iflag)
    return

def calc_overlap(r1, r2):
# the 1st node
  xx1 = np.conj(r1.nodes[0])
  yy1 = r2.nodes[0]
  rtmp = np.tensordot(xx1,yy1, axes=([0,1],[0,1]))
  for i in range(1,r1.length):
#------------------------------------------------------
    xx1 = np.conj((r1.nodes[i]))
    yy1 = r2.nodes[i]
    rtmp = np.tensordot(rtmp,yy1,axes=([1],[0]))
    rtmp = np.tensordot(xx1,rtmp,axes=([0,1],[0,1]))
  return rtmp[0,0]

def calc_element(r1, r2, ope):
  rtmp = np.ones((1,1,1),dtype=np.complex128)
  for i in range(r1.length):
    xx1 = np.conj(r1.node[i])
    yy1 = r2.nodes[i]
    mat1 = ope.nodes[i]
    rtmp2 = np.tensordot(xx1,rtmp,axes=((0),(0)))
    rtmp3 = np.tensordot(rtmp2,mat1,axes=((0,2),(1,0)))
    rtmp = np.tensordot(rtmp3,yy1,axes=((1,2),(0,1)))
  return rtmp[0,0,0]

def calc_rho(rin):
# the first two nodes
  rho = np.dot(rin.nodes[0][0,:,:],rin.nodes[1][:,0,:])
#------------------------------------------------------
# intermediate ones
  for i in range(2,rin.length-1):
    rho = np.dot(rho,rin.nodes[i][:,0,:])
#------------------------------------------------------
# the right node
  rho = np.dot(rho, rin.nodes[rin.length-1][:,:,0])
  return rho

def MPO2MPS(pall):
  rout = MPS(pall.length,pall.nb)
  for inode in pall.nodes:
    if(len(inode.shape)==4):
      ra1, rmid1, rmid2, ra2 = inode.shape
      vtmp = np.reshape(inode,(ra1,rmid1*rmid2,ra2),order='F')
    else:
      vtmp = inode
    rout.nodes.append(vtmp)
  return rout

def MPS2MPO(rin):
  pall = MPO(rin.length,rin.nb)
  for inode in rin.nodes:
    ra1, rmid0, ra2 = inode.shape
    vtmp = np.reshape(inode,(ra1,int(np.sqrt(rmid0)),int(np.sqrt(rmid0)),ra2),order='F')
    pall.nodes.append(vtmp)
  return pall

def print_mps(rin,iflag="rank"):
  nlen = rin.length
  print("========================================")
  print("nlen=",nlen)
  for i in range(nlen):
    print("i=",i)
    print("shape of nodes", rin.nodes[i].shape)
  if (iflag == "rank"):
    print("========================================")
    return
  elif (iflag == "data"):
    print("data for the tensor")
    for i in range(nlen):
      print("----------- node:",i,"-----------------------------")
      for j in range(rin.nodes[i].shape[1]):
        print(rin.nodes[i][:,j,:])
      print("--------end node:",i,"-----------------------------")
  return

def print_mpo(rin,iflag="rank"):
  nlen = rin.length
  print("========================================")
  print("nlen=",nlen)
  for i in range(nlen):
    print("i=",i)
    print("shape of nodes", rin.nodes[i].shape)
  if (iflag == "rank"):
    print("========================================")
    return
  elif (iflag == "data"):
    print("data for the tensor")
    for i in range(nlen):
      print("----------- node:",i,"-----------------------------")
      if(len(inode.shape)==4):
        for j in range(rin.nodes[i].shape[1]):
          for k in range(rin.nodes[i].shape[2]):
            print(rin.nodes[i][:,j,k,:])
      else:
        for j in range(rin.nodes[i].shape[1]):
          print(rin.nodes[i][:,j,:])
      print("--------end node:",i,"-----------------------------")
  return
