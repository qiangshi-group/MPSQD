import numpy as np
import sys
from ...utils import MPS, MPS2MPO
from .construct_term import getlist
from . import construct_simple as cs
from . import construct_term as ct
def construct_tds(params,construct_type):
  nlevel = params['nlevel']
  if(params['custom_nb']):
    nb = np.append(np.array([params['ndvr']]),params['nb1'])
  else:
    nbv = params['nbv']
    if (nbv <= 0):
      print("Wrong value: nbv")
      sys.exit()
    nb = nbv*np.ones(nlevel+1,dtype=int)
    nb[0] = params['ndvr']
  vmid=[]
  hsys=[]
  xtmp1=[]
  xtmp2=[]
  for k in range(nlevel):
    tmpmat=np.zeros((nb[k+1],nb[k+1]),dtype=np.float64)
    for i in range(nb[k+1]):
      jj = i*nb[k+1]+i
      tmpmat[i,i] = params['omega'][k] * (i+0.5)
    hsys.append(tmpmat.copy())

    tmpmat=np.zeros((nb[k+1],nb[k+1]),dtype=np.float64)
    for i in range(nb[k+1]):
      for j in range(nb[k+1]):
        if (i == j-1):
          tmpmat[i,j] = np.sqrt(0.5 * j)
        if (i == j+1):
          tmpmat[i,j] = np.sqrt(0.5 * i)
    xtmp1.append(tmpmat.copy())

    tmpmat=np.zeros((nb[k+1],nb[k+1]),dtype=np.float64)
    for i in range(nb[k+1]):
      for j in range(nb[k+1]):
        if (i == j):
          tmpmat[i,j] = (0.5+j)
        if (i == j+2):
          tmpmat[i,j] = 0.5*np.sqrt(i*(i-1))
        if (i == j-2):
          tmpmat[i,j] = 0.5*np.sqrt(j*(j-1))
    xtmp2.append(tmpmat.copy())

    tmpmat=np.zeros((nb[k+1]),dtype=np.float64)
    tmpmat[0]=1.0
    vmid.append(tmpmat.copy())

  waveall = MPS(nlevel+1,nb)
  vl = np.zeros((1, nb[0], params['nrtt']),dtype=np.complex128)
  vl[0,params['init_state'],0] = 1.0
  waveall.nodes.append(vl)
  for i in range(nlevel-1):
    vtmp = np.zeros((params['nrtt'], nb[i+1], params['nrtt']),dtype=np.complex128)
    vtmp[0,:,0] = vmid[i]
    waveall.nodes.append(vtmp.copy())
  vr = np.zeros((params['nrtt'], nb[nlevel], 1),dtype=np.complex128)
  vr[0,:,0] = vmid[nlevel-1]
  waveall.nodes.append(vr)

  for k in range(nlevel):
    xtmp1[k]=np.reshape(xtmp1[k],(1,nb[k+1]**2,1),order='F')
    xtmp2[k]=np.reshape(xtmp2[k],(1,nb[k+1]**2,1),order='F')
    hsys[k]=np.reshape(hsys[k],(1,nb[k+1]**2,1),order='F')
  if construct_type == "term":
    cons_func = ct.construct
  elif construct_type == "simple":
    cons_func = cs.construct
  elif construct_type == None:
    if(time_cal(nlevel,nb,params['coef1'],params['coef2'])):  #calculate the time of two different method
      cons_func = ct.construct  #construct the MPO by terms
    else:
      cons_func = cs.construct  #construct the MPO by elements
  pall = cons_func(nlevel,nb,params['e0'],params['coef1'],params['coef2'],hsys,xtmp1,xtmp2,params['small'],params['nrmax'],params['need_trun'])
  return MPS2MPO(pall), waveall

def time_cal(nlevel,nb,coef1,coef2):
  #calculate the time of element-order method
  time1=0
  for i in range(nb[0]):
    for j in range(i,nb[0]):
      for k1 in range(nlevel):
        if (np.abs(coef1[i,j,k1]) > 1e-12 or np.abs(coef2[i,j,k1,k1]) > 1e-12):
          time1+=1
        for k2 in range(k1):
          if (np.abs(coef2[i,j,k2,k1]) > 1e-12):
            time1+=1

  #calculate the time of term-order method
  time2=0
  for k1 in range(nlevel):
    if(np.abs([coef2[i,i,k1,k1] for i in range(nb[0])]).max() > 1e-12):
      time2+=1
    if(np.abs([coef1[i,i,k1] for i in range(nb[0])]).max() > 1e-12):
      time2+=1
    for k2 in range(k1):
      if(np.abs([coef2[i,i,k2,k1] for i in range(nb[0])]).max() > 1e-12):
        time2+=1

  al = {}
  for i in range(nb[0]-1):
    for j in range(i+1,nb[0]):
      al[(i,j)] = 1

  for k in range(nlevel):
    for key in al.keys():
      if (np.abs(coef1[key[0],key[1],k]) > 1e-12):
        al[key] = 0
    while True:
      re,al = getlist(nb[0],al)
      if (len(re) == 0):
        break
      for key in re:
        time2+=1

  for k in range(nlevel):
    for key in al.keys():
      if (np.abs(coef2[key[0],key[1],k,k]) > 1e-12):
        al[key] = 0
    while True:
      re,al = getlist(nb[0],al)
      if (len(re) == 0):
        break
      for key in re:
        time2+=1

  for k1 in range(nlevel):
    for k2 in range(k1):
      for key in al.keys():
        if (np.abs(coef2[key[0],key[1],k2,k1]) > 1e-12):
          al[key] = 0
      while True:
        re,al = getlist(nb[0],al)
        if (len(re) == 0):
          break
        for key in re:
          time2+=1

  return(time1>time2)
