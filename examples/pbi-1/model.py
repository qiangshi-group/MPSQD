import numpy as np

au2ev = 27.2113845
au2cm = 219474.63068

hsmode = 10
nstate = 6
nmode = hsmode*nstate


e0 = np.zeros((nstate,nstate),dtype=np.float64)
for i in range(nstate-1):
  e0[i,i+1] = -0.064
  e0[i+1,i] = -0.064
e0 = e0/au2ev

omega = np.zeros(nmode,dtype=np.float64)
for i in range(nstate):
  omega[i*hsmode+0] = 206
  omega[i*hsmode+1] = 211
  omega[i*hsmode+2] = 540
  omega[i*hsmode+3] = 552
  omega[i*hsmode+4] = 751
  omega[i*hsmode+5] = 1325
  omega[i*hsmode+6] = 1371
  omega[i*hsmode+7] = 1469
  omega[i*hsmode+8] = 1570
  omega[i*hsmode+9] = 1628
omega = omega/au2cm

coef1 = np.zeros((nstate,nstate,nmode),dtype=np.float64)
coef2 = np.zeros((nstate,nstate,nmode,nmode),dtype=np.float64)

for i in range(nstate):
  coef1[i,i,i*hsmode+0] = np.sqrt(2*0.197)*omega[0]
  coef1[i,i,i*hsmode+1] = np.sqrt(2*0.215)*omega[1]
  coef1[i,i,i*hsmode+2] = np.sqrt(2*0.019)*omega[2]
  coef1[i,i,i*hsmode+3] = np.sqrt(2*0.037)*omega[3]
  coef1[i,i,i*hsmode+4] = np.sqrt(2*0.033)*omega[4]
  coef1[i,i,i*hsmode+5] = np.sqrt(2*0.010)*omega[5]
  coef1[i,i,i*hsmode+6] = np.sqrt(2*0.208)*omega[6]
  coef1[i,i,i*hsmode+7] = np.sqrt(2*0.042)*omega[7]
  coef1[i,i,i*hsmode+8] = np.sqrt(2*0.083)*omega[8]
  coef1[i,i,i*hsmode+9] = np.sqrt(2*0.039)*omega[9]

