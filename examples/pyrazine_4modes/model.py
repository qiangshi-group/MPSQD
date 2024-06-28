# this is the 4-mode model of pyrazine....
import numpy as np

# number of modes...
nmode = 4

nstate = 2

# physical parameters, couplings etc...
au2ev = 27.2113845
au2fs = 2.418884254e-2

delta = 0.8460/2.0/au2ev
e0 = np.zeros((nstate,nstate),dtype=np.float64)
e0[0,0] = -delta
e0[1,1] = delta

# the 4 mode model used 6a, 1, 9a, 10a
#=============================================
# the omega terms
omega = np.zeros(4,dtype=np.float64)
# B1g
omega[0] = 0.1139  # 10a 0.1139 0.1133 0.1285 # w10a  = 0.1139  ,ev
# Ag terms                                    
omega[1] = 0.0739  # 6a 0.0739 0.0737 0.0808  # w6a   = 0.0739  ,ev
omega[2] = 0.1258  # 1  0.1258 0.1269 0.1387  # w1    = 0.1258  ,ev
omega[3] = 0.1525  # 9a 0.1525 0.1564 0.1673  # w9a   = 0.1525  ,ev
# this is the unit transform
omega = omega/au2ev
#=================================================
# A and B terms 
#acoef = np.zeros(4,dtype=np.float64)
#bcoef = np.zeros(4,dtype=np.float64)
#ccoef = np.zeros(4,dtype=np.float64)
coef1 = np.zeros((nstate,nstate,nmode),dtype=np.float64)
coef2 = np.zeros((nstate,nstate,nmode,nmode),dtype=np.float64)

# 6a 1 9a 
coef1[0,0,1] = 0.09806   #kl_6a_s1  =   0.09806 , ev
coef1[0,0,2] = 0.05033   #kl_1_s1  =   0.05033 , ev
coef1[0,0,3] = 0.14521   #kl_9a_s1  =   0.14521 , ev

coef1[1,1,1] = -0.13535  #kl_6a_s2  =  -0.13545 , ev
coef1[1,1,2] = 0.171     #kl_1_s2  =   0.17100 , ev
coef1[1,1,3] = 0.03746   #kl_9a_s2  =   0.03746 , ev
                     
#acoef = acoef/au2ev #!*au2a
#bcoef = bcoef/au2ev #!*au2a
#================================================
# the first order C terms, only for 10a 
                             # linear, off-diagonal coupling coefficients
coef1[0,1,0] = 0.20804           #lambda =   0.20804 , ev
#ccoef = ccoef/au2ev 
coef1 = coef1/au2ev

#=================================================
# second order A and B terms
#a2coef = np.zeros((4,4),dtype=np.float64)
#b2coef = np.zeros((4,4),dtype=np.float64)

                               # quadratic, on-diagonal coupling coefficients
                               # H(1,1)
coef2[0,0,0,0] = -0.01159         # kq_10a_s1  =  -0.01159 , ev
coef2[0,0,1,1] = 0.0000           # kq_6a_s1  =   0.00000 , ev
coef2[0,0,2,2] = 0.00             # kq_1_s1  =   0.00000 , ev
coef2[0,0,3,3] = 0.00             # kq_9a_s1  =  0.00000 , ev
                                       # H(2,2)
coef2[1,1,0,0] = -0.01159         # kq_10a_s2  =  -0.01159 , ev
coef2[1,1,1,1] = 0.00             # kq_6a_s2  =  0.00000 , ev
coef2[1,1,2,2] = 0.00             # kq_1_s2  =   0.00000 , ev
coef2[1,1,3,3] = 0.00             # kq_9a_s2  =   0.00000 , ev

                             # bilinear, on-diagonal coupling coefficients
                             # H(1,1)
coef2[0,0,1,2] =  0.00108       # kb_6ax1_s1  =   0.00108 , ev
coef2[0,0,2,3] = -0.00474       # kb_1x9a_s1  =  -0.00474 , ev
coef2[0,0,1,3] =  0.00204       # kb_6ax9a_s1  =   0.00204 , ev
                             # H(2,2)
coef2[1,1,1,2] = -0.00298        # kb_6ax1_s2  =  -0.00298 , ev
coef2[1,1,2,3] = -0.00155        # kb_1x9a_s2  =  -0.00155 , ev
coef2[1,1,1,3] =  0.00189       # kb_6ax9a_s2  =   0.00189 , ev

#a2coef = a2coef/au2ev  #!*au2a**2
#b2coef = b2coef/au2ev  #!*au2a**2

#==================================================
# the C terms
#c2coef = np.zeros((4,4),dtype=np.float64)
                          # bilinear, off-diagonal coupling coefficients
                          # H(1,2) and H(2,1)
coef2[0,1,0,2] =  0.00553     #kb_1x10a  =   0.00553 , ev
coef2[0,1,0,1] =  0.01000    #kb_6ax10a  =  0.01000 , ev
coef2[0,1,0,3] =  0.00126     #kb_9ax10a  =   0.00126 , ev

#c2coef = c2coef/au2ev #*au2a**2

coef2 = coef2/au2ev
