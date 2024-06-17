# this is the 24-mode model of pyrazine....
# ALL energy units in eV...
import numpy as np

# number of modes...
nmode = 24

nstate = 2

# physical parameters, couplings etc...
au2ev = 27.2113845
au2fs = 2.418884254e-2

# the energy offsets...
delta = 0.8460/2.0/au2ev
#=============================================
# the omega terms
omega = np.zeros(nmode,dtype=np.float64)

# B1g
omega[0] = 0.1139      # 10a
# Ag terms
omega[1] = 0.0739      # 6a
omega[2] = 0.1258      # 1
omega[3] = 0.1525      # 9a
omega[4] = 0.1961      # 8a
omega[5] = 0.3788      # 2
# B2g
omega[6] = 0.0937      # 4
omega[7] = 0.1219      # 5
# B3g
omega[8] = 0.0873      # 6b
omega[9] = 0.1669      # 3
omega[10] = 0.1891      # 8b
omega[11] = 0.3769      # 7b
# Au
omega[12] = 0.0423      # 16a
omega[13] = 0.1190      # 17a
# B1u
omega[14] = 0.1266      # 12
omega[15] = 0.1408      # 18a
omega[16] = 0.1840      # 19a
omega[17] = 0.3734      # 13
# B2u
omega[18] = 0.1318      # 18b
omega[19] = 0.1425      # 14
omega[20] = 0.1756      # 19b
omega[21] = 0.3798      # 20b
# B3u
omega[22] = 0.0521      # 16b
omega[23] = 0.0973      # 11

omega = omega/au2ev
#=================================================
# A, B, and C terms 
acoef = np.zeros(nmode,dtype=np.float64)
bcoef = np.zeros(nmode,dtype=np.float64)
ccoef = np.zeros(nmode,dtype=np.float64)
# A for s1 and B for s2
# linear, on-diagonal coupling coefficients
# H(1,1)
acoef[1] = 0.09806      # 6a
acoef[2] = 0.05033      # 1
acoef[3] = 0.14521      # 9a
acoef[4] = -0.04448      # 8a
acoef[5] = -0.02473      # 2
# H(2,2)
bcoef[1] = -0.13545      # 6a
bcoef[2] = 0.17100      # 1
bcoef[3] = 0.03746      # 9a
bcoef[4] = 0.01677      # 8a
bcoef[5] = -0.01619      # 2

# the first order C terms, only for 10a
ccoef[0] = 0.20804   #10a

acoef = acoef/au2ev
bcoef = bcoef/au2ev 
ccoef = ccoef/au2ev 

#=================================================
# the second order A, B, C terms
# ?? some are set to zero intentioanlly ????
a2coef = np.zeros((nmode,nmode),dtype=np.float64)
b2coef = np.zeros((nmode,nmode),dtype=np.float64)
c2coef = np.zeros((nmode,nmode),dtype=np.float64)

# quadratic, on-diagonal coupling coefficients
# H(1,1)
a2coef[0,0] = -0.01159      # 10a
a2coef[1,1] = 0.00000      # 6a
a2coef[2,2] = 0.00000      # 1
a2coef[3,3] = 0.00000      # 9a
a2coef[4,4] = 0.00000      # 8a
a2coef[5,5] = 0.00000      # 2
a2coef[6,6] = -0.02252      # 4
a2coef[7,7] = -0.01825      # 5
a2coef[8,8] = -0.00741      # 6b
a2coef[9,9] = 0.05183      # 3
a2coef[10,10] = -0.05733      # 8b
a2coef[11,11] = -0.00333      # 7b
a2coef[12,12] = 0.01145      # 16a
a2coef[13,13] = -0.02040      # 17a
a2coef[14,14] = -0.04819      # 12
a2coef[15,15] = -0.00792      # 18a
a2coef[16,16] = -0.02429      # 19a
a2coef[17,17] = -0.00492      # 13
a2coef[18,18] = -0.00277      # 18b
a2coef[19,19] = 0.03924      # 14
a2coef[20,20] = 0.00992      # 19b
a2coef[21,21] = -0.00110      # 20b
a2coef[22,22] = -0.02176      # 16b
a2coef[23,23] = 0.00315      # 11
# H(2,2)
b2coef[0,0] = -0.01159      # 10a
b2coef[1,1] = 0.00000      # 6a
b2coef[2,2] = 0.00000      # 1
b2coef[3,3] = 0.00000      # 9a
b2coef[4,4] = 0.00000      # 8a
b2coef[5,5] = 0.00000      # 2
b2coef[6,6] = -0.03445      # 4
b2coef[7,7] = -0.00265      # 5
b2coef[8,8] = -0.00385      # 6b
b2coef[9,9] = 0.04842      # 3
b2coef[10,10] = -0.06332      # 8b
b2coef[11,11] = -0.00040      # 7b
b2coef[12,12] = -0.01459      # 16a
b2coef[13,13] = -0.00618      # 17a
b2coef[14,14] = -0.00840      # 12
b2coef[15,15] = 0.00429      # 18a
b2coef[16,16] = -0.00734      # 19a
b2coef[17,17] = 0.00062      # 13
b2coef[18,18] = -0.01179      # 18b
b2coef[19,19] = 0.04000      # 14
b2coef[20,20] = 0.01246      # 19b
b2coef[21,21] = 0.00069      # 20b
b2coef[22,22] = -0.02214      # 16b
b2coef[23,23] = -0.00496      # 11

# bilinear, on-diagonal coupling coefficients
# H(1,1)
a2coef[2,5] = -0.00163      # 1, 2
a2coef[1,2] = 0.00108      # 6a, 1
a2coef[2,4] = -0.00154      # 1, 8a
a2coef[2,3] = -0.00474      # 1, 9a
a2coef[1,5] = -0.00285      # 6a, 2
a2coef[4,5] = 0.00143      # 8a, 2
a2coef[3,5] = 0.00474      # 9a, 2
a2coef[1,4] = 0.00135      # 6a, 8a
a2coef[1,3] = 0.00204      # 6a, 9a
a2coef[3,4] = 0.00872      # 9a, 8a
a2coef[12,13] = 0.00100      # 16a, 17a
a2coef[6,7] = -0.00049      # 4, 5
a2coef[8,9] = 0.01321      # 6b, 3
a2coef[8,10] = -0.00717      # 6b, 8b
a2coef[8,11] = 0.00515      # 6b, 7b
a2coef[9,10] = -0.03942      # 3, 8b
a2coef[9,11] = 0.00170      # 3, 7b
a2coef[10,11] = -0.00204      # 8b, 7b
a2coef[14,15] = 0.00525      # 12, 18a
a2coef[14,16] = -0.00485      # 12, 19a
a2coef[14,17] = -0.00326      # 12, 13
a2coef[15,16] = 0.00852      # 18a, 19a
a2coef[15,17] = 0.00888      # 18a, 13
a2coef[16,17] = -0.00443      # 19a, 13
a2coef[18,19] = 0.00016      # 18b, 14
a2coef[18,20] = -0.00250      # 18b, 19b
a2coef[18,21] = 0.00357      # 18b, 20b
a2coef[19,20] = -0.00197      # 14, 19b
a2coef[19,21] = -0.00355      # 14, 20b
a2coef[20,21] = 0.00623      # 19b, 20b
a2coef[22,23] = -0.00624      # 16b, 11
# H(2,2)
b2coef[2,5] = -0.00600      # 1, 2
b2coef[1,2] = -0.00298      # 6a, 1
b2coef[2,4] = -0.00311      # 1, 8a
b2coef[2,3] = -0.00155      # 1, 9a
b2coef[1,5] = -0.00128      # 6a, 2
b2coef[4,5] = 0.00713      # 8a, 2
b2coef[3,5] = 0.00334      # 9a, 2
b2coef[1,4] = 0.00203      # 6a, 8a
b2coef[1,3] = 0.00189      # 6a, 9a
b2coef[3,4] = 0.01194      # 9a, 8a
b2coef[12,13] = -0.00091      # 16a, 17a
b2coef[6,7] = 0.00911      # 4, 5
b2coef[8,9] = -0.00661      # 6b, 3
b2coef[8,10] = 0.00429      # 6b, 8b
b2coef[8,11] = -0.00246      # 6b, 7b
b2coef[9,10] = -0.03034      # 3, 8b
b2coef[9,11] = -0.00185      # 3, 7b
b2coef[10,11] = -0.00388      # 8b, 7b
b2coef[14,15] = 0.00536      # 12, 18a
b2coef[14,16] = -0.00097      # 12, 19a
b2coef[14,17] = 0.00034      # 12, 13
b2coef[15,16] = 0.00209      # 18a, 19a
b2coef[15,17] = -0.00049      # 18a, 13
b2coef[16,17] = 0.00346      # 19a, 13
b2coef[18,19] = -0.00884      # 18b, 14
b2coef[18,20] = 0.07000      # 18b, 19b
b2coef[18,21] = -0.01249      # 18b, 20b
b2coef[19,20] = -0.05000      # 14, 19b
b2coef[19,21] = 0.00265      # 14, 20b
b2coef[20,21] = -0.00422      # 19b, 20b
b2coef[22,23] = -0.00261      # 16b, 11

# bilinear, off-diagonal coupling coefficients
# H(1,2) and H(2,1)
c2coef[0,2] = 0.00553      # 10a, 1
c2coef[0,5] = 0.00514      # 10a, 2
c2coef[0,1] = 0.01000      # 10a, 6a
c2coef[0,4] = 0.00799      # 10a, 8a
c2coef[0,3] = 0.00126      # 10a, 9a
c2coef[6,9] = -0.00466      # 3, 4
c2coef[6,8] = -0.01372      # 6b, 4
c2coef[6,11] = -0.00031      # 7b, 4
c2coef[6,10] = 0.00329      # 8b, 4
c2coef[7,9] = -0.00914      # 3, 5
c2coef[7,8] = 0.00598      # 6b, 5
c2coef[7,11] = 0.00500      # 7b, 5
c2coef[7,10] = 0.00961      # 8b, 5
c2coef[12,14] = -0.01056      # 12, 16a
c2coef[12,17] = -0.00226      # 13, 16a
c2coef[12,15] = 0.00559      # 18a, 16a
c2coef[12,16] = 0.00401      # 19a, 16a
c2coef[13,14] = -0.01200      # 12, 17a
c2coef[13,17] = -0.00396      # 13, 17a
c2coef[13,15] = -0.00213      # 18a, 17a
c2coef[13,16] = 0.00328      # 19a, 17a
c2coef[19,23] = -0.01780      # 14, 11
c2coef[18,23] = 0.01281      # 18b, 11
c2coef[20,23] = 0.00134      # 19b, 11
c2coef[21,23] = -0.00481      # 20b, 11
c2coef[19,22] = -0.00009      # 14, 16b
c2coef[18,22] = 0.00118      # 18b, 16b
c2coef[20,22] = -0.00285      # 19b, 16b
c2coef[21,22] = -0.00095      # 20b, 16b

#===================================
a2coef = a2coef/au2ev
b2coef = b2coef/au2ev
c2coef = c2coef/au2ev

# primary basis size 
nb1 = np.zeros(nmode,dtype=int)
nb1[0] = 40
nb1[1] = 32
nb1[2] = 20
nb1[3] = 12
nb1[4] = 8
nb1[5] = 4
nb1[6] = 24
nb1[7] = 8
nb1[8] = 8
nb1[9] = 8
nb1[10] = 24
nb1[11] = 4
nb1[12] = 24
nb1[13] = 6
nb1[14] = 20
nb1[15] = 6
nb1[16] = 6
nb1[17] = 4
nb1[18] = 80
nb1[19] = 20
nb1[20] = 72
nb1[21] = 6
nb1[22] = 32
nb1[23] = 6

