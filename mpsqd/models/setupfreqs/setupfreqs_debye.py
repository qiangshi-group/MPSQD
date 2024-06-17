import numpy as np

#calculate the parameters in exponential decomposition of the bath correlation function 
#C(t) = \sum_k {[gval(k) + eye*gval_I(k)]*exp(-wval(k)*t)}
def setupfreqs_debye(nlevel,eta,omega,beta):

    wval = np.zeros(nlevel,dtype=np.complex128)
    gval = np.zeros(nlevel,dtype=np.complex128)
    gval_i = np.zeros(nlevel,dtype=np.complex128)

    gval[0] = eta*omega/2.0/np.tan(0.5*beta*omega)
    gval_i[0] = -eta*omega/2.0
    wval[0] = omega
    for i in range(1,nlevel,1):
        omegatmp = 2.0*np.pi*i/beta
        gval[i] = -2.0*eta*omega/beta*omegatmp/(omega**2-omegatmp**2)
        wval[i] = omegatmp

#this is the dephasing term
    k0 = (eta/beta-gval[0])/omega
    for i in range(2,nlevel+1,1):
        k0 = k0-gval[i-1]/wval[i-1]

#normalization factor used for filtering
    a0coef = 0.0
    normfac = np.empty(nlevel,dtype=np.float128)
    for i in range(nlevel):
        normfac[i] = np.sqrt(abs(gval[i]))
        a0coef += np.real(gval[i])
    normfac = np.sqrt(a0coef)*np.ones(nlevel)
    return wval, gval, gval_i, k0, normfac
