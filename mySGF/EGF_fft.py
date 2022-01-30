from numba import jit,njit
import numpy as np

# @jit
# def EGF(u,f,n,nsplit,r0,r,xi,Vs,Vr,tau):
#     nfreq = len(u)
#     U = np.zeros(nfreq,dtype=np.complex)
#     nk = (n-1)*nsplit
#     tdelay = (r-r0)/Vs + xi/Vr
#     for iomega in range(nfreq):
#         omega = 2*np.pi*f[iomega]
#         for i in range(n):
#             for j in range(n):
#                 ij = np.exp(-1j*omega*tdelay[i,j])
#                 for k in range(nk):
#                     F = np.exp(-k/nk)/nsplit/(1-np.exp(-1))
#                     ij += F*np.exp(-1j*omega*(k*tau/nk))
#                 U[iomega] += r0/r[i,j] * ij * u[iomega]
#     return U

@njit
def add(ones,u,U,f,n,nsplit,r0,r,xi,Vs,Vr,tau):
    nk = (n-1)*nsplit
    tdelay = (r-r0)/Vs + xi/Vr
    omega = 2*np.pi*f
    for i in range(n):
        for j in range(n):
            delay = np.exp(-1j*omega*tdelay[i,j])
            Fij = ones*1.0
            for k in range(nk):
                coef = np.exp(-k/nk)/nsplit/(1-np.exp(-1))
                Fij += coef*np.exp(-1j*omega*(k*tau/nk))
            U += r0/r[i,j] * Fij * delay * u
    return U

def EGF(u,f,n,nsplit,r0,r,xi,Vs,Vr,tau):
    U = np.zeros(len(u),dtype=np.complex)
    ones = np.ones_like(u,dtype=np.complex)
    U = add(ones,u,U,f,n,nsplit,r0,r,xi,Vs,Vr,tau)
    return U
