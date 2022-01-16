from numba import jit
import numpy as np
import math

@jit
def add(ntime,nl,nw,nd,nsplit,tsplit,inexp,u,r0,r,tdelay):
    U = np.zeros(ntime)
    u[0] = 0
    for t in range(ntime):
        for i in range(nl):
            for j in range(nw):
                tt = t-tdelay[i,j]
                Uij = u[max(0,tt)]
                # Uij = u[tt]
                for k in range((nd-1)*nsplit):
                    # U[t] += u[t-tdelay[i,j]-k*tsplit]/nsplit
                    tt = t-tdelay[i,j]-k*tsplit
                    Uij += np.exp(-k*inexp)/nsplit/(1-np.exp(-1)) * u[max(0,tt)]
                    # Uij += np.exp(-tk/tau)/nsplit/(1-np.exp(-1)) * u[tt]
                U[t] += r0/r[i,j] * Uij
    return U

def EGF(dt,u,nd,nsplit,r0,r,xi,Vs,Vr,tau):
    nl,nw = r.shape
    ntime = len(u)
    tdelay = np.array(((r-r0)/Vs + xi/Vr)/dt, dtype=np.int64)
    tsplit = math.floor(tau/(nd-1)/nsplit/dt)
    inexp = tsplit*dt/tau
    U = add(ntime,nl,nw,nd,nsplit,tsplit,inexp,u,r0,r,tdelay)
    return U
