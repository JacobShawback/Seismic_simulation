from numba import jit,njit
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import parameter,coordinate,EGF

L,W = parameter.Lm,parameter.Wm
S1,S2 = parameter.Sa1,parameter.Sa2
l1,l2 = np.sqrt(S1),np.sqrt(S2)
N1,N2 = parameter.Na1,parameter.Na2

@jit
def newmark(acc0,Dis,Vel,Acc,beta,dt,m,c,k,N):
    a1,a2 = 0.0,0.0
    for i in range(1,N):
        a1 = m + 0.5*dt*c + beta*dt**2*k
        a2 = -m*acc0[i] - c*(Vel[i-1]+0.5*dt*Acc[i-1]) - k*(Dis[i-1]+dt*Vel[i-1]+(0.5-beta)*dt**2*Acc[i-1])
        Acc[i] = a2/a1
        Vel[i] = Vel[i-1] + dt*(Acc[i-1]+Acc[i])/2
        Dis[i] = Dis[i-1] + dt*Vel[i-1] + (0.5-beta)*dt**2*Acc[i-1] + beta*dt**2*Acc[i]
    return Dis,Vel,Acc

def ResponseSpectrum(acc,dt=0.01,h=0.02,T=0.38):
    m = 1
    N = len(acc)
    beta = 1/6
    Dis = np.zeros(N)
    Vel = np.zeros(N)
    Acc = np.zeros(N)
    w = 2 * np.pi / T
    k = m * w**2
    c = 2 * m * w * h
    Dis,Vel,Acc = newmark(acc,Dis,Vel,Acc,beta,dt,m,c,k,N)
    # Zacc = Acc + acc
    MaxDis = np.max(np.abs(Dis))
    return MaxDis

# XY1はアスペリティ1の原点側のコーナーの座標
def check1(x,y,l):
    return x+l<L and y+l<W

def check2(x1,y1,x2,y2):
    c0 = check1(x2,y2,l2)
    c1 = x2-x1<-l2 or l1<x2-x1
    c2 = y2-y1<-l2 or l1<y2-y1
    return c0 and c1 and c2

def search(acc0,dt=0.01,nmax=50):
    n_acc = len(acc0)
    u,v,a = np.zeros(n_acc),np.zeros(n_acc),np.zeros(n_acc)
    u,_,_ = newmark(-acc0,u,v,a,1/6,dt,1,0,0,n_acc)

    n = nmax
    xmin,xmax = 0,L
    ymin,ymax = 0,W
    dx,dy = (xmax-xmin)/n,(ymax-ymin)/n
    c = coordinate.c
    obs = coordinate.pos_obs
    r0 = coordinate.r0
    Vs,Vr = parameter.Vs,parameter.Vr
    tau1,tau2 = parameter.taua1,parameter.taua2
    nsplit1 = math.floor(tau1/(N1-1)/dt)
    nsplit2 = math.floor(tau2/(N2-1)/dt)
    Acc = np.zeros_like(u)
    Sd,best_xy1,best_xy2,best_hypo = 0,(0,0),(0,0),(0,0)
    best_U = 0
    for i in range(n):
        for j in range(n):
            x1 = xmin+i*dx
            y1 = ymin+j*dy
            if not check1(x1,y1,l1):
                continue
            pos1,r1 = c.mesh_pos(x1,y1,l1,N1,obs)
            for k in range(nmax):
                for l in range(nmax):
                    x2 = xmin+k*dx
                    y2 = ymin+l*dy
                    if not check2(x1,y1,x2,y2):
                        continue
                    pos2,r2 = c.mesh_pos(x2,y2,l2,N2,obs)
                    xy_hypo,xi1,xi2 = c.mesh_hypo(pos1,pos2,(x1,y1),(x2,y2),l1,l2)
                    U1 = EGF.EGF(dt,u,N1,nsplit1,r0,r1,xi1,Vs,Vr,tau1)
                    U2 = EGF.EGF(dt,u,N2,nsplit2,r0,r2,xi2,Vs,Vr,tau2)
                    U = U1+U2
                    Acc[1:-1] = (U[2:]-2*U[1:-1]+U[:-2])/dt**2
                    Sdijkl = ResponseSpectrum(Acc,dt,T=0.38)
                    if Sdijkl > Sd:
                        Sd = Sdijkl
                        best_xy1 = x1,y1
                        best_xy2 = x2,y2
                        best_hypo = xy_hypo
                        best_U = U
            print(f'{(n*i+j+1)/n**2*100:.1f}%')
    return best_U,best_xy1,best_xy2,best_hypo

def output(U,xy1,xy2,xy_hypo,time0,dt=0.01):
    Acc = np.zeros_like(U)
    Acc[1:-1] = (U[2:]-2*U[1:-1]+U[:-2])/dt**2
    np.savetxt('data/egf_acc.txt',np.stack([time0,Acc],axis=1))
    np.savetxt('data/U.txt',U)

    fig,ax = plt.subplots()
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]', labelpad=4.0)
    rec0 = pat.Rectangle((0,0),L,W,fill=False,edgecolor='black')
    rec1 = pat.Rectangle(xy1,l1,l1,fill=True,edgecolor='black',facecolor='red',alpha=0.3)
    rec2 = pat.Rectangle(xy2,l2,l2,fill=True,edgecolor='black',facecolor='blue',alpha=0.3)
    ax.plot() #label
    # ax.legend()
    ax.add_patch(rec0)
    ax.add_patch(rec1)
    ax.add_patch(rec2)
    ax.plot(xy_hypo[0],xy_hypo[1],marker='*',markersize=15,color='black')
    ymargin = 0.1*W
    ax.set_ylim(W+ymargin,-ymargin)
    fig.savefig('fig.png')
    plt.close(fig)
