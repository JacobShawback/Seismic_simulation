from os import getuid
from numpy.lib.twodim_base import fliplr
from numba import jit,njit
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.fftpack import fft,ifft,fftfreq
import parameter,coordinate,EGF,output_format

output_format.Format.params()

L,W = parameter.Lm,parameter.Wm
S1,S2 = parameter.Sa1,parameter.Sa2
l1,l2 = np.sqrt(S1),np.sqrt(S2)
N1,N2 = parameter.Na1,parameter.Na2

class Dataset:
    def __init__(self,xy1,xy2,pos1,r1,pos2,r2,hypo,xi1,xi2,u,U1,U2):
        self.xy1,self.xy2,self.pos1,self.r1,self.pos2,self.r2,self.hypo,self.xi1,self.xi2,self.u,self.U1,self.U2 \
            = xy1,xy2,pos1,r1,pos2,r2,hypo,xi1,xi2,u,U1,U2
        self.U = U1+U2

def integration(wave,dt,low=0.2,high=50):
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        coef = (0.0+1.0j)*2.0*math.pi*f

        if abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0) / coef
        elif abs(f) <= high:
            w2[i] = w[i] / coef
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0) / coef
        else:
            w2[i] = 0.0 + 0.0j

    wave_int = np.real(np.fft.ifft(w2))
    return wave_int

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

def cut(acc,dt=0.01):
    freq = fftfreq(len(acc),d=dt)
    fftacc = fft(acc)
    fs = 0.5
    fftacc[np.abs(freq)<fs] = 0
    acc = np.real(ifft(fftacc))
    return acc

def output(acc0,u,U,xy1,xy2,xy_hypo,time0,dt=0.01,fname=''):
    n = len(acc0)

    Acc = np.zeros_like(U)
    Acc[1:-1] = (U[2:]-2*U[1:-1]+U[:-2])/dt**2
    cutAcc = cut(Acc,dt)
    np.savetxt(f'data/{fname}_egf_acc.txt',np.stack([time0,cutAcc],axis=1))

    fig,ax = plt.subplots()
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]', labelpad=4.0)
    rec0 = pat.Rectangle((0,0),L,W,fill=False,edgecolor='black')
    rec1 = pat.Rectangle(xy1,l1,l1,fill=True,edgecolor='black',facecolor='red',alpha=0.3)
    rec2 = pat.Rectangle(xy2,l2,l2,fill=True,edgecolor='black',facecolor='blue',alpha=0.3)
    ax.plot() #label
    ax.add_patch(rec0)
    ax.add_patch(rec1)
    ax.add_patch(rec2)
    ax.plot(xy_hypo[0],xy_hypo[1],marker='*',markersize=15,color='black')
    ymargin = 0.1*W
    ax.set_ylim(W+ymargin,-ymargin)
    fig.savefig('fig.png')
    plt.close(fig)

    freq = fftfreq(n,dt)
    fftu = np.abs(fft(u)[:int(n / 2 - 1)])
    fftU = np.abs(fft(U)[:int(n / 2 - 1)])
    ffta = np.abs(fft(acc0)[:int(n / 2 - 1)])
    fftA = np.abs(fft(Acc)[:int(n / 2 - 1)])

    plt.figure(figsize=[5,5])
    plt.plot(freq[:int(n/2-1)],fftU,label='after')
    plt.plot(freq[:int(n/2-1)],fftu,label='before')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier spectrum of displacement')
    plt.xlim([2e-1,1e2])
    # plt.ylim([1e-4,1e4])
    # plt.grid()
    plt.legend()
    plt.savefig(f'fig/{fname}u_spectrum.png',bbox_inches="tight",pad_inches=0.05)

    plt.figure(figsize=[5,5])
    plt.plot(freq[:int(n/2-1)],fftA,label='after')
    plt.plot(freq[:int(n/2-1)],ffta,label='before')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier spectrum of acceleration')
    plt.xlim([2e-1,1e2])
    # plt.ylim([1e-4,1e4])
    # plt.grid()
    plt.legend()
    plt.savefig(f'fig/{fname}acc_spectrum.png',bbox_inches="tight",pad_inches=0.05)

    plt.figure(figsize=[5,5])
    plt.plot(freq[:int(n/2-1)],fftU/fftu)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Ratio of displacement fourier spectrum')
    plt.xlim([2e-1,1e2])
    # plt.ylim([1e-4,1e4])
    # plt.grid()
    plt.savefig(f'fig/{fname}ratio_spectrum.png',bbox_inches="tight",pad_inches=0.05)

def search(acc0,time0,dt=0.01,nmax=50):
    v = integration(acc0,dt)
    u = integration(v,dt)

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
    Sd = 0
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
                        d = Dataset((x1,y1),(x2,y2),pos1,r1,pos2,r2,xy_hypo,xi1,xi2,u,U1,U2)
            print(f'{(n*i+j+1)/n**2*100:.1f}%')
    output(acc0,u,d.U,d.xy1,d.xy2,d.hypo,time0,dt)
    output(acc0,u,d.U1,d.xy1,d.xy2,d.hypo,time0,dt,'U1')
    output(acc0,u,d.U2,d.xy1,d.xy2,d.hypo,time0,dt,'U2')
    return d
