from ast import Pass
from os import getuid
from traceback import print_tb
from numpy.lib.twodim_base import fliplr
from numba import jit,njit
import numpy as np
import math,pickle
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.fftpack import fft,ifft,fftfreq
import parameter,coordinate,output_format
from EGF_fft import EGF

output_format.Format.params()

L,W = parameter.Lm,parameter.Wm
S1,S2 = parameter.Sa1,parameter.Sa2
l1,l2 = np.sqrt(S1),np.sqrt(S2)
N1,N2 = parameter.Na1,parameter.Na2

def add_zeros(acc0,time0,dt0,k=2):
    N = len(acc0)
    n = 1
    while 2**(n-k)<N:
        n += 1
    print(n)
    zeros = np.zeros(2**(n-1)-int(N/2))
    acc0 = np.concatenate([zeros,acc0,zeros])
    t0 = time0[0] - dt0*len(zeros)
    tend = time0[-1] + dt0*len(zeros)
    time0 = np.linspace(t0,tend,len(acc0))
    print(f'length {N} to {len(acc0)} ({len(acc0)/N*100:.0f}% increase)')
    return acc0,time0,N

def delete_zeros(wave,time,N):
    k = int(len(wave)/2)
    n = int(N/2)
    wave = wave[k-n:k+n]
    time = time[k-n:k+n]
    return wave,time

def interpolation(acc0,time0=None,dt0=0.01,div=10):
    n = len(acc0)
    acc = np.zeros((div*n))
    if time0 is None:
        time0 = np.linspace(0,dt0*n,n)
    time = np.linspace(time0[0],time0[-1],div*n)

    for i in range(n-1):
        a,b = acc0[i:i+2]
        for j in range(div):
            acc[div*(i-1)+j] = a+(b-a)*(j-1)/div
    return acc,time,dt0/div

def integration(wave,dt,low=0.2,high=50):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        coef = (0.0+1.0j)*2.0*math.pi*f
        if np.abs(coef) <= 0.0:
            w2[i] = 0.0j
        elif abs(f) < low0:
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

def differentiation(wave,dt,low=0.2):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        coef = (0.0+1.0j)*2.0*math.pi*f
        if abs(coef) <= 0.0:
            w2[i] = 0.0j
        elif abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0) * coef
        elif abs(f) <= high:
            w2[i] = w[i] * coef
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0) * coef
        else:
            w2[i] = 0.0 + 0.0j

    wave_int = np.real(np.fft.ifft(w2))
    return wave_int

def cut_filter(wave,dt,low=0.2):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        if abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0)
        elif abs(f) <= high:
            w2[i] = w[i]
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0)
        else:
            w2[i] = 0.0 + 0.0j

    wave2 = np.real(np.fft.ifft(w2))
    return wave2

def fftUtoAcc(fftU,dt):
    U = np.real(np.fft.ifft(fftU))
    Acc = differentiation(differentiation(U,dt),dt)
    return Acc

def AcctofftU(Acc,dt):
    U = integration(integration(Acc,dt),dt)
    fftU = np.fft.fft(U)
    freq = np.fft.fftfreq(len(Acc),d=dt)
    return fftU,freq


class Dataset:
    def __init__(self,load=False,fname='search_data'):
        if load:
            self.set_tuple(fname)

    def init(self,xy1,xy2,pos1,r1,pos2,r2,hypo,xi1,xi2,
                    n1,nsplit1,n2,nsplit2,r0,tau1,tau2,Vs,Vr
    ):
        self.xy1,self.xy2,self.pos1,self.r1,self.pos2,self.r2,self.hypo,self.xi1,self.xi2 \
            = xy1,xy2,pos1,r1,pos2,r2,hypo,xi1,xi2
        self.n1,self.nsplit1,self.n2,self.nsplit2,self.r0,self.tau1,self.tau2,self.Vs,self.Vr \
            = n1,nsplit1,n2,nsplit2,r0,tau1,tau2,Vs,Vr

    def get_tuple(self):
        item = (self.xy1,self.xy2,self.pos1,self.r1,self.pos2,self.r2,self.hypo,self.xi1,self.xi2,
                self.n1,self.nsplit1,self.n2,self.nsplit2,self.r0,self.tau1,self.tau2,self.Vs,self.Vr)
        return item

    def set_tuple(self,fname='search_data'):
        with open('data/'+fname+'.pickle','rb') as f:
            item = pickle.load(f)
        (self.xy1,self.xy2,self.pos1,self.r1,self.pos2,self.r2,self.hypo,self.xi1,self.xi2,
            self.n1,self.nsplit1,self.n2,self.nsplit2,self.r0,self.tau1,self.tau2,self.Vs,self.Vr) = item

    def save(self,fname='search_data'):
        with open('data/'+fname+'.pickle','wb') as f:
            pickle.dump(self.get_tuple(),f)

    def egf(self,acc,time,dt):
        d = self
        u,freq = AcctofftU(acc,dt)
        U1 = EGF(u,freq,d.n1,d.nsplit1,d.r0,d.r1,d.xi1,d.Vs,d.Vr,d.tau1)
        U2 = EGF(u,freq,d.n2,d.nsplit2,d.r0,d.r2,d.xi2,d.Vs,d.Vr,d.tau2)
        U = U1+U2
        return u,U,freq



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

def output(acc0,u,U,freq,xy1,xy2,xy_hypo,time0,dt=0.01,n_add=None,fname=''):
    n = len(acc0)

    Acc = fftUtoAcc(U,dt)
    Acc = cut_filter(Acc,dt,low=0.5)
    if n_add is None:
        cut_acc,cut_time = Acc,time0
    else:
        cut_acc,cut_time = delete_zeros(Acc,time0,n_add)
    np.savetxt(f'data/{fname}_egf_acc.txt',np.stack([cut_time,cut_acc],axis=1))

    fig,ax = plt.subplots()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('acceleration [m/s^2]', labelpad=4.0)
    ax.plot(time0,acc0) #label
    # ax.set_xlim(0,120)
    fig.savefig('fig/acc_check0.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('acceleration [m/s^2]', labelpad=4.0)
    ax.plot(cut_time,cut_acc) #label
    fig.savefig('fig/acc_check.png')
    plt.close(fig)

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
    fig.savefig('fig/position.png')
    plt.close(fig)

    xlim = [5e-1,1e2]
    ylim = [1e-2,1e6]
    nn = int(n/2-1)
    freq = freq[:nn]
    u,U = np.abs(u[:nn]),np.abs(U[:nn])

    Acc0 = fftUtoAcc(AcctofftU(acc0,dt)[0],dt)
    plt.figure(figsize=[5,5])
    plt.plot(freq,np.abs(fft(Acc)[:nn]),label='after')
    plt.plot(freq,np.abs(fft(Acc0)[:nn]),label='before')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier spectrum of acceleration')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f'fig/{fname}acc_spectrum.png',bbox_inches="tight",pad_inches=0.05)

    plt.figure(figsize=[5,5])
    plt.plot(freq,U,label='after')
    plt.plot(freq,u,label='before')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier spectrum of displacement')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f'fig/{fname}u_spectrum.png',bbox_inches="tight",pad_inches=0.05)

    plt.figure(figsize=[5,5])
    plt.plot(freq,U/u)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Ratio of displacement fourier spectrum')
    plt.xlim(xlim)
    # plt.ylim([1e-4,1e4])
    plt.savefig(f'fig/{fname}ratio_spectrum.png',bbox_inches="tight",pad_inches=0.05)

def search(acc0,time0,dt=0.01,n=50,div=10):
    acc0,time0,n_add = add_zeros(acc0,time0,dt)
    # u,U: freq domain
    u,freq = AcctofftU(acc0,dt)

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
            for k in range(n):
                for l in range(n):
                    x2 = xmin+k*dx
                    y2 = ymin+l*dy
                    if not check2(x1,y1,x2,y2):
                        continue
                    pos2,r2 = c.mesh_pos(x2,y2,l2,N2,obs)
                    xy_hypo,xi1,xi2 = c.mesh_hypo(pos1,pos2,(x1,y1),(x2,y2),l1,l2)
                    U1 = EGF(u,freq,N1,nsplit1,r0,r1,xi1,Vs,Vr,tau1)
                    U2 = EGF(u,freq,N2,nsplit2,r0,r2,xi2,Vs,Vr,tau2)
                    U = U1+U2
                    Acc = fftUtoAcc(U,dt)
                    Sdijkl = ResponseSpectrum(Acc,dt,T=0.38)
                    if Sdijkl > Sd:
                        Sd = Sdijkl
                        d = Dataset()
                        d.init((x1,y1),(x2,y2),pos1,r1,pos2,r2,xy_hypo,xi1,xi2,
                        N1,nsplit1,N2,nsplit2,r0,tau1,tau2,Vs,Vr)
            print(f'{(n*i+j+1)/n**2*100:.1f}%')

    print('Search finished')
    if div > 1:
        acc0,time0,dt = interpolation(acc0,time0,dt,div=div)
    u,freq = AcctofftU(acc0,dt)
    U1 = EGF(u,freq,d.n1,d.nsplit1,d.r0,d.r1,d.xi1,d.Vs,d.Vr,d.tau1)
    U2 = EGF(u,freq,d.n2,d.nsplit2,d.r0,d.r2,d.xi2,d.Vs,d.Vr,d.tau2)
    U = U1+U2
    print('EGF finished')
    output(acc0,u,U,freq,d.xy1,d.xy2,d.hypo,time0,dt,n_add)
    print('All finished')
    return d
