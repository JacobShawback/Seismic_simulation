import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from myPack.constitution import *
from myPack.output_format import Format
Format.params()


ndiv = 100
total_time = 20
period = 3
nseq = total_time * ndiv
amp = 3

x = np.sin(np.linspace(0,total_time/period*2*np.pi,nseq))
x *= np.sin(np.linspace(0,np.pi,nseq))
x *= amp

slip = Slip(k1=1,k2=0.05,dyield=1)
sb = Slip_Bilinear(k=1,alpha=0.05,dyield=1,slip_rate=0.85)
for i in range(nseq):
    slip.sheer(x[i])
    sb.sheer(x[i])
    slip.push()
    sb.push()

fig,ax = plt.subplots(figsize=(5,5))
ax.plot(x,slip.F,label='slip') #label
ax.legend()
ax.grid()
fig.savefig('fig/slip.png')
plt.close(fig)

fig,ax = plt.subplots(figsize=(5,5))
ax.plot(x,sb.F,label='sb') #label
ax.legend()
ax.grid()
fig.savefig('fig/slip_bilinear.png')
plt.close(fig)

def plot_gif(x,y,path='fig/constitution.gif',ylim=None):
    npage = 200
    fig = plt.figure(figsize=(5,5))
    ims = []
    ni = int(np.floor(nseq/npage))
    if ylim is not None:
        plt.ylim(*ylim)
    for i in range(npage):
        start = (i-3)*ni if i-3>0 else 0
        end = i*ni
        im1 = plt.plot(x[:end],y[:end],color='black',lw=1)
        im2 = plt.plot(x[start:end],y[start:end],color='red',lw=2)
        im3 = plt.plot(x[end-1],y[end-1],marker='.',markersize=10,color='red')
        ims.append(im1+im2+im3)
    interval = total_time/npage*1000
    ani = animation.ArtistAnimation(fig,ims,interval=interval)
    ani.save(path)

ylim = np.abs(slip.F).max() * 1.1
ylim = (-ylim,ylim)

# plot_gif(x,slip.F,'fig/slip.gif',ylim)

def plot_gif2(x,ys,fig,ax,path='fig/constitution.gif',ylim=None):
    npage = 200
    ims = []
    ni = int(np.floor(nseq/npage))
    if ylim is not None:
        plt.ylim(*ylim)
    for i in range(npage):
        start = (i-3)*ni if i-3>0 else 0
        end = i*ni
        im = None
        for i in range(len(ax)):
            y = ys[i]
            if im is None:
                im = ax[i].plot(x[:end],y[:end],color='black',lw=1)
            else:
                im += ax[i].plot(x[:end],y[:end],color='black',lw=1)
            im += ax[i].plot(x[start:end],y[start:end],color='red',lw=2)
            im += ax[i].plot(x[end-1],y[end-1],marker='.',markersize=10,color='red')
        ims.append(im)
    interval = total_time/npage*1000
    ani = animation.ArtistAnimation(fig,ims,interval=interval)
    ani.save(path)

ys = sb.F,sb.slip.F,sb.bilinear.F
fig = plt.figure(figsize=(12,12))
ax = [fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(224)]
plot_gif2(x,ys,fig,ax,'fig/slip_bilinear.gif',ylim)