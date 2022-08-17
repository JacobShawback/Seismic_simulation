import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from myPack.constitution import *
from myPack.output_format import Format
# Format.params()


ndiv = 100
total_time = 20
period = 3
nseq = total_time * ndiv
amp1 = 0.1
amp2 = 0.2

x0 = np.sin(np.linspace(0,total_time/period*2*np.pi,nseq))
x0 *= np.sin(np.linspace(0,np.pi,nseq))
x = x0*amp1
x2 = x0*amp2


sb = Slip_Bilinear2(2000e3,3)
sb2 = Slip_Bilinear2(2000e3,3)
for i in range(nseq):
    sb.shear(x[i])
    sb.push()
    sb2.shear(x2[i])
    sb2.push()

fig,ax = plt.subplots(figsize=(5,5))
ax.plot(x,sb.F,label='sb') #label
# ax.legend()
# ax.grid()
fig.savefig('fig/slip_bilinear.png')
plt.close(fig)

fig,ax = plt.subplots(figsize=(5,5))
ax.plot(x2,sb2.F,label='sb') #label
# ax.legend()
# ax.grid()
fig.savefig('fig/slip_bilinear2.png')
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


def plot_gif2(x,ys,fig,ax,path='fig/constitution.gif',ylim=None):
    npage = 100
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

# ylim = np.abs(sb.F).max() * 1.1
# ylim = (-ylim,ylim)

# ys = sb.F,sb.slip.F,sb.bilinear.F
# fig = plt.figure(figsize=(12,12))
# ax = [fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(224)]
# plot_gif2(x,ys,fig,ax,'fig/slip_bilinear.gif',ylim)

# ylim = np.abs(sb2.F).max() * 1.1
# ylim = (-ylim,ylim)

# ys = sb2.F,sb2.slip.F,sb2.bilinear.F
# fig = plt.figure(figsize=(12,12))
# ax = [fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(224)]
# plot_gif2(x2,ys,fig,ax,'fig/slip_bilinear2.gif',ylim)