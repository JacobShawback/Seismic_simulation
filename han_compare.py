import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from myPack.output_format import Format
Format.params()

a1 = np.loadtxt('data/acc1.dat')
a1 = (a1-a1.mean())/100
a2 = np.loadtxt('data/acc2.dat')
a2 = (a2-a2.mean())
a3 = np.loadtxt('data/acc3.dat')
a3 = (a3-a3.mean())

dt = 0.01
t = np.linspace(0,len(a1)*dt,len(a1))
label = ['K-NET羽曳野','工学的基盤','対象構造物']
acc = [a1,a2,a3]
max_acc = np.array([np.abs(a).max() for a in acc])
margin = 1.1
ylim = (-margin*max_acc.max(),margin*max_acc.max())


for i in range(3):
    fig,ax = plt.subplots(figsize=[6,3])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Acceleration [m/sec^2]', labelpad=4.0)
    ax.plot(t,acc[i],label=f'max: {max_acc[i]:.1f}') #label
    ax.legend()
    ax.set_xlim(20,50)
    ax.set_ylim(*ylim)
    fig.savefig(f'fig/han/time-{i+1}.png')
    plt.close(fig)


fig,ax = plt.subplots()
for i in range(3):
    A = np.fft.fft(acc[i])
    freqlist = np.fft.fftfreq(len(acc[i]),dt)
    ax.plot(freqlist[:int(len(A)/2)],np.abs(A[:int(len(A)/2)]),label=label[i]) #label
    ax.semilogx()
    ax.semilogy()
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Spectrum of Acceleration', labelpad=4.0)
ax.set_xlim(1e-2,1e2)
ax.set_ylim(1e-3,1e4)
ax.legend()
fig.savefig(f'fig/han/freq.png')
plt.close(fig)
