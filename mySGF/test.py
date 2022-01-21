from distutils.archive_util import make_archive
from numpy.lib.twodim_base import fliplr
from numba import jit,njit
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.fftpack import fft,ifft,fftfreq
import parameter,coordinate,EGF,search

input_csv = np.loadtxt('data/acc02.txt')
acc0 = input_csv[:,1]
time0 = input_csv[:,0]
dt = 0.01
n = len(acc0)

# acc0 = np.sin(np.linspace(0,10*np.pi,n)) + np.sin(np.linspace(0,55*np.pi,n)) + np.sin(np.linspace(0,100*np.pi,n))
# acc0 -= acc0.mean()

u,v,a = np.zeros(n),np.zeros(n),np.zeros(n)
v = search.integration(acc0,dt)
u = search.integration(v,dt)

fig,ax = plt.subplots()
ax.plot(u) #label
# ax.legend()
# ax.semilogx()
plt.show()
plt.close(fig)

freq = fftfreq(n,dt)
fftu = np.abs(fft(u)[:int(n / 2 - 1)])
fftv = np.abs(fft(v)[:int(n / 2 - 1)])
ffta = np.abs(fft(acc0)[:int(n / 2 - 1)])

# plt.figure(figsize=[5,5])
# plt.plot(freq[:int(n/2-1)],fftu,label='before')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('fig/aaa.png',bbox_inches="tight",pad_inches=0.05)

plt.figure(figsize=[5,5])
plt.plot(freq[:int(n/2-1)],ffta,label='before')
plt.xscale('log')
plt.yscale('log')
plt.savefig('fig/bbb.png',bbox_inches="tight",pad_inches=0.05)

plt.figure(figsize=[5,5])
plt.plot(freq[:int(n/2-1)],fftv,label='before')
plt.xscale('log')
plt.yscale('log')
plt.savefig('fig/ccc.png',bbox_inches="tight",pad_inches=0.05)
