import numpy as np
import coordinate
from search_fft import search
from scipy.fftpack import fft, ifft, fftfreq


def drop(wave,dt,low=0.2):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        if np.abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif np.abs(f) < low:
            w2[i] = w[i] * (np.abs(f)-low0)/(low-low0)
        elif np.abs(f) <= high:
            w2[i] = w[i]
        elif np.abs(f) <= high0:
            w2[i] = w[i] * (np.abs(f)-high0)/(high-high0)
        else:
            w2[i] = 0.0 + 0.0j
    wave = np.real(np.fft.ifft(w2))
    return wave

input_csv = np.loadtxt('data/acc0.txt')
acc = input_csv[:,1]
time = input_csv[:,0]
dt = 0.01

d = search(acc,time,dt,n=5,div=1)
d.save()

c = coordinate.c
print('xy_hypo',d.hypo)
print('pos_hypo',c.pos(*d.hypo))