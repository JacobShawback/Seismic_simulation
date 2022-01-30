from ast import Pass
from os import getuid
from traceback import print_tb
from numpy.lib.twodim_base import fliplr
from numba import jit,njit
import numpy as np
import math,pickle,time
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.fftpack import fft,ifft,fftfreq
import parameter,coordinate,output_format
from search_fft import Dataset,output,add_zeros,delete_zeros

class PrintTime:
    def __init__(self):
        self.k = 1
        self.t0 = time.time()

    def print(self,msg=None):
        t = time.time()-self.t0
        msg = '' if msg is None else f'[{msg}]   '
        print(f'{msg}output:{self.k} ({t:.1f} sec)')

ptime = PrintTime()

input_csv = np.loadtxt('data/acc0.txt')
acc = input_csv[:,1]
t0 = input_csv[:,0]
dt = 0.01
# acc,t0,n_add = add_zeros(acc,t0,dt,k=1)

d = Dataset(load=True)
u,U,freq = d.egf(acc,t0,dt)
ptime.print('egf')
# u,_ = delete_zeros(np.real(np.fft.ifft(u)),t0,n_add)
# U,_ = delete_zeros(np.real(np.fft.ifft(U)),t0,n_add)
# acc,t0 = delete_zeros(acc,t0,n_add)
# ptime.print('dlt')
# u,U = np.fft.fft(u),np.fft.fft(U)
# freq = np.fft.fftfreq(len(u),dt)
# ptime.print('fft')
output(acc,u,U,freq,d.xy1,d.xy2,d.hypo,t0,dt,n_add=None)
ptime.print('end')

c = coordinate.c
print('xy_hypo',d.hypo)
print('pos_hypo',c.pos(*d.hypo))