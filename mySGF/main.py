import numpy as np
import search,coordinate

def Interpolation(acc0,time0=None,dt0=0.01,div=10):
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

# input_csv = np.loadtxt('data/acc0.txt')
input_csv = np.loadtxt('data/acc02.txt')
acc = input_csv[:,1]
time = input_csv[:,0]
dt = 0.01
# acc,time,dt = Interpolation(acc,time,dt,div=4)

d = search.search(acc,time,dt,nmax=5)

c = coordinate.c
print('xy_hypo',d.hypo)
print('pos_hypo',c.pos(*d.hypo))