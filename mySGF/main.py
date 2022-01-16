import numpy as np
import search,coordinate

input_csv = np.loadtxt('data/acc0.txt')
acc0 = input_csv[:,1]
time0 = input_csv[:,0]
dt = 0.01

args = search.search(acc0,dt,nmax=10)
search.output(*args,time0)
U,xy1,xy2,xy_hypo = args

c = coordinate.c
print('xy_hypo',xy_hypo)
print('pos_hypo',c.pos(*xy_hypo))