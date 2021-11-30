import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from myPack.constitution import *


ndiv = 100
total_time = 20 + 1
nseq = total_time * ndiv
amp = 3
x = np.zeros(nseq)
# points = np.random.normal(0,1,2*total_time)
points = np.random.rand(2*total_time)*2 - 1
points[0] = 0
points = points / np.max(np.abs(points))
div = math.floor(ndiv/2)
for n in range(2*total_time-2):
    x[ndiv+n*div:ndiv+(n+1)*div] = np.linspace(points[n],points[n+1],div)
x *= amp


# linear1 = Linear(2)
# linear2 = Linear(5)

# dict1 = {'k1':3,'k2':1,'dyield':1}
# dict2 = {'k1':5,'k2':3,'dyield':0.5}
# slip11 = Slip(**dict1)
# slip12 = Slip(**dict1)
# slip13 = Slip(**dict1)
# slip2 = Slip(**dict2)
# bilinear = Bilinear(**dict2)

# model = Combined([[slip11,slip2],[slip12,bilinear],[slip13,linear1]])
# model = Slip_Bilinear(1,0.05,1,0.85)
model = Slip_Bilinear(1,0.05,1,1)
# f = np.zeros([3,nseq])
for i in range(nseq):
    # f[:,i] = model.sheer(x[i]*np.ones(3))
    # model.push()

    model.sheer(x[i])
    model.push()

fig,ax = plt.subplots(figsize=(5,6))
# ax.plot(x,f[0],label='slip1+slip2') #label
# ax.plot(x,f[1],label='slip1+bilinear') #label
# ax.plot(x,f[2],label='slip1+linear1') #label
# ax.plot(x,model.model[0][0].F,label='slip1') #label
# ax.plot(x,model.model[0][1].F,label='slip2') #label
# ax.plot(x,model.model[1][0].F,label='slip') #label
# ax.plot(x,model.model[1][1].F,label='bilinear') #label
# ax.plot(x,model.model[2][0].F,label='linear2') #label

ax.plot(x,model.F,label='bilinear') #label
ax.legend()
ax.grid()
# ax.semilogx()
fig.savefig('constitution.png')
plt.close(fig)


fig = plt.figure(figsize=(5,6))
ims = []
k = 500
y = model.F
ni = int(np.floor(nseq/k))
for i in range(k):
    start = (i-3)*ni if i-3>0 else 0
    end = i*ni
    im1 = plt.plot(x[:end],y[:end],color='black',lw=1)
    im2 = plt.plot(x[start:end],y[start:end],color='red',lw=2)
    im3 = plt.plot(x[end],y[end],marker='.',markersize=10,color='red')

    ims.append(im1+im2+im3)

ani = animation.ArtistAnimation(fig,ims,interval=20)
ani.save('constitution.gif')
